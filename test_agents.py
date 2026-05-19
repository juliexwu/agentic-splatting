import ffmpeg
import pathlib
import pycolmap
import open3d as o3d
import os
import subprocess
import yaml
import argparse

from gsplat.rendering import rasterization
from typing import TypedDict
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

# instruct necessary for following yaml str
ACTOR_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct" 
CRITIC_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

actor_client = InferenceClient(model=ACTOR_MODEL, token=os.getenv("HF_TOKEN"))
critic_client = InferenceClient(model=CRITIC_MODEL, token=os.getenv("HF_TOKEN"))


# This is where we choose what information comes in and out of the agents
class State(TypedDict):
    feedback: str
    approved: bool
    iteration: int
    video_name: str
    video_path: str
    initial_point_cloud_path: str
    config_path: str
    result_dir: str
    input_mode: int


def call_llm(prompt: str) -> str:
    response = actor_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def call_vlm(prompt: str, image_paths: list[str]) -> str:
    import base64
    content = []
    for path in image_paths:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text": prompt})
    response = critic_client.chat.completions.create(
        messages=[{"role": "user", "content": content}],
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()


#### TODO: need to implement rasterization to turn GS to 2d novel view imgs 
def get_render_images(result_dir: str, max_images: int = 4) -> list[str]:
    result_dir = pathlib.Path(result_dir)
    for subdir in [result_dir / "renders", result_dir]:
        images = sorted(subdir.glob("*.png"))[:max_images]
        if images:
            return [str(p) for p in images]
    return []


def extract_frames(video_path, output_dir, fps=2):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        ffmpeg.input(str(video_path))
        .filter("fps", fps=fps)
        .output(str(output_dir / "frame_%05d.jpg"), **{"q:v": 2})
        .run(overwrite_output=True)
    )


def images_to_point_cloud(
    image_dir,
    output_dir,
    match_method="exhaustive",
    dense=False,
    use_gpu=True,
):
    image_dir = pathlib.Path(image_dir)
    output_dir = pathlib.Path(output_dir)
    database_path = output_dir / "database.db"
    mvs_path = output_dir / "mvs"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu

    try:
        pycolmap.extract_features(database_path, image_dir, device=device)
    except ValueError: 
        print("[!] CUDA SIFT unavailable, retrying on CPU.")
        device = pycolmap.Device.cpu
        pycolmap.extract_features(database_path, image_dir, device=device)

    if match_method == "sequential":
        pycolmap.match_sequential(database_path, device=device)
    elif match_method == "exhaustive":
        pycolmap.match_exhaustive(database_path, device=device)
    else:
        raise ValueError("match_method must be 'sequential' or 'exhaustive'")

    maps = pycolmap.incremental_mapping(database_path, image_dir, output_dir)
    if not maps:
        raise RuntimeError("Reconstruction failed — check image overlap and quality")

    reconstruction = maps[0]
    reconstruction.write(output_dir)
    reconstruction.export_PLY(output_dir / "sparse.ply")

    if dense:
        mvs_path.mkdir(exist_ok=True)
        pycolmap.undistort_images(mvs_path, output_dir, image_dir)
        pycolmap.patch_match_stereo(mvs_path)
        pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)

    return reconstruction


# Multi-agent gsplat pipeline
def run_gsplat_training(data_dir, output_dir, config_path):
    # 1. Load config/hyperparameters written by your Coder Agent
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 2. Build the command-line argument array dynamically
    cmd = [
        "python",
        "gsplat/examples/simple_trainer.py",
        "default",
        "--data_dir",
        str(data_dir),
        "--result_dir",
        str(output_dir),
        "--max_steps",
        str(cfg["training"]["max_steps"]),
        "--strategy.grow_grad2d",
        str(cfg["training"]["densify_grad_threshold"]),
        "--strategy.prune_opa",
        str(cfg["training"]["opacity_cull_threshold"]),
        "--data_factor",
        "1",
        "--disable_viewer",  # Kept headless for the autonomous pipeline
        "--save_ply",
    ]

    print("Starting gsplat training with agent configs...")
    subprocess.run(cmd, check=True)


def extract_eval_metrics(result_dir):
    """
    Finds the latest mathematical logging outputs from the training run.
    Most configurations output metrics directly to a final json or tensorboard file.
    """
    result_dir = pathlib.Path(result_dir)
    # Mocking standard baseline evaluation dictionary metrics
    # In practice, you would parse the training log file generated by gsplat
    metrics = {"psnr": 24.5, "ssim": 0.82, "lpips": 0.18}
    return metrics


def mock_vlm_critic(metrics, current_loop_iteration):
    """
    Simulates the VLM feedback step. For the first few loops, it identifies
    rendering errors and feeds specific structural critiques back to the Actor.
    """
    print("[+] Packaging diagnostic frames and metrics for VLM Critic evaluation...")

    # Hardcoded simulation behavior matching typical VLM visual assertions
    if current_loop_iteration == 1:
        feedback = {
            "status": "REJECTED",
            "critique": "The scene contains heavy floating cloud artifacts in the background. Texture sharpness is adequate, but structural borders are muddy.",
            "recommendation": "Increase the opacity_cull_threshold to clean out background fog, and slightly lower the densify_grad_threshold.",
        }
    elif current_loop_iteration == 2:
        feedback = {
            "status": "REJECTED",
            "critique": "Background clouding has drastically improved. However, complex central surface details are blurry.",
            "recommendation": "Lower your densify_grad_threshold further to encourage tighter geometric splits, and extend max_steps.",
        }
    else:
        feedback = {
            "status": "ACCEPTED",
            "critique": "Visual elements are sharp and clean. Background floaters have been culled successfully. Convergence reached.",
            "recommendation": "None",
        }

    return feedback


def mock_actor_agent_rewrite(vlm_feedback, config_path):
    """
    Simulates an LLM Coder Engineer parsing the VLM recommendation
    and rewriting the hyperparameter config file safely.
    """
    print(f"[+] Actor Agent is parsing feedback and updating config file properties...")

    # Read current configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    rec = vlm_feedback["recommendation"]

    # Rule-engine fallback simulating the logical parameter adjustment
    if "opacity_cull_threshold" in rec:
        cfg["training"]["opacity_cull_threshold"] = round(
            cfg["training"]["opacity_cull_threshold"] + 0.02, 3
        )
    if "densify_grad_threshold" in rec:
        cfg["training"]["densify_grad_threshold"] = round(
            cfg["training"]["densify_grad_threshold"] - 0.00005, 5
        )
    if "extend max_steps" in rec:
        cfg["training"]["max_steps"] += 2000

    # Write out updated configuration for the next loop pass
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)


# def render_diagnostic_views(checkpoint_path, output_image_dir):
#     # Load the trained gaussian parameters (xyz, opacities, scales, shs)
#     ckpt = torch.load(checkpoint_path)

#     # Define a set of camera paths (e.g., 4 cardinal directions + 1 nadir overhead view)
#     diagnostic_cameras = generate_eval_cameras()

#     for idx, camera in enumerate(diagnostic_cameras):
#         # Render the novel viewpoint using gsplat's rapid rasterizer
#         render_results = rasterization(
#             means3d=ckpt["means3d"],
#             quats=ckpt["quats"],
#             scales=ckpt["scales"],
#             opacities=ckpt["opacities"],
#             colors=ckpt["colors"],
#             viewmats=camera.viewmat,
#             K=camera.K,
#             width=camera.width,
#             height=camera.height
#         )
#         # Save frame out to disk for the VLM to ingest
#         save_image(render_results["image"], f"{output_image_dir}/eval_view_{idx}.png")


def prefiltering_agent(state: State) -> State:
    print("Prefiltering Agent running...")
    image_dir = f"input/images/{state['video_name']}"
    output_dir = f"output/{state['video_name']}"
    
    if state["input_mode"] == 0:
        extract_frames(state["video_path"], image_dir)

        images_to_point_cloud(
            image_dir=image_dir,
            output_dir=output_dir,
            match_method="sequential",
            dense=False,
        )

    else:
        images_to_point_cloud(
            image_dir=image_dir,
            output_dir=output_dir,
            match_method="sequential",
            dense=False,
        )

    return {**state, "initial_point_cloud_path": output_dir}


def actor_agent(state: State) -> State:
    print(f"[Actor Agent] Iteration {state['iteration']} - updating config and launching training...")
    config_path = state["config_path"]

    if state["iteration"] == 0 or not os.path.exists(config_path):
        cfg = {
            "training": {
                "max_steps": 2000,
                "densify_grad_threshold": 0.0002,
                "opacity_cull_threshold": 0.05,
            }
        }
    else:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        current_config_str = yaml.dump(cfg)

        prompt = (
            "You are an expert ML Engineer specializing in 3D Gaussian Splatting.\n"
            "A visual critic has reviewed the latest render and provided this feedback:\n\n"
            f"{state['feedback']}\n\n"
            "Current training config:\n"
            f"{current_config_str}\n"
            "Rewrite the config YAML with adjusted hyperparameters to address the critique.\n"
            "Reply with ONLY valid YAML in the exact same structure. No explanation.\n"
            "training:\n"
            "  max_steps: <int>\n"
            "  densify_grad_threshold: <float>\n"
            "  opacity_cull_threshold: <float>"
        )

        response = call_llm(prompt)
        try:
            parsed = yaml.safe_load(response)
            if isinstance(parsed, dict) and "training" in parsed:
                cfg = parsed
            else:
                print("[Actor] LLM response missing 'training' key, keeping current config.")
        except yaml.YAMLError:
            print("[Actor] Failed to parse LLM YAML response, keeping current config.")

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    run_gsplat_training(
        data_dir=state["initial_point_cloud_path"],
        output_dir=state["result_dir"],
        config_path=config_path,
    )

    return {**state}


def critic_agent(state: State) -> State:
    print(f"[Critic Agent] Evaluating results from iteration {state['iteration']}...")
    metrics = extract_eval_metrics(state["result_dir"])
    print(f"[+] Metrics: PSNR={metrics['psnr']}, SSIM={metrics['ssim']}, LPIPS={metrics['lpips']}")

    image_paths = get_render_images(state["result_dir"]) # TODO: 
    print(f"[+] Found {len(image_paths)} render(s) for visual evaluation.")

    prompt = (
        "You are an expert Machine Learning Critic specializing in 3D Gaussian Splatting.\n"
        "Analyze the rendered novel viewpoints above alongside these training metrics:\n\n"
        f"PSNR: {metrics['psnr']}, SSIM: {metrics['ssim']}, LPIPS: {metrics['lpips']}\n\n"
        "Look specifically for:\n"
        "1. Floaters / clouding (foggy artifacts in empty space)\n"
        "2. Blurring (loss of fine surface detail)\n"
        "3. Anisotropic stretching (needle-shaped Gaussians)\n\n"
        "Reply in exactly this format:\n"
        "STATUS: ACCEPTED or REJECTED\n"
        "CRITIQUE: <one sentence assessment>\n"
        "RECOMMENDATION: <specific hyperparameter adjustments if REJECTED, else None>"
    )

    if image_paths:
        response = call_vlm(prompt, image_paths)
    else:
        print("[Critic] No render images found, falling back to metrics-only evaluation.")
        response = call_llm(prompt)

    approved = "STATUS: ACCEPTED" in response
    print(f"[Critic]: {response}")

    return {
        **state,
        "feedback": response,
        "approved": approved,
        "iteration": state["iteration"] + 1,
    }


# Put termination here
def route(state: State) -> str:
    if state["approved"] or state["iteration"] >= 3:
        return "end"
    return "actor_agent"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Gaussian Splatting Pipeline")
    parser.add_argument(
        "--mode",
        type=int,
        required=True,
        choices=[0, 1],
        help="0 for video, 1 for images",
    )
    args = parser.parse_args()

    graph = StateGraph(State)
    graph.add_node("prefiltering_agent", prefiltering_agent)
    graph.add_node("actor_agent", actor_agent)
    graph.add_node("critic_agent", critic_agent)

    graph.set_entry_point("prefiltering_agent")
    graph.add_edge("prefiltering_agent", "actor_agent")
    graph.add_edge("actor_agent", "critic_agent")
    graph.add_conditional_edges(
        "critic_agent",
        route,
        {
            "actor_agent": "actor_agent",
            "end": END,
        },
    )

    pipeline = graph.compile()

    result = pipeline.invoke(
        {
            "feedback": "",
            "approved": False,
            "iteration": 0,
            "video_path": "input/video.mp4",
            "video_name": "video",
            "initial_point_cloud_path": "",
            "config_path": "output/config.yaml",
            "result_dir": "output/splat_results",
            "input_mode": args.mode,
        }
    )

    exit(0) ### remove if you want to view 
    print("\n[+] Launching interactive viewer for final output...")
    colmap_out_dir = f"output/{result['video_name']}"
    pcd = o3d.io.read_point_cloud(f"{colmap_out_dir}/sparse.ply")
    o3d.visualization.draw_geometries([pcd])

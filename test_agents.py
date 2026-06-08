import ffmpeg
import pathlib
import pycolmap
import open3d as o3d
import os
import shutil
import subprocess
import yaml
import argparse
import torch

from PIL import Image
from gsplat.rendering import rasterization
from typing import TypedDict
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from google import genai
from pydantic import BaseModel

load_dotenv()

# ACTOR_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# CRITIC_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# actor_client = InferenceClient(model=ACTOR_MODEL, token=os.getenv("HF_TOKEN"))
# critic_client = InferenceClient(model=CRITIC_MODEL, token=os.getenv("HF_TOKEN"))

client = genai.Client(api_key=os.getenv("GEMINI_KEY"))
GEMINI_MODEL = "gemma-4-31b-it"
GEMINI_FALLBACK = "gemma-4-26b-a4b-it"


class TrainingConfig(BaseModel):
    max_steps: int
    opacity_cull_threshold: float
    # learning_rate_position: float
    # learning_rate_feature: float
    # learning_rate_opacity: float
    # learning_rate_scaling: float
    # learning_rate_rotation: float
    # lr_position_final: float
    # lr_delay_steps: int
    # lr_delay_mult: float

    densify_grad_threshold: float
    # densify_from_step: int
    densify_until_step: int
    # densification_interval: int
    densify_size_threshold: float
    max_num_gaussians: int


# class TrainingConfig(BaseModel):
#     max_steps: int
#     densify_grad_threshold: float
#     opacity_cull_threshold: float


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
    skip_preprocessing: bool


def call_llm(prompt: str) -> str:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    return response.text.strip()


def call_vlm(prompt: str, image_paths: list[str]) -> str:
    contents = []

    for path in image_paths:
        img = Image.open(path)
        contents.append(img)

    contents.append(prompt)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
        )
    except:
        print("Main VLM failed, using fallback model...")
        response = client.models.generate_content(
            model=GEMINI_FALLBACK,
            contents=contents,
        )

    return response.text.strip()


def _generate_eval_cameras(
    scene_center: torch.Tensor,
    radius: float,
    width: int,
    height: int,
    device: torch.device,
    n_orbit: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate evaluation camera viewmats (world-to-cam, [C, 4, 4]) and
    intrinsics Ks ([C, 3, 3]) for novel-view diagnostics.

    Produces `n_orbit` cameras evenly spaced in azimuth around `scene_center`
    at elevation ~30°, plus one overhead (nadir) view.

    Returns:
        viewmats: [C, 4, 4] float32 world-to-cam transforms
        Ks:       [C, 3, 3] float32 pinhole intrinsics
    """
    import torch
    import math

    C = n_orbit + 1  # orbit views + nadir
    viewmats = torch.zeros(C, 4, 4, dtype=torch.float32, device=device)
    elev_rad = math.radians(30)

    for i in range(n_orbit):
        azimuth = 2 * math.pi * i / n_orbit
        # Camera position on a sphere around the scene centre
        cam_pos = scene_center + torch.tensor(
            [
                radius * math.cos(elev_rad) * math.cos(azimuth),
                radius * math.sin(elev_rad),
                radius * math.cos(elev_rad) * math.sin(azimuth),
            ],
            dtype=torch.float32,
            device=device,
        )
        # Look-at: forward = scene_center - cam_pos, world up = (0,1,0)
        forward = torch.nn.functional.normalize(scene_center - cam_pos, dim=0)
        world_up = torch.tensor([0.0, 1.0, 0.0], device=device)
        right = torch.nn.functional.normalize(
            torch.linalg.cross(forward, world_up), dim=0
        )
        up = torch.linalg.cross(right, forward)

        # Rotation matrix (rows = right, up, -forward in camera convention)
        R = torch.stack([right, up, -forward], dim=0)  # [3, 3]
        t = -R @ cam_pos  # [3]

        viewmats[i, :3, :3] = R
        viewmats[i, :3, 3] = t
        viewmats[i, 3, 3] = 1.0

    # Nadir (top-down) view: camera directly above, looking straight down
    nadir_pos = scene_center + torch.tensor(
        [0.0, radius, 0.0], dtype=torch.float32, device=device
    )
    forward_nadir = torch.nn.functional.normalize(scene_center - nadir_pos, dim=0)
    right_nadir = torch.tensor([1.0, 0.0, 0.0], device=device)
    up_nadir = torch.linalg.cross(right_nadir, forward_nadir)

    R_nadir = torch.stack([right_nadir, up_nadir, -forward_nadir], dim=0)
    t_nadir = -R_nadir @ nadir_pos
    viewmats[n_orbit, :3, :3] = R_nadir
    viewmats[n_orbit, :3, 3] = t_nadir
    viewmats[n_orbit, 3, 3] = 1.0

    # Simple pinhole intrinsics: 60° horizontal FoV
    fx = width / (2 * math.tan(math.radians(60) / 2))
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    K = torch.tensor(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device
    )
    Ks = K.unsqueeze(0).expand(C, -1, -1).contiguous()  # [C, 3, 3]

    return viewmats, Ks


def render_diagnostic_views(
    checkpoint_path: str,
    output_image_dir: str,
    width: int = 800,
    height: int = 600,
) -> list[str]:
    """
    Load a gsplat checkpoint, render novel-view diagnostic images with
    gsplat's rasterization kernel, and save them as PNGs.

    Args:
        checkpoint_path: Path to the .ckpt file saved by simple_trainer.py.
        output_image_dir: Directory where eval_view_N.png frames are written.
        width:  Render width in pixels.
        height: Render height in pixels.

    Returns:
        List of absolute paths to the saved PNG files.
    """
    import torch
    from torchvision.utils import save_image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = pathlib.Path(output_image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load checkpoint
    # ------------------------------------------------------------------ #
    ckpt = torch.load(checkpoint_path, map_location=device)

    # simple_trainer.py saves a nested dict under the key "splats"
    splats = ckpt.get("splats", ckpt)

    means = splats["means"].to(device)  # [N, 3]
    quats = splats["quats"].to(device)  # [N, 4]  wxyz
    scales = torch.exp(splats["scales"]).to(device)  # [N, 3]  log-space in ckpt
    opacities = torch.sigmoid(splats["opacities"]).to(device)  # [N]

    # Colours: may be raw RGB [N,3] or SH coefficients [N, K, 3]
    colors_raw = splats.get("sh0", splats.get("colors", None))
    if colors_raw is None:
        raise KeyError("Checkpoint has neither 'sh0' nor 'colors' key.")
    colors = colors_raw.to(device)  # let rasterization handle SH if needed

    # ------------------------------------------------------------------ #
    # 2. Build diagnostic cameras around the scene centroid
    # ------------------------------------------------------------------ #
    scene_center = means.mean(dim=0)  # [3]
    # Use the 90th-percentile distance as the orbit radius so the splat fits
    dists = torch.linalg.norm(means - scene_center, dim=-1)
    radius = float(torch.quantile(dists, 0.90).item()) * 2.5
    radius = max(radius, 1.0)  # guard against degenerate scenes

    viewmats, Ks = _generate_eval_cameras(
        scene_center=scene_center,
        radius=radius,
        width=width,
        height=height,
        device=device,
    )

    # ------------------------------------------------------------------ #
    # 3. Rasterize
    # ------------------------------------------------------------------ #
    # rasterization() expects colors shaped [N, D] for plain RGB (D=3)
    # or [N, K, 3] for SH; it handles the SH→RGB conversion internally
    # when sh_degree is passed.
    sh_degree = None
    if colors.ndim == 3:
        # SH coefficients: [N, K, 3] — infer degree from K
        K_coeffs = colors.shape[1]
        sh_degree = int(K_coeffs**0.5) - 1  # 1→deg0, 4→deg1, 9→deg2 …

    with torch.no_grad():
        render_colors, render_alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            sh_degree=sh_degree,
            render_mode="RGB",
            packed=True,
        )
    # render_colors: [C, H, W, 3] in [0, 1]

    # ------------------------------------------------------------------ #
    # 4. Save frames
    # ------------------------------------------------------------------ #
    saved_paths: list[str] = []
    for idx in range(render_colors.shape[0]):
        frame = render_colors[idx]  # [H, W, 3]
        frame = frame.permute(2, 0, 1)  # [3, H, W]  — torchvision convention
        frame = frame.clamp(0.0, 1.0)
        out_path = output_dir / f"eval_view_{idx:02d}.png"
        save_image(frame, str(out_path))
        saved_paths.append(str(out_path))
        print(f"  [render] Saved {out_path}")

    return saved_paths


def get_render_images(result_dir: str, max_images: int = 4) -> list[str]:
    """
    Return paths to render images produced by simple_trainer's eval pass.

    simple_trainer saves side-by-side (GT | render) images to:
      {result_dir}/renders/val_{i:04d}.png

    These are written automatically when --eval_steps fires.  We prefer the
    trainer's own renders (they show GT alongside the render, which is more
    useful for the critic) and only fall back to render_diagnostic_views() if
    the renders/ directory is empty (e.g. eval hasn't run yet).

    Args:
        result_dir:  The gsplat output directory.
        max_images:  Maximum number of images to return to the critic.

    Returns:
        List of PNG paths (up to `max_images`), or [] if nothing is available.
    """
    result_dir = pathlib.Path(result_dir)
    render_dir = result_dir / "renders"

    # ------------------------------------------------------------------ #
    # 1. Use trainer's own eval renders (val_NNNN.png) — best option
    # ------------------------------------------------------------------ #
    if render_dir.exists():
        val_renders = sorted(render_dir.glob("val_*.png"))
        if val_renders:
            # Space them out so we get a spread across the val set
            step = max(1, len(val_renders) // max_images)
            chosen = val_renders[::step][:max_images]
            print(
                f"[render] Using {len(chosen)} trainer eval renders from {render_dir}"
            )
            return [str(p) for p in chosen]

    # ------------------------------------------------------------------ #
    # 2. Fall back to rendering novel views from the checkpoint ourselves
    # ------------------------------------------------------------------ #
    ckpt_dir = result_dir / "ckpts"
    checkpoint_path = None

    if ckpt_dir.exists():
        ckpts = (
            list(ckpt_dir.glob("ckpt_*_rank0.pt"))
            or list(ckpt_dir.glob("*.pt"))
            or list(ckpt_dir.glob("*.ckpt"))
        )
        if ckpts:
            checkpoint_path = max(ckpts, key=lambda p: p.stat().st_mtime)

    if checkpoint_path is not None:
        print(
            f"[render] No trainer renders found; rendering from checkpoint: {checkpoint_path}"
        )
        try:
            saved = render_diagnostic_views(
                checkpoint_path=str(checkpoint_path),
                output_image_dir=str(render_dir),
            )
            return saved[:max_images]
        except Exception as exc:
            print(f"[render] Rendering failed ({exc}), no images available.")

    print("[render] No renders or checkpoints found.")
    return []


def extract_frames(video_path, output_dir, fps=2, cuda=False):  # cuda=False for gpu
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cuda:
        (
            ffmpeg.input(str(video_path), hwaccel="cuda", hwaccel_output_format="cuda")
            .filter("hwdownload")
            .filter("format", "nv12")
            .filter("fps", fps=fps)
            .output(str(output_dir / "frame_%05d.jpg"), **{"q:v": 2})
            .run(overwrite_output=True)
        )

        for frame_path in sorted(output_dir.glob("frame_*.jpg")):
            img = Image.open(frame_path)
            img.transpose(Image.FLIP_TOP_BOTTOM).save(frame_path)

    else:
        (
            ffmpeg.input(str(video_path))
            .filter("fps", fps=fps)
            .output(str(output_dir / "frame%05d.jpg"), **{"q:v": 2})
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
    if database_path.exists():
        database_path.unlink()

    device = pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu

    if dsmlp:
        options = pycolmap.FeatureExtractionOptions()
        options.num_threads = 6
        options.max_image_size = 3200

        pycolmap.extract_features(database_path, image_dir, extraction_options=options, device=device)
    else:
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

    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    maps = pycolmap.incremental_mapping(database_path, image_dir, sparse_dir)
    if not maps:
        raise RuntimeError("Reconstruction failed — check image overlap and quality")

    reconstruction = maps[0]
    reconstruction.export_PLY(output_dir / "sparse.ply")

    if dense:
        mvs_path.mkdir(exist_ok=True)
        pycolmap.undistort_images(mvs_path, output_dir, image_dir)
        pycolmap.patch_match_stereo(mvs_path)
        pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)

    return reconstruction


def run_gsplat_training(data_dir, output_dir, config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    max_steps = cfg["training"]["max_steps"]

    # --save_steps triggers BOTH the checkpoint .pt write AND the train stat JSON.
    # simple_trainer fires it when: step in [i - 1 for i in cfg.save_steps]
    # So pass max_steps directly; the trainer will save at step max_steps-1 (the
    # last step) AND always saves at the very last step regardless.
    #
    # --eval_steps triggers the val pass (PSNR/SSIM/LPIPS) with the same
    # off-by-one logic: eval fires at step (eval_step - 1).
    # We pass max_steps so eval runs at the end of training.

    cmd = [
        "python",
        "gsplat/examples/simple_trainer.py",
        "default",
        "--data_dir",
        str(data_dir),
        "--result_dir",
        str(output_dir),
        "--max_steps",
        str(max_steps),
        "--save_steps",
        str(max_steps),
        "--eval_steps",
        str(max_steps),
        "--strategy.grow_grad2d",
        str(cfg["training"]["densify_grad_threshold"]),
        "--strategy.prune_opa",
        str(cfg["training"]["opacity_cull_threshold"]),
        "--strategy.refine_stop_iter",
        str(cfg["training"]["densify_until_step"]),
        "--strategy.cap_max",
        str(cfg["training"]["max_num_gaussians"]),
        "--strategy.prune_scale3d",
        str(cfg["training"]["densify_size_threshold"]),
        "--data_factor",
        "1",
        "--disable_viewer",
        "--render_traj_factor",
        "1",
        "--render_traj_max_frames",
        "250",
        "--save_ply",
    ]

    print("Starting gsplat training with agent configs...")
    subprocess.run(cmd, check=True)


def extract_eval_metrics(result_dir: str) -> dict:
    """
    Parse evaluation metrics written by simple_trainer.py.

    simple_trainer writes two kinds of JSON into {result_dir}/stats/:

      val_step{step:05d}.json   — eval pass: contains psnr, ssim, lpips,
                                   ellipse_time, num_GS
      train_step{step:05d}.json — training pass: contains mem, ellipse_time,
                                   num_GS  (NO loss, NO psnr)

    We scan all JSON files for the one with the richest metrics (any file that
    has psnr wins; otherwise we report what we can from training stats).
    """
    import json

    result_dir = pathlib.Path(result_dir)
    stats_dir = result_dir / "stats"

    fallback = {"psnr": None, "ssim": None, "lpips": None, "num_gs": None}

    if not stats_dir.exists():
        print("[metrics] stats/ directory not found — returning empty metrics.")
        return fallback

    def _load(path):
        with open(path) as f:
            return json.load(f)

    # ── 1. Prefer any file that has PSNR (written by eval pass) ──────────
    # simple_trainer names them val_step*.json but scan all to be safe.
    all_json = sorted(stats_dir.glob("*.json"))
    if not all_json:
        print("[metrics] No JSON stats files found — returning empty metrics.")
        return fallback

    eval_candidates = []
    for p in all_json:
        try:
            d = _load(p)
            if "psnr" in d or "PSNR" in d:
                eval_candidates.append((p, d))
        except Exception:
            pass

    if eval_candidates:
        # Pick the one with the highest step number (latest mtime as tiebreak)
        latest_path, data = max(eval_candidates, key=lambda t: t[0].stat().st_mtime)
        metrics = {
            "psnr": data.get("psnr", data.get("PSNR")),
            "ssim": data.get("ssim", data.get("SSIM")),
            "lpips": data.get("lpips", data.get("LPIPS")),
            "num_gs": data.get("num_GS", data.get("num_gs")),
        }
        print(f"[metrics] Loaded eval metrics from {latest_path.name}: {metrics}")
        return metrics

    # ── 2. No eval stats yet — extract what we can from training stats ───
    # Training JSONs have: {"mem": float, "ellipse_time": float, "num_GS": int}
    train_candidates = []
    for p in all_json:
        try:
            d = _load(p)
            if "num_GS" in d or "mem" in d:
                train_candidates.append((p, d))
        except Exception:
            pass

    if train_candidates:
        latest_path, data = max(train_candidates, key=lambda t: t[0].stat().st_mtime)
        metrics = {
            **fallback,
            "num_gs": data.get("num_GS"),
            "mem_gb": data.get("mem"),
            "ellipse_time": data.get("ellipse_time"),
        }
        print(
            f"[metrics] No eval stats yet; loaded train stats from {latest_path.name}: {metrics}"
        )
        return metrics

    print("[metrics] No parseable stats files found — returning empty metrics.")
    return fallback


def prefiltering_agent(state: State) -> State:
    print("Prefiltering Agent running...")

    output_dir = f"output/{state['video_name']}"
    image_dir = f"{output_dir}/images"

    if state["skip_preprocessing"]:
        print("[Prefiltering] Skipping frame extraction and COLMAP.")

        return {
            **state,
            "initial_point_cloud_path": output_dir,
        }

    if state["input_mode"] == 0:
        extract_frames(state["video_path"], image_dir, cuda=(not dsmlp))
        images_to_point_cloud(
            image_dir=image_dir,
            output_dir=output_dir,
            match_method="sequential",
            dense=False,
            use_gpu=(not dsmlp),
        )
    else:
        images_to_point_cloud(
            image_dir=image_dir,
            output_dir=output_dir,
            match_method="sequential",
            dense=False,
            use_gpu=(not dsmlp),
        )

    return {
        **state,
        "initial_point_cloud_path": output_dir,
    }


def actor_agent(state: State) -> State:
    print(
        f"[Actor Agent] Iteration {state['iteration']+1} - updating config and launching training..."
    )
    config_path = state["config_path"]

    DEFAULT_TRAINING_CONFIG = TrainingConfig(
        max_steps=2000,
        opacity_cull_threshold=0.05,
        # learning_rate_position=1.6e-4,
        # learning_rate_feature=2.5e-3,
        # learning_rate_opacity=5e-2,
        # learning_rate_scaling=5e-3,
        # learning_rate_rotation=1e-3,
        # lr_position_final=1.6e-6,
        # lr_delay_steps=0,
        # lr_delay_mult=0.01,
        densify_grad_threshold=0.0002,
        # densify_from_step=500,
        densify_until_step=15000,
        # densification_interval=100,
        densify_size_threshold=0.01,
        max_num_gaussians=500_000,
    )

    if state["iteration"] == 0 or not os.path.exists(config_path):
        cfg = {"training": DEFAULT_TRAINING_CONFIG.model_dump()}
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
            "Return updated hyperparameters in .json format to address the critique."
        )
        print(prompt)

        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": TrainingConfig,
                },
            )
            print(response)
            parsed = response.parsed
            print(parsed)
            cfg = {"training": parsed.model_dump()}
            print(f"[Actor] Updated config: {cfg['training']}")
        except Exception as e:
            print(response)
            print(f"[Actor] Structured output failed ({e}), keeping current config.")

    config_dir = os.path.dirname(config_path)
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    run_gsplat_training(
        data_dir=state["initial_point_cloud_path"],
        output_dir=state["result_dir"],
        config_path=config_path,
    )

    # Archive final checkpoint and PLY for this iteration
    result_dir = pathlib.Path(state["result_dir"])
    archive_dir = result_dir / f"iter_{state['iteration']}"
    archive_dir.mkdir(exist_ok=True)

    for src in sorted((result_dir / "ckpts").glob("*.pt")):
        shutil.copy2(src, archive_dir / src.name)
    for src in sorted((result_dir / "ply").glob("*.ply")):
        shutil.copy2(src, archive_dir / src.name)

    print(f"[Actor] Archived checkpoint and PLY to {archive_dir}")

    return {**state}


def critic_agent(state: State) -> State:
    print(f"[Critic Agent] Evaluating results from iteration {state['iteration']}...")
    metrics = extract_eval_metrics(state["result_dir"])

    def _fmt(v, decimals=4):
        if v is None:
            return "N/A"
        try:
            return f"{float(v):.{decimals}f}"
        except (TypeError, ValueError):
            return str(v)

    print(
        f"[+] Metrics: PSNR={_fmt(metrics['psnr'], 3)}, "
        f"SSIM={_fmt(metrics['ssim'], 4)}, "
        f"LPIPS={_fmt(metrics['lpips'], 4)}, "
        f"num_GS={metrics.get('num_gs', 'N/A')}"
    )

    image_paths = get_render_images(state["result_dir"])
    print(f"[+] Found {len(image_paths)} render(s) for visual evaluation.")

    # Build a metrics summary the critic LLM can reason about.
    # When PSNR is unavailable (eval not yet run) we report training diagnostics.
    if metrics.get("psnr") is not None:
        metrics_str = (
            f"PSNR: {_fmt(metrics['psnr'], 3)} dB (higher is better, target >27), "
            f"SSIM: {_fmt(metrics['ssim'], 4)} (higher is better, target >0.85), "
            f"LPIPS: {_fmt(metrics['lpips'], 4)} (lower is better, target <0.20), "
            f"Gaussian count: {metrics.get('num_gs', 'N/A')}"
        )
    else:
        metrics_str = (
            f"Eval metrics not yet available. "
            f"Training stats — Gaussian count: {metrics.get('num_gs', 'N/A')}, "
            f"GPU mem: {_fmt(metrics.get('mem_gb'), 2)} GB, "
            f"time per image: {_fmt(metrics.get('ellipse_time'), 2)}s"
        )

    prompt = (
        "You are an expert Machine Learning Critic specializing in 3D Gaussian Splatting.\n"
        "Analyze the rendered novel viewpoints above alongside these training metrics:\n\n"
        f"{metrics_str}\n\n"
        "Look specifically for:\n"
        "1. Floaters / clouding (foggy artifacts in empty space)\n"
        "2. Blurring (loss of fine surface detail)\n"
        "3. Anisotropic stretching (needle-shaped Gaussians)\n\n"
        "Reply in exactly this format:\n"
        "STATUS: ACCEPTED or REJECTED\n"
        "CRITIQUE: <one sentence assessment>\n"
        "RECOMMENDATION: <specific hyperparameter adjustments if REJECTED, else None>"
    )

    print(prompt)
    if image_paths:
        response = call_vlm(prompt, image_paths)
    else:
        print(
            "[Critic] No render images found, falling back to metrics-only evaluation."
        )
        response = call_llm(prompt)

    approved = "STATUS: ACCEPTED" in response
    print(f"[Critic]: {response}")

    return {
        **state,
        "feedback": response,
        "approved": approved,
        "iteration": state["iteration"] + 1,
    }


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
    parser.add_argument("--dsmlp", action='store_true', help="include this flag if running on DSMLP")

    parser.add_argument(
        "--skip_preprocessing",
        action="store_true",
        help="Skip frame extraction and COLMAP reconstruction",
    )
    args = parser.parse_args()

    global dsmlp
    dsmlp = args.dsmlp

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
            "skip_preprocessing": args.skip_preprocessing,
        }
    )

    exit(0)
    print("\n[+] Launching interactive viewer for final output...")
    colmap_out_dir = f"output/{result['video_name']}"
    pcd = o3d.io.read_point_cloud(f"{colmap_out_dir}/sparse.ply")
    o3d.visualization.draw_geometries([pcd])

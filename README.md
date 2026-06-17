# Agentic Splatting

 Multi-agent pipeline for video-to-3D Gaussian Splatting. A **Critic agent** (VLM) evaluates 2D rendered output and feeds structured visual feedback to an **Actor agent** (LLM coder) that rewrites the GS training hyperparameter config — looping until convergence.

```
Video → Frames → COLMAP (sparse reconstruct) → gsplat training → VLM Critic → Actor/Coder rewrite → repeat
```


## Setup

### 1. Clone this repo

```bash
git clone https://github.com/you/agentic-splatting.git
cd agentic-splatting
```

### 2. Create and activate the conda environment

```bash
conda create -n 252splat python=3.10 -y
conda activate 252splat
```

On UCSD DSMLP, the following may be necessary:
```bash
python3.10 -m ensurepip --default-pip
```

### 3. Install CUDA toolkit into the environment

- Match to the CUDA version your PyTorch build expects. Check with `nvcc --version` and cross-reference [PyTorch's compatibility table](https://pytorch.org/get-started/locally/).

- I used CUDA 12.4 on my machine but idk what the DSMLP has.

```bash
# Example for CUDA 11.8:
conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit -y
conda install -c conda-forge cxx-compiler ninja -y
```

### 4. Install PyTorch

Install the version matching your CUDA toolkit:

```bash
# Example for CUDA 12.4:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

On DSMLP, this may be necessary instead of the above:
```bash
# Example for CUDA 12.4:
python3.10 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 5. Install gcc 13
```bash
conda install -c conda-forge gcc=13 gxx=13 -n 252splat --force-reinstall
git submodule update --init --recursive
export CUDAHOSTCXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
```

On DSMLP, replace the first line with:
```bash
conda install -c conda-forge gcc=12 gxx=12 -n 252splat --force-reinstall
```

### 6. Clone and build gsplat locally

> **Why a local clone?** The `pip install gsplat` package does not include `examples/simple_trainer.py`, which this pipeline calls directly. The clone is required to get that script and all dependencies on disk. The `.so` CUDA extension is also compiled locally against your specific CUDA/PyTorch versions.

```bash
git clone https://github.com/nytseng/gsplat.git # fork bc update to pycolmap SceneManager
cd gsplat && pip install . --no-build-isolation --no-cache-dir
cd ..
```

A custom implementation of `agentic-splatting/gsplat/gsplat/strategy/default.py` is located at `default_custom.py`. In order to use the custom implementation, copy-paste teh contents of `default_custom.py` into `agentic-splatting/gsplat/gsplat/strategy/default.py`.

On DSMLP, the second line may need to be replaced with:
```bash
cd gsplat && python3.10 -m pip install . --no-build-isolation --no-cache-dir
```

Verify the build succeeded:

```bash
python -c "from gsplat.cuda import csrc; print('gsplat CUDA extension loaded successfully')"
python -c "from gsplat import rasterization; print('rasterization ready')"
```

### 7. Install remaining dependencies
Note: requirements.txt is for cu12 which is compatible with cu12.x

```bash
pip install --no-build-isolation -r requirements.txt
```

On DSMLP, this worked better instead:
```bash
python3.10 -m pip install --no-build-isolation -r requirements_dsmlp.txt

# To get Open3D to work:
# for libX11.so.6:
conda install -c conda-forge xorg-libx11
# for libGL.so.1:
conda install -c conda-forge xorg-libxext xorg-libxrender
conda install -c conda-forge freeglut

conda install -c conda-forge ffmpeg
```

---

## Project Structure
- Upload your input images or video to input/
- After running python test_agents.py, output/ should be populated as such.
```
agentic-splatting/
├── input/
│   ├── images/          # Extracted frames go here
│   └── zoo.mp4          # Example input video
├── output/
│   ├── database.db      # COLMAP feature database 
│   ├── sparse/           # COLMAP SfM reconstruction to be input to gsplat
│   ├── sparse.ply       # COLMAP sparse point cloud
│   ├── config.yaml      # Agent-managed training config
│   └── splat_results/   # gsplat training output
│       └── ckpts/        # final trained Gaussians (.pt)
│       └── ply/          # Gaussian point clouds for viewing (.ply)
│       └── ...        
├── gsplat/              # Local clone 
├── test_agents.py       # Main pipeline 
├── requirements.txt
└── README.md
```

---

## Usage

### Option A: Start from a video

Uncomment the `extract_frames` block in `test_agents.py` and run:

```bash
python test_agents.py
```

This will extract frames from `input/zoo.mp4` at 2 fps into `input/images/`, then exit. Re-comment the block and run again to proceed with the full pipeline.

### Option B: Start from pre-extracted frames

Place your images in `input/images/` and run:

```bash
python test_agents.py
```

The pipeline will:
1. Run COLMAP to generate camera poses and a sparse point cloud
2. Initialize a baseline training config
3. Enter the autonomous Actor-Critic loop (up to 5 iterations)
4. Launch an Open3D viewer on the final sparse point cloud

---

## Agent Loop

Each iteration of the loop:

| Step | Agent | Action |
|------|-------|--------|
| A | — | Train gsplat with current config |
| B | — | Extract eval metrics (PSNR, SSIM, LPIPS) |
| C | VLM Critic | Evaluate rendered frames, return structured feedback |
| D | Actor (LLM Coder) | Parse critique, rewrite `config.yaml` hyperparameters |

The loop exits when the Critic returns `ACCEPTED` or `MAX_LOOP_ATTEMPTS` (default: 5) is reached.

---

## Troubleshooting

**`ImportError: cannot import name '_ext' from 'gsplat.cuda'`**  
The extension was renamed. Use `from gsplat.cuda import csrc` — or verify with `python -c "import gsplat.cuda as c; print(dir(c))"`.

**`csrc.so` missing after install**  
The CUDA extension failed to compile. Check that `nvcc --version` and `torch.version.cuda` match exactly, then rebuild:
```bash
cd gsplat && rm -rf build/ dist/ *.egg-info
TORCH_CUDA_ARCH_LIST="8.6" pip install . --no-build-isolation --no-cache-dir -v
```
Replace `8.6` with your GPU's compute capability: `python -c "import torch; cap=torch.cuda.get_device_capability(); print(f'{cap[0]}.{cap[1]}')"`.

**`ModuleNotFoundError: No module named 'cv2'`**  
```bash
pip install opencv-python-headless
```



# UV SET UP
- Using python 3.10
- Install uv by running `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Run `uv sync` to install all packages
- Run `uv run main.py` to run main.py
- Run `uv add "package_name"` to install a package
- Make .env file and put huggingface token in like this HF_TOKEN =  

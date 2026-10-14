"""install.py: set up cartoon-me in one command, then prove it works.

  python install.py

It does every step in order and stops at the first one that fails, saying which and why:
  1. checks the computer (Python 3.10-3.12, git, ffmpeg, NVIDIA card, disk)
  2. gets LivePortrait next to this folder (../LivePortrait) if it is not there
  3. LivePortrait's own Python environment + its models
  4. cartoon-me's own Python environment
  5. a real test: draws a cartoon from a sample clip and makes a 2-second talking cartoon video

Everything goes into the two .venv folders. It never installs into your computer's main Python.
The last line is PASS or FAIL. Only PASS means it is set up."""
import os, platform, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LP = Path(os.environ.get("LIVEPORTRAIT_DIR", HERE.parent / "LivePortrait"))
WIN = os.name == "nt"
MAC = platform.system() == "Darwin"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
TEST = HERE / "work" / "_install_test"
# LivePortrait prints emoji. On Windows that crashes when an AI tool captures the output, unless Python writes UTF-8.
os.environ["PYTHONUTF8"] = "1"; os.environ["PYTHONIOENCODING"] = "utf-8"


def say(msg):
    print(f"\n== {msg}", flush=True)


def fail(step, why):
    print(f"\nFAIL at step {step}: {why}", flush=True)
    sys.exit(1)


def run(step, cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], **kw)
    if r.returncode:
        fail(step, f"this command failed (exit {r.returncode}): {' '.join(str(c) for c in cmd[:4])} ...")


def venv_python(folder):
    return folder / ".venv" / ("Scripts/python.exe" if WIN else "bin/python")


def make_venv(step, folder):
    py = venv_python(folder)
    if not py.exists():
        run(step, [sys.executable, "-m", "venv", folder / ".venv"])
    run(step, [py, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    return py


def free_gpu_memory():
    """A local AI (Ollama) running this install can hold most of the card. Unload it so the test has room.
    Ollama loads it again by itself the next time it is asked something."""
    if not shutil.which("nvidia-smi"):
        return
    free = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.split()[0])
    if free >= 14000 or not shutil.which("ollama"):
        return
    rows = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout.splitlines()[1:]
    for name in [r.split()[0] for r in rows if r.strip()]:
        print(f"  graphics card is busy: unloading Ollama model {name} for the test (it reloads on its own)", flush=True)
        subprocess.run(["ollama", "stop", name])


# ---------------------------------------------------------------- 1
say("1/5 Checking the computer")
v = sys.version_info
if not ((3, 10) <= (v.major, v.minor) <= (3, 12)):
    fail(1, f"this is Python {v.major}.{v.minor}. Use Python 3.10, 3.11 or 3.12"
            + (" (on Windows: py -3.12 install.py)" if WIN else ""))
print(f"  Python {v.major}.{v.minor} ok")
for tool in ("git", "ffmpeg"):
    if not shutil.which(tool):
        fail(1, f"{tool} is not installed or not on PATH. Install it"
                + (f" (winget install {'Git.Git' if tool == 'git' else 'Gyan.FFmpeg'})" if WIN else "")
                + ", open a new terminal, and run this again")
    print(f"  {tool} ok")
if shutil.which("nvidia-smi"):
    name, mem = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True).stdout.strip().splitlines()[0].split(", ")
    print(f"  graphics card: {name}, {int(mem) // 1024} GB")
    if int(mem) < 16000:
        print("  WARNING: under 16 GB is untested. It may run out of memory.")
elif MAC:
    print("  no NVIDIA card, but this is a Mac: it should work, slowly. Untested.")
    TORCH_INDEX = None
else:
    fail(1, "no NVIDIA graphics card found (nvidia-smi is missing). cartoon-me needs one")
free_gb = shutil.disk_usage(HERE).free // 2**30
print(f"  free disk: {free_gb} GB")
if free_gb < 30:
    fail(1, f"only {free_gb} GB free. It needs about 30 GB (models + two Python environments)")

# ---------------------------------------------------------------- 2
say("2/5 Getting LivePortrait")
if not (LP / "inference.py").exists():
    run(2, ["git", "clone", "--depth", "1", "https://github.com/KwaiVGI/LivePortrait", LP])
print(f"  LivePortrait at {LP}")

# ---------------------------------------------------------------- 3
say("3/5 LivePortrait's Python environment and models (a few GB, takes a while)")
lp_py = make_venv(3, LP)
if TORCH_INDEX:
    # the versions LivePortrait was tested with here (RTX 4090, Windows 11, Python 3.12)
    run(3, [lp_py, "-m", "pip", "install", "-q", "torch==2.6.0", "torchvision==0.21.0", "--index-url", TORCH_INDEX.replace("cu128", "cu124")])
    # LivePortrait pins lmdb 1.4.1, which has no build for new Pythons on Windows. Same list, lmdb unpinned.
    reqs = [l for l in (LP / "requirements_base.txt").read_text().splitlines() if l.strip() and not l.startswith("lmdb")]
    reqs += ["lmdb", "onnxruntime-gpu==1.18.0", "requests"]  # requests: LivePortrait imports it but does not list it
    run(3, [lp_py, "-m", "pip", "install", "-q", *reqs])
else:
    run(3, [lp_py, "-m", "pip", "install", "-q", "torch", "torchvision"])
    run(3, [lp_py, "-m", "pip", "install", "-q", "-r", LP / "requirements_macOS.txt"])
run(3, [lp_py, "-m", "pip", "install", "-q", "huggingface_hub"])
if not (LP / "pretrained_weights/liveportrait/landmark.onnx").exists():
    run(3, [lp_py, "-c", "from huggingface_hub import snapshot_download as d; "
            f"d('KlingTeam/LivePortrait', local_dir=r'{LP / 'pretrained_weights'}', allow_patterns=['liveportrait/*', 'insightface/*'])"])
print("  LivePortrait's own example (proves the card, the models and the code work together):")
free_gpu_memory()
run(3, [lp_py, "inference.py", "-s", "assets/examples/source/s9.jpg", "-d", "assets/examples/driving/d0.mp4",
        "-o", TEST / "liveportrait"], cwd=LP)
if not list((TEST / "liveportrait").glob("*.mp4")):
    fail(3, "LivePortrait ran but made no video")

# ---------------------------------------------------------------- 4
say("4/5 cartoon-me's Python environment")
py = make_venv(4, HERE)
if TORCH_INDEX:
    run(4, [py, "-m", "pip", "install", "-q", "torch", "--index-url", TORCH_INDEX])
else:
    run(4, [py, "-m", "pip", "install", "-q", "torch"])
run(4, [py, "-m", "pip", "install", "-q", "-r", HERE / "requirements.txt"])
run(4, [py, "-c", "import torch, diffusers, mediapipe, faster_whisper, cv2; "
        "assert torch.cuda.is_available() or torch.backends.mps.is_available(), 'torch cannot see the graphics card'; print('  imports ok')"])

# ---------------------------------------------------------------- 5
say("5/5 Real test: draw a cartoon, then make a 2-second talking cartoon (first run downloads ~15 GB of FLUX.2 Klein)")
TEST.mkdir(parents=True, exist_ok=True)
clip = TEST / "sample.mp4"
shutil.copy(LP / "assets/examples/driving/d0.mp4", clip)
free_gpu_memory()
run(5, [py, HERE / "cartoonme.py", "draw", clip, "--at", "0.5", "--count", "1"], cwd=HERE)
out = TEST / "sample_cartoon.mp4"
out.unlink(missing_ok=True)
free_gpu_memory()
run(5, [py, HERE / "cartoonme.py", "make", clip, "--pick", "0", "--cartoon", "0-2", "--out", out], cwd=HERE)
if not out.exists() or out.stat().st_size < 10_000:
    fail(5, "the test video was not made")

act = r".venv\Scripts\activate" if WIN else "source .venv/bin/activate"
print(f"""
Test video: {out}
To use it on your own video:
  cd {HERE}
  {act}
  python cartoonme.py draw my_video.mp4 --at 1.0
  python cartoonme.py make my_video.mp4 --pick N --cartoon "0-8 swipe" --captions

PASS""")

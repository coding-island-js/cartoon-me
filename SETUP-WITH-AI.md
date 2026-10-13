# Setup instructions for an AI coding agent

You are setting up cartoon-me for the person you are working with. Do the steps in order. Run the commands yourself. Stop and tell the person in plain words if a check fails; do not work around a missing GPU.

## 1. Check the machine
- `nvidia-smi` must show an NVIDIA GPU. Report its name and memory. Under 16 GB: warn that it is untested and may run out of memory.
- No NVIDIA GPU but an Apple Silicon Mac: tell them it should work but slowly and is untested; continue only if they say yes.
- Need: Python 3.10–3.12, git, ffmpeg on PATH. Install what is missing with the system package manager (winget, brew, apt), asking first.
- Need about 20 GB of free disk.

## 2. Get the code
In the folder the person chose:
```
git clone https://github.com/coding-island-js/cartoon-me
git clone https://github.com/KwaiVGI/LivePortrait
```
The two folders must sit side by side (cartoon-me finds `../LivePortrait`), or set `LIVEPORTRAIT_DIR`.

## 3. LivePortrait
Follow `LivePortrait/readme.md` exactly: create `LivePortrait/.venv`, install its requirements, download its pretrained weights into `LivePortrait/pretrained_weights`. Verify with its own example command before going on.

## 4. cartoon-me
```
cd cartoon-me
python -m venv .venv
(activate it)
pip install torch --index-url https://download.pytorch.org/whl/cu128     # CUDA; on a Mac: pip install torch
pip install -r requirements.txt
```
The FLUX.2 Klein 4B weights (~15 GB) download on the first `draw`. If Hugging Face asks for a login, have the person run `huggingface-cli login`.

## 5. Test
Ask the person for a short clip of themselves talking (10–20 s, facing the camera). Then:
```
python cartoonme.py draw their_clip.mp4 --at 1.0
```
Show them `work/<clip name>/pick.jpg` and ask which number they like. Then:
```
python cartoonme.py make their_clip.mp4 --pick N --cartoon "0-5" --captions
```
Done when `<clip name>_cartoon.mp4` plays and the first 5 seconds are a talking cartoon. Tell them how to run it on their own videos (see README "Do it yourself" steps 4–5).

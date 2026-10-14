# cartoon-me

Turn yourself into a talking cartoon. Free, on your own computer.

You film yourself talking. It draws a cartoon of you. Then the cartoon copies you: your head moves, its head moves. You talk, it talks. And if you sweep your hand across the screen, the real you shows up behind your hand.

Nothing gets uploaded. No account. No credits that run out.

## The easy way: let your AI set it up

If you use an AI coding tool (Claude Code, Codex, Cursor), open it in an empty folder and say:

> Set this up for me: https://github.com/coding-island-js/cartoon-me

It reads [`SETUP-WITH-AI.md`](SETUP-WITH-AI.md) and runs one installer that checks your computer, installs everything and makes a test video.

**For the AI tool:** clone the repo, `cd cartoon-me`, run `python install.py --background` (it returns at once), then run `python install.py --status` every few minutes until it says PASS or FAIL. It takes 20 to 60 minutes. Do not install anything else yourself, and do not say it is set up until `--status` says PASS. Details: [`SETUP-WITH-AI.md`](SETUP-WITH-AI.md).

### No AI subscription? Use a free one on your own PC

Tested: [OpenCode](https://opencode.ai) running gpt-oss 20B through [Ollama](https://ollama.com) set this up on its own, no paid AI. (Qwen3 30B did not: it said "done" when it wasn't.)

1. Install Ollama and OpenCode (`npm i -g opencode-ai`), then `ollama pull gpt-oss:20b` (13 GB).
2. Give the model room to work. Save this as `Modelfile`, then run `ollama create gpt-oss-20b-32k -f Modelfile`:
   ```
   FROM gpt-oss:20b
   PARAMETER num_ctx 32768
   ```
3. In an empty folder, save this as `opencode.json`:
   ```json
   {
     "provider": { "ollama": { "npm": "@ai-sdk/openai-compatible", "options": { "baseURL": "http://localhost:11434/v1" },
                               "models": { "gpt-oss-20b-32k": { "tools": true } } } },
     "model": "ollama/gpt-oss-20b-32k"
   }
   ```
4. Run `opencode` there and say the line above. If it only lists the commands, say "go ahead". Then ask "is it done?" every few minutes until it says PASS.

The model and the installer share the graphics card. The installer unloads the model while it runs its test, and Ollama loads it again by itself.

## What you need

- A Windows or Linux PC with an NVIDIA graphics card. Built and tested on an RTX 4090 (24 GB). Smaller cards: not tested yet.
- About 30 GB of free disk space (models plus two Python environments).
- Newer Macs (M1 and up): should work, slower. Not tested yet. If you try it, open an issue and tell me.

## Do it yourself

1. Install Python 3.12, [ffmpeg](https://ffmpeg.org/download.html) and git.
2. Get this repo and run the installer:
   ```
   git clone https://github.com/coding-island-js/cartoon-me
   cd cartoon-me
   python install.py
   ```
   It gets LivePortrait, makes its own Python environments (never your main Python), downloads the models and makes a test video. The first run takes 20 to 60 minutes. The last line says PASS or FAIL, and a FAIL says which step and why.
3. Turn on its Python environment: `.venv\Scripts\activate` (Mac/Linux: `source .venv/bin/activate`).
4. Draw your cartoon. It makes a few versions and a sheet to pick from:
   ```
   python cartoonme.py draw my_video.mp4 --at 1.0 --describe "grey beard, black polo shirt"
   ```
5. Make the video. List the parts (in seconds) that should be the cartoon. Add `swipe` to a part if you sweep your hand at the end of it:
   ```
   python cartoonme.py make my_video.mp4 --pick 2 --cartoon "0-11 swipe, 35-63, 73-81 swipe" --captions
   ```
   After each cartoon part it prints a mouth check (timing and size, compared to your real mouth).

## How to film it

- Camera on something steady. Face the camera.
- In the parts that will be cartoon, keep your hands below your chin.
- For the swipe: raise one hand and sweep it slowly across your face until it leaves the frame. Keep talking.
- Same shirt and seat the whole time. The cartoon is drawn from one picture.

## How it works

| Step | Tool | What it does |
|---|---|---|
| 1 | [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | Redraws one frame of your video as a cartoon, a few versions |
| 2 | [LivePortrait](https://github.com/KwaiVGI/LivePortrait) | Moves the cartoon with your real head and mouth |
| 3 | [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) | Finds your hand in every frame for the swipe |
| 4 | OpenCV + ffmpeg | Blends cartoon and real along your hand, keeps your hand real |
| 5 | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Word-timed captions |
| 6 | MediaPipe Face Landmarker, `lipmeter.py` | Checks the cartoon's mouth follows yours, with no one watching |

Two things I learned the hard way, both built in now:
- LivePortrait copies your mouth *relative to the first frame*. If a cartoon part starts mid-word, "open" becomes the resting mouth and the cartoon barely moves (I measured 4%). So every part now starts from a frame where your mouth is closed.
- `--lips` boosts the movement, but it boosts the whole face. At 1.5 the glasses bent and the eyebrows smeared. Default is 1.0. Try 1.15 if the mouth check says it's weak.

## Licenses

This repo: MIT. The tools keep their own licenses: FLUX.2 Klein 4B (Apache-2.0), MediaPipe (Apache-2.0), faster-whisper (MIT), LivePortrait code (MIT). **LivePortrait uses InsightFace face-detection models, which are for non-commercial research use.** Fine for your own videos. Check that license before selling anything made with it.

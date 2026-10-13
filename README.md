# cartoon-me

Turn yourself into a talking cartoon. Free, on your own computer.

You film yourself talking. It draws a cartoon of you. Then the cartoon copies you: your head moves, its head moves. You talk, it talks. And if you sweep your hand across the screen, the real you shows up behind your hand.

Nothing gets uploaded. No account. No credits that run out.

## The easy way: let your AI set it up

If you use an AI coding tool (Claude Code, Codex, Cursor), open it in an empty folder and say:

> Set this up for me: https://github.com/coding-island-js/cartoon-me

It reads [`SETUP-WITH-AI.md`](SETUP-WITH-AI.md), checks your computer, installs everything and runs a test.

## What you need

- A Windows or Linux PC with an NVIDIA graphics card. Built and tested on an RTX 4090 (24 GB). Smaller cards: not tested yet.
- About 20 GB of free disk space for the models.
- Newer Macs (M1 and up): should work, slower. Not tested yet. If you try it, open an issue and tell me.

## Do it yourself

1. Install Python 3.12, [ffmpeg](https://ffmpeg.org/download.html) and git.
2. Get this repo and its tools:
   ```
   git clone https://github.com/coding-island-js/cartoon-me
   git clone https://github.com/KwaiVGI/LivePortrait
   ```
   Follow LivePortrait's own install steps inside `LivePortrait/` (it has its own Python environment and downloads its own weights).
3. In `cartoon-me/`:
   ```
   python -m venv .venv
   .venv\Scripts\activate          (Mac/Linux: source .venv/bin/activate)
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements.txt
   ```
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

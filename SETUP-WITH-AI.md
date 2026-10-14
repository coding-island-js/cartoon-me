# Setup instructions for an AI coding agent

You are setting up cartoon-me for the person you are working with. There is one command. Do not install anything yourself, do not use npm, and do not pip install into the system Python.

## 1. Get the code
In the folder the person chose:
```
git clone https://github.com/coding-island-js/cartoon-me
cd cartoon-me
```

## 2. Run the installer
```
python install.py --background
```
It returns at once and keeps installing on its own (20 to 60 minutes, about 20 GB of downloads). Then check it every few minutes:
```
python install.py --status
```
On Windows, if `python` is not 3.10 to 3.12, use `py -3.12` instead of `python` in both commands.

It checks the computer, gets LivePortrait into `../LivePortrait`, makes two separate Python environments, downloads the models, and makes a short test video. Keep checking until it says PASS or FAIL. If it says STOPPED, run `python install.py --background` again: it skips what is already done.

## 3. Read the status
- **`PASS`**: setup worked. Tell the person, and show them the "To use it on your own video" lines it printed.
- **`FAIL at step N: ...`**: tell the person that line, word for word. If it says something is missing (git, ffmpeg, Python), offer to install that one thing with winget, brew or apt, ask first, then run `python install.py --background` again. It skips what is already done.

Never tell the person it is set up unless `--status` says PASS.

## 4. Their first video
Ask for a short clip of them talking (10 to 20 seconds, facing the camera). Then, inside `cartoon-me` with `.venv` active:
```
python cartoonme.py draw their_clip.mp4 --at 1.0
```
Show them `work/<clip name>/pick.jpg` and ask which number they like. Then:
```
python cartoonme.py make their_clip.mp4 --pick N --cartoon "0-5" --captions
```
Done when `<clip name>_cartoon.mp4` plays and the first 5 seconds are a talking cartoon.

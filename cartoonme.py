"""cartoon-me: turn parts of a talking-head video into a talking cartoon of you, with a hand-swipe back to real.
Everything runs on your own computer with open models. See README.md.

  python cartoonme.py draw  my_video.mp4 --at 1.0             # makes a few cartoon versions + a sheet to pick from
  python cartoonme.py make  my_video.mp4 --pick 2 --cartoon "0-11 swipe, 45-75, 160-176 swipe" --captions

--cartoon lists the parts (in seconds) that become the cartoon. "swipe" on a part means: somewhere in its last
6 seconds you sweep your hand across the frame, and the real you is revealed behind your hand. Parts without
"swipe" switch with a plain cut."""
import argparse, json, os, re, shutil, subprocess, sys
from pathlib import Path
import cv2, numpy as np

HERE = Path(__file__).resolve().parent
LP_DIR = Path(os.environ.get("LIVEPORTRAIT_DIR", HERE.parent / "LivePortrait"))
LP_PY = os.environ.get("LIVEPORTRAIT_PYTHON") or str(next((p for p in [LP_DIR / ".venv/Scripts/python.exe", LP_DIR / ".venv/bin/python"] if p.exists()), sys.executable))
FLUX_PY = os.environ.get("FLUX_PYTHON", sys.executable)
HAND_MODEL = HERE / "models/hand_landmarker.task"
HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
FPS = 30

def run(cmd, **kw):
    r = subprocess.run([str(c) for c in cmd], **kw)
    if r.returncode: sys.exit(f"failed: {' '.join(map(str, cmd[:3]))} ...")
    return r

def work_dir(video):
    d = HERE / "work" / Path(video).stem; d.mkdir(parents=True, exist_ok=True); return d

def normalized(video, d):
    """Constant 30 fps copy, so frame numbers and seconds always agree (phone and Clipchamp exports often vary)."""
    out = d / "src.mp4"
    if not out.exists() or out.stat().st_mtime < Path(video).stat().st_mtime:
        run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vf", f"fps={FPS}", "-c:v", "libx264", "-crf", "14", "-preset", "fast",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", out])
    return out

# ---------------------------------------------------------------- draw
def draw(a):
    d = work_dir(a.video); frame = d / "frame.png"
    run(["ffmpeg", "-v", "error", "-y", "-ss", str(a.at), "-i", a.video, "-frames:v", "1", frame])
    cmd = [FLUX_PY, HERE / "draw_cartoon.py", frame, d, "--count", str(a.count)]
    if a.style: cmd += ["--style", a.style]
    if a.describe: cmd += ["--describe", a.describe]
    run(cmd)
    # one sheet with all versions numbered, so picking is one look
    tiles = []
    for i in range(a.count):
        im = cv2.imread(str(d / f"cartoon_{i}.png")); im = cv2.resize(im, (640, int(640 * im.shape[0] / im.shape[1])))
        cv2.putText(im, str(i), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 0, 0), 12); cv2.putText(im, str(i), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (255, 255, 255), 4)
        tiles.append(im)
    while len(tiles) % 2: tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
    cv2.imwrite(str(d / "pick.jpg"), sheet)
    print(f"\nLook at {d / 'pick.jpg'} and pick a number. Then:\n  python cartoonme.py make {a.video} --pick N --cartoon \"0-8 swipe\"")

# ---------------------------------------------------------------- make
def parse_parts(text):
    parts = []
    for chunk in text.split(","):
        m = re.fullmatch(r"\s*([\d.]+)\s*-\s*([\d.]+)\s*(swipe)?\s*", chunk)
        if not m: sys.exit(f'cannot read part "{chunk}". Use like "0-11 swipe, 45-75"')
        parts.append({"s": float(m[1]), "e": float(m[2]), "swipe": bool(m[3])})
    return sorted(parts, key=lambda p: p["s"])

PAD = 8   # closed-mouth frames put in front of each part, then cut off again

def neutral_frame(src, d):
    """The frame where the real mouth is most closed. LivePortrait copies mouth movement RELATIVE to the first
    driving frame, so a part that starts mid-word makes 'open' the resting mouth and the cartoon barely moves
    (measured 2026-10-13: 8% of the real movement; with this frame first, 65%)."""
    png = d / "neutral.png"
    if not png.exists():
        import lipmeter
        m = lipmeter.series(src)[0]; i = int(np.nanargmin(m))
        run(["ffmpeg", "-v", "error", "-y", "-ss", f"{i / FPS:.3f}", "-i", src, "-frames:v", "1", png])
    return png

def animate(cartoon_png, src, part, d, k, w, h, lips=1.0):
    """LivePortrait: the cartoon picture copies the head and mouth of the real video for this part."""
    drive, out_dir = d / f"drive_{k}.mp4", d / f"lp_{k}"
    run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-t", f"{PAD / FPS:.4f}", "-r", str(FPS), "-i", neutral_frame(src, d),
         "-ss", f"{part['s']:.3f}", "-to", f"{part['e']:.3f}", "-i", src, "-filter_complex",
         f"[0:v]fps={FPS},format=yuv420p[a];[1:v]fps={FPS},format=yuv420p[b];[a][b]concat=n=2:v=1[v]", "-map", "[v]",
         "-c:v", "libx264", "-crf", "14", drive])
    shutil.rmtree(out_dir, ignore_errors=True); out_dir.mkdir()
    r = subprocess.run([LP_PY, "inference.py", "-s", str(cartoon_png.resolve()), "-d", str(drive.resolve()), "-o", str(out_dir.resolve()),
                        "--flag_crop_driving_video", "--source_max_dim", str(max(w, h)), "--driving_multiplier", str(lips)],
                       cwd=LP_DIR, capture_output=True, text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    raw = next((p for p in out_dir.glob("*.mp4") if "_concat" not in p.name), None)
    if not raw: sys.exit(f"LivePortrait made no video for part {k}. Its last words:\n{(r.stderr or r.stdout)[-800:]}\n"
                         "Usually: the cartoon face is too flat, pick a more lifelike version.")
    vid = out_dir / "part.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-vf", f"trim=start_frame={PAD},setpts=PTS-STARTPTS", "-c:v", "libx264", "-crf", "14", vid])
    return vid

def check_lips(src, part, vid):
    """Same ruler as lipmeter.py: does the cartoon mouth follow the real one?"""
    import lipmeter
    r, _ = lipmeter.series(src, part["s"], part["e"] - part["s"]); t, tp = lipmeter.series(vid)
    s = lipmeter.score(r, t, tp)
    note = "" if s["size"] > 0.5 and s["timing"] > 0.8 else "  <- weak mouth movement, try --lips 1.15"
    print(f"  mouth check: timing {s['timing']}, size {s['size']}, wobble {s['jitter']}{note}", flush=True)

def track_hand(src, f0, f1, d, k):
    """MediaPipe finds 21 points on the hand in every frame between f0 and f1."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    if not HAND_MODEL.exists():
        import urllib.request; HAND_MODEL.parent.mkdir(exist_ok=True); urllib.request.urlretrieve(HAND_URL, HAND_MODEL)
    lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL)), running_mode=vision.RunningMode.VIDEO,
        num_hands=1, min_hand_detection_confidence=0.3, min_tracking_confidence=0.3))
    cap = cv2.VideoCapture(str(src)); cap.set(cv2.CAP_PROP_POS_FRAMES, f0); out = []
    for i in range(f0, f1):
        ok, f = cap.read()
        if not ok: break
        h, w = f.shape[:2]
        r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), int(i * 1000 / FPS))
        out.append([[p.x * w, p.y * h] for p in r.hand_landmarks[0]] if r.hand_landmarks else None)
    json.dump(out, open(d / f"hands_{k}.json", "w"))
    return out

def swipe_plan(hands, w):
    """Where the cartoon/real line sits in each frame of the swipe, and which side is still cartoon."""
    xs = [None if p is None else float(np.mean([q[0] for q in p])) for p in hands]
    idx = [i for i, x in enumerate(xs) if x is not None]
    if len(idx) < 3: return None
    first, last = idx[0], idx[-1]
    left = xs[last] < xs[first]                    # hand travels right to left: cartoon stays on the left of it
    entry, exitx = (1.16 * w, -0.16 * w) if left else (-0.16 * w, 1.16 * w)
    ramp = FPS // 4
    line = np.full(len(hands), np.nan)
    line[:first] = entry
    line[max(0, first - ramp):first] = np.linspace(entry, xs[first], first - max(0, first - ramp) + 1)[:-1]
    line[first:last + 1] = np.interp(np.arange(first, last + 1), idx, [xs[i] for i in idx])
    done = min(len(hands), last + 1 + ramp)
    line[last + 1:done] = np.linspace(xs[last], exitx, done - last - 1 + 1)[1:]
    return {"line": line, "left": left, "first": first, "last": last, "done": done}

def hand_mask(p, w, h):
    """The hand plus a forearm running off screen, softened: these pixels always stay real."""
    m = np.zeros((h, w), np.uint8)
    if p is None: return m
    sc = w / 1920
    cv2.fillConvexPoly(m, cv2.convexHull(np.array(p, np.int32)), 255)
    w0, m9 = np.array(p[0], float), np.array(p[9], float); dv = w0 - m9; dv /= (np.linalg.norm(dv) + 1e-6)
    width = np.linalg.norm(np.array(p[5]) - np.array(p[17])) * 0.75
    n = np.array([-dv[1], dv[0]]); far = w0 + dv * 1400 * sc
    cv2.fillConvexPoly(m, np.array([w0 + n * width, w0 - n * width, far - n * width * 1.3, far + n * width * 1.3], np.int32), 255)
    k = max(3, int(45 * sc)); b = max(3, int(31 * sc) | 1)
    return cv2.GaussianBlur(cv2.dilate(m, np.ones((k, k), np.uint8)), (b, b), 0)

def captions(src, d, w, h):
    from faster_whisper import WhisperModel
    try: m = WhisperModel("large-v3", device="cuda", compute_type="float16")
    except Exception: m = WhisperModel("small", device="cpu", compute_type="int8")
    segs, _ = m.transcribe(str(src), word_timestamps=True, language="en", vad_filter=True)
    words = [{"w": x.word.strip(), "s": x.start, "e": x.end} for s in segs for x in s.words]
    json.dump(words, open(d / "words.json", "w"), indent=0)
    chunks, cur = [], []
    for x in words:
        cur.append(x)
        if len(cur) >= 4 or re.search(r"[.,?!]$", x["w"]): chunks.append(cur); cur = []
    if cur: chunks.append(cur)
    t = lambda s: f"{int(s // 3600)}:{int(s % 3600 // 60):02d}:{s % 60:05.2f}"
    size, margin = int(h * 0.068), int(h * 0.065)
    lines = []
    for i, c in enumerate(chunks):
        nxt = chunks[i + 1][0]["s"] if i + 1 < len(chunks) else None
        e = nxt if nxt is not None and nxt - c[-1]["e"] < 0.6 else c[-1]["e"] + 0.25
        lines.append(f"Dialogue: 0,{t(c[0]['s'])},{t(e)},C,,0,0,0,,{' '.join(x['w'] for x in c)}")
    ass = d / "caps.ass"
    ass.write_text(f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: C,Arial,{size},&H00FFFFFF,&H00FFFFFF,&H00161616,&H64000000,-1,0,0,0,100,100,0,0,1,{max(2, size // 12)},2,2,80,80,{margin},1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""" + "\n".join(lines) + "\n", encoding="utf-8")
    return ass

def make(a):
    d = work_dir(a.video); src = normalized(a.video, d)
    cap = cv2.VideoCapture(str(src)); w, h = int(cap.get(3)), int(cap.get(4)); n = int(cap.get(7))
    pic = d / f"cartoon_{a.pick}.png"
    if not pic.exists(): sys.exit(f"no {pic}. Run draw first.")
    cartoon = d / "cartoon_picked.png"; cv2.imwrite(str(cartoon), cv2.resize(cv2.imread(str(pic)), (w, h), interpolation=cv2.INTER_LANCZOS4))

    parts = parse_parts(a.cartoon)
    for k, p in enumerate(parts):
        p["f0"], p["f1"] = round(p["s"] * FPS), min(n, round(p["e"] * FPS))
        keep = d / f"lp_{k}" / "keep.mp4"
        if k in a.keep:
            if not keep.exists(): shutil.copy(next(q for q in (d / f"lp_{k}").glob("*.mp4") if "_concat" not in q.name), keep)
            print(f"part {k}: keeping the cartoon from last time", flush=True); p["lp"] = keep
        else:
            print(f"part {k}: {p['s']}-{p['e']}s, animating the cartoon...", flush=True)
            p["lp"] = animate(cartoon, src, p, d, k, w, h, a.lips)
            check_lips(src, p, p["lp"])
        if p["swipe"]:
            t0 = max(p["f0"], p["f1"] - 6 * FPS)
            plan = swipe_plan(track_hand(src, t0, p["f1"], d, k), w)
            if plan is None: print(f"  no hand found in the last 6 s of part {k}: it will end on a plain cut")
            else: plan["t0"] = t0; plan["hands"] = json.load(open(d / f"hands_{k}.json"))
            p["plan"] = plan

    print("putting it together...", flush=True)
    body = d / "body.mp4"
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
                           "-i", str(src), "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "15", "-preset", "fast",
                           "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", str(body)], stdin=subprocess.PIPE)
    xx = np.arange(w, dtype=np.float32)[None, :]; edge = 18 * w / 1920
    cap = cv2.VideoCapture(str(src)); part = None; lpcap = None; held = None; prev = None
    for i in range(n):
        ok, f = cap.read()
        if not ok: break
        cur = next((p for p in parts if p["f0"] <= i < p["f1"]), None)
        if cur is not part:
            part = cur; held = prev = None
            lpcap = cv2.VideoCapture(str(cur["lp"])) if cur else None
        if part:
            ok2, toon = lpcap.read()
            toon = cv2.resize(toon, (w, h), interpolation=cv2.INTER_LANCZOS4) if ok2 else held
            plan = part.get("plan")
            j = i - plan["t0"] if plan else -1
            if toon is None: pass
            elif not plan or j < plan["first"] - FPS // 4:
                f = toon; held = toon                                      # plain cartoon
            elif j < plan["done"]:                                         # the swipe
                bx = plan["line"][j]
                hp = plan["hands"][j] if j < len(plan["hands"]) else None
                if hp is not None: prev = hp
                elif j > plan["last"]: prev = None
                if not (0.385 * w < bx < 0.615 * w) or held is None: held = toon   # hand over the face: keep the last clean cartoon
                al = np.clip(((bx - xx) if plan["left"] else (xx - bx)) / edge + 0.5, 0, 1)
                al = np.repeat(al, h, 0) * (1 - hand_mask(prev, w, h).astype(np.float32) / 255)
                f = (held * al[..., None] + f * (1 - al[..., None])).astype(np.uint8)
            # after the hand has left: real
        ff.stdin.write(f.tobytes())
    ff.stdin.close(); ff.wait()

    out = Path(a.out or f"{Path(a.video).stem}_cartoon.mp4")
    if a.captions:
        print("writing captions...", flush=True)
        ass = captions(src, d, w, h)
        rel = os.path.relpath(ass, Path.cwd()).replace("\\", "/").replace(":", "\\:")
        run(["ffmpeg", "-v", "error", "-y", "-i", body, "-vf", f"ass={rel}", "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-c:a", "copy", out])
    else:
        shutil.copy(body, out)
    print("done:", out.resolve())

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("draw", help="make cartoon versions of one frame")
p.add_argument("video"); p.add_argument("--at", type=float, default=1.0, help="second to take the picture from (face the camera, no hand)")
p.add_argument("--count", type=int, default=4); p.add_argument("--style", help="optional picture whose art style to copy")
p.add_argument("--describe", default="", help='a few words so the cartoon keeps your look, e.g. "grey beard, black polo"')
p = sub.add_parser("make", help="build the video")
p.add_argument("video"); p.add_argument("--pick", type=int, required=True)
p.add_argument("--cartoon", required=True, help='parts in seconds, e.g. "0-11 swipe, 45-75"')
p.add_argument("--captions", action="store_true"); p.add_argument("--out")
p.add_argument("--lips", type=float, default=1.0, help="boost the cartoon face movement (1.0 = same). Above 1.0 the mouth moves more but "
               "the eyes and glasses blur: measured 2026-10-13, 1.5 lost 19%% eye detail and 30%% on head turns")
p.add_argument("--keep", type=lambda s: [int(x) for x in s.split(",")], default=[], help="part numbers to reuse from the last run, e.g. 0")
a = ap.parse_args()
draw(a) if a.cmd == "draw" else make(a)

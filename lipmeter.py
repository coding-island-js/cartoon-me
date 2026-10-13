"""Measures how well a cartoon's mouth follows the real mouth, with no human watching.
MediaPipe Face Landmarker finds 478 face points per frame. Mouth opening = inner-lip gap / eye-corner distance.
  python lipmeter.py real.mp4 START cartoon.mp4 [SECONDS]
Score: timing (correlation of the two mouth curves), size (cartoon movement / real movement, 1.0 = same),
jitter (frame-to-frame wobble of the cartoon's face points; lower is steadier)."""
import sys
from pathlib import Path
import cv2, numpy as np

MODEL = Path(__file__).resolve().parent / "models/face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# eyes, eyebrows and the glasses around them
EYES = [33, 133, 159, 145, 263, 362, 386, 374, 70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
LAST_FACE = {}

def sharp(f, pts, eye):
    """Detail inside the box around these points: variance of the Laplacian on a fixed-size crop. Blur lowers it."""
    x0, y0 = pts.min(0) - eye * .15; x1, y1 = pts.max(0) + eye * .15
    c = f[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)]
    if c.size == 0: return np.nan
    g = cv2.cvtColor(cv2.resize(c, (256, int(256 * c.shape[0] / max(1, c.shape[1])) or 1)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

def face_quality():
    """From the last series() call: eye-area detail (median), worst whole-face detail (5th percentile, catches
    smeared head turns), and how much the eyebrows move (std of brow-to-eye distance)."""
    e, fs, b = (np.array(LAST_FACE[k], float) for k in ("eye_sharp", "face_sharp", "brow"))
    return {"eye_detail": round(float(np.nanmedian(e)), 1), "worst_face_detail": round(float(np.nanpercentile(fs, 5)), 1),
            "brow_move": round(float(np.nanstd(b)), 4)}

def series(video, start=0.0, seconds=None):
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    if not MODEL.exists():
        import urllib.request; MODEL.parent.mkdir(exist_ok=True); urllib.request.urlretrieve(MODEL_URL, MODEL)
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)), running_mode=vision.RunningMode.VIDEO, num_faces=1,
        min_face_detection_confidence=0.3, min_tracking_confidence=0.3))
    cap = cv2.VideoCapture(str(video)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    n = int((seconds or 1e9) * fps); mouth, pts = [], []
    global LAST_FACE; LAST_FACE = {"eye_sharp": [], "face_sharp": [], "brow": []}
    for i in range(n):
        ok, f = cap.read()
        if not ok: break
        r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), int(i * 1000 / fps))
        if not r.face_landmarks: mouth.append(np.nan); pts.append(None); continue
        p = np.array([[q.x * f.shape[1], q.y * f.shape[0]] for q in r.face_landmarks[0]])
        eye = np.linalg.norm(p[33] - p[263]) + 1e-6
        LAST_FACE["eye_sharp"].append(sharp(f, p[EYES], eye))
        LAST_FACE["face_sharp"].append(sharp(f, p, eye))
        LAST_FACE["brow"].append((np.linalg.norm(p[105] - p[159]) + np.linalg.norm(p[334] - p[386])) / 2 / eye)
        mouth.append(np.linalg.norm(p[13] - p[14]) / eye)
        pts.append(p[[1, 33, 263, 61, 291, 199]] / eye)          # nose, eyes, mouth corners, chin
    return np.array(mouth), pts

def score(real, toon, toon_pts):
    n = min(len(real), len(toon)); a, b = real[:n], toon[:n]
    ok = ~np.isnan(a) & ~np.isnan(b)
    found = ok.mean()
    if ok.sum() < 10: return {"found": round(float(found), 2), "timing": 0, "size": 0, "jitter": 9, "total": 0}
    timing = float(np.corrcoef(a[ok], b[ok])[0, 1])
    size = float((np.percentile(b[ok], 90) - np.percentile(b[ok], 10)) / (np.percentile(a[ok], 90) - np.percentile(a[ok], 10) + 1e-6))
    q = [p for p in toon_pts[:n] if p is not None]
    jit = float(np.mean([np.abs(q[i + 1] - 2 * q[i] + q[i - 1]).mean() for i in range(1, len(q) - 1)])) if len(q) > 3 else 9
    # one number to rank by: timing matters most, size should land near 1, wobble is punished
    total = found * (timing - abs(np.log(max(size, 1e-3))) * 0.5 - jit * 20)
    return {"found": round(float(found), 2), "timing": round(timing, 3), "size": round(size, 2), "jitter": round(jit, 4), "total": round(float(total), 3)}

if __name__ == "__main__":
    real, start, toon = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    secs = float(sys.argv[4]) if len(sys.argv) > 4 else None
    r, _ = series(real, start, secs); t, tp = series(toon, 0, secs)
    print(score(r, t, tp))

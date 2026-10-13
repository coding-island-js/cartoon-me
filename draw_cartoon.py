"""Step 1: redraw one frame of your video as a cartoon, several versions, with FLUX.2 Klein 4B (open, Apache-2.0).
Run by cartoonme.py inside a Python that has a recent diffusers (0.37+) and torch.
  python draw_cartoon.py frame.png out_dir --count 4 [--style style.jpg] [--model PATH_OR_HF_ID]"""
import argparse, os
from pathlib import Path
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

PROMPT = ("Redraw image 1 as a soft, clean 2D animated cartoon illustration{style}: gentle shading, thin dark outlines, "
          "clearly drawn, not a photo. The same person: keep their exact skin tone, age, face shape and facial hair. "
          "Same clothes, same hair, same glasses and hat if any. {describe}"
          "Keep image 1's exact framing: the head stays exactly as large and in the same place, same shoulders, same room behind. "
          "Remove any raised hand completely. Looking straight at the camera, mouth closed, slight smile. No text.")

ap = argparse.ArgumentParser()
ap.add_argument("frame"); ap.add_argument("out_dir")
ap.add_argument("--count", type=int, default=4)
ap.add_argument("--style", help="optional picture whose art style to copy")
ap.add_argument("--describe", default="", help='extra words about you, e.g. "grey beard, black polo shirt"')
ap.add_argument("--model", default=os.environ.get("FLUX_MODEL", "black-forest-labs/FLUX.2-klein-4B"))
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
pipe = Flux2KleinPipeline.from_pretrained(a.model, torch_dtype=torch.bfloat16 if dev != "cpu" else torch.float32).to(dev)

src = Image.open(a.frame).convert("RGB")
# about one megapixel, same shape as the video, sides a multiple of 16
s = (1_050_000 / (src.width * src.height)) ** 0.5
W, H = int(src.width * s) // 16 * 16, int(src.height * s) // 16 * 16
images = [src.resize((W, H))]
if a.style: images.append(Image.open(a.style).convert("RGB"))
prompt = PROMPT.format(style=" in the exact art style of image 2" if a.style else "", describe=a.describe)

out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
for seed in range(a.count):
    g = torch.Generator(dev if dev != "mps" else "cpu").manual_seed(seed)
    img = pipe(prompt=prompt, image=images, width=W, height=H, num_inference_steps=4, generator=g).images[0]
    img.resize((src.width, src.height), Image.LANCZOS).save(out / f"cartoon_{seed}.png")
    print("made", out / f"cartoon_{seed}.png", flush=True)

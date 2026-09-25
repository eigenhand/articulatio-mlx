"""Compare this loop with mlx-audio's on the same prompts.

    python scripts/bench.py MODEL [--ref ref.wav --ref-text "..."] [--out DIR]
"""
import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from articulatio_mlx import SAMPLE_RATE, Articulatio  # noqa: E402

TEXT = ("Am dritten Oktober fährt der Zug um acht Uhr dreißig ab. "
        "Bitte seien Sie pünktlich, denn die Straßenbahn fährt heute nicht.")

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--ref")
ap.add_argument("--ref-text")
ap.add_argument("--out")
a = ap.parse_args()
cases = [("design", dict(instruct="Eine ruhige, freundliche Männerstimme."))]
if a.ref:
    cases += [("clone", dict(ref_audio=a.ref, ref_text=a.ref_text)),
              ("clone + instruction, cfg 3",
               dict(ref_audio=a.ref, ref_text=a.ref_text, instruct="Speak slowly with a serious tone.", cfg_scale=3.0))]

eng = Articulatio.load(a.model)
for _, kw in cases:  # warm up (compilation, Metal kernels)
    eng.generate("Hallo.", seed=0, **kw)
    list(eng.m.generate("Hallo.", seed=0, max_tokens=5, **kw))

print(f"| case | audio | mlx-audio RTF | articulatio-mlx RTF | speedup | first audio (stream) |")
print(f"|---|---:|---:|---:|---:|---:|")
for name, kw in cases:
    t0 = time.perf_counter()
    ref_audio = mx.concatenate([r.audio for r in eng.m.generate(TEXT, seed=1, **kw)])
    mx.eval(ref_audio)
    t_ref = time.perf_counter() - t0
    d_ref = ref_audio.shape[0] / SAMPLE_RATE

    t0 = time.perf_counter()
    audio, _ = eng.generate(TEXT, seed=1, **kw)
    t_ours = time.perf_counter() - t0
    d_ours = audio.shape[0] / SAMPLE_RATE

    t0 = time.perf_counter()
    first = None
    for _chunk in eng.stream(TEXT, seed=1, **kw):
        first = first or time.perf_counter() - t0
    r_ref, r_ours = t_ref / d_ref, t_ours / d_ours
    print(f"| {name} | {d_ours:.1f} s | {r_ref:.2f} | {r_ours:.2f} | {r_ref / r_ours:.1f}x | {first:.2f} s |")
    if a.out:
        sf.write(f"{a.out}/{name.split()[0]}-{'cfg' if 'cfg_scale' in kw else 'plain'}.wav", np.array(audio), SAMPLE_RATE)

"""Command line: articulatio-mlx MODEL --text ... --output out.wav"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

from .engine import SAMPLE_RATE, Articulatio, Stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="articulatio-mlx", description=__doc__)
    ap.add_argument("model", help="MLX model directory (converted with mlx_audio.convert)")
    ap.add_argument("--text", required=True)
    ap.add_argument("--instruction", "--instruct", dest="instruct")
    ap.add_argument("--ref-audio")
    ap.add_argument("--ref-text")
    ap.add_argument("--cfg-scale", type=float)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--max-frames", type=int, default=750, help="80 ms each")
    ap.add_argument("--output", default="output.wav")
    ap.add_argument("--stream", action="store_true",
                    help="decode in chunks while generating and report the time to first audio")
    ap.add_argument("--play", action="store_true", help="play the result with afplay")
    ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args(argv)
    if bool(a.ref_audio) != bool(a.ref_text):
        ap.error("--ref-audio and --ref-text go together")

    t_load = time.perf_counter()
    eng = Articulatio.load(a.model, compile=not a.no_compile)
    t_load = time.perf_counter() - t_load
    kw = dict(instruct=a.instruct, ref_audio=a.ref_audio, ref_text=a.ref_text, cfg_scale=a.cfg_scale,
              temperature=a.temperature, top_k=a.top_k, seed=a.seed, max_frames=a.max_frames)

    t0 = time.perf_counter()
    if a.stream:
        chunks, first = [], None
        for chunk in eng.stream(a.text, **kw):
            first = first or time.perf_counter() - t0
            chunks.append(np.array(chunk))
        audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        extra = f", first audio after {first:.2f} s" if first else ""
    else:
        out, st = eng.generate(a.text, **kw)
        audio = np.array(out)
        extra = f" (prefill {st.prefill_s:.2f} s, frames + decode {st.loop_s:.2f} s)"
    total = time.perf_counter() - t0
    dur = len(audio) / SAMPLE_RATE
    sf.write(a.output, audio, SAMPLE_RATE)
    print(f"wrote {a.output}: {dur:.2f} s audio in {total:.2f} s, RTF {total / max(dur, 1e-6):.2f}"
          f"{extra}; model loaded in {t_load:.1f} s", file=sys.stderr)
    if a.play:
        subprocess.run(["afplay", a.output], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

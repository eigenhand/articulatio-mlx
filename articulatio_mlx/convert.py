"""Convert a merged Hugging Face checkpoint to MLX with separate bit widths.

    python -m articulatio_mlx.convert MERGED_HF_DIR OUT_DIR --bits 4 --depth-bits 8

Like ``mlx_audio.convert --quantize``, but the depth decoder can keep more bits
than the backbone. The depth decoder runs 15 dependent steps per audio frame and
feeds its codes back into the backbone, so quantization error there compounds
over long passages (the C++ engine keeps it at higher precision by default).
"""
from __future__ import annotations

import argparse


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m articulatio_mlx.convert", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hf_path", help="merged checkpoint (scripts/merge_lora.py of articulatio-training)")
    ap.add_argument("mlx_path")
    ap.add_argument("--bits", type=int, default=4, help="backbone, text encoder and heads")
    ap.add_argument("--depth-bits", type=int, default=8,
                    help="depth decoder; 0 keeps it unquantized (float16)")
    ap.add_argument("--group-size", type=int, default=64)
    a = ap.parse_args(argv)

    import mlx_audio.convert as conv

    base = conv.build_quant_predicate

    def build(model, name=None):
        ok = base(model, name)

        def predicate(path, module):
            if not ok(path, module):
                return False
            # The depth decoder's audio embedding is tied to the backbone's during conversion;
            # quantizing it differently would change the backbone's too. Only its layers get more bits.
            if path.startswith("depth_decoder.") and "embed_tokens" not in path:
                if a.depth_bits == 0:
                    return False
                return {"group_size": a.group_size, "bits": a.depth_bits, "mode": "affine"}
            return True
        return predicate

    conv.build_quant_predicate = build
    conv.convert(a.hf_path, a.mlx_path, quantize=True, q_group_size=a.group_size, q_bits=a.bits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

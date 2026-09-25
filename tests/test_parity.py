"""Greedy decoding must give exactly mlx-audio's codes, and streaming decode its audio.

Needs a converted model: ARTICULATIO_MLX_MODEL=/path/to/model pytest
(ARTICULATIO_MLX_REF=ref.wav and ARTICULATIO_MLX_REF_TEXT add the cloning cases).
"""
import os

import mlx.core as mx
import numpy as np
import pytest

MODEL = os.environ.get("ARTICULATIO_MLX_MODEL")
REF = os.environ.get("ARTICULATIO_MLX_REF")
REF_TEXT = os.environ.get("ARTICULATIO_MLX_REF_TEXT")
pytestmark = pytest.mark.skipif(not MODEL, reason="set ARTICULATIO_MLX_MODEL")
TEXT = "Guten Morgen, heute ist ein schöner Tag."

CASES = [("design", dict(instruct="Eine ruhige Männerstimme."))]
if REF and REF_TEXT:
    CASES += [("clone", dict(ref_audio=REF, ref_text=REF_TEXT)),
              ("clone-cfg", dict(ref_audio=REF, ref_text=REF_TEXT, instruct="Speak slowly.", cfg_scale=3.0))]


@pytest.fixture(scope="module")
def eng():
    from articulatio_mlx import Articulatio
    return Articulatio.load(MODEL)


@pytest.mark.parametrize("name,kw", CASES, ids=[c[0] for c in CASES])
def test_greedy_codes_match_mlx_audio(eng, name, kw):
    m = eng.m
    ref, orig = [], m._depth_tokens

    def hook(*a, **k):
        frame = orig(*a, **k)
        ref.append(frame)
        return frame

    m._depth_tokens = hook
    try:
        list(m.generate(TEXT, temperature=0.0, top_k=0, max_tokens=60, **kw))
    finally:
        m._depth_tokens = orig
    ours = [f.tolist() for f in eng.frames(TEXT, temperature=0.0, top_k=0, max_frames=60, **kw)]
    assert ours == ref


def test_stream_matches_full_decode(eng):
    kw = dict(instruct="Eine ruhige Männerstimme.", seed=3)
    full = eng.m._decode_codes(mx.stack(list(eng.frames(TEXT, **kw)))[None].astype(mx.int32))
    streamed = mx.concatenate(list(eng.stream(TEXT, chunk_frames=5, first_chunk_frames=2, **kw)))
    assert full.shape == streamed.shape
    assert float(mx.abs(full - streamed).max()) < 1e-3

"""Fix for mlx-audio 0.5.6's streaming codec decoder.

``DecoderBlockUpsample.step`` does overlap-add for its transposed convolution
but carries the overlapping tail *including* the bias, so the bias is added
twice at the start of every streamed chunk. The streamed audio then jumps at
each chunk boundary (an audible click about once per second). Carrying the
tail without the bias makes streaming decode match the one-shot decode.
"""
from __future__ import annotations

import mlx.core as mx


def _step(self, x: mx.array) -> mx.array:
    y = self.conv(x)
    if self._overflow is not None:
        n = self._overflow.shape[1]
        y = mx.concatenate([y[:, :n, :] + self._overflow, y[:, n:, :]], axis=1)
    if self.trim_right > 0:
        tail = y[:, -self.trim_right:, :]
        if "bias" in self.conv:
            tail = tail - self.conv.bias
        self._overflow = tail
        y = y[:, : -self.trim_right, :]
    return y


def apply() -> None:
    from mlx_audio.tts.models.qwen3_tts import speech_tokenizer as st
    st.DecoderBlockUpsample.step = _step

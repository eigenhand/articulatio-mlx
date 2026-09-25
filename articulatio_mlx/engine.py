"""Fast generation loop for Breeze TTS 2 models (Articulatio-DE) on MLX.

The model itself (weights, text encoder, audio codec, prompt layout) is loaded
with mlx-audio. This module replaces its per-frame loop:

- the depth decoder keeps a KV cache, so each of its 15 steps processes one
  token instead of re-running the whole prefix;
- sampling stays on the GPU, so there is one host sync per audio frame instead
  of 16;
- with classifier-free guidance the conditional and unconditional passes run
  as one batch of two instead of two separate passes;
- the 15 depth-decoder steps of a frame are compiled with ``mx.compile``.

With greedy decoding the codes are identical to mlx-audio's (tests/test_parity.py).
"""
from __future__ import annotations

import functools
import os
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

import mlx.core as mx
from mlx_audio.lm.models.base import create_attention_mask
from mlx_audio.lm.models.cache import KVCache

SAMPLE_RATE = 24_000


def _sample(logits: mx.array, temperature: float, top_k: int, key: Optional[mx.array]) -> mx.array:
    """Temperature and top-k sampling on the GPU. logits [1, V] -> token [1]."""
    if temperature == 0:
        return mx.argmax(logits, axis=-1)
    logits = logits / temperature
    if top_k and top_k < logits.shape[-1]:
        kth = mx.min(mx.topk(logits, top_k, axis=-1), axis=-1, keepdims=True)
        logits = mx.where(logits < kth, -mx.inf, logits)
    return mx.random.categorical(logits, axis=-1, key=key)


@dataclass
class Stats:
    frames: int = 0
    prefill_s: float = 0.0
    loop_s: float = 0.0
    decode_s: float = 0.0
    first_audio_s: Optional[float] = None
    extra: dict = field(default_factory=dict)

    @property
    def audio_s(self) -> float:
        return self.frames * 1920 / SAMPLE_RATE


class Articulatio:
    """Wraps an mlx-audio Breeze TTS 2 model with a faster frame loop."""

    def __init__(self, model, compile: bool = True):
        self.m = model
        self.bb = model.backbone_model
        self.dd = model.depth_decoder.model
        self.heads = model.depth_decoder.codebooks_head.weight
        self.n_cb = model.num_codebooks
        self.eos = model.vocab_size                # the backbone's extra EOS class
        self.dd_vocab = self.dd.vocab_size
        codec_vocab = model.config.codec_vocab_size
        mask = mx.zeros((self.dd_vocab,))
        if codec_vocab < self.eos:                 # padding/control ids are never sampled
            mask = mask.at[codec_vocab:self.eos].add(-mx.inf)
        self.mask_dd = mask
        self.mask_bb = mx.concatenate([mask[: self.eos], mx.zeros((1,))])
        self.compile = compile
        self._frame_fns: dict = {}
        self._ref_codes: dict = {}
        encode = model._encode_reference

        def encode_cached(ref_audio):
            # Encoding the reference clip is deterministic and costs a few hundred ms.
            if not isinstance(ref_audio, (str, os.PathLike)):
                return encode(ref_audio)
            st = os.stat(ref_audio)
            key = (os.path.abspath(ref_audio), st.st_mtime_ns, st.st_size)
            if key not in self._ref_codes:
                self._ref_codes[key] = encode(ref_audio)
                mx.eval(self._ref_codes[key])
            return self._ref_codes[key]

        model._encode_reference = encode_cached
        self._key = mx.random.key(0)

    @classmethod
    def load(cls, path: str, compile: bool = True, cache_limit_mb: int = 512) -> "Articulatio":
        """Load an MLX model directory. ``cache_limit_mb`` caps MLX's pool of freed
        buffers, which otherwise keeps several GB after a generation (no speed cost)."""
        from mlx_audio.tts import load
        from . import codec_fix
        codec_fix.apply()
        if cache_limit_mb is not None:
            mx.set_cache_limit(cache_limit_mb << 20)
        return cls(load(path), compile=compile)

    # -- one frame ---------------------------------------------------------

    def _next_key(self) -> mx.array:
        self._key, key = mx.random.split(self._key)
        return key

    def _dd_embed(self, tok: mx.array, pos: int) -> mx.array:
        # Position p >= 1 carries codebook p-1, offset into its own vocabulary block.
        return self.dd.inputs_embeds_projector(self.dd.embed_tokens(tok + (pos - 1) * self.dd_vocab))

    def _dd_step(self, x: mx.array, cache: list) -> mx.array:
        mask = create_attention_mask(x, cache[0])
        for layer, c in zip(self.dd.layers, cache):
            x = layer(x, mask, c)
        return self.dd.norm(x)[:, -1, :]

    def _frame(self, first, hidden, key, *, cfg, temperature, top_k):
        """Depth-decode one frame. first [1], hidden [B, H] (B = 2: cond, uncond) -> codes [16]."""
        B = hidden.shape[0]
        keys = mx.random.split(key, self.n_cb - 1)
        cache = [KVCache() for _ in self.dd.layers]
        h = hidden
        if self.dd.backbone_hidden_state_projector is not None:
            h = self.dd.backbone_hidden_state_projector(h)
        x = mx.concatenate(
            [self.dd.inputs_embeds_projector(h)[:, None, :],
             self._dd_embed(mx.broadcast_to(first.reshape(1, 1), (B, 1)), 1)], axis=1)
        codes = [first.reshape(1)]
        for i in range(self.n_cb - 1):
            logits = self._dd_step(x, cache) @ self.heads[i]
            if B == 2:
                logits = logits[1:2] + cfg * (logits[0:1] - logits[1:2])
            tok = _sample(logits + self.mask_dd, temperature, top_k, keys[i])
            codes.append(tok)
            if i < self.n_cb - 2:
                x = self._dd_embed(mx.broadcast_to(tok.reshape(1, 1), (B, 1)), i + 2)
        return mx.concatenate(codes)

    def _frame_fn(self, B, cfg, temperature, top_k):
        k = (B, cfg, temperature, top_k)
        if k not in self._frame_fns:
            fn = functools.partial(self._frame, cfg=cfg, temperature=temperature, top_k=top_k)
            self._frame_fns[k] = mx.compile(fn) if self.compile else fn
        return self._frame_fns[k]

    def _bb_step(self, x: mx.array, cache: list, pad: mx.array) -> mx.array:
        """Backbone step for a left-padded batch; x [B, L, H] -> last hidden [B, H]."""
        B, L, _ = x.shape
        start = cache[0].offset
        k_pos = mx.arange(start + L)
        allowed = k_pos[None, :] >= pad[:, None]                         # [B, K]: not padding
        if L > 1:
            q_pos = start + mx.arange(L)
            allowed = allowed[:, None, :] & (k_pos[None, None, :] <= q_pos[None, :, None])
        else:
            allowed = allowed[:, None, :]
        mask = mx.where(allowed, 0.0, -mx.inf).astype(x.dtype)[:, None]  # [B, 1, L, K]
        for layer, c in zip(self.bb.layers, cache):
            x = layer(x, mask, c)
        return self.bb.norm(x)[:, -1, :]

    def _first(self, hidden, cfg, temperature, top_k):
        logits = self.m.lm_head(hidden)
        if hidden.shape[0] == 2:
            logits = logits[1:2] + cfg * (logits[0:1] - logits[1:2])
        return _sample(logits[..., : self.eos + 1] + self.mask_bb, temperature, top_k, self._next_key())

    # -- generation --------------------------------------------------------

    def frames(self, text: str, *, instruct: Optional[str] = None, ref_audio=None,
               ref_text: Optional[str] = None, voice: Optional[str] = None,
               cfg_scale: Optional[float] = None, temperature: float = 0.9, top_k: int = 50,
               max_frames: int = 750, seed: Optional[int] = None,
               stats: Optional[Stats] = None) -> Iterator[mx.array]:
        """Yield the codes of each audio frame ([16] int32, evaluated)."""
        if seed is not None:
            mx.random.seed(seed)
            self._key = mx.random.key(seed)
        stats = stats if stats is not None else Stats()
        t0 = time.perf_counter()
        m = self.m
        kw = dict(voice=voice, ref_audio=ref_audio, ref_text=ref_text)
        use_cfg = bool(instruct) and cfg_scale not in (None, 1.0)
        cfg = 1.0 if cfg_scale is None else float(cfg_scale)
        prompts = [m._prompt_embeddings(text, instruct=instruct, **kw)]
        if use_cfg:  # the unconditional prompt is the same prompt without the instruction
            prompts.append(m._prompt_embeddings(text, instruct=None, **kw))
        cache = self.bb.make_cache()
        if len(prompts) == 1:
            pad = None
            hidden = self.bb(input_embeddings=prompts[0], cache=cache)[:, -1, :]
        else:
            # One batch of two: left-pad the shorter prompt and mask the padding.
            # RoPE only sees relative positions, so the shift changes nothing.
            L = max(p.shape[1] for p in prompts)
            pad = mx.array([L - p.shape[1] for p in prompts])
            x = mx.concatenate([mx.pad(p, [(0, 0), (L - p.shape[1], 0), (0, 0)]) for p in prompts])
            hidden = self._bb_step(x, cache, pad)
        first = self._first(hidden, cfg, temperature, top_k)
        mx.eval(first, hidden)
        stats.prefill_s = time.perf_counter() - t0
        frame_fn = self._frame_fn(len(prompts), cfg, temperature, top_k)

        def step(first, hidden):
            """Lazy graph: this frame's codes, and the next frame's first token and hidden state."""
            frame = frame_fn(first, hidden, self._next_key())
            ids = frame.reshape(1, 1, self.n_cb)
            if pad is None:
                hidden = self.bb(input_ids=ids, cache=cache)[:, -1, :]
            else:
                emb = self.bb.embed_tokens(ids)
                hidden = self._bb_step(mx.broadcast_to(emb, (2,) + emb.shape[1:]), cache, pad)
            return frame, self._first(hidden, cfg, temperature, top_k), hidden

        # Pipelined: frame t+1 is queued on the GPU before frame t is read back, so the GPU
        # never idles while Python builds the next graph. After EOS one extra frame has been
        # computed and is dropped.
        t1 = time.perf_counter()
        if max_frames <= 0 or int(first.item()) == self.eos:
            return
        cur = step(first, hidden)
        mx.async_eval(*cur)
        for n in range(max_frames):
            nxt = step(cur[1], cur[2]) if n + 1 < max_frames else None
            if nxt is not None:
                mx.async_eval(*nxt)
            frame, first_next, _ = cur
            done = int(first_next.item()) == self.eos
            stats.frames += 1
            stats.loop_s = time.perf_counter() - t1
            yield frame
            if done or nxt is None:
                return
            cur = nxt

    def generate(self, text: str, **kw) -> tuple[mx.array, Stats]:
        """Generate the whole utterance; returns (audio [samples] float32, stats).

        Decodes in chunks with the streaming decoder (identical output), which keeps
        the codec's activations small: peak memory ~4 GB instead of ~6.6 GB at 4 bit."""
        stats = Stats()
        dec = self.m.audio_tokenizer.decoder
        dec.reset_streaming_state()
        chunks, pending = [], []
        for frame in self.frames(text, stats=stats, **kw):
            pending.append(frame)
            if len(pending) == 25:
                t0 = time.perf_counter()
                chunks.append(self._decode_chunk(dec, pending))
                stats.decode_s += time.perf_counter() - t0
                pending = []
        t0 = time.perf_counter()
        if pending:
            chunks.append(self._decode_chunk(dec, pending))
        audio = mx.concatenate(chunks) if chunks else mx.zeros((0,), mx.float32)
        mx.eval(audio)
        stats.decode_s += time.perf_counter() - t0
        return audio, stats

    def stream(self, text: str, chunk_frames: int = 12, first_chunk_frames: int = 3,
               **kw) -> Iterator[mx.array]:
        """Yield audio chunks while generating. One frame is 80 ms of audio; the first
        chunk is short so playback can start early, later chunks are ``chunk_frames``."""
        dec = self.m.audio_tokenizer.decoder
        dec.reset_streaming_state()
        pending, want = [], first_chunk_frames
        for frame in self.frames(text, **kw):
            pending.append(frame)
            if len(pending) == want:
                yield self._decode_chunk(dec, pending)
                pending, want = [], chunk_frames
        if pending:
            yield self._decode_chunk(dec, pending)

    def _decode_chunk(self, dec, frames):
        codes = mx.stack(frames)[None].astype(mx.int32)            # [1, T, 16]
        audio = self.m._audio_vector(dec.streaming_step(mx.transpose(codes, (0, 2, 1))))
        mx.eval(audio)
        return audio

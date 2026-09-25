# articulatio-mlx

Real-time German text-to-speech on Apple Silicon: fast [MLX](https://github.com/ml-explore/mlx)
inference for [Articulatio-DE](https://huggingface.co/eigenhand/Articulatio-DE) and other
[Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) models. Faster than real time on an M3
Pro, with or without guidance, streaming with the first audio after about 0.4 s — and the cloned
German is as intelligible as real recordings (word error rate 4.6 % for the 4-bit MLX model, 4.4 % for
the real recordings of the same sentences).

Model loading, the text encoder, the audio codec and the prompt layout come from
[mlx-audio](https://github.com/Blaizzy/mlx-audio) (pinned to 0.5.6). This package replaces its
per-frame generation loop and fixes its streaming decoder.

## Demo

https://github.com/user-attachments/assets/aaee58ac-1228-48b6-b9f9-7ac4150b0338

13 s of German in a voice the model created itself, one sentence per piece, `speed` 1.25 with the pitch
preserved and 150 ms between sentences: *"Guten Morgen! Heute ist Donnerstag, der fünfundzwanzigste
September. Draußen sind es achtzehn Grad, am Nachmittag zieht von Westen ein Gewitter auf. Vergiss also
den Regenschirm nicht, wenn du später noch zum Bahnhof fährst."*

Synthetic speech generated with Articulatio-DE (GGUF Q8_0 on
[articulatio.cpp](https://github.com/eigenhand/articulatio.cpp); articulatio-mlx produces the same kind of
output); the voice belongs to no real person. Derived from Breeze TTS 2 by BreezeBlue and licensed for research and non-commercial use only: the
clip is an output of the model and falls under the
[BreezeBlue Research and Non-Commercial License](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE),
not under the Apache 2.0 license of this repository.

## Speed

M3 Pro (18 GB), MLX 0.32.2, the released Articulatio-DE in 4 bit, the same German sentence (~7 s of
audio) and seed. RTF = compute time / audio duration, below 1 is faster than
realtime. `python scripts/bench.py MODEL --ref ref.wav --ref-text "..."` reproduces it.

| Case | mlx-audio | articulatio-mlx | Speedup | First audio (streaming) |
|---|---:|---:|---:|---:|
| Voice design | 2.02 | 0.69 | 2.9x | 0.34 s |
| Voice clone | 1.97 | 0.69 | 2.9x | 0.38 s |
| Clone + instruction, `cfg_scale` 3 | 3.74 | 0.87 | 4.3x | 0.61 s |

Memory: the process peaks at about 5.7 GB (3.1 GB of weights). The codec decodes in chunks, and
MLX's pool of freed buffers is capped at 512 MB (`Articulatio.load(..., cache_limit_mb=...)`);
without both, the same run held up to 13 GB.

For comparison, the C++ engine (`articulatio.cpp` built with ggml's Metal backend, Q4_K) runs at about
RTF 4.4 on the same machine, more than four times slower than real time.

## What is different

Each 80 ms audio frame is one backbone step plus 15 dependent depth-decoder steps. mlx-audio
re-runs the depth decoder over the whole prefix at every step, reads each sampled token back to
the CPU (16 syncs per frame) and runs guidance as two separate passes. Here:

- the depth decoder keeps a KV cache, and a frame's 15 steps are one compiled graph;
- sampling (temperature, top-k) stays on the GPU, and frames are pipelined: the next frame is
  queued before the current one is read back, so the GPU never waits for Python (after the end
  of speech one extra frame is computed and dropped);
- with guidance, conditional and unconditional prompts run as one batch of two: the shorter
  prompt is left-padded and masked, which RoPE's relative positions leave unchanged;
- the encoded reference clip is cached per file.

What remains is memory bandwidth, not compute: a frame reads the backbone once and the depth
decoder 15 times, about 3.4 GB at 4 bit, in 47 ms. MLX's quantized matrix-vector kernels stream
at most about 100 GB/s on an M3 Pro, and the depth decoder's layers take exactly the sum of their
kernel times; fusing projections, wired memory and decoding the codec on a second stream brought
nothing. That is also why the GPU and CPU look lightly loaded: both mostly wait for memory.
Beyond this, only fewer bits in the depth decoder or a chip with more bandwidth help.

With greedy decoding the codes are identical to mlx-audio's, frame by frame, including
guidance and the end of speech (`tests/test_parity.py`).

**Streaming fix.** mlx-audio's streaming codec decoder adds the bias of its transposed
convolutions twice where chunks overlap, which puts a click at every chunk boundary.
[`codec_fix.py`](articulatio_mlx/codec_fix.py) corrects that; streamed audio now matches the
one-shot decode to within 1e-4.

## Install

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .
```

Requires Apple Silicon and about 6 GB of free unified memory. Download the 4-bit Articulatio-DE:

```bash
hf download eigenhand/Articulatio-DE-MLX --local-dir articulatio-de-mlx-4bit
```

Any other Breeze TTS 2 model in mlx-audio's format works too, for example one converted from a merged
checkpoint (the adapter merged with `scripts/merge_lora.py` of
[articulatio-training](https://github.com/eigenhand/articulatio-training)):

```bash
python -m mlx_audio.convert --hf-path merged/ --mlx-path articulatio-de-mlx-4bit --quantize --q-bits 4
```

To keep the depth decoder at 8 bit while the rest is 4 bit (3.1 GB instead of 2.9 GB, about 30 %
slower; in the release evaluation of Articulatio-DE it measured no better than plain 4 bit):

```bash
python -m articulatio_mlx.convert merged/ articulatio-de-mlx-4bit-dd8 --bits 4 --depth-bits 8
```

## Use

```bash
articulatio-mlx articulatio-de-mlx-4bit --text "Guten Morgen, heute ist ein schöner Tag." \
  --ref-audio reference.wav --ref-text "Exakter Wortlaut der Sprachprobe." --output out.wav --play
```

The very first run after installing is slower (in one test RTF 2.2 instead of 0.7): the weights come
from disk and the Metal kernels are compiled once. Every later call, and every call after the first in
one process, runs at the speed above.

`--instruction` describes the voice (voice design) or, together with a reference, the delivery;
`--cfg-scale` sets the guidance (on a cloned voice an instruction needs about 3); `--stream` decodes while generating and reports the time to first
audio, `--seed` makes a run repeatable.

```python
from articulatio_mlx import Articulatio

tts = Articulatio.load("articulatio-de-mlx-4bit")
audio, stats = tts.generate("Guten Morgen.", ref_audio="reference.wav", ref_text="...")
for chunk in tts.stream("Guten Morgen.", instruct="Eine ruhige Männerstimme."):
    ...  # float32 chunks at 24 kHz; the first is 3 frames (240 ms), later ones 12 frames
```

## Limitations

- The standard conversion quantizes the depth decoder to 4 bit as well. Its errors feed back
  into the backbone every frame; the C++ engine found that this degrades long passages (muffled
  after roughly 45 s). Check long texts, or split them into sentences.
- `engine.py` uses internals of mlx-audio's Breeze model; before raising the pin, run
  `ARTICULATIO_MLX_MODEL=... pytest`.
- No minimum length before EOS yet (articulatio.cpp blocks EOS for the first steps of a piece);
  in the release evaluation one prompt in 232 ended with no audio.
- No sentence splitting yet: long text in one piece derails after 15–40 s with this model; pass
  one sentence at a time.
- `top_p` and `repetition_penalty` are not implemented (mlx-audio's defaults, 1.0, do nothing).
- No server yet; `articulatio.cpp` has the HTTP and WebSocket API.

## License

Code: Apache 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). The model weights and the audio
they generate are under the
[BreezeBlue Research and Non-Commercial License](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE):
research and non-commercial use only, and only with voices whose speakers have consented.

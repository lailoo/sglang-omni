# LongCat-AudioDiT on SGLang-Omni

This adapter serves LongCat-AudioDiT 1B and 3.5B through `/v1/audio/speech`
using one serialized GPU stage. Plain text uses CFG; a single audio reference
with its transcript uses APG voice cloning. The result is a complete waveform.
The adapter calls the external `AudioDiTModel` and `batch_inference.infer_one`;
it does not implement diffusion layers, model batching, or streaming.

This is a review implementation. See the [validation report and open
items](../../../docs/developer_reference/longcat_audiodit_validation.md)
before deployment. Historical measurements are not validation of the current
rebased PR commit.

## Environment and ModelScope Assets

Use a GPU environment with Omni's pinned PyTorch 2.13.0 stack. Historical
experiments used Python 3.12.3, PyTorch 2.13.0+cu130, Transformers 5.12.1, and
one RTX 5090 with 32 GB of VRAM. The ModelScope downloader below requires
Python 3.11 or later and the `modelscope` package.

The external [LongCat-AudioDiT source](https://github.com/meituan-longcat/LongCat-AudioDiT)
must be checked out at `12c76b51d2a8aa6b6c9af5b25cd5ff8f7aa8178a` and
its runtime dependencies installed without replacing Omni's pinned stack.
An installable, namespaced dependency contract is still an open item: the
current implementation imports top-level `audiodit`, `batch_inference`, and
`utils` from that checkout through `PYTHONPATH`.

Run from the Omni repository root after setting `MODEL_ROOT` to your asset
directory:

```bash
python examples/longcat_audiodit/download_models.py --model 1B \
  --revision 33ed105fc69428d58f20cc95fe189fb3a9115df5 --output-dir "$MODEL_ROOT"
python examples/longcat_audiodit/download_models.py --model 3.5B \
  --revision 61df56cd748712367b9c364899761ccbfc0fcb2d --output-dir "$MODEL_ROOT"
python examples/longcat_audiodit/download_models.py --model tokenizer \
  --revision bba3c3bdfcc89dc6203f35cc5d381836c3d49c7b --output-dir "$MODEL_ROOT"
```

The downloader checks file sizes, rejects LFS pointers, verifies the published
checkpoint SHA256 values, and records a manifest. The tokenizer directory
must contain the serialized Fast tokenizer. Under Transformers 5.12.1,
reconstructing UMT5 with `AutoTokenizer` changed the unknown-token index for
some rare Chinese characters; `PreTrainedTokenizerFast` preserves that backend.

## Serve

Set `SOURCE_ROOT` to the pinned AudioDiT checkout and `REFERENCE_ROOT` to the
directory containing audio that the server may read:

```bash
export PYTHONPATH="$SOURCE_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
sgl-omni serve --model-path "$MODEL_ROOT/LongCat-AudioDiT-1B" \
  --host 127.0.0.1 --port 8000 \
  --allowed-local-media-path "$REFERENCE_ROOT" \
  --inference.factory.tokenizer_path "$MODEL_ROOT/umt5-base"
```

Replace `1B` with `3.5B` for the larger checkpoint. Defaults are 16 steps,
guidance strength 4.0, and initial seed 1024. Sampling rate and maximum
duration come from the checkpoint. Reference audio may be an allowlisted
`file://` URL or a base64 WAV data URI and must have matching `ref_text`.

```bash
curl -sS http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"The weather is clear today.","response_format":"wav"}' \
  --output plain.wav
```

For ordered comparison with the reference implementation, start a fresh
server for each model and dataset split, omit per-request seeds, and do not
send a generation probe before the first measured sample. Explicit request
seeds currently reset global Torch RNG state and affect later unseeded
requests; isolation is pending. Increasing HTTP concurrency currently adds
queueing, not model-level batching.

## Validation on the GPU Host

The focused test entry points are:

```bash
python -m pytest -q \
  tests/unit_test/longcat_audiodit/test_pipeline.py \
  tests/unit_test/pipeline/test_simple_scheduler_concurrent.py \
  tests/unit_test/profiler/test_views.py
```

These model unit tests substitute the external model; they do not establish
real-model quality. Run both checkpoint sizes through the actual HTTP service
and the quality/performance protocol in the validation report before merging.
The current PR preparation did not run local tests, pre-commit, or new GPU
experiments.

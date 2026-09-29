# LongCat-AudioDiT: Review Scope and Historical Validation

## Status

This draft ports the existing experimental adapter onto main
`20329946256a7e4ee3a7de22cfe215893c9c7147`. It includes model discovery,
single-stage speech generation, `inference_mode`, SimpleScheduler timing,
focused tests, and a ModelScope downloader. It is intended for reviewing
changes before completing the remaining integration work.

No local tests, pre-commit checks, or GPU experiments were run while preparing
this PR. Historical test and benchmark results below apply to the experimental
Omni baseline `69ab0ed5e16ef7ca2071f616ba2bac1b201b8c8e` with the adapter and
profiling changes, not to this rebased PR commit. Model tests now use the
scheduler's inbox/outbox contract; the port also accounts for main's renamed
scheduler fields and profiler test helpers.

## Implementation and Adaptation Problems

| Area | Change and reason |
| --- | --- |
| Discovery | Map checkpoint type `audiodit` to `AudioDiTForConditionalGeneration`; one configuration serves both checkpoint sizes. |
| Generation | Reuse the fixed external model. Plain text follows CFG; one reference plus transcript follows APG. Keep 16 steps and guidance 4.0. |
| Tokenization | Load serialized `PreTrainedTokenizerFast`. The Transformers 5.12.1 default reconstruction changed UMT5's unknown-token index from 3 to 2; 14 of 3508 inputs differed before this correction. |
| Audio | Reuse Omni's speech API and waveform output. Path references need the media allowlist; inline bytes are passed through a temporary WAV file. |
| Comparison | Compare decoded sample counts and PCM16 values: SoundFile and Omni quantize differently, so container hashes alone misidentify regressions. |
| Observability | Add scheduler queue-entry and compute start/end events, attach the stage name, and expose queue/compute/completion intervals. |
| Performance | Use `torch.inference_mode()` around actual inference. Keep the serialized execution contract. |

See the [model setup](../../sglang_omni/models/longcat_audiodit/README.md).
The external model requires runtime dependencies compatible with Omni's pinned
Torch stack. In the historical environment, torchaudio v2.11.0 was compiled
against Torch 2.13.0; installing a wheel built against Torch 2.11 was not a
compatible substitute. ASR/SIM scoring used a separate environment with
Transformers 4.44.2 and jiwer 3.0.5, retaining Torch 2.13.0+cu130.

## Fixed Historical Protocol

| Component | Version or revision |
| --- | --- |
| Hardware | One RTX 5090, 32607 MiB VRAM |
| Runtime | Python 3.12.3; PyTorch 2.13.0+cu130; Transformers 5.12.1 |
| AudioDiT source | `12c76b51d2a8aa6b6c9af5b25cd5ff8f7aa8178a` |
| ModelScope 1B | `33ed105fc69428d58f20cc95fe189fb3a9115df5` |
| ModelScope 3.5B | `61df56cd748712367b9c364899761ccbfc0fcb2d` |
| ModelScope UMT5 | `bba3c3bdfcc89dc6203f35cc5d381836c3d49c7b` |
| Seed-TTS data mirror | `zhaochenyang20/seed-tts-eval`, revision `8f5e1aa2a35d42f42e940074c1983358b9491f89` |
| Seed-TTS scoring code | `BytedanceSpeech/seed-tts-eval`, commit `752f4297f090c46bb1a55a1f7439e5944ddefe8d` |

For each model, evaluate EN 1088, ZH 2020, and ZH-Hard 400, totaling 7016
requests across both models. Each model/split starts a fresh process, seeds
once after loading with 1024, and generates in metadata order without
per-request seeds or prior probes. The server binds to localhost and returns
24 kHz WAV. Compare against the same fixed external source with identical
references, transcripts, tokenizer, APG settings, and request order.

## Quality Results

Both the original `no_grad` service and the optimized service completed
7016/7016 requests, with matching sample counts and PCM16 maximum differences
of at most one integer unit against the reference-source output. All 7016
decoded PCM hashes matched exactly between the two Omni versions. Saved Omni
WAVs were independently scored with the same ASR/SIM evaluator.

| Model | Split | Omni WER/CER | Source baseline WER/CER | Omni SIM | Source baseline SIM |
| --- | --- | ---: | ---: | ---: | ---: |
| 1B | EN | 2.07750% | 2.12046% | 0.762552 | 0.762647 |
| 1B | ZH | 1.16272% | 1.17168% | 0.811803 | 0.811817 |
| 1B | ZH-Hard | 6.44382% | 6.47912% | 0.787058 | 0.787047 |
| 3.5B | EN | 1.81340% | 1.82931% | 0.785381 | 0.785479 |
| 3.5B | ZH | 1.13544% | 1.14681% | 0.818541 | 0.818572 |
| 3.5B | ZH-Hard | 6.03660% | 6.02346% | 0.796361 | 0.796379 |

The maximum aggregate error-rate difference is 0.043 percentage points and
all mean SIM differences are below 0.00010. Small PCM differences can change
ASR transcripts. Omni's English WER exceeds the published values by roughly
0.30 and 0.31 percentage points: these results are a close reproduction, not
an exact paper reproduction. Full scores concern reference-conditioned APG;
plain CFG has only historical HTTP smoke and mock unit-test coverage.

## Performance and Profiling

The historical load study used the first 32 EN requests plus four warmups per
concurrency level, with one original run and two optimized runs on an exclusive
GPU. Profiling ran separately. Throughput is successful requests per second:

| Model and run | Concurrency 1 | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: |
| 1B original | 1.014 | 1.065 | 1.026 | 1.073 |
| 1B inference_mode | 1.303 | 1.367 | 1.362 | 1.371 |
| 1B optimized repeat | 1.301 | 1.369 | 1.384 | 1.385 |
| 3.5B original | 0.726 | 0.742 | 0.753 | 0.744 |
| 3.5B inference_mode | 0.743 | 0.760 | 0.760 | 0.759 |
| 3.5B optimized repeat | 0.740 | 0.760 | 0.760 | 0.759 |

The 1B optimized repeat improved throughput by 28.3-34.9% over the sole
original run. Concurrency-8 HTTP P95 changed from 7.686 to 5.820 seconds.
The 3.5B differences of approximately 1-2% do not establish a reliable gain.
These small load cohorts do not establish production stability or precise
tail percentiles.

At concurrency 4, 1B mean scheduler queue/compute times changed from
2285.9/946.5 ms to 1763.2/735.3 ms. The 3.5B compute interval stayed around
1329 ms. More HTTP concurrency increased queueing without true model batching.
Sampled EN memory peaks were approximately 6871/16179 MiB for 1B/3.5B; longer
ZH-Hard requests reached about 8.8/20.9 GiB.

The short CUDA traces contained 1294/1737 ms (1B) and 3717/4468 ms (3.5B)
of FP32 GEMM/summed kernel durations. These sums are not GPU wall-clock
utilization. Scheduler-thread CPU operators were missing, so the data do not
identify an exact CPU bottleneck. A 3.5B TF32 probe was faster but failed the
PCM16 tolerance on all 12 requests; TF32 is not enabled in this implementation.

## Open Items Before Merge

- Re-run focused tests and both checkpoint smokes on the rebased PR commit
  using the target GPU environment. Historical focused tests passed 28/28;
  the current commit has not been tested. Run pre-commit on that environment.
- Isolate explicitly seeded requests without changing the ordered unseeded
  baseline. Current global RNG resets affect subsequent unseeded requests.
- Map adapter input errors to HTTP 400 after the worker/client boundary;
  multi-reference rejection currently becomes HTTP 500. Public validation
  catches some other malformed inputs earlier.
- Establish a fixed, installable dependency and license contract for the
  external AudioDiT source; top-level `utils` imports can collide.
- Extend real-audio base64, malformed-request, cancellation/error, and batch
  profiler coverage. Existing fake-model tests do not validate audio decoding
  or sampling numerics.
- Coordinate the generic profiler work with upstream
  [#2419](https://github.com/sgl-project/sglang-omni/pull/2419) and
  [#2378](https://github.com/sgl-project/sglang-omni/pull/2378). At the recorded
  review point both were open; neither replaces the complete SimpleScheduler
  event addition here. Recheck their state before upstream submission.

## Further Optimization Candidates

Measure model-internal CPU/GPU stages before selecting the next change.
Repeated-reference conditioning reuse should evaluate the existing
`ReferenceEncodeService`, including cache identity, memory bounds, and RNG
effects. CPU preparation/GPU overlap needs evidence that preparation is
material. True batching needs per-sample durations, reference lengths, masks,
output trimming, RNG isolation, and memory-aware admission. These are proposed
experiments, not implemented or measured gains in this PR.

For a new performance study, reuse the speech benchmark's repeat, arrival-rate,
seed, and fingerprint support; establish A/A noise, then alternate matched
A/B runs. Any batch, kernel, or dtype change needs a declared quality gate and
independent scoring. Raw audio, weights, full traces, and machine credentials
are not included in the repository.

# dart_llm

A tiny language model (LLM) written in pure Dart. It implements a minimal GPT-style network with a custom Tensor/autograd engine, trains on a small built‑in text sample, and prints a sampled continuation.

## Prerequisites

* **Dart SDK**: 3.7.2 (or compatible with `pubspec.yaml`)
  - Check: `dart --version`

## Setup

```bash
dart pub get
```

## Run

```bash
dart run bin/dart_llm.dart
```

What you’ll see:

* **Training progress**: lines like `iter 25 train=… val=…` up to 300 iters (default).
* **Sample output**: a generated text continuation from the built‑in prompt.

## What this does

* Tokenization: byte‑level via `ByteTokenizer` (vocab size 256).
* Model: small GPT with multi‑head self‑attention, MLP, layer norm.
* Data: short embedded excerpt in `_loadToyText()` in `bin/dart_llm.dart`.
* Objective: next‑token prediction with cross‑entropy; optimizer: AdamW.

## Adjusting hyperparameters

Edit `main()` in `bin/dart_llm.dart`:

* **Block size**: `final blockSize = 64;`
* **Model dims**: `nEmbed`, `nHead`, `nLayer`
* **Training**: `batchSize`, `iters`, optimizer lr/weightDecay
* **Prompt and sample length**: `prompt`, `sample(model, ids, steps, rng)`

For better samples, increase data size (replace `_loadToyText()` with your own corpus) and/or increase iters/model size, noting runtime cost.

## Running tests

```bash
dart test
```

## Troubleshooting

* __SDK mismatch__: Ensure `dart --version` satisfies `environment.sdk` in `pubspec.yaml`.
* __Slow/garbled samples__: Expected with tiny data and few iters. Increase data/iters/model.
* __Out of memory / performance__: Reduce `nEmbed`, `nLayer`, `blockSize`, or `batchSize`.

## Project structure

* `bin/dart_llm.dart` – all model, tensor ops, training loop, and main entry.
* `test/` – example tests.
* `pubspec.yaml` – SDK constraints and dev dependencies.

## License

MIT (or project’s chosen license). Update as appropriate.

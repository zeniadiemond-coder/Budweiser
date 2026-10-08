# TrainV6

A Python toolkit that finetunes the timbre (.vtb2) and pitch (.vpit) models of a VOCALOID6 voice package on a recorded corpus.

Prerequisites: Python 3.13+, PyTorch (a CUDA GPU is used by default), and a VOCALOID6 voice package to start from.

# READ.

You cannot train a pitch model for English or Chinese. This **will** be added.
You cannot finetune nor train a timing model from scratch. This **will** be added.
You cannot train a pitch model from scratch. This **will** be added.
You cannot train/finetune several speakers at once. This **will** be added.
From-scratch training is experimental. 

## Install

```sh
pip install -e .
```

## Usage

```sh
trainv6 <command> [options]
```

Run `trainv6 <command> --help` for all options. The timbre pipeline is, in order:

```sh
trainv6 build-dataset --base-package <voice-package> --corpus <corpus> --out <dataset>
trainv6 train <base.vtb2> <dataset> <run-directory>
trainv6 evaluate <run-directory>/best.pt <corpus> <dataset> <eval-directory>
trainv6 package <run-directory>/best.pt <voice-package> <out-package>
```

The pitch pipeline is:

```sh
trainv6 pitch-build <base.vpit> <corpus-manifest.json> <pitch-dataset> --scores <scores> --kana-table <table>
trainv6 pitch-train <base.vpit> <pitch-dataset> <run-directory> --base-row <row>
trainv6 pitch-package <run-directory>/best.pt <voice-package> <out-package> --name <character>
```

## Configs

Every command takes `--config file.json` (repeatable). A config is a JSON object of option values, keyed by option name as written on the command line, with `-` or `_`; later files override earlier ones and command-line options override all of them. Positional arguments cannot come from a config.

```json
{"steps": 600000, "lr": 1e-3, "final-crop": 1088, "session-rows": 4}
```

## Corpus

A corpus is either a directory of `<name>.wav` recordings with `<name>.lab` phoneme labels (also accepted as `wav/` and `lab/` subdirectories), or a `manifest.json`:

```json
{
  "recordings": [
    {"id": "song_000", "wav": "/path/song_000.wav", "split": "train",
     "labels": [{"start_100ns": 0, "end_100ns": 5000000, "phoneme": "sil"}]}
  ]
}
```

Labels are converted to model phonemes according to `--language` of `build-dataset`: `ja` takes UTAU/NNSVS symbols, `en` takes ARPAbet (stress digits ignored), `zh` takes the model's own phoneme names (e.g. `iE_n+0`). The engine's language classes (0 Japanese, 1 English, 2 Chinese) and these phoneme names were checked against captures of the engine. Without a manifest, `--validation` holds out that many recordings, chosen by a hash of their names.

`pitch-build` also needs the manifest, a scores directory with one `<id>/` folder per recording id holding `<id>.ust` and `<id>.lab`, and an oto2lab kana-to-phonemes table. Every lyric of the score must expand through the table to the phonemes in the `.lab`; a song where one does not is skipped and listed in the dataset's `index.json`.

## Timbre training

`build-dataset` takes the model, its native speaker row and its `ParamExp` control from the voice package, then measures harmonic and noise targets, phase and decay from the recordings and writes them as shards.

`train` runs three stages. `--steps` sets the total (default 6000, at most 1000000) and splits it 5%/25%/70%; `--stage-steps` gives the three lengths explicitly instead:

1. Teacher-forced training on the recorded targets
2. The probability of feeding the model its own output rises from 0 to 1
3. Fully free-running training on longer crops (`--final-crop`, default 1088 frames; `--crop` defaults to 544), which is where the audible gains come from

The defaults follow the recipe stored in Yamaha's models: AdamW (β2 0.999, weight decay 0.01), learning rate 1e-3 decaying to `--lr-end` 2e-5, and an exponential moving average of the weights (decay 0.999) that is what gets validated and saved. The encoder is shared between voices in Yamaha's models, so it is frozen unless `--train-encoder` or `--scratch`. On an 8 GB card the CUDA graphs of these crops may not fit; use `--no-cuda-graph`.

By default it adds a new embedding row, named `--speaker-name` (default: the run directory's name), and trains that. The row starts as a copy of the dataset's speaker row, or from a `rank-speakers` result given as `--speaker-init`. `--adapt-speaker` trains the dataset's speaker row in place instead. `--scratch` randomly initializes the synthesis, encoder and control nets and the dataset's speaker row, and implies `--adapt-speaker`. `--resume` continues from `state.pt`. `--extra-speaker NAME=DATASET` (repeatable) trains further new speakers in the same run, each from its own dataset built for the same base model and language, for example a Soft and a Power speaker of one voice. Each starts from its dataset's speaker row, all are exported under their names, and validation is the mean over speakers. `package` and `--speaker NAME` choose which one the package points at. `--session-rows N` adds N new rows instead of one, trains each on its own contiguous group of recordings, and keeps the row that validates best.

The run directory holds `best.pt` (lowest free-running validation error), `last.pt`, `state.pt` and `log.jsonl`. Judge a checkpoint by `evaluate`, not by the training loss. It renders base and trained audio next to the recording with the reference decoder, and writes `metrics.json`. The plain `base` and `trained` columns use the constant control a shipped voice runs with; `_frame_control` columns use a control contour measured from the recording itself and flatter the model. `--calibrate` searches the constant control that zeroes the loudness difference and prints the matching `ParamExp`.

Other commands:

- `refresh-targets <base.vtb2> <corpus> <dataset>`: re-measure the targets of an existing dataset
- `add-control <corpus> <dataset>`: derive the per-frame vocal-effort control from the recordings and store it in the shards
- `rank-speakers <base.vtb2> <dataset> <out>`: rank the base model's speaker rows by how well they fit the corpus
- `rank-bases <build-base.vtb2> <dataset> <out> <candidate.vtb2>...`: rank other base models' rows on the same dataset; incompatible models are listed as skipped. Build a new dataset for the winner before training on it
- `record-history <checkpoint> <dataset> <out>`: record free-running histories of a checkpoint, which `train --histories` mixes back in
- `export <checkpoint> <out.vtb2>`: write the trained model alone

The feedback noise mode (`--noisy`) must match the mode the model is exported with. It defaults to the base model's.

## Pitch training

`pitch-build` extracts f0 from the recordings and aligns it with the scores, then writes the dataset with phoneme-boundary jitter (`--variants`, `--jitter`) to make training robust to timing-model mismatch.

`pitch-train` adds a new pitch row and trains it towards `--base-row`, an existing row of the base model. The new row starts from `--init` (a row number, or `mean`; default `--base-row`). It trains in two stages, set by `--stage-steps` (default 2500 1500): teacher-forced, then a ramp to free-running. `pitch-export` writes the model alone under the speaker name given by `--name`. `pitch-package` copies a voice package and replaces its pitch model, naming the new row and the character `--name`.

## Timing training

No.

## Voice package

The input and output packages have the layout OpenV6 reads: `v6.yaml`, `bank.json` and the model files it names. `package` and `pitch-package` copy the base package and replace the trained model, and update `NPIdx` in `bank.json` when the speaker or pitch row changed. `package` refuses a checkpoint whose base model is not the one in the package.

Set `ParamExp` in `bank.json` to what the model trained with: the control input is `(ParamExp - 3000) / 3000`.

## Layout

- `trainv6/`: the CLI, dataset, training, evaluation and export code
- `trainv6/reference/`: the host-free reference synthesizer used to render audio and verify exports

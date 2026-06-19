# Speech Emotion Recognition with Annotation Uncertainty

This repository contains the implementation for the paper:

**Learning from Annotation Uncertainty: Entropy-Aware Curriculum for Speech Emotion Recognition**

The code supports speech emotion recognition (SER) experiments using WavLM-based models, distribution-based supervision, and entropy-aware analysis of annotation uncertainty on MSP-Podcast 2.0.

## Overview

Most SER systems train on hard consensus labels, collapsing annotator disagreement into a single emotion category. This repository implements an alternative framework that models annotator uncertainty using emotion-label distributions and entropy-aware training strategies.

The implementation includes:

- WavLM-based SER training
- hard-label and distribution-based supervision
- primary and merged annotation-distribution targets
- entropy-aware curriculum learning
- multitask categorical emotion and VAD modeling
- evaluation utilities for categorical and distributional metrics

## Repository structure

- `train_ser.py` — main training script
- `dataio_msp_podcast.py` — MSP-Podcast data loading utilities
- `prepare_msp_podcast.py` — MSP-Podcast preparation utilities
- `validate_config.py` — configuration validation
- `hparams/wavlm_ser_example.yaml` — example WavLM SER training configuration
- `utils/` — model, augmentation, logging, metric, and training utilities


## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

## Data

This repository does **not** include MSP-Podcast data, audio files, extracted features, model checkpoints, or generated experiment outputs. It also does not include generated feature manifests or prepared JSON splits.

Users must obtain MSP-Podcast according to the official dataset license and update local paths in the configuration file before running experiments.

Example paths to update:

```yaml
data_folder: /path/to/MSP-PODCAST2.0/Audios
json_root: /path/to/MSP-PODCAST2.0/i26_json
rir_folder: /path/to/RIRS_NOISES/real_rirs_isotropic_noises
```

To create the default hard-label manifest directory expected by `hparams/wavlm_ser_example.yaml`, run:

```bash
python prepare_msp_podcast.py \
  --data_root /path/to/MSP-PODCAST2.0/Audios \
  --labels_folder /path/to/MSP-PODCAST2.0/Labels \
  --output_dir /path/to/MSP-PODCAST2.0/i26_json/hard_full9 \
  --class_preset 9 \
  --merge_preset none \
  --consensus_only
```

## Example usage

```bash
python train_ser.py hparams/wavlm_ser_example.yaml
```

## Citation

If you use this repository, please cite the associated paper:

```bibtex
@inproceedings{omidi2026annotationuncertainty,
  title     = {Learning from Annotation Uncertainty: Entropy-Aware Curriculum for Speech Emotion Recognition},
  author    = {Omidi, Zahra and Hansen, John H. L.},
  booktitle = {Proceedings of Interspeech},
  year      = {2026}
}
```

## Notes

This repository is intended as a public code release accompanying the paper above. It excludes private data, local machine paths, cluster-specific scripts, and generated experiment artifacts.

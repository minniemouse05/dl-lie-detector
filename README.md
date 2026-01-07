# 3 Modalities and 1 Lie Detector

Calibrated Deep Learning for Real-World Deception Detection

**Authors:** Minnie Liang, Charis Ching, Renee Hong

**[Read the full paper](https://dl-lie-detector.vercel.app)**

---

## Overview

We investigate deception detection using three modalities—**video**, **audio**, and **text transcripts**—from real court trial clips. Beyond accuracy and F1, we focus on **probability calibration** (ECE) so model confidence is meaningful in high-stakes settings.

## Key Contributions

- **Real-world data:** clips from actual court trials (either deceptive or truthful)
- **Strong unimodal baselines:** Fine-tuned RoBERTa (text), ResNet18 (audio), ResNet18+Transformer (video)
- **Multimodal fusion:** Gated fusion across all three modalities; bimodal (video+text) variant
- **Calibration-first evaluation:** ECE metrics with temperature and Platt scaling

---

## Dataset

| Property | Value |
|----------|-------|
| Total clips | 121 (61 deceptive, 60 truthful) |
| Duration | 20–50 seconds each |
| Source | Real court trial footage |

Each clip includes video frames, extracted audio, and human-annotated transcripts. Labels are based on trial outcomes and corroborating evidence.

---

## Models

### Text: RoBERTa

- Tokenized to 256 tokens max
- Fine-tuned encoder with differential learning rates

### Audio: ResNet18

- Mel spectrograms (64 bins, 224×224)
- Pretrained ResNet18 with custom classifier head

### Video: ResNet18 + Transformer

- 20 uniformly sampled frames → ResNet18 features
- 2-layer Transformer encoder with 4-head attention
- Also tested: Bidirectional GRU baseline

### Fusion

- **Trimodal:** Gated fusion of video, audio, text embeddings → MLP classifier
- **Bimodal:** Video + text concatenation → MLP classifier

---

## Results

### Cross-Validation Accuracy

| Model | Accuracy |
|-------|----------|
| Frozen RoBERTa | 0.48 |
| Fine-tuned RoBERTa | 0.70 |
| Audio (ResNet18) | 0.69 |
| Video (GRU) | 0.70 |
| Video (Transformer) | **0.73** |

### Test Set Performance

| Model | Accuracy | AUC | F1 | ECE (raw) |
|-------|----------|-----|-----|-----------|
| Video Transformer | **0.80** | 0.846 | 0.783 | 0.23 |
| Video + Text | 0.76 | **0.853** | 0.727 | 0.11 |

Video is the strongest single modality. The bimodal model trades slight accuracy for better calibration, which is important for legal contexts where confident errors are costly.

---

## Limitations

- **Limited dataset:** 121 clips from similar English-speaking courtrooms
- **Label noise:** Truth labels inferred from verdicts; potential systemic bias
- **Overfitting risk:** Large models on limited data
- **Sequential training:** Encoders trained separately before fusion

---

## Future Work

- Joint multimodal training (contrastive learning)
- Fairness evaluation across demographic groups
- More expressive calibration and uncertainty estimation

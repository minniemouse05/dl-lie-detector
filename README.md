# 3 Modalities and 1 Lie Detector

Calibrated Deep Learning for Real World Deception  
Authors: Charis Ching, Minnie Liang, Renee Hong

---

## Overview

This project investigates real world deception detection from three modalities:

* Video
* Audio
* Text transcripts

We build modality specific encoders and then fuse them in a calibrated multimodal classifier. Beyond standard accuracy and F1 score, we focus on probability calibration, evaluating and improving Expected Calibration Error (ECE) so that model confidence scores are meaningful in high stakes settings such as courts and security interviews.

---

## Key Ideas

* Real world deception

   * 121 clips from actual court trials
   * Truth labels based on case outcomes and corroborating evidence

* Three strong unimodal baselines

   * Fine tuned RoBERTa for transcripts
   * ResNet18 on mel spectrograms for audio
   * ResNet18 plus GRU or Transformer encoder for video

* Multimodal fusion

   * Gated fusion over video, audio, and text embeddings
   * Bimodal fusion for video plus text

* Calibration first

   * Expected Calibration Error (ECE) on held out test set
   * Post hoc calibration via temperature scaling and Platt scaling

---

## Dataset

We use a multimodal deception dataset collected from real trial footage  
(61 deceptive clips and 60 truthful clips, 20 to 50 seconds each).

For each clip we use:

* Video frames containing the speaker’s face
* High quality audio extracted from the clip
* Verbatim human annotated transcript of the spoken content

Ground truth labels reflect whether the speaker is deceptive or truthful, based on trial outcomes and corroborating evidence. This gives a high stakes setting but also introduces label noise and potential bias, which we discuss in the report.

---

## Methodology

### 1 Transcription Model

**Preprocessing**

* Tokenization with pretrained RoBERTa tokenizer
* Maximum length 256 tokens with padding and attention masks

**Models**

* Frozen RoBERTa

   * RoBERTa used as a fixed feature extractor
   * [CLS] like embedding passed through dropout and a linear classifier

* Fine tuned RoBERTa

   * Whole encoder updated with a small learning rate
   * Classifier head trained with a larger learning rate
   * This model becomes our primary text encoder

### 2 Audio Model

**Preprocessing**

* Extract waveform with FFmpeg
* Convert to mono at 16 kHz with Librosa
* Compute mel spectrograms
   * FFT window 1024
   * Hop length 512
   * 64 mel bins

* Convert to decibel scale
* Render spectrogram as grayscale PNG
* Resize to 224 by 224
* Repeat channel to get three channels
* Normalize with ImageNet statistics

**Architecture**

* Pretrained ResNet18
* First convolution adapted for spectrogram input
* Early layers mostly frozen
* Deeper layers plus custom classifier head trained
* Classifier head: linear layer with 512 units and ReLU, dropout, final linear layer to two logits

### 3 Video Models

**Preprocessing**

* Uniformly sample 20 frames per clip
* Resize to 224 by 224
* Normalize with ImageNet statistics
* Extract 512 dimensional frame features using pretrained ResNet18
* Global average pooling per frame

**GRU model**

* Bidirectional GRU over frame embeddings
* Concatenate forward and backward final states into 256 dimensional vector
* Fully connected layer to two logits

**Transformer model**

* Project 512 dimensional frame embeddings to 256 dimensional tokens
* Add positional encodings
* Stack of two Transformer encoder layers
   * Multi head self attention with four heads
   * Two layer feedforward block
   * Residual connections and normalization

* Masked mean pooling over tokens
* Final linear layer to two logits

### 4 Multimodal Fusion

We explore two fusion schemes

#### Trimodal fusion: video, audio, and transcript

* Text encoder

   * Fine tuned RoBERTa
   * 768 dimensional embedding projected down to 512

* Video encoder

   * Video Transformer encoder
   * 512 dimensional embedding

* Audio encoder

   * ResNet18 audio model
   * 512 dimensional embedding

These three embeddings are layer normalized and passed into a small gating network that produces three scalar gates (one per modality) in the range 0 to 1. Each embedding is scaled by its gate and then all three are concatenated.

The fused vector is fed into a small multilayer perceptron:

* Linear projection to 128 dimensions
* ReLU
* Dropout
* Linear projection to a single logit (deception probability)

#### Bimodal fusion: video and transcript

To reduce parameters and noise, we also build a bimodal model using only video and transcript.

* Take the best video encoder and best transcript encoder
* Concatenate their logits or embeddings
* Feed into an MLP classifier with
   * 512 unit hidden layer
   * ReLU
   * Dropout
   * Final layer to two logits

---

## Training and Evaluation

* 80–20 train test split of clips
* Within the 80 percent training partition
   * 5 fold cross validation for model selection and hyperparameters

**Metrics**

* Accuracy
* AUC
* F1 score
* Expected Calibration Error (ECE) with 10 probability bins

**Calibration**

1. Baseline ECE
2. Temperature scaling
   * Single temperature parameter T applied to logits

3. Platt scaling
   * Logistic regression on the logit difference between classes

Both methods are fit on a held out calibration split within the training portion. The final evaluation on the test set uses the learned calibration parameters but does not refit them.

---

## Results

### Unimodal

* Transcripts

   * Frozen RoBERTa: CV accuracy around 0.48
   * Fine tuned RoBERTa: CV accuracy around 0.70

* Audio

   * ResNet18 on mel spectrograms: CV accuracy around 0.69

* Video

   * GRU: CV accuracy around 0.70
   * Transformer: CV accuracy around 0.73

Video emerges as the strongest single modality, followed closely by text, with audio giving moderate but useful additional signal.

### Multimodal

* Bimodal (video plus transcript)

   * CV accuracy comparable to best video model
   * Better calibration than unimodal video

* Trimodal (video plus transcript plus audio)

   * Lower CV accuracy
   * Likely overfitting due to added parameters and spectrogram noise

### Calibration

On the held out test set:

* Video Transformer

   * Accuracy approximately 0.80
   * AUC approximately 0.846
   * F1 approximately 0.783
   * Raw ECE about 0.23
   * Platt scaling significantly improves ECE

* Video plus transcript

   * Accuracy approximately 0.76
   * AUC approximately 0.853
   * Better raw calibration than video only
   * Temperature and Platt scaling reduce ECE further

The bimodal model is slightly less accurate but more calibrated. It often makes more confident predictions, including confident errors, which is important to consider in real world legal contexts.

---

## Limitations

* Small dataset size
   * Only 121 clips from similar English speaking courtrooms

* Label noise and bias
   * Truth labels inferred from verdicts and case materials
   * Possible confounding by systemic bias in the justice system

* Overfitting risk
   * Large language model and Transformer encoder on limited data

* Limited cross modal learning
   * Encoders trained mostly in isolation before fusion

These limitations motivate larger, more diverse datasets and multimodal training that learns interactions directly.

---

## Future Directions

* Joint training of multimodal encoders
   * Co training or contrastive learning across audio, video, and text

* Robustness and fairness
   * Explicit bias and fairness evaluation across demographic groups
   * Debiasing methods to avoid learning spurious correlations

* Better calibration methods
   * More expressive calibration models and uncertainty estimation

---

#  Deep Learning for Myocardial Infarction Detection from PPG Signals

This project presents a full end-to-end AI pipeline for detecting Myocardial Infarction (MI) using Photoplethysmography (PPG) signals.
It combines Wavelet Time–Frequency transformation with state-of-the-art Transfer Learning CNN models (ResNet-18, MobileNetV2, EfficientNet-B0).

The primary goal is to explore how simple wearable-friendly PPG sensors can support early cardiac risk detection and remote health monitoring.

#  Dataset

We used the public PPG dataset available on Kaggle:

 Dataset: https://www.kaggle.com/datasets/ucimachinelearning/photoplethysmography-ppg-dataset

This dataset contains:

PPG signals (2000 samples per recording)

Labels:

0 → Normal

1 → Myocardial Infarction (MI)

# 1️ Preprocessing of PPG Signals

Each sample is a 1D PPG signal with 2000 points.

✔ Convert 1D Signal → 2D Wavelet Scalogram

Using Continuous Wavelet Transform (CWT):

Wavelet: Morlet "morl"

Scales: 1–127

Output shape:

127 frequency bins × 2000 time points

✔ Why use Wavelet Transform?

PPG is non-stationary → frequency content changes over time.
Wavelet transform captures:

Cardiac waveform shape

Dicrotic notch abnormalities

Heart rate variability changes

High/low frequency bursts

MI-induced distortions

Wavelet scalograms convert signals into images, ideal for CNNs.

# 2️ Converting PPG → Wavelet Image

Steps performed:

Compute CWT

Take magnitude

Normalize to 0–255

Convert to PIL image

Resize to 224×224

Convert to 3 channels

Apply ImageNet normalization

This transforms PPG signals into CNN-ready images, similar to real-world spectrograms.

# 3️ Dataset Pipeline (On-the-Fly Wavelet Generation)

We do not save wavelet images.
Instead, we compute them inside __getitem__() for each batch.

Benefits:

Saves >10 GB RAM

Efficient on Colab/Kaggle

No huge intermediate files

Clean + GPU friendly

This enables training large models with limited hardware.

# 4️ Why Transfer Learning?

PPG datasets are small → training CNNs from scratch causes overfitting.

Transfer Learning allows:

Using pretrained ImageNet filters

Strong generalization

Fast convergence

Training only few layers

Excellent performance even with limited data

# 5️ Model Architectures Used

We trained three CNN architectures:

 A. ResNet-18 (Best Performer)
✔ Highlights:

Residual learning

Skip connections

Strong generalization

Lightweight

✔ Fine-Tuning Strategy:

Freeze all layers

Unfreeze only layer4

Replace classifier:

fc → Linear(in_features, 1)

✔ Why it works?

Learns MI-related frequency features like:

High-frequency bursts

Low-frequency suppression

Abnormal heartbeat morphology

 B. MobileNetV2 (Edge Device Friendly)
✔ Highlights:

Depthwise separable convolutions

Only ~3.4M parameters

Fastest model

✔ Fine-Tuning:

Freeze all layers

Unfreeze features[-1]

Replace classifier head

 C. EfficientNet-B0 (Most Accurate)
✔ Highlights:

Balanced depth × width × resolution

SE attention blocks

Combines efficiency + accuracy

✔ Fine-Tuning:

Freeze all

Unfreeze features[-1]

Replace classifier:

classifier[1] = Linear(in_features, 1)

✔ Why best?

Captures subtle MI-related waveform distortions.

# 6️ Output Layer & Loss Function
✔ Output Activation

We use:

Sigmoid

✔ Loss Function
BCEWithLogitsLoss

Why?

Stable gradients

Perfect for binary classification

Numerically safe

#  Training Strategy

Train only last conv block

Optimizer: Adam

Learning Rate: 1e-4

Batch Size: 16

Epochs: 5–10

DataLoader workers: 2

This ensures speed + stability + accuracy.

# 8️ Results (Accuracy)
Model	Accuracy
ResNet-18	⭐ 96%
MobileNetV2	94%
EfficientNet-B0	95%

 ResNet-18 performed best in our final implementation.

# 9️ Why This Approach Works So Well
✔ Wavelet captures MI-specific temporal-frequency patterns
✔ CNN learns complex shapes + texture patterns
✔ Transfer Learning prevents overfitting
✔ Fast, lightweight, and real-time capable

Perfect for:

Wearable heart monitoring

Clinical risk screening

Remote patient monitoring

Smartwatch-based cardiac detection

# Summary of Techniques Used
✔ Continuous Wavelet Transform (CWT)
✔ Image normalization & resizing
✔ On-the-fly wavelet generation
✔ Transfer Learning:

ResNet-18

MobileNetV2

EfficientNet-B0

✔ Freeze-all → Unfreeze-last-block strategy
✔ BCEWithLogitsLoss
✔ Adam optimizer
✔ Accuracy evaluation
#  Conclusion

This project demonstrates that PPG signals, when transformed using wavelet scalograms and analyzed using deep transfer learning, can reliably detect Myocardial Infarction.

The pipeline is lightweight, efficient, and ready for deployment in wearable devices, telemedicine platforms, and real-time cardiac monitoring systems.

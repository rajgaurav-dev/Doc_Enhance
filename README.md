# Document Image Restoration – Experimental Study

This repository presents a comparative study of deep learning–based approaches for **document image restoration**, focusing on real-world degradations such as noise, blur, shadows, uneven illumination, and geometric distortions.

The work is structured into **two experiments**, where the second experiment is motivated directly by the observations and limitations of the first.

---

## 📌 Motivation

Document images captured using mobile cameras often suffer from:
- Motion blur and sensor noise
- Strong shadows and non-uniform lighting
- Perspective distortion and page warping
- Poor contrast affecting text readability

Generic image restoration models are not always well-suited for these challenges.  
This repository explores that hypothesis empirically.

---

## 🧪 Experiments Overview

### 🔬 Experiment 1: NAFNet (Baseline)
- Uses **NAFNet**, a lightweight CNN-based image restoration model
- Fine-tuned on paired document images
- Serves as a **baseline** to evaluate generic restoration capability

➡️ Result: Handles denoising and mild blur but struggles with shadows and geometry.

---

### 🔬 Experiment 2: DocRes (Document-Specific)
- Uses a **task-aware document restoration architecture**
- Handles multiple restoration tasks such as:
  - Deblurring
  - Deshadowing
  - Dewarping
  - Appearance enhancement
  - Binarization
- Uses **prompt-based conditioning and Transformer attention**

➡️ Result: Significantly improved restoration quality for real document images.

---

## 📁 Repository Structure



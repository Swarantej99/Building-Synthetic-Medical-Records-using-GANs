# Building Synthetic Medical Records using GANs

---

### Overview

This project demonstrates how to generate **synthetic medical records** using **Generative Adversarial Networks (GANs)** in Python.  
By simulating realistic, privacy-safe healthcare data, we can enable **data augmentation**, **testing**, and **machine learning experimentation** without exposing actual patient information.

The focus is on **data preprocessing discipline**, **training stability**, and **ethical generative modeling**.

---

### Project Objectives

- Build a **GAN** from scratch using **PyTorch**  
- Preprocess structured healthcare data (numeric + categorical)  
- Train a **Generator** and **Discriminator** to produce realistic records  
- Explore **privacy-compliant synthetic dataset creation**  
- Demonstrate responsible AI practices in healthcare data synthesis  

---

### Technologies Used

- **Python 3.9+**  
- **PyTorch** – Deep learning framework  
- **Scikit-learn** – Encoding and normalization  
- **Pandas**, **NumPy** – Data manipulation  
- **Matplotlib / Seaborn** – Plotting and visualization  

---

### Dataset

The dataset (`Follow-up_Records.csv`) contains anonymized patient-level attributes such as:

| Category | Example Features |
|-----------|-----------------|
| Demographics | Age, Weight, BMI |
| Clinical | Heart Rate, Systolic BP, HbA1c |
| Lifestyle | Exercise Frequency, Sleep Hours |
| Binary Conditions | Neuropathy, Retinopathy, UTI |
| Categorical Notes | Follow-up outcomes |

> ⚠️ You may replace this dataset with your own medical dataset.  
Ensure categorical columns are **encoded** and numeric columns **scaled between [-1, 1]**.

---

### Model Architecture

#### Generator  
- Input: Random noise (`latent_dim = 64`)  
- Hidden layers with **LeakyReLU** activations  
- Output activation: **Tanh**

#### Discriminator  
- Input: Real or synthetic record  
- Hidden layers with **LeakyReLU**  
- Output activation: **Sigmoid** (real vs fake)

Both models train in an adversarial setup where the generator tries to **fool** the discriminator while the discriminator tries to **differentiate**.

---

### Training Loop

The training follows a two-step adversarial approach:

1. **Train the Discriminator** on real and fake samples.  
2. **Train the Generator** to fool the Discriminator.

**Loss Function:** Binary Cross-Entropy (BCE)  
**Optimizer:** Adam (lr = 0.0002, betas = (0.5, 0.999))  
**Epochs:** ~2000  

**Example training log:**
Epoch [0/2000]  D_loss: 0.6890  G_loss: 0.6736

Epoch [200/2000]  D_loss: 0.1810  G_loss: 1.6598

Epoch [400/2000]  D_loss: 0.7586  G_loss: 1.5357

Epoch [600/2000]  D_loss: 0.2532  G_loss: 2.6024

Epoch [800/2000]  D_loss: 0.2025  G_loss: 3.7295

Epoch [1000/2000]  D_loss: 0.2216  G_loss: 2.2439

Epoch [1200/2000]  D_loss: 0.1681  G_loss: 2.2328

Epoch [1400/2000]  D_loss: 0.0373  G_loss: 2.5638

Epoch [1600/2000]  D_loss: 0.0565  G_loss: 2.1592

Epoch [1800/2000]  D_loss: 0.0421  G_loss: 3.5040

### Ethical Use

This project promotes **ethical and responsible AI** practices.  
Generated data should be used **only for research, model prototyping, or educational** purposes — **not for real-world medical decisions**.

### Common Pitfalls

- Skipping data normalization for **Tanh** output  
- Overtraining **Generator** leading to mode collapse  
- Ignoring column-wise scaling and encoding consistency  
- Using large batch sizes for small medical datasets 

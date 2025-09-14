# 🌌 Convolutional Neural Networks for Galaxy Morphological Classification

Welcome to the repository of my **Bachelor Thesis** at **Vrije Universiteit Amsterdam (Department of Mathematics)**.  

This work applies **Convolutional Neural Networks (CNNs)** to the classification of galaxy morphology using astronomical imaging data.

The aim is to automate a traditionally manual process, improving **accuracy, scalability, and reproducibility**, and to provide tools that support the astronomical community in analyzing the massive datasets produced by modern sky surveys.

---

## 📖 Abstract

Since the dawn of humanity, we have shown a timeless fascination with the heavens, seeking to understand the universe that surrounds us.  
In today’s age of **data-intensive science**, the study of galaxy morphology plays a fundamental role in astrophysics.  

The objective of this thesis is to apply **Convolutional Neural Networks (CNNs)** to galaxy morphological classification. A mathematical and computational approach is developed to enhance the **accuracy, scalability, and automation** of this task, traditionally performed manually.

This work aims to contribute a complementary tool for the astronomical community that can accelerate the analysis of large-scale datasets from modern sky surveys, and thus support humanity’s ongoing effort to understand the universe.

---

## 📄 Thesis
- Full text: [Thesis.pdf](./Thesis.pdf)  
- Author: **Antón Fidalgo Pérez**  
- Supervisor: **dr. Hannah Rocio Santa Cruz Baur**  
- Faculty of Science, Vrije Universiteit Amsterdam  
- Year: 2025  

---

## 📂 Repository Structure
```

.
├── Thesis.pdf                        # Full thesis document
├── Thesis CNN.ipynb                  # Main Jupyter Notebook (code & results)
├── Requirements.txt                  # Dependencies
└── README.md                         # This file

````

---

## ⚙️ Requirements & Setup

Clone the repository (or download) and install dependencies:

```bash
pip install -r requirements.txt
````

**requirements.txt**

```
# Data manipulation and visualization
numpy
pandas
matplotlib
seaborn
opencv-python
scipy

# Progress bar
tqdm

# Deep learning
tensorflow

# Machine learning utilities
scikit-learn
```

---

## 🧠 Methodology

1. **Dataset**

   * Based on **Galaxy Zoo** (original dataset \~500,000 galaxies).
   * Reduced to \~120,000 balanced images across 8 morphological classes.

2. **Preprocessing**

   * Image resizing & normalization
   * Label encoding and one-hot encoding
   * Train/validation/test splits

3. **CNN Architecture**

   * Convolutional layers: 32 → 64 → 128 filters (3x3 kernels)
   * ReLU activations, Batch Normalization
   * MaxPooling and Dropout for regularization
   * Global Average Pooling
   * Dense(256) fully-connected layer
   * Output: Softmax over 8 classes

4. **Training**

   * Optimizer: **Adam**
   * Loss: **Categorical Crossentropy**
   * Callbacks: EarlyStopping, ReduceLROnPlateau
   * Framework: TensorFlow/Keras
   * Hardware: NVIDIA RTX 3060 GPU (CUDA)

5. **Evaluation**

   * Metrics: Accuracy, F1, Precision, Recall, ROC-AUC
   * Confusion matrix analysis
   * Tests for robustness under image rotations
---

## 🖼 Visualizations

### Galaxy Samples
Example inputs from the dataset:
<p align="center">
  <img src="./Images/Image%20Sample.png" alt="Galaxy Samples" width="500"/>
</p>

---

### CNN Model Architecture
Overview of the convolutional network used for classification:
<p align="center">
  <img src="./Images/CNN%20Model.png" alt="CNN Model" width="500"/>
</p>

---

### Training Performance
Accuracy and loss curves during training:
<p align="center">
  <img src="./Images/Model%20Accuracy%20and%20Loss.png" alt="Training Curves" width="500"/>
</p>

---

### Evaluation Metrics
Per-class precision, recall, F1, accuracy, and AUC:
<p align="center">
  <img src="./Images/Evaluation%20Metrics%20per%20Class.png" alt="Evaluation Metrics" width="500"/>
</p>

---

### Model Comparison
Performance compared with Zhu et al. (2019):
<p align="center">
  <img src="./Images/ResNets%20(Zhu%202019)%20Comparison%20%20.png" alt="Model Comparison" width="500"/>
</p>

---

## 🙌 Citation

If you use this work, please cite:

```bibtex
@thesis{Fidalgo2025,
  title={Convolutional Neural Networks for Galaxy Morphological Classification},
  author={Antón Fidalgo Pérez},
  school={Vrije Universiteit Amsterdam},
  year={2025},
  url={https://github.com/Antonfidalgo/Convolutional-Neural-Networks-for-Galaxy-Morphological-Classification}
}
```
---

## 📜 License

* **Code**: MIT License — see [`LICENSE`](./LICENSE)
* **Thesis (PDF)**: CC BY-NC 4.0 (Non-Commercial, Attribution required)

---

## 👤 Author

**Antón Fidalgo Pérez**

Mathematician & Data Analyst

Vrije Universiteit Amsterdam

Inditex Netherlands

📧 antonfidalgoperez@gmail.com

💼 [LinkedIn Profile](https://www.linkedin.com/in/antonfidalgo)

🔗 [GitHub Profile](https://github.com/Antonfidalgo)

---

<p align="center">
  <img src="./Images/Galaxy%20Design.png" alt="Galaxy Morphology Diagram" width="400"/>
</p>

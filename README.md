[![lang-en](https://img.shields.io/badge/lang-en-orange.svg)](README.md)
[![lang-tr](https://img.shields.io/badge/lang-tr-blue.svg)](README.tr.md)

> 📌 Looking for notebooks?  
> 📎 [Jump to Kaggle Notebooks ⬇️](#-kaggle-notebooks)

---

# 📊 Sentiment Analysis Project with Deep Learning (CNN, LSTM, GRU)

This project is a comprehensive **Natural Language Processing (NLP)** study aimed at performing **sentiment analysis** on user comments and textual sentences. Our goal is to classify texts as **Positive**, **Negative**, or **Neutral** using and comparing different deep learning models.

## 🚀 Models Used
- **Convolutional Neural Network (CNN)**
- **Gated Recurrent Unit (GRU)**
- **Long Short-Term Memory (LSTM)**

---

## 📌 Insights Gained During the Project

### 1️⃣ Starting with BERT and Transition Process
- Initially aimed to leverage **BERT** for its strong contextual capabilities.
- However, due to **excessive training times** and **high hardware requirements**, we **transitioned to a lighter GRU-based model**.

---

### 2️⃣ CNN Model: Improvements and Challenges

#### ⚠️ Overfitting Problem
**To address the overfitting encountered in early trials, the following solutions were applied:**
- Increased `Dropout` rates.
- Added `L2 Regularization`.
- Included an additional `Conv1D` layer and used `LeakyReLU` activation.
- Reduced dense layer sizes to lower model complexity.
- Data was split into 75% training / 13% validation / 12% test.

#### 🧠 Architectural Changes
- **Replaced basic Sequential structure with Functional API** to build a `Multi-Scale` CNN architecture.
- Added parallel `Conv1D` layers with 4 different kernel sizes and combined `GlobalMax + GlobalAvgPooling`.
- Restructured dense layers as 512→256→128.

#### 🔁 Data Augmentation & Class Imbalance
- Applied 4x **augmentation** for the Negative class.
- Assigned 5x **class weight** to the Negative class.
- These highly specific methods **prevented the model from learning the true data distribution**.

#### 🛠️ Hyperparameter Tuning
| Parameter        | Before | After |
|------------------|--------|--------|
| Vocabulary Size  | 15K    | 25K    |
| Sequence Length  | 120    | 200    |
| Embedding Dim    | 128    | 200    |
| Batch Size       | 32     | 8      |
| Epoch            | 15     | 25     |

#### ⚙️ Loss Function and Optimization
- Used `Focal Loss` (`γ=3.0`, `α=0.4`)
- Applied threshold tuning.

#### 📈 CNN Model Results
- F1-score for Negative class: **91%**
- Overfitting risk was minimized.
- Careful tuning of `validation_split`, `max_len`, and `patience` led to better performance.

---

### 3️⃣ GRU Model: Restructuring

#### Initial Problems:
- Observed **low test accuracy and high loss** in the beginning.

#### Applied Changes:
- Model structure: `GRU(64)` → `GRU(64)` + `BatchNorm` + `GlobalMaxPooling1D`
- Dense layers: `Dense(64)` + `Dense(32)`
- Dropout: 0.3 → 0.5, also added `SpatialDropout1D(0.3)`
- Optimizer: `Adam(lr=0.0005)`
- Advanced text cleaning (URLs, Turkish characters, minimum length)
- EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
- Added L2 regularization

#### 📊 GRU Model Results
- F1-score for Negative class: **77–78%**
- Data split: **60% / 20% / 20%** for training/validation/test

---

### 4️⃣ LSTM Model: Overall Success

- Test accuracy: **92.9%**
- Macro F1-score: **89.3%**
- Overfitting difference only **1%**
- F1-score for Negative class: **77–78%**
- Early stopped training at 9 epochs

---

## ⚠️ Observations and Evaluations

- **Negative class** was the most challenging across all models.
- Methods like excessive augmentation and focal loss **did not always yield performance gains**.
- **CNN model** showed the best performance especially for the `negative` class.
- Overfitting was **kept under control** across all models, with differences within the 0.1–0.2 range.

---

## ✅ Conclusion

- **For Neutral and Positive classes**, all models delivered high and consistent performance.
- **For the Negative class, the CNN architecture not only provided the highest performance but also emerged as the most stable model overall.**
- The technical gains and hyperparameter configurations achieved during training serve as a strong reference for similar NLP projects.

---

## 🧠 Keywords
`#SentimentAnalysis` `#CNN` `#GRU` `#LSTM` `#DeepLearning` `#NLP` `#FocalLoss` `#Overfitting` `#ModelComparison` `#TurkishTextClassification`

---

## ▶️ Live Preview
Below is the demonstration video of the project:

[![Project Demo](https://img.youtube.com/vi/vdBjLsf7te4/0.jpg)](https://www.youtube.com/watch?v=vdBjLsf7te4)

---

## 🔗 Kaggle Notebooks

Below are the links to Kaggle notebooks used in this project:

- [📘 GRU-based Sentiment Analysis](https://www.kaggle.com/code/ceblock/turkish-sentiment-analysis-w-gru-deep-learning)
- [📘 LSTM-based Sentiment Analysis](https://www.kaggle.com/code/ceblock/turkish-sentiment-analysis-w-lstm-deep-learning)
- [📘 CNN-based Sentiment Analysis](https://www.kaggle.com/code/ceblock/turkish-sentiment-analysis-w-cnn-deep-learning)

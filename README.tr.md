[![lang-en](https://img.shields.io/badge/lang-en-orange.svg)](README.md)
[![lang-tr](https://img.shields.io/badge/lang-tr-blue.svg)](README.tr.md)

> 📌 Kaggle notebooklarını görmek için:  
> 📎 [Kaggle Projelerine Git ⬇️](#-kaggle-notebooklar)

---

# 📊 Derin Öğrenme ile Duygu Analizi Projesi (CNN, LSTM, GRU)

Bu proje, kullanıcı yorumları ve metin cümleleri üzerinden **duygu analizi** gerçekleştirmeyi hedefleyen kapsamlı bir **doğal dil işleme (NLP)** çalışmasıdır. Amacımız; metinleri **Pozitif**, **Negatif** veya **Nötr** olarak sınıflandırmak için farklı derin öğrenme modellerini karşılaştırmalı olarak incelemektir.

## 🚀 Kullanılan Modeller
- **Convolutional Neural Network (CNN)**
- **Gated Recurrent Unit (GRU)**
- **Long Short-Term Memory (LSTM)**

---

## 📌 Proje Sürecinde Edinilen Tecrübeler

### 1️⃣ BERT ile Başlangıç ve Geçiş Süreci
- Başlangıçta **BERT** modelinin güçlü bağlamsal yeteneklerinden yararlanmak hedeflendi.
- Ancak **aşırı uzun eğitim süreleri** ve **yüksek donanım gereksinimi** nedeniyle **GRU tabanlı daha hafif modele geçiş** yapıldı.

---

### 2️⃣ CNN Modeli: Geliştirmeler ve Zorluklar

#### ⚠️ Aşırı Öğrenme (Overfitting) Sorunu
**İlk denemelerde karşılaşılan overfitting sorunu için şu çözümler uygulandı:**
- `Dropout` oranları artırıldı.
- `L2 Regularization` eklendi.
- Ek `Conv1D` katmanı ve `LeakyReLU` aktivasyon fonksiyonu kullanıldı.
- Dense katmanlar küçültülerek model karmaşıklığı azaltıldı.
- Veri %75 eğitim / %13 doğrulama / %12 test olarak bölündü.

#### 🧠 Mimari Değişiklikler
- **Basit Sequential yapı yerine Functional API** ile çok ölçekli (`Multi-Scale`) CNN mimarisi kuruldu.
- 4 farklı kernel boyutuyla paralel `Conv1D` katmanları ve `GlobalMax + GlobalAvgPooling` eklendi.
- Dense katmanlar 512→256→128 şeklinde yeniden yapılandırıldı.

#### 🔁 Veri Artırma & Sınıf Dengesizliği
- Negatif sınıf için 4 kat **veri artırma (augmentation)** uygulandı.
- Negatif sınıfa 5 kat fazla **class weight** verildi.
- Aşırı özel bu yöntemler, modelin **gerçek veri dağılımını öğrenmesini engelledi**.

#### 🛠️ Hiperparametre Ayarlamaları
| Parametre       | Önce | Sonra |
|-----------------|------|-------|
| Vocabulary Size | 15K  | 25K   |
| Sequence Length | 120  | 200   |
| Embedding Dim   | 128  | 200   |
| Batch Size      | 32   | 8     |
| Epoch           | 15   | 25    |

#### ⚙️ Kayıp Fonksiyonu ve Optimizasyon
- `Focal Loss` kullanıldı (`γ=3.0`, `α=0.4`)
- Eşik ayarı (`threshold tuning`) uygulandı.

#### 📈 CNN Modeli Sonuçları
- Negatif sınıf F1-skoru: **%91**
- Aşırı öğrenme riski minimumda tutuldu.
- Hassas `validation_split`, `max_len`, `patience` gibi ayarlamalarla başarı artırıldı.

---

### 3️⃣ GRU Modeli: Yeniden Yapılanma

#### İlk Sorunlar:
- Başlangıçta **düşük test doğruluğu ve yüksek kayıp** gözlemlendi.

#### Uygulanan Değişiklikler:
- Model yapısı: `GRU(64)` → `GRU(64)` + `BatchNorm` + `GlobalMaxPooling1D`
- Dense katmanlar: `Dense(64)` + `Dense(32)`
- Dropout oranı: 0.3 → 0.5, ek olarak `SpatialDropout1D(0.3)`
- Optimizer: `Adam(lr=0.0005)`
- Gelişmiş metin temizleme (URL, Türkçe karakter, minimum uzunluk)
- EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
- L2 regülarizasyon eklendi

#### 📊 GRU Modeli Sonuçları
- Negatif sınıf F1-skoru: **%77–78**
- Eğitim/Doğrulama/Test bölünümü: **%60 / %20 / %20**

---

### 4️⃣ LSTM Modeli: Genel Başarı

- Test doğruluğu: **%92.9**
- Makro F1-skor: **%89.3**
- Aşırı öğrenme farkı sadece **%1**
- Negatif sınıf F1-skoru: **%77–78**
- Eğitim erken durduruldu (9 epoch)

---

## ⚠️ Gözlemler ve Değerlendirmeler

- **Negatif sınıf**, tüm modellerde en zor sınıf oldu.
- Aşırı veri artırımı ve focal loss gibi yöntemler **her zaman performans artışı sağlamadı**.
- **CNN modeli**, özellikle `negatif` sınıfta en yüksek başarıyı gösterdi.
- Aşırı öğrenme (overfitting) tüm modellerde **kontrol altında tutuldu**, farklar %0.1–0.2 bandında kaldı.

---

## ✅ Sonuç

- **Nötr ve Pozitif sınıflarda**, tüm modeller yüksek ve istikrarlı performans sergilemiştir.
- **Negatif sınıf özelinde, CNN mimarisi yalnızca yüksek başarı sağlamakla kalmamış, aynı zamanda genel olarak en kararlı sonuçları veren model olarak öne çıkmıştır.**
- Eğitim sürecinde elde edilen teknik kazanımlar ve hiperparametre ayarları, benzer NLP projeleri için güçlü bir referans niteliği taşımaktadır.

---

## 🧠 Anahtar Kelimeler
`#SentimentAnalysis` `#CNN` `#GRU` `#LSTM` `#DeepLearning` `#NLP` `#FocalLoss` `#Overfitting` `#ModelComparison` `#TurkishTextClassification`

---

## ▶️ Tanıtım Videosu
Aşağıda projenin tanıtım videosu yer almaktadır:

[![Project Demo](https://img.youtube.com/vi/vdBjLsf7te4/0.jpg)](https://www.youtube.com/watch?v=vdBjLsf7te4)

---

## 🔗 Kaggle Notebooklar

- [📘 GRU Tabanlı Duygu Analizi](https://www.kaggle.com/code/ceblock/turkish-sentiment-analysis-w-gru-deep-learning)  
- [📘 LSTM Tabanlı Duygu Analizi](https://www.kaggle.com/code/ceblock/turkish-sentiment-analysis-w-lstm-deep-learning)  
- [📘 CNN Tabanlı Duygu Analizi](https://www.kaggle.com/code/ceblock/turkish-sentiment-analysis-w-cnn-deep-learning)

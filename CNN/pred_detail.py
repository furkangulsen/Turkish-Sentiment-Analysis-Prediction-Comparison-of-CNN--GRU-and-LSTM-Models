import os
# ---- Logları ve uyarıları en erken aşamada bastır ----
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
import sys
import re
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.preprocessing.sequence import pad_sequences
# ---- Rich kütüphanesi ile gelişmiş UI ----
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich.text import Text
from rich import box
import matplotlib.pyplot as plt
import seaborn as sns

console = Console()

# ---- Dosya yolları ----
MODEL_PATH = 'CNN_balanced_model_v6.keras'
TOKENIZER_PATH = 'tokenizer_balanced_v6.pickle'
LABELENCODER_PATH = 'labelencoder_balanced_v6.pickle'

# ---- Metin temizleme ----
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-ZçğıöşüÇĞIİÖŞÜ\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---- Güven seviyesi hesaplama ----
def calculate_confidence_metrics(probabilities):
    """Her tahmin için detaylı güven metrikleri hesaplar"""
    metrics = []
    for prob in probabilities:
        max_prob = np.max(prob)
        second_max = np.partition(prob, -2)[-2]
        
        # Güven metrikleri
        confidence = max_prob
        certainty = max_prob - second_max  # En yüksek ile ikinci en yüksek arasındaki fark
        entropy = -np.sum(prob * np.log(prob + 1e-8))  # Entropi (belirsizlik)
        
        # Güven seviyesi kategorisi
        if confidence > 0.8 and certainty > 0.3:
            confidence_level = "Çok Yüksek"
            confidence_color = "bright_green"
        elif confidence > 0.6 and certainty > 0.2:
            confidence_level = "Yüksek"
            confidence_color = "green"
        elif confidence > 0.4:
            confidence_level = "Orta"
            confidence_color = "yellow"
        else:
            confidence_level = "Düşük"
            confidence_color = "red"
        
        metrics.append({
            'confidence': confidence,
            'certainty': certainty,
            'entropy': entropy,
            'confidence_level': confidence_level,
            'confidence_color': confidence_color
        })
    
    return metrics

# ---- Sınıf dağılımını analiz etme ----
def analyze_class_distribution(probabilities, class_names):
    """Tüm sınıflar için olasılık dağılımını analiz eder"""
    distribution_analysis = {}
    
    for i, class_name in enumerate(class_names):
        class_probs = probabilities[:, i]
        distribution_analysis[class_name] = {
            'ortalama': np.mean(class_probs),
            'std': np.std(class_probs),
            'min': np.min(class_probs),
            'max': np.max(class_probs),
            'medyan': np.median(class_probs)
        }
    
    return distribution_analysis

# ---- Model yükleme ----
console.log("[bold cyan]CNN Model yükleniyor (derlemeden)...[/]")
model = load_model(
    MODEL_PATH,
    custom_objects={'Orthogonal': Orthogonal},
    compile=False
)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
console.log("[bold green]CNN Model yüklendi![/]")

console.log("[bold cyan]Tokenizer yükleniyor...[/]")
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
console.log("[bold green]Tokenizer yüklendi![/]")

console.log("[bold cyan]LabelEncoder yükleniyor...[/]")
with open(LABELENCODER_PATH, 'rb') as f:
    le = pickle.load(f)
console.log("[bold green]LabelEncoder yüklendi![/]")

# ---- Yorumları alma ----
if len(sys.argv) > 1:
    yorumlar = sys.argv[1:]
else:
    yorumlar = [
        "Bu ürün gerçekten çok kötüydü, hiç memnun kalmadım.",
        "Harika bir alışverişti, teşekkür ederim!",
        "Kargo çok yavaş geldi ama ürün güzel.",
        "Bu kişi Bilgisayar Mühendisi mezunu.",
        "Orta seviyede bir ürün, fena değil ama çok da iyi değil.",
        "Kesinlikle tavsiye etmiyorum, berbat kalite!",
        "Ürün beklediğimden daha iyi çıktı, çok memnun kaldım.",
        "Teslimat sorunsuz ve hızlıydı, teşekkürler.",
        "Ambalaj çok özensizdi, biraz daha dikkat edilmeli.",
        "Ürün tasarımı güzel ama işlevselliği tatmin etmedi.",
        "Gayet güzel bir alışveriş deneyimiydi, tekrar alabilirim.",
        "Teknik destek çok yetersizdi, sorunuma çözüm bulamadılar.",
        "Ortalama bir ürün, ne kötü ne de çok iyi.",
        "Kargo paketlemesi mükemmeldi, sorunsuz ulaştı.",
        "Bu ürün fiyatına göre beklentimi karşılamadı, hayal kırıklığı yaşadım.",
        "Bu cihazın modeli X123, özellikleri standart.",
        "Sipariş numaram 456789, teslim tarihi 10 Haziran.",
        "Ürün açıklaması oldukça detaylı ve bilgilendiriciydi."
    ]


# ---- Ön işleme ve tahmin ----
console.log("[bold cyan]CNN ile tahmin işlemi başlıyor...[/]")
temiz_metinler = [clean_text(y) for y in track(yorumlar, description="Metinler temizleniyor...")]
seqs = tokenizer.texts_to_sequences(temiz_metinler)
maxlen = model.input_shape[1]
padded = pad_sequences(seqs, maxlen=maxlen, padding='post', truncating='post')

# Tahmin yapma
probs = model.predict(padded, verbose=0)
predicted_indices = np.argmax(probs, axis=1)
predicted_labels = le.inverse_transform(predicted_indices)

# Sınıf isimlerini al
all_classes = le.classes_
confidence_metrics = calculate_confidence_metrics(probs)

# ---- Ana Sonuçlar Tablosu ----
main_table = Table(title="🎯 CNN Detaylı Tahmin Sonuçları", expand=True, box=box.ROUNDED)
main_table.add_column("#", justify="right", style="dim", width=3)
main_table.add_column("Yorum", justify="left", width=40)
main_table.add_column("Ana Etiket", justify="center", width=12)
main_table.add_column("Güven", justify="center", width=8)
main_table.add_column("Güven Seviyesi", justify="center", width=12)
main_table.add_column("Belirsizlik", justify="center", width=10)

for i, (txt, lab, metrics) in enumerate(zip(yorumlar, predicted_labels, confidence_metrics), 1):
    # Ana etiket rengi
    lab_lower = lab.lower()
    if lab_lower.startswith('pos'):
        label_color = "bright_green"
    elif lab_lower.startswith('neg'):
        label_color = "bright_red"
    else:
        label_color = "bright_yellow"
    
    # Kısa metin gösterimi
    short_text = txt[:37] + "..." if len(txt) > 40 else txt
    
    main_table.add_row(
        str(i),
        short_text,
        f"[{label_color}]{lab}[/{label_color}]",
        f"{metrics['confidence']:.3f}",
        f"[{metrics['confidence_color']}]{metrics['confidence_level']}[/{metrics['confidence_color']}]",
        f"{metrics['entropy']:.3f}"
    )

console.print(main_table)

# ---- Detaylı Olasılık Dağılımı ----
console.print("\n")
prob_table = Table(title="📊 CNN - Tüm Sınıflar İçin Olasılık Dağılımı", expand=True, box=box.DOUBLE_EDGE)
prob_table.add_column("Yorum #", justify="center", width=8)

# Tüm sınıflar için sütun ekle
for class_name in all_classes:
    prob_table.add_column(f"{class_name}", justify="center", width=10)

# Her yorum için olasılık değerlerini ekle
for i, prob_row in enumerate(probs, 1):
    row_data = [str(i)]
    for j, prob_val in enumerate(prob_row):
        class_name = all_classes[j]
        # En yüksek olasılığı vurgula
        if j == predicted_indices[i-1]:
            color = "bright_green" if prob_val > 0.6 else "green"
            row_data.append(f"[{color}]{prob_val:.3f}[/{color}]")
        elif prob_val > 0.1:  # Düşük ama dikkate değer olasılıklar
            row_data.append(f"[yellow]{prob_val:.3f}[/yellow]")
        else:
            row_data.append(f"{prob_val:.3f}")
    
    prob_table.add_row(*row_data)

console.print(prob_table)

# ---- İstatistiksel Analiz ----
distribution_stats = analyze_class_distribution(probs, all_classes)

stats_table = Table(title="📈 CNN - Sınıf Bazlı İstatistiksel Analiz", expand=True)
stats_table.add_column("Sınıf", justify="left")
stats_table.add_column("Ortalama", justify="center")
stats_table.add_column("Std. Sapma", justify="center")
stats_table.add_column("Min", justify="center")
stats_table.add_column("Max", justify="center")
stats_table.add_column("Medyan", justify="center")

for class_name, stats in distribution_stats.items():
    stats_table.add_row(
        class_name,
        f"{stats['ortalama']:.3f}",
        f"{stats['std']:.3f}",
        f"{stats['min']:.3f}",
        f"{stats['max']:.3f}",
        f"{stats['medyan']:.3f}"
    )

console.print("\n")
console.print(stats_table)

# ---- Özet Panel ----
toplam_yorum = len(yorumlar)
yuksek_guven = sum(1 for m in confidence_metrics if m['confidence'] > 0.7)
dusuk_guven = sum(1 for m in confidence_metrics if m['confidence'] < 0.5)

ozet_text = f"""
📊 CNN Model - Genel Özet:
• Toplam Yorum: {toplam_yorum}
• Yüksek Güvenli Tahmin: {yuksek_guven} ({yuksek_guven/toplam_yorum*100:.1f}%)
• Düşük Güvenli Tahmin: {dusuk_guven} ({dusuk_guven/toplam_yorum*100:.1f}%)
• Ortalama Güven: {np.mean([m['confidence'] for m in confidence_metrics]):.3f}
• Tespit Edilen Sınıflar: {len(all_classes)}
"""

console.print(Panel(ozet_text, title="🎯 CNN Analiz Özeti", border_style="green"))

# ---- Alternatif Tahminler ----
console.print("\n")
alt_table = Table(title="🔄 CNN - İkinci ve Üçüncü En Yüksek Olasılıklar", expand=True)
alt_table.add_column("#", justify="center", width=3)
alt_table.add_column("1. Tahmin", justify="center", width=15)
alt_table.add_column("2. Tahmin", justify="center", width=15)
alt_table.add_column("3. Tahmin", justify="center", width=15)

for i, prob_row in enumerate(probs, 1):
    # En yüksek 3 olasılığı bul
    top3_indices = np.argsort(prob_row)[-3:][::-1]
    
    predictions_str = []
    for idx in top3_indices:
        class_name = all_classes[idx]
        prob_val = prob_row[idx]
        predictions_str.append(f"{class_name}\n({prob_val:.3f})")
    
    alt_table.add_row(str(i), *predictions_str)

console.print(alt_table)

print("\n" + "="*80)
print("✅ CNN modeli ile detaylı analiz tamamlandı!")
print("💡 Bu analiz size CNN modelinin her tahmini için güven seviyesi, alternatif tahminler ve")
print("   tüm sınıflar için olasılık dağılımını göstermektedir.")
print("="*80)
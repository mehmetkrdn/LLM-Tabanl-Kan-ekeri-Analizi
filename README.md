# 🩺 Diyabet Asistanı

Görüntü + RAG + XGBoost + LLM Tabanlı Glukoz Tahmini ve Yorumlama Sistemi

Bu proje, diyabetli bireylerin öğün sonrası (t+120 dk) glukoz seviyelerini tahmin etmek ve bu tahmini güvenli, kısa ve açıklayıcı bir metinsel yorumla desteklemek amacıyla geliştirilmiş uçtan uca bir yapay zekâ tabanlı karar destek sistemidir.

Sistem, kullanıcıdan minimum manuel giriş alarak şu adımları otomatik olarak gerçekleştirir:

Öğün Fotoğrafı
   ↓
Yemek Tanıma (Food-101 + ResNet50)
   ↓
Besin Bilgisi Geri Getirme (RAG)
   ↓
Sayısal Glukoz Tahmini (XGBoost)
   ↓
Metinsel Yorumlama (LLM + LoRA)

# Projenin Amacı

Bu çalışmanın temel amaçları şunlardır:

Öğün fotoğrafından otomatik yemek tanıma yapmak

Tanımlanan yemeğe ait makro besin bilgilerini (karbonhidrat, protein, yağ) Retrieval-Augmented Generation (RAG) yaklaşımıyla geri getirmek

Bu bilgiler ve CGM geçmişi kullanılarak t+120 dakika glukoz değerini sayısal olarak tahmin etmek

Sayısal tahmini, tıbbi tavsiye içermeyen, güvenli ve kısa bir LLM çıktısı ile açıklamak

RAG + LLM entegrasyonunun diyabet destek sistemlerindeki etkinliğini deneysel olarak göstermek

Bu sistem, karar verici değil, bilgilendirici ve açıklayıcı bir yardımcı olarak tasarlanmıştır.

🧠 Kullanılan Modeller (Tam İsimleriyle)
🍽️ Yemek Tanıma

ResNet50

Food-101 Dataset

Best Validation Accuracy: %81.83

📚 RAG (Besin Bilgisi Geri Getirme)

Embedding Model: sentence-transformers/all-MiniLM-L6-v2

Retrieval: Dense Retrieval

Similarity Metric: Cosine Similarity

Top-K: Ayarlanabilir

📈 Glukoz Tahmini

Model: XGBoost Regressor

Girdiler:

Karbonhidrat, Protein, Yağ

CGM geçmişi (t-120, t-60, t0)

🤖 Metinsel Yorumlama (LLM + LoRA)

LoRA ile fine-tune edilmiş aşağıdaki büyük dil modelleri kullanılmıştır:

# google/gemma-2b-it

# meta-llama/Llama-2-7b-chat-hf

# mistralai/Mistral-7B-Instruct-v0.2

# Qwen/Qwen2.5-3B-Instruct

⚠️ LLM yalnızca yorumlayıcı rolündedir.
Sayısal tahmin üretmez, tıbbi tavsiye vermez.

# 🧩 Sistem Özellikleri

✅ Uçtan uca otomatik akış

✅ Sabit ve doğrulanabilir 4 satırlık LLM çıktı formatı

✅ Placeholder ve halüsinasyon engelleme

✅ Türkçe çıktı zorunluluğu

✅ Normal / Uyarı durumu sınıflandırması

✅ Deneysel loglama ve tekrar edilebilirlik

# 📊 Performans Değerlendirme Metrikleri
🔹 LLM Çıktı Formatı

4 satır format uyumu

Format başarı oranı

🔹 Durum Performansı

Accuracy (Normal / Uyarı)

Confusion Matrix

False Alarm Rate

Recall

Macro-F1

🔹 Metin Kalitesi

BLEU

ROUGE-L (F1)

🔹 RAG Performansı

Coverage (geri getirme başarısı)

Best similarity (avg / median / min / max)

Top-K ortalama benzerlik

# 🖥️ Uygulama

Uygulama Streamlit tabanlıdır ve tek dosya üzerinden çalışır:

streamlit run app.py

# Kurulum
pip install -r requirements.txt / !pip install -r requirements.txt

# Gerekli başlıca kütüphaneler:
torch
torchvision
xgboost
sentence-transformers
transformers
peft
streamlit
sacrebleu
rouge-score
# ⚠️ Yasal ve Etik Uyarı
Bu proje akademik ve araştırma amaçlıdır.
❌ Tıbbi teşhis veya tedavi önermez
❌ İnsülin / ilaç dozu hesaplamaz
✅ Yalnızca bilgilendirici ve açıklayıcı çıktı üretir
📌 Akademik Katkı
Bu çalışma:
Görüntü işleme
Retrieval-Augmented Generation
Tabular regresyon
Large Language Models
yaklaşımlarını tek bir diyabet destek sistemi altında birleştirerek literatüre bütüncül bir örnek sunmaktadır.

# 👤 Geliştirici
# Mehmet Kordon

# 🏥 MedBot — Chatbot Konsultasi Medis Bahasa Indonesia

Proyek Praktikum NLP (Semester 3): chatbot tanya-jawab kesehatan berbahasa Indonesia menggunakan **TF-IDF + Cosine Similarity** (dan opsi Sentence-BERT) di atas 5.000 pasang tanya-jawab asli pasien–dokter dari Alodokter.com.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DHOsss/medbot-nlp/blob/main/dataset/medical_chatbot_NLP.ipynb)

## ✨ Fitur

- Dataset: 5.000 QnA medis bahasa Indonesia (Alodokter) dengan auto-kategorisasi 16 kategori
- Preprocessing: lowercase, hapus stopwords (Indonesia + Inggris), lemmatization
- Retrieval jawaban: TF-IDF (n-gram 1–2) + cosine similarity, atau Sentence-BERT
- Konteks percakapan (context-aware) & threshold "tidak yakin"
- 2 tampilan UI: ipywidgets (di notebook) dan **Gradio** (web, `http://127.0.0.1:7860`)
- Analisis dataset + evaluasi batch

## 📁 Struktur

```
├── dataset/
│   ├── medical_chatbot_NLP.ipynb   ← notebook utama (chatbot)
│   ├── qna_alodokter_5k.csv        ← 5.000 QnA (id, judul, pertanyaan, dokter, jawaban)
│   └── README.md                   ← dokumentasi dataset
├── README.md
└── .gitignore
```

## ▶️ Cara Menjalankan

### Google Colab (paling gampang)
1. Klik badge **Open in Colab** di atas
2. *Runtime → Run all*
3. Dataset otomatis terunduh dari repo ini

### Lokal (Jupyter / VS Code)
```bash
pip install nltk scikit-learn pandas numpy ipywidgets sentence-transformers gradio
jupyter notebook dataset/medical_chatbot_NLP.ipynb
```
Run cell dari atas ke bawah. UI Gradio terbuka otomatis di browser.

## 📊 Sumber Dataset

- Asal: [abid/indonesia-medical-qna](https://huggingface.co/datasets/abid/indonesia-medical-qna) (scrape Alodokter.com)
- Di-sampling 5.000 baris (seed=42), dibersihkan: karakter rusak dihapus, duplikat dibuang

## ⚠️ Disclaimer

Chatbot ini **untuk edukasi saja**, bukan pengganti konsultasi dokter. Keadaan darurat: **119**.

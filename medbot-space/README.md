---
title: MedBot - Chatbot Konsultasi Medis Indonesia
emoji: 🏥
colorFrom: blue
colorTo: indigo
tags:
- nlp
- medical
- chatbot
- indonesian
- tf-idf
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: true
license: mit
---

# MedBot — Chatbot Konsultasi Medis Bahasa Indonesia 🏥

Aplikasi Praktikum NLP (Semester 3): chatbot tanya-jawab kesehatan berbahasa Indonesia
menggunakan **TF-IDF + Cosine Similarity** di atas 5.000 pasang tanya-jawab asli
pasien–dokter dari Alodokter.com.

## Cara Kerja
1. Dataset: 5.000 QnA medis (Alodokter) + auto-kategorisasi 16 kategori
2. Preprocessing: lowercase, stopwords Indonesia+Inggris, lemmatization (NLTK)
3. Retrieval: TF-IDF (unigram+bigram) → cosine similarity → keyword boost → threshold 0.15
4. Aturan khusus: deteksi sapaan & kondisi darurat (119)

⚠️ Edukasi saja, bukan pengganti dokter. Darurat: **119**

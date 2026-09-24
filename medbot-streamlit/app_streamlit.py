# MedBot — versi web Streamlit (deploy gratis di Streamlit Community Cloud)
# file: medbot-streamlit/app_streamlit.py
# cara lokal  : streamlit run app_streamlit.py
# cara deploy : https://share.streamlit.io (hubungkan repo GitHub)

import re
import os
import random

import numpy as np
import pandas as pd
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= 1. NLTK =================
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

indonesian_stopwords = {
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk',
    'pada', 'adalah', 'atau', 'juga', 'dalam', 'tidak', 'akan', 'ada',
    'saya', 'kamu', 'anda', 'ia', 'mereka', 'kami', 'kita', 'bisa',
    'sudah', 'bila', 'jika', 'maka', 'oleh', 'karena', 'apa',
    'bagaimana', 'berapa', 'kapan', 'dimana', 'siapa', 'apakah', 'cara',
    'lebih', 'sangat', 'dapat', 'nya', 'pun', 'lagi', 'belum',
    'telah', 'namun', 'tapi', 'serta', 'meski', 'agar', 'supaya', 'hal',
    'the', 'is', 'are', 'was', 'what', 'how', 'why', 'when', 'where'
}
english_stopwords = set(stopwords.words('english'))
all_stopwords = indonesian_stopwords | english_stopwords


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in all_stopwords and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


# ================= 2. DATASET + ENGINE (cache biar cepat) =================
CATEGORY_KEYWORDS = {
    "demam berdarah":      ["demam berdarah", "dbd", "dengue"],
    "covid":               ["covid", "corona", "vaksin"],
    "diabetes":            ["diabetes", "gula darah", "kencing manis"],
    "hipertensi":          ["hipertensi", "tekanan darah", "darah tinggi"],
    "kolesterol":          ["kolesterol", "lemak darah", "trigliserida"],
    "jantung":             ["jantung", "nyeri dada", "berdebar"],
    "asma":                ["asma", "sesak nafas", "sesak napas"],
    "batuk pilek":         ["batuk", "pilek", "flu", "tenggorok"],
    "maag":                ["maag", "lambung", "gastritis", "gerd", "mual"],
    "sakit kepala":        ["sakit kepala", "migrain", "pusing"],
    "kehamilan":           ["hamil", "kehamilan", "melahirkan", "haid", "menstruasi"],
    "kesehatan anak":      ["bayi", "anak saya", "balita", "imunisasi", "asi"],
    "kesehatan mental":    ["stres", "cemas", "depresi", "psikolog", "susah tidur", "insomnia"],
    "pertolongan pertama": ["pertolongan pertama", "p3k", "luka", "patah", "terkilir", "kejang"],
    "demam":               ["demam", "panas"],
}


def categorize(text):
    t = str(text).lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(k in t for k in keywords):
            return cat
    return "umum"


@st.cache_resource(show_spinner="Menyiapkan dataset & model TF-IDF...")
def load_engine():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qna_alodokter_5k.csv")
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"pertanyaan": "question", "jawaban": "answer"})
    df = df.dropna(subset=["question", "answer"]).reset_index(drop=True)
    df["category"] = (df["judul"].fillna("") + " " + df["question"]).apply(categorize)
    df["index_text"] = (df["judul"].fillna("") + ". " + df["question"]).str.lower()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(df["index_text"])
    return df, vectorizer, tfidf_matrix


df, vectorizer, tfidf_matrix = load_engine()
THRESHOLD = 0.15
TOP_K = 3


def get_response(user_input, history):
    user_input = user_input.strip()
    if not user_input:
        return "Silakan ketik pertanyaan."

    # rule: sapaan & darurat
    t = user_input.lower()
    if re.search(r'\b(halo|hai|hi|hello)\b', t):
        return "👋 Halo! Ada yang bisa saya bantu?"
    if re.search(r'(sesak.*berat|nyeri dada.*berat|tidak.*bernapas|pingsan)', t):
        return "🚨 DARURAT! Hubungi 119 atau segera ke IGD!"

    # query dengan konteks percakapan
    query = (history[-1] + " " + user_input) if history else user_input
    ql = query.lower()

    scores = cosine_similarity(vectorizer.transform([ql]), tfidf_matrix).flatten()
    top_idx = np.argsort(-scores)[:TOP_K]

    if scores[top_idx[0]] < THRESHOLD:
        return "🤔 Tidak menemukan jawaban yang cukup relevan."

    # keyword boost
    boosted = []
    for idx in top_idx:
        bonus = sum(1 for w in user_input.split() if w in df.iloc[idx]['question'])
        boosted.append((idx, scores[idx] + 0.05 * bonus))
    best_idx = sorted(boosted, key=lambda x: x[1], reverse=True)[0][0]

    row = df.iloc[best_idx]
    return (
        f"**[{row['category'].upper()} | TF-IDF]**\n\n"
        f"{row['answer']}\n\n"
        f"---\n"
        f"⚠️ *Untuk kondisi serius, konsultasikan ke dokter.*"
    )


# ================= 3. UI STREAMLIT =================
st.set_page_config(page_title="MedBot — Chatbot Medis Indonesia", page_icon="🏥", layout="centered")

st.markdown(
    """
    <div style="background:linear-gradient(135deg,#1a1f5e,#2d3a8c,#1565c0);
                padding:22px;border-radius:15px;color:white;text-align:center;">
        <h2 style="margin:0;">🏥 MedBot — Chatbot Konsultasi Medis</h2>
        <p style="margin:6px 0 0;opacity:.9;">🟢 Online · TF-IDF + Cosine Similarity · 5.000 QnA Alodokter</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("⚠️ Edukasi saja, bukan pengganti dokter. Darurat: 119")

# inisialisasi riwayat chat
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# tombol pertanyaan cepat
cols = st.columns(5)
quick = ["🤕 Sakit Kepala", "🩸 Diabetes", "❤️ Jantung", "🫁 Asma", "🧠 Mental"]
for col, label in zip(cols, quick):
    if col.button(label, use_container_width=True):
        q = label.split(" ", 1)[1].lower()
        st.session_state.pending = q

# tampilkan riwayat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# input chat
prompt = st.chat_input("Ketik pertanyaan kesehatan...")
if "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("MedBot sedang mengetik..."):
            response = get_response(prompt, st.session_state.history)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.history.append(prompt)

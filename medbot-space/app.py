# MedBot — aplikasi web chatbot konsultasi medis Indonesia
# Dibangun dari notebook medical_chatbot_NLP.ipynb (Praktikum NLP)
# Jalankan: python app.py  → buka http://127.0.0.1:7860

import re
import random
import time

import numpy as np
import pandas as pd
import nltk
import gradio as gr
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= 1. SIAPKAN NLTK =================
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

lemmatizer = WordNetLemmatizer()

# stopwords bahasa indonesia (gabungan manual + inggris dari nltk)
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
    """lowercase, buang tanda baca/angka, tokenisasi, hapus stopwords, lemmatize"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in all_stopwords and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


# ================= 2. LOAD DATASET + KATEGORI =================
df = pd.read_csv("qna_alodokter_5k.csv")
df = df.rename(columns={"pertanyaan": "question", "jawaban": "answer"})
df = df.dropna(subset=["question", "answer"]).reset_index(drop=True)

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


df["category"] = (df["judul"].fillna("") + " " + df["question"]).apply(categorize)
df["processed_question"] = df["question"].apply(preprocess_text)


# ================= 3. ENGINE CHATBOT =================
class MedicalChatbotEngineV3:
    """TF-IDF + cosine similarity + keyword boost, dengan konteks percakapan."""

    def __init__(self, dataframe, threshold=0.15, top_k=3):
        self.df = dataframe
        self.threshold = threshold
        self.top_k = top_k
        self.conversation_history = []

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_features=5000, sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["processed_question"])

        self.rules = {
            "emergency": {
                "patterns": [r"(sesak.*berat|nyeri dada.*berat|tidak.*bernapas|pingsan)"],
                "responses": ["🚨 DARURAT! Hubungi 119 atau segera ke IGD!"],
            },
            "greeting": {
                "patterns": [r"\b(halo|hai|hi|hello)\b"],
                "responses": ["👋 Halo! Ada yang bisa saya bantu?"],
            },
        }

    def _check_rules(self, text):
        for intent, data in self.rules.items():
            for pattern in data["patterns"]:
                if re.search(pattern, text.lower()):
                    return random.choice(data["responses"])
        return None

    def _search_tfidf(self, query):
        processed = preprocess_text(query)
        vec = self.vectorizer.transform([processed])
        scores = cosine_similarity(vec, self.tfidf_matrix).flatten()
        top_results = np.argsort(scores)[::-1][: self.top_k]
        return [(idx, scores[idx]) for idx in top_results]

    def _build_context_query(self, user_input):
        if len(self.conversation_history) > 0:
            return self.conversation_history[-1] + " " + user_input
        return user_input

    def get_response(self, user_input):
        if not user_input.strip():
            return "Silakan ketik pertanyaan."

        rule = self._check_rules(user_input)
        if rule:
            return rule

        query = self._build_context_query(user_input)
        results = self._search_tfidf(query)
        method = "TF-IDF"

        best_idx, best_score = results[0]
        if best_score < self.threshold:
            return "🤔 Tidak menemukan jawaban yang cukup relevan."

        # keyword boost: bonus untuk kata input yang muncul di pertanyaan FAQ
        boosted = []
        for idx, score in results:
            text = self.df.iloc[idx]["question"]
            bonus = sum(1 for word in user_input.split() if word in text)
            boosted.append((idx, score + 0.05 * bonus))

        best_idx = sorted(boosted, key=lambda x: x[1], reverse=True)[0][0]
        row = self.df.iloc[best_idx]
        self.conversation_history.append(user_input)

        return (
            f"[Kategori: {row['category']} | {method}]\n\n"
            f"{row['answer']}\n\n"
            f"─────────────────\n"
            f"⚠️ Untuk kondisi serius, konsultasikan ke dokter."
        )


bot = MedicalChatbotEngineV3(df)


# ================= 4. UI GRADIO =================
def respond(message, history):
    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "⏳ MedBot sedang mengetik"})
    yield "", history

    time.sleep(0.6)
    response = bot.get_response(message)

    history[-1] = {"role": "assistant", "content": response}
    yield "", history


def respond_chip(message, history):
    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": bot.get_response(message)})
    return "", history


def clear_chat():
    bot.conversation_history = []
    return []


custom_css = """
.gradio-container { font-family: 'Plus Jakarta Sans', sans-serif; background: #f0f4ff; }
.chatbot { height: 420px; }
.header { background: linear-gradient(135deg, #1a1f5e, #2d3a8c, #1565c0);
          padding: 20px; border-radius: 15px; color: white; }
.disclaimer { background: #fff8e1; padding: 10px; font-size: 12px;
              border-left: 4px solid orange; margin-top: 10px; }
"""

with gr.Blocks(title="MedBot — Chatbot Medis Indonesia") as demo:
    gr.Markdown(
        """
        <div class="header">
            <h2>🏥 MedBot v3 — Smart Medical Assistant</h2>
            <p>🟢 Online · Context-Aware Semantic Engine</p>
        </div>
        """
    )
    gr.Markdown(
        """
        <div class="disclaimer">
        ⚠️ Edukasi saja. Bukan pengganti dokter. Darurat: 119
        </div>
        """
    )

    chatbot = gr.Chatbot(height=420)
    msg = gr.Textbox(placeholder="Ketik pertanyaan kesehatan...")

    with gr.Row():
        send = gr.Button("Kirim 📤")
        clear = gr.Button("Clear 🗑️")

    with gr.Row():
        chip1 = gr.Button("🤕 Sakit Kepala")
        chip2 = gr.Button("🩸 Diabetes")
        chip3 = gr.Button("❤️ Jantung")
        chip4 = gr.Button("🫁 Asma")
        chip5 = gr.Button("🧠 Mental")

    send.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(clear_chat, None, chatbot, queue=False)

    chip1.click(lambda h: respond_chip("sakit kepala", h), chatbot, [msg, chatbot])
    chip2.click(lambda h: respond_chip("diabetes", h), chatbot, [msg, chatbot])
    chip3.click(lambda h: respond_chip("penyakit jantung", h), chatbot, [msg, chatbot])
    chip4.click(lambda h: respond_chip("asma", h), chatbot, [msg, chatbot])
    chip5.click(lambda h: respond_chip("kesehatan mental", h), chatbot, [msg, chatbot])


if __name__ == "__main__":
    # di gradio 6, parameter css dipindah ke launch()
    demo.launch(css=custom_css)

# Dataset Medical untuk Praktikum NLP

## `qna_alodokter_5k.csv` — Tanya Jawab Medis Bahasa Indonesia

- **Sumber**: https://huggingface.co/datasets/abid/indonesia-medical-qna
- **Asal data**: Alodokter.com (pasien bertanya, dokter menjawab)
- **Ukuran**: 5.000 baris (sampling acak seed=42 dari ~681 ribu data asli)
- **Kolom**:
  | Kolom | Isi |
  |---|---|
  | `id` | ID unik pertanyaan |
  | `judul` | Judul/topik pertanyaan |
  | `pertanyaan` | Pertanyaan pasien (sudah dibersihkan dari HTML) |
  | `dokter` | Nama dokter penjawab |
  | `jawaban` | Jawaban dokter (sudah dibersihkan dari HTML) |

**Catatan pembersihan**: karakter rusak (encoding) dihapus, spasi/newline berlebih dirapikan, duplikat dibuang.

**Pemakaian di notebook** (`medical_chatbot_NLP.ipynb`):

```python
import pandas as pd
df = pd.read_csv("qna_alodokter_5k.csv")
# kolom 'pertanyaan' & 'jawaban' di-rename jadi 'question' & 'answer'
# untuk dipakai engine MedicalChatbotEngineV3 (TF-IDF + cosine similarity)
```

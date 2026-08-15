# Demo - Riset Nama Gender

Live web demo untuk klasifikasi gender dari nama Indonesia menggunakan **8 neural models** yang dilatih di notebook.

## Fitur

1. **Single Model** - prediksi dengan CharBiLSTM (F1 0.9589 rata-rata lima seed),
   dengan confidence bar L vs P. Disebut single, bukan best, karena CharBiGRU
   unggul 0.04 poin dengan interval yang melewati nol, jadi menyebut salah satu
   terbaik adalah seleksi yang tidak didukung data.
2. **Compare 8 Models** - bandingkan prediksi semua model neural side-by-side, highlight model yang disagree
3. **Attention Weights** - visualisasi karakter mana yang paling diperhatikan model saat prediksi (heatmap + bar chart)

## Stack

- **Backend**: FastAPI + uvicorn (Python 3.11, PyTorch 2.11 + CUDA 12.8)
- **Frontend**: vanilla HTML/CSS/JS (no framework, no build step)
- **Inference**: 7 model `.pt` files + 2 tokenizers `.pkl` dari `../results/`

## Struktur

```
demo/
├── app.py              # FastAPI app + 4 endpoints
├── inference.py        # Predictor class (load models, predict, attention)
├── launch_demo.bat     # One-click launcher (Windows)
├── README.md
└── static/
    ├── index.html
    ├── style.css
    └── script.js
```

## Cara jalanin

### Opsi 1 - Double-click bat file

```
demo\launch_demo.bat
```

Bat bakal otomatis:
1. Activate conda env `riset-gender`
2. Start uvicorn di `127.0.0.1:8000`

Buka browser: <http://127.0.0.1:8000>

### Opsi 2 - Manual via Anaconda Prompt

```bash
conda activate riset-gender
cd D:\MyPaper\RisetNamaGender\demo
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

## API Endpoints

| Method | Path | Body | Output |
|---|---|---|---|
| GET | `/` | - | serve `index.html` |
| GET | `/api/models` | - | list 7 model names + device info |
| POST | `/api/predict` | `{name, model}` | single prediction + confidence |
| POST | `/api/compare` | `{name}` | all 7 models side-by-side |
| POST | `/api/attention` | `{name, model}` | prediction + per-token attention weights |

**Auto-generated API docs**: <http://127.0.0.1:8000/docs> (Swagger UI)

### Example curl

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"name": "BANOWATI LARASATI", "model": "CharBiLSTM"}'
```

Response:
```json
{
  "model": "CharBiLSTM",
  "name": "BANOWATI LARASATI",
  "label": "P",
  "label_desc": "Perempuan",
  "confidence": 0.9282,
  "prob_female": 0.9282,
  "prob_male": 0.0718
}
```

## Insight buat paper

Demo ini bagus buat **interactive showcase** di paper Sinta 2 lo:

1. **Pilih nama** dengan suffix berbeda (e.g. WULANDARI, RAHMANTO, GANDHI, DEVI) -> tunjukin model bisa generalize.
2. **Compare tab** - tunjukin pola: Word models sering disagree pada nama langka, Char models lebih konsisten.
3. **Attention tab** - bukti kuat bahwa **CharBiLSTM belajar pola suffix** (huruf akhir paling diperhatikan untuk kebanyakan nama Indonesia).

Pas demo ke reviewer/penguji, contoh nama ambigu (`SETIA`, `WAHYU`, `DIAN`) bagus buat tunjukin **kalibrasi confidence** model.

## Troubleshooting

**Server gagal start dengan "DLL load" error**:
- Pastikan kernel/env udah aktif: `conda activate riset-gender`
- Pastikan `KMP_DUPLICATE_LIB_OK=TRUE` (sudah di-set otomatis di `inference.py`)

**Port 8000 already in use**:
- Ganti port di `launch_demo.bat`: `--port 8001`

**Frontend gak load (blank page)**:
- Cek browser console (F12). Kalo error CORS, restart server.
- Pastikan `static/` folder ada di sebelah `app.py`.

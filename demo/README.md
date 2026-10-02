# Demo - Riset Nama Gender

Live web demo for gender classification from Indonesian names, using **8 neural models** trained in the notebooks.

## Features

1. **Single Model** - prediction with CharBiLSTM (F1 0.9675, mean of five seeds),
   with an L vs P confidence bar. It is called single, not best, because CharBiGRU
   is ahead by 0.04 points with an interval that crosses zero, so naming either
   model the best would be a selection the data does not support.
2. **Compare 8 Models** - side-by-side predictions from all neural models, with the models that disagree with the consensus highlighted
3. **Attention Weights** - visualization of which characters the model attends to when predicting (heatmap + bar chart)

## Stack

- **Backend**: FastAPI + uvicorn (Python 3.11, PyTorch 2.11 + CUDA 12.8)
- **Frontend**: vanilla HTML/CSS/JS (no framework, no build step)
- **Inference**: 8 model `.pt` files from `../models/` + 2 tokenizer `.pkl` files from `../tokenizers/`

## Structure

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

## How to run

### Option 1 - Double-click the bat file

```
demo\launch_demo.bat
```

The bat file will:
1. Activate the conda env `riset-gender`
2. Start uvicorn on `127.0.0.1:8000`

Then open <http://127.0.0.1:8000> in a browser.

### Option 2 - Manually from an Anaconda Prompt

```bash
conda activate riset-gender
cd demo
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

## API Endpoints

| Method | Path | Body | Output |
|---|---|---|---|
| GET | `/` | - | serves `index.html` |
| GET | `/api/models` | - | list of the 8 model names + device info |
| POST | `/api/predict` | `{name, model}` | single prediction + confidence |
| POST | `/api/compare` | `{name}` | all 8 models side-by-side |
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
  "label_desc": "Female",
  "confidence": 0.9282,
  "prob_female": 0.9282,
  "prob_male": 0.0718
}
```

## Suggested inputs

- **Names with different suffixes** (e.g. WULANDARI, RAHMANTO, GANDHI, DEVI) show how the models handle common Indonesian endings.
- **Compare tab** shows where the word-level models and the character-level models give different labels, which happens mostly for rare names.
- **Attention tab** shows which characters receive the most attention. The attention weights describe where a model looks, and they are not by themselves evidence of what the model has learned.
- **Ambiguous names** (`SETIA`, `WAHYU`, `DIAN`) are useful for inspecting the confidence values the models output.

## Troubleshooting

**Server fails to start with a "DLL load" error**:
- Make sure the env is active: `conda activate riset-gender`
- Make sure `KMP_DUPLICATE_LIB_OK=TRUE` is set (it is set automatically in `inference.py`)

**Port 8000 already in use**:
- Change the port in `launch_demo.bat`: `--port 8001`

**Frontend does not load (blank page)**:
- Check the browser console (F12). If there is a CORS error, restart the server.
- Make sure the `static/` folder is next to `app.py`.

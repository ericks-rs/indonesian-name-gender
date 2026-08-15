"""indonamegender, gender classification dari nama Indonesia.

Dilatih atas 202,134 nama unik 1990-2025 dengan novel-name temporal split. Nama
dideduplikasi menjadi satu baris per nama, dan setiap nama masuk partisi tahun
kemunculan pertamanya. Training 169,329 nama 1990-2021, development 16,882 nama
2022-2023 dipakai untuk memilih epoch, evaluasi 15,923 nama 2024-2025 disentuh
sekali. Tidak ada nama evaluasi yang pernah muncul di training pada tahun mana pun.

Angka F1 di bawah berasal dari rata-rata lima seed pada partisi 2024-2025.

Quick start:
    from indonamegender import GenderPredictor

    gp = GenderPredictor()  # default CharBiGRU
    result = gp.predict("BANOWATI LARASATI")
    # {'name': 'BANOWATI LARASATI', 'gender': 'Female', 'confidence': 0.9949}
"""
from .predictor import GenderPredictor

__version__ = "0.1.0"
__all__ = ["GenderPredictor"]

AVAILABLE_MODELS = [
    "CharBiGRU",       # default, 86,065 params, F1 0.9593
    "CharBiLSTM",      # 114,097 params, F1 0.9589
    "CharBiRNN",       # 30,001 params, F1 0.9574
    "CharTransformer", # 605,953 params, F1 0.9554
    "WordTransformer", # 6,126,721 params, F1 0.9365
    "WordBiLSTM",      # 2,544,289 params, F1 0.9344
    "WordBiGRU",       # 2,507,041 params, F1 0.9335
    "WordBiRNN",       # 2,432,545 params, F1 0.9323
]

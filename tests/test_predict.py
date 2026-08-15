import pytest
from indonamegender import GenderPredictor

def test_predict_female():
    gp = GenderPredictor()
    r = gp.predict("BANOWATI LARASATI")
    assert r["gender"] == "Female"
    assert r["confidence"] > 0.8
    assert r["model"] == "CharBiGRU"

def test_predict_male():
    gp = GenderPredictor()
    r = gp.predict("GATOTKACA WIRAWAN")
    assert r["gender"] == "Male"
    assert r["confidence"] > 0.8

def test_batch():
    gp = GenderPredictor()
    results = gp.predict_batch(["DEWI", "BUDI"])
    assert len(results) == 2
    assert all("gender" in r for r in results)

def test_attention():
    gp = GenderPredictor()
    r = gp.get_attention("DEWI SRI")
    assert "tokens" in r
    assert "attention" in r
    assert len(r["tokens"]) == len(r["attention"])
    assert abs(sum(r["attention"]) - 1.0) < 0.01

def test_invalid_model():
    with pytest.raises(ValueError):
        GenderPredictor(model="NonexistentModel")

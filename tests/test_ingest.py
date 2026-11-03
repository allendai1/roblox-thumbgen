import csv
from pathlib import Path

from PIL import Image

from thumbgen.data.ingest import dhash, filter_row, hamming, load_metadata
from thumbgen.data.upscale import Upscaler, fit_to_bucket

CFG = {"filters": {"min_aspect": 1.5, "max_aspect": 2.0}, "upscale": {"min_source_width": 384}}


def test_filter_row():
    assert filter_row(Image.new("RGB", (768, 432)), CFG) is None
    assert "aspect" in filter_row(Image.new("RGB", (512, 512)), CFG)
    assert "too small" in filter_row(Image.new("RGB", (320, 180)), CFG)


def test_fit_to_bucket_and_lanczos_upscale():
    img = Image.new("RGB", (768, 432), (200, 50, 50))
    up = Upscaler("lanczos", scale=2)(img)
    assert up.size == (1536, 864)
    out = fit_to_bucket(up, 1344, 768)
    assert out.size == (1344, 768)


def test_dhash_near_duplicates():
    a = Image.linear_gradient("L").resize((768, 432)).convert("RGB")
    b = a.resize((700, 400))
    assert hamming(dhash(a), dhash(b)) <= 4


def test_load_metadata_csv_and_json(tmp_path: Path):
    p = tmp_path / "m.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "tags", "game_name", "genre"])
        w.writeheader()
        w.writerow({"filename": "a.png", "tags": "x, y", "game_name": "G", "genre": "horror"})
    rows = load_metadata(str(p))
    assert rows[0]["genre"] == "horror"
    j = tmp_path / "m.json"
    j.write_text('{"a.png": {"tags": "x"}, "b.png": "y, z"}')
    rows = load_metadata(str(j))
    assert rows[1] == {"filename": "b.png", "tags": "y, z"}

from typing import List

from fastapi.testclient import TestClient
from PIL import Image

from thumbgen.app import server


class FakeGen:
    base, lora_path, _pipe = "fake/base", "", None

    def generate(self, prompt, negative, seeds: List[int], width, height, steps, cfg, lora_scale=None):
        return [Image.new("RGB", (width, height), (i * 20 % 255, 0, 0)) for i, _ in enumerate(seeds)]

    def upscale_1080p(self, img, prompt, negative, seed):
        return img.resize((1920, 1080))


def make_client(tmp_path):
    server.OUTPUT_DIR = tmp_path
    app = server.create_app("configs/presets.yaml", generator=FakeGen())
    return TestClient(app)


def test_presets_and_prompt(tmp_path):
    c = make_client(tmp_path)
    assert c.get("/api/presets").json()["styles"] == ["3d", "anime"]
    r = c.post("/api/prompt", json={"preset": "vs_faceoff", "style": "anime"})
    assert r.status_code == 200 and "anime style" in r.json()["positive"]
    assert c.post("/api/prompt", json={"preset": "nope"}).status_code == 400


def test_generate_saves_files(tmp_path):
    c = make_client(tmp_path)
    r = c.post("/api/generate", json={"preset": "face_closeup", "count": 2, "seed": 5})
    assert r.status_code == 200
    res = r.json()["results"]
    assert [x["seed"] for x in res] == [5, 6]
    assert all((tmp_path / x["url"].split("/")[-1]).exists() for x in res)
    assert c.get(res[0]["url"]).status_code == 200
    assert len(c.get("/api/history").json()) == 1


def test_index_served(tmp_path):
    c = make_client(tmp_path)
    assert "Thumbnail Generator" in c.get("/").text

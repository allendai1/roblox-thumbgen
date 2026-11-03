import yaml
from thumbgen.app.prompts import PromptBuilder

PB = PromptBuilder.from_yaml("configs/presets.yaml")


def test_every_preset_expands_without_leftover_slots():
    for pid in PB.presets:
        for style in ("3d", "anime"):
            pos, neg, meta = PB.build(pid, style=style, palette="orange_teal", seed=1)
            assert "{" not in pos and "}" not in pos, (pid, pos)
            assert pos.startswith("rbxthumb, ")
            assert "text" in neg
            assert meta["text_side"] in ("left", "right", "center")


def test_style_phrase_switches():
    a, _, _ = PB.build("face_closeup", style="3d", seed=1)
    b, _, _ = PB.build("face_closeup", style="anime", seed=1)
    assert "3d render" in a and "anime style" in b


def test_slots_and_text_side_override():
    pos, _, meta = PB.build("face_closeup", slots={"character": "noob avatar"}, text_side="left", seed=3)
    assert "noob avatar" in pos
    assert meta["subject_side"] == "right" if "subject_side" in meta else True
    assert "empty uncluttered background on the left side" in pos


def test_random_palette_is_deterministic_per_seed():
    a = PB.build("loot_reveal", palette="random", seed=7)[2]["palette"]
    b = PB.build("loot_reveal", palette="random", seed=7)[2]["palette"]
    assert a == b and a in PB.cfg["palettes"]


def test_catalog_shape():
    c = PB.catalog()
    assert {p["id"] for p in c["presets"]} == set(PB.presets)
    assert c["styles"] == ["3d", "anime"]


def test_strip_weights():
    from thumbgen.app.pipeline import strip_weights
    assert strip_weights("a, (big face:1.25), b") == "a, big face, b"

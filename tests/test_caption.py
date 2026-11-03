from thumbgen.data.caption import approx_tokens, build_caption, clean_sentence, split_tags


def test_split_tags_handles_blank_and_separators():
    assert split_tags(None) == []
    assert split_tags("") == []
    assert split_tags("Zombie, night; flashlight | Dark") == ["zombie", "night", "flashlight", "dark"]


def test_clean_sentence_strips_preamble():
    assert clean_sentence("The image shows a blocky avatar running.") == "a blocky avatar running"


def test_build_caption_order_and_budget():
    cap = build_caption("rbxthumb", "3d render", "Horror", ["zombie", "night"], "an avatar runs from a monster",
                        ["glowing eyes", "zombie", "fog", "dark"] + [f"tag{i}" for i in range(200)], max_tokens=60)
    parts = cap.split(", ")
    assert parts[:3] == ["rbxthumb", "3d render", "horror"]
    assert "zombie" in parts and parts.count("zombie") == 1
    assert approx_tokens(cap) <= 60


def test_missing_genre_falls_back():
    cap = build_caption("rbxthumb", "anime style", None, [], "", [])
    assert cap == "rbxthumb, anime style, game"

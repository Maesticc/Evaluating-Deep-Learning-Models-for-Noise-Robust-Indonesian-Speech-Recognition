from src.audio_validation import speech_in_noise_flags


def test_speech_in_noise_flags():
    rows = speech_in_noise_flags(["", "a i", "saya pergi ke pasar hari ini"], [10.0, 10.0, 2.0], 2.0)
    assert [r["speech_suspected"] for r in rows] == [False, False, True]
    assert rows[1]["transcribed_chars"] == 2 and rows[2]["chars_per_sec"] == 11.5
    assert speech_in_noise_flags(["x"], [0.0], 2.0)[0]["chars_per_sec"] == 0.0


def _cfg():
    from src.utils import load_config

    return load_config(None)


def test_select_for_listening_covers_review_cases():
    import pandas as pd

    from src.audio_validation import select_for_listening

    noise = pd.DataFrame({
        "noise_id": ["noise_test/A_1", "noise_train/A_2", "noise_train/A_3", "noise_train/A_4", "noise_train/B_1"],
        "noise_path": ["t/A_1.wav", "n/A_2.wav", "n/A_3.wav", "n/A_4.wav", "n/B_1.wav"],
        "noise_pool": ["noise_test", "noise_train", "noise_train", "noise_train", "noise_train"],
        "noise_category": ["A", "A", "A", "A", "B"], "quality_status": ["valid"] * 5})
    speech = pd.DataFrame({
        "utt_id": ["rate", "clip_bad", "clip_ok", "plain_tr", "plain_te"],
        "audio_path": ["r.wav", "c1.wav", "c2.wav", "p1.wav", "p2.wav"],
        "official_split": ["train", "train", "val", "train", "test"], "transcript": ["t"] * 5,
        "audit_flags": ["too_long", "possible_clipping", "possible_clipping", "", ""],
        "auto_review": ["valid: span", "invalid: clipped", "valid: clipped", "", ""],
        "eligible_primary": [True, False, True, True, True]})
    sheet = select_for_listening(speech, noise, _cfg())
    noise_ids = set(sheet.loc[sheet["manifest"] == "noise", "id"])
    assert "noise_test/A_1" in noise_ids and "noise_train/B_1" not in noise_ids  # B is not an evaluation category
    assert len(noise_ids & {"noise_train/A_2", "noise_train/A_3", "noise_train/A_4"}) == 2
    assert set(sheet.loc[sheet["manifest"] == "speech", "id"]) == {"rate", "clip_bad", "clip_ok", "plain_tr", "plain_te"}
    assert (sheet["decision"] == "").all()


def test_merge_listening_decisions():
    import pandas as pd
    import pytest

    from src.audio_validation import merge_listening_decisions

    old = pd.DataFrame({"manifest": ["noise"], "id": ["n1"], "decision": ["valid"], "note": ["earlier"]})
    sheet = pd.DataFrame({"manifest": ["noise", "speech"], "id": ["n1", "u1"],
                          "decision": [" Invalid", ""], "note": ["talking in background", ""]})
    merged = merge_listening_decisions(old, sheet)
    assert merged.to_dict("records") == [
        {"manifest": "noise", "id": "n1", "decision": "invalid", "note": "listening: talking in background"}]
    with pytest.raises(ValueError):
        merge_listening_decisions(old, sheet.assign(decision=["maybe", ""]))

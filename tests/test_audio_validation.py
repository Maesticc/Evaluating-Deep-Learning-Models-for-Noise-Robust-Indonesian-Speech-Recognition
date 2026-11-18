from src.audio_validation import speech_in_noise_flags


def test_speech_in_noise_flags():
    rows = speech_in_noise_flags(["", "a i", "saya pergi ke pasar hari ini"], [10.0, 10.0, 2.0], 2.0)
    assert [r["speech_suspected"] for r in rows] == [False, False, True]
    assert rows[1]["transcribed_chars"] == 2 and rows[2]["chars_per_sec"] == 11.5
    assert speech_in_noise_flags(["x"], [0.0], 2.0)[0]["chars_per_sec"] == 0.0

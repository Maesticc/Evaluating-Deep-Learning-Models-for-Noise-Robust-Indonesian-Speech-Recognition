from src.utils import load_config


def test_models_match_the_plan():
    """Plan v5 sections 7 and 9.1: three systems; Whisper Large-v3 set to Indonesian transcription."""
    models = load_config(None)["models"]
    assert set(models) == {"custom_cnn_bilstm_ctc", "wav2vec2", "whisper"}
    assert models["wav2vec2"]["checkpoint"] == "indonesian-nlp/wav2vec2-large-xlsr-indonesian"
    w = models["whisper"]
    assert (w["checkpoint"], w["language"], w["task"]) == ("openai/whisper-large-v3", "indonesian", "transcribe")


def test_environment_info_records_every_package_version():
    from src.utils import environment_info

    info = environment_info()
    for pkg in ("numpy", "pandas", "torch", "transformers", "jiwer", "librosa", "soundfile"):
        assert info[pkg] not in (None, "unknown"), pkg

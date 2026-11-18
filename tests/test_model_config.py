from src.utils import load_config


def test_whisper_checkpoints_share_generation_settings():
    """Large-v3 and Turbo must differ only in checkpoint and batch size to stay comparable."""
    models = load_config(None)["models"]
    a, b = models["whisper"], models["whisper_turbo"]
    for key in ("language", "task", "num_beams", "max_new_tokens", "fp16"):
        assert a[key] == b[key], key
    assert a["checkpoint"] == "openai/whisper-large-v3"
    assert b["checkpoint"] == "openai/whisper-large-v3-turbo"


def test_environment_info_records_every_package_version():
    from src.utils import environment_info

    info = environment_info()
    for pkg in ("numpy", "pandas", "torch", "transformers", "jiwer", "librosa", "soundfile"):
        assert info[pkg] not in (None, "unknown"), pkg

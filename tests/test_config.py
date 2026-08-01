"""Tests for Settings loading and .env.example coverage."""

from pathlib import Path

from mh_rag.config import Settings


def test_settings_load_with_defaults(monkeypatch):
    for key in list(Settings.model_fields):
        monkeypatch.delenv(key.upper(), raising=False)
    monkeypatch.setattr(Settings, "model_config", {**Settings.model_config, "env_file": None})

    settings = Settings()
    assert settings.embedding_dim == 1024
    assert settings.chunk_tokens == 500
    assert settings.depth_max == 3
    assert settings.er_auto_merge_threshold == 0.92
    assert settings.token_budget == 3000


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("TOKEN_BUDGET", "1500")
    assert Settings().token_budget == 1500


def test_env_example_covers_all_fields():
    env_example = Path(__file__).resolve().parents[1] / ".env.example"
    keys: list[str] = []
    for line in env_example.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        keys.append(stripped.split("=", 1)[0])

    expected = {name.upper() for name in Settings.model_fields}
    assert sorted(keys) == sorted(expected)
    assert len(keys) == len(set(keys))

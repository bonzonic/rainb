from rainb.speech import ASR, MY, US


def test_malaysian_without_a_gpu_falls_back_to_us_with_a_note(monkeypatch):
    monkeypatch.setattr(ASR, "us_backend", "parakeet")
    monkeypatch.setattr(ASR, "my_ok", False)
    accent, note = ASR.resolve(MY)
    assert accent == US and "GPU" in note
    assert ASR.resolve(US) == (US, None)


def test_malaysian_with_a_gpu_is_kept(monkeypatch):
    monkeypatch.setattr(ASR, "us_backend", "nemo")
    monkeypatch.setattr(ASR, "my_ok", True)
    assert ASR.resolve(MY) == (MY, None)
    assert "Malaysian" in ASR.name(MY) and "Parakeet" in ASR.name(US)

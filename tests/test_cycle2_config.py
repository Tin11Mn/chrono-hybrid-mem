from app.main import need_select_by_bm25_from_environment


def test_bm25_selection_defaults_on_for_cycle2(monkeypatch):
    monkeypatch.delenv("MEMORY_NEED_SELECT_BY_BM25", raising=False)
    assert need_select_by_bm25_from_environment() is True


def test_bm25_selection_can_be_disabled_for_ablation(monkeypatch):
    monkeypatch.setenv("MEMORY_NEED_SELECT_BY_BM25", "false")
    assert need_select_by_bm25_from_environment() is False

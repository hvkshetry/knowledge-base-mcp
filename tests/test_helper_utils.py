import pytest
from pathlib import Path

import ingest_blocks
import server


@pytest.fixture(autouse=True)
def _ensure_dirs(tmp_path, monkeypatch):
    plan_dir = tmp_path / "plans"
    plan_dir.mkdir()
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    monkeypatch.setattr(server, "PLAN_DIR", plan_dir)
    monkeypatch.setattr(server, "ARTIFACT_DIR", artifact_dir)
    yield plan_dir, artifact_dir


@pytest.mark.parametrize(
    "table_tokens,text_density,multicol,images,vector,expected",
    [
        (120, 500, 0.0, False, 50, 0.98),
        (0, 1600, 0.0, False, 10, 0.95),
        (40, 900, 1.0, True, 60, 0.60),
        (10, 800, 0.0, False, 20, 0.80),
    ],
)
def test_calculate_page_confidence(table_tokens, text_density, multicol, images, vector, expected):
    result = ingest_blocks._calculate_page_confidence(
        table_tokens=table_tokens,
        text_density=text_density,
        multicolumn_score=multicol,
        has_images=images,
        vector_lines=vector,
    )
    assert pytest.approx(result, rel=1e-2) == expected


def test_validate_artifact_path_allows_plan_dir(_ensure_dirs):
    plan_dir, _ = _ensure_dirs
    target = plan_dir / "doc.plan.json"
    target.write_text("{}", encoding="utf-8")
    resolved = server._validate_artifact_path(target)
    assert resolved == target.resolve()


def test_validate_artifact_path_rejects_escape(tmp_path, _ensure_dirs):
    with pytest.raises(ValueError):
        server._validate_artifact_path(tmp_path / "outside.json")

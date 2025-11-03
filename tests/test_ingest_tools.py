import asyncio
import json
from pathlib import Path

import pytest

import server


@pytest.fixture(autouse=True)
def _setup_artifact_dirs(monkeypatch):
    Path("data/ingest_artifacts").mkdir(parents=True, exist_ok=True)
    Path("data/ingest_plans").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(server, "ingest_upsert_tool", server.ingest_upsert_tool.fn)
    monkeypatch.setattr(server, "ingest_upsert_batch", server.ingest_upsert_batch.fn)
    monkeypatch.setattr(server, "ingest_generate_summary_tool", server.ingest_generate_summary_tool.fn)
    monkeypatch.setattr(server, "ingest_validate_extraction", server.ingest_validate_extraction.fn)
    yield


class StubContext:
    def __init__(self, subjects=None):
        self.metadata = {"subjects": subjects or ["pytest"]}


def make_ctx() -> StubContext:
    return StubContext()


@pytest.mark.asyncio
async def test_ingest_upsert_success(monkeypatch):
    doc_id = "test-doc"
    artifact_dir = Path("data/ingest_artifacts") / doc_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = artifact_dir / "chunks.json"
    chunk_path.write_text(json.dumps({
        "doc_id": doc_id,
        "chunks": [{"chunk_start": 0, "chunk_end": 10, "text": "hello"}],
        "plan_hash": "abc123",
    }), encoding="utf-8")

    async def fake_perform_upsert(doc_id, collection_name, chunks_artifact, **kwargs):
        return {
            "status": "ok",
            "chunks_upserted": 1,
            "qdrant_points": 1,
            "fts_rows": 1,
        }

    monkeypatch.setattr(server, "_perform_upsert", fake_perform_upsert)
    ctx = make_ctx()

    result = await server.ingest_upsert_tool(
        ctx,
        doc_id=doc_id,
        collection="kb",
        chunks_artifact=str(chunk_path),
    )

    assert result["status"] == "ok"
    assert result["chunks_upserted"] == 1


@pytest.mark.asyncio
async def test_ingest_upsert_batch_parallel(monkeypatch):
    docs = []
    for idx in range(3):
        doc_id = f"doc-{idx}"
        docs.append(doc_id)
        artifact_dir = Path("data/ingest_artifacts") / doc_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = artifact_dir / "chunks.json"
        chunk_path.write_text(json.dumps({"doc_id": doc_id, "chunks": [{"chunk_start": 0, "chunk_end": 1}], "plan_hash": "xyz"}), encoding="utf-8")

    async def fake_perform_upsert(doc_id, collection_name, chunks_artifact, **kwargs):
        return {"status": "ok", "chunks_upserted": 1, "qdrant_points": 1, "fts_rows": 1}

    monkeypatch.setattr(server, "_perform_upsert", fake_perform_upsert)
    specs = [{"doc_id": d, "chunks_artifact": str(Path("data/ingest_artifacts") / d / "chunks.json")} for d in docs]
    ctx = make_ctx()
    result = await server.ingest_upsert_batch(ctx, specs, collection="kb", parallel=2)

    assert result["successful"] == len(docs)
    assert result["failed"] == 0


@pytest.mark.asyncio
async def test_ingest_generate_summary_tracks_provenance(monkeypatch):
    doc_id = "summary-doc"
    plan_path = server._plan_file_path(doc_id)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps({"doc_id": doc_id, "triage": {"pages": []}}), encoding="utf-8")

    captured = {}

    def fake_upsert_summary_entry(collection, doc_id, section_path, summary, element_ids, metadata, summary_path=server.SUMMARY_DB_PATH):
        captured["collection"] = collection
        captured["doc_id"] = doc_id
        captured["section_path"] = section_path
        captured["summary"] = summary
        captured["metadata"] = metadata

    monkeypatch.setattr(server, "upsert_summary_entry", fake_upsert_summary_entry)
    ctx = make_ctx()

    result = await server.ingest_generate_summary_tool(
        ctx,
        doc_id=doc_id,
        summary_text="This section explains the recycle loop.",
        section_path=["Chapter 1", "Overview"],
        collection="kb",
        element_ids=["elem-1"],
        summary_metadata={"model": "claude-sonnet-4", "prompt_sha": "abc"},
        client_id="claude-code",
        client_model="claude-sonnet-4",
    )

    assert result["status"] == "ok"
    assert captured["summary"].startswith("This section")
    plan = server._load_plan(doc_id)
    orchestration = plan.get("client_orchestration", {})
    assert orchestration.get("client_id") == "claude-code"
    assert orchestration.get("client_model") == "claude-sonnet-4"
    assert orchestration.get("decisions")


def test_validate_extraction_path_security():
    result = asyncio.run(server.ingest_validate_extraction(make_ctx(), artifact_ref="../../etc/passwd"))
    assert result["error"] == "artifact_not_found"

    doc_id = "validate-doc"
    artifact_dir = Path("data/ingest_artifacts") / doc_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    blocks_path = artifact_dir / "blocks.json"
    blocks_path.write_text(json.dumps({"blocks": [{"text": "Heading", "page": 1, "type": "heading"}]}), encoding="utf-8")

    async def run_validation():
        ctx = make_ctx()
        return await server.ingest_validate_extraction(ctx, str(blocks_path))

    valid_result = asyncio.run(run_validation())
    assert valid_result["valid"] is True
    assert valid_result["stats"]["block_count"] == 1

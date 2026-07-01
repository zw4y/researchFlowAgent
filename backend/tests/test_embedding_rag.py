from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from app.container import AppContainer
from app.core.config import Settings
from app.core.errors import AppError
from app.main import app
from app.providers.dashscope import DashScopeEmbeddingProvider, DashScopeRerankProvider
from app.rag.index_profile import IndexProfile
from app.rag.vector_store import LlamaIndexVectorStore
from app.services.retrieval import RetrievalService
from httpx import ASGITransport, AsyncClient
from llama_index.core.schema import NodeWithScore, TextNode
from reportlab.pdfgen import canvas

from tests.doubles import (
    DeterministicChatProvider,
    DeterministicEmbeddingProvider,
    DeterministicRerankProvider,
    DeterministicToolSelector,
)


def embedding_provider() -> DashScopeEmbeddingProvider:
    return DashScopeEmbeddingProvider(
        api_key="test-key",
        model="text-embedding-v4",
        dimensions=1024,
        batch_size=10,
        timeout_seconds=30,
        max_retries=3,
        query_instruction="Retrieve research paper evidence.",
    )


@pytest.mark.asyncio
async def test_dashscope_embedding_batches_documents_and_preserves_order(monkeypatch) -> None:
    calls: list[dict] = []

    def mock_call(**kwargs):
        calls.append(kwargs)
        embeddings = [
            {"text_index": index, "embedding": [float(index)] * 1024}
            for index in reversed(range(len(kwargs["input"])))
        ]
        return SimpleNamespace(
            status_code=200,
            output={"embeddings": embeddings},
            request_id="embedding-request",
        )

    monkeypatch.setattr("app.providers.dashscope.TextEmbedding.call", mock_call)
    provider = embedding_provider()
    vectors = await provider.embed_documents([f"document {index}" for index in range(12)])

    assert [len(call["input"]) for call in calls] == [10, 2]
    assert all(call["text_type"] == "document" for call in calls)
    assert all(call["instruct"] is None for call in calls)
    assert all(call["dimension"] == 1024 for call in calls)
    assert [vector[0] for vector in vectors[:3]] == [0.0, 1.0, 2.0]
    assert len(vectors) == 12


@pytest.mark.asyncio
async def test_dashscope_query_uses_query_type_and_instruction(monkeypatch) -> None:
    captured: dict = {}

    def mock_call(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            status_code=200,
            output={"embeddings": [{"text_index": 0, "embedding": [0.25] * 1024}]},
            request_id="query-request",
        )

    monkeypatch.setattr("app.providers.dashscope.TextEmbedding.call", mock_call)
    vector = await embedding_provider().embed_query("What is the main finding?")

    assert captured["text_type"] == "query"
    assert captured["instruct"] == "Retrieve research paper evidence."
    assert len(vector) == 1024


@pytest.mark.asyncio
async def test_dashscope_embedding_rejects_invalid_dimensions(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.providers.dashscope.TextEmbedding.call",
        lambda **kwargs: SimpleNamespace(
            status_code=200,
            output={"embeddings": [{"text_index": 0, "embedding": [0.1] * 8}]},
        ),
    )

    with pytest.raises(AppError, match="向量数量或维度不正确"):
        await embedding_provider().embed_query("query")


@pytest.mark.asyncio
async def test_dashscope_rerank_returns_provider_order(monkeypatch) -> None:
    captured: dict = {}

    def mock_call(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            status_code=200,
            output=SimpleNamespace(
                results=[
                    SimpleNamespace(index=2, relevance_score=0.96),
                    SimpleNamespace(index=0, relevance_score=0.74),
                ]
            ),
            request_id="rerank-request",
        )

    monkeypatch.setattr("app.providers.dashscope.TextReRank.call", mock_call)
    provider = DashScopeRerankProvider(
        api_key="test-key",
        model="qwen3-rerank",
        timeout_seconds=30,
        max_retries=3,
        instruction="Rank paper passages.",
    )
    results = await provider.rerank("query", ["a", "b", "c"], top_n=2)

    assert [(item.index, item.score) for item in results] == [(2, 0.96), (0, 0.74)]
    assert captured["top_n"] == 2
    assert captured["return_documents"] is False
    assert captured["instruct"] == "Rank paper passages."


class FailingRerankProvider:
    name = "failing"
    model_name = "failing-rerank"
    enabled = True
    configured = True

    async def rerank(self, query: str, documents: list[str], top_n: int):
        del query, documents, top_n
        raise RuntimeError("provider unavailable")


@pytest.mark.asyncio
async def test_rerank_failure_falls_back_to_vector_order() -> None:
    settings = Settings(rerank_top_n=2)
    provider = DeterministicEmbeddingProvider()
    profile = IndexProfile.build(settings, provider)
    service = RetrievalService(
        settings,
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        FailingRerankProvider(),
        profile,
    )
    candidates = [
        NodeWithScore(node=TextNode(text=f"document {index}"), score=1.0 - index * 0.1)
        for index in range(3)
    ]

    selected, status = await service._rerank("query", candidates)

    assert status == "rerank_fallback"
    assert selected == candidates[:2]

def test_metric_query_prioritizes_table_summaries() -> None:
    settings = Settings(rerank_top_n=3)
    provider = DeterministicEmbeddingProvider()
    profile = IndexProfile.build(settings, provider)
    service = RetrievalService(
        settings,
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        FailingRerankProvider(),
        profile,
    )
    candidates = [
        NodeWithScore(
            node=TextNode(id_="table-9", text="[Table key rows] MSRS | ISFM"),
            score=0.8,
        ),
        NodeWithScore(node=TextNode(id_="body-9", text="table body 9"), score=0.9),
        NodeWithScore(
            node=TextNode(id_="table-10", text="[Table key rows] MRI-PET | ISFM"),
            score=0.7,
        ),
        NodeWithScore(node=TextNode(id_="body-10", text="table body 10"), score=0.85),
    ]

    selected = service._prioritize_table_summaries(
        "每个数据集的具体指标",
        candidates,
        [candidates[1], candidates[3]],
    )

    assert [item.node.node_id for item in selected] == [
        "table-9",
        "table-10",
        "body-9",
    ]


def test_index_profile_changes_with_model_or_chunking() -> None:
    provider = DeterministicEmbeddingProvider()
    first = IndexProfile.build(Settings(chunk_size_tokens=800), provider)
    changed_chunk = IndexProfile.build(Settings(chunk_size_tokens=600), provider)
    changed_ocr = IndexProfile.build(Settings(table_ocr_enabled=False), provider)

    assert first.profile_id != changed_chunk.profile_id
    assert first.profile_id != changed_ocr.profile_id
    assert first.collection_name.endswith(first.profile_id)


def make_run_dir(label: str) -> Path:
    path = Path("data/test-runs") / f"{label}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def make_sample_pdf(path: Path) -> None:
    document = canvas.Canvas(str(path))
    document.setTitle("Transformer Study")
    document.drawString(72, 760, "Transformer Study")
    document.drawString(72, 730, "Attention improves long range dependency modeling.")
    document.showPage()
    document.drawString(72, 760, "RAG answers retain page-level source evidence.")
    document.save()


@pytest.mark.asyncio
async def test_qdrant_local_survives_client_restart() -> None:
    run_dir = make_run_dir("qdrant-persistence")
    settings = Settings(
        app_env="test",
        vector_store_mode="qdrant_local",
        qdrant_path=run_dir / "qdrant",
        qdrant_collection="persistence_test",
    )
    provider = DeterministicEmbeddingProvider()
    profile = IndexProfile.build(settings, provider)
    node = TextNode(
        id_="f42bb92b-5e7e-41d1-b9a3-d5c4cbbb7164",
        text="Attention improves long range dependency modeling.",
        metadata={
            "paper_id": "paper-1",
            "paper_title": "Transformer Study",
            "page": 1,
            "chunk_id": "f42bb92b-5e7e-41d1-b9a3-d5c4cbbb7164",
            "chunk_index": 0,
            "index_profile": profile.profile_id,
        },
    )
    node.excluded_embed_metadata_keys = list(node.metadata)
    node.excluded_llm_metadata_keys = list(node.metadata)

    first = LlamaIndexVectorStore(settings, profile, provider)
    await first.add_nodes([node])
    assert await first.current_point_count() == 1
    await first.close()

    second = LlamaIndexVectorStore(settings, profile, provider)
    results = await second.retrieve("attention dependency", ["paper-1"], 5)
    assert [item.node.node_id for item in results] == ["f42bb92b-5e7e-41d1-b9a3-d5c4cbbb7164"]
    assert await second.current_point_count() == 1
    await second.close()


@pytest.mark.asyncio
async def test_index_status_and_reindex_api() -> None:
    run_dir = make_run_dir("reindex-api")
    pdf_path = run_dir / "paper.pdf"
    make_sample_pdf(pdf_path)
    settings = Settings(
        app_env="test",
        database_url=f"sqlite+aiosqlite:///{run_dir / 'test.db'}",
        upload_dir=run_dir / "uploads",
        vector_store_mode="memory",
        retrieval_score_threshold=0,
    )
    instance = AppContainer(
        settings,
        chat_provider=DeterministicChatProvider(),
        tool_selector=DeterministicToolSelector(),
        embedding_provider=DeterministicEmbeddingProvider(),
        rerank_provider=DeterministicRerankProvider(),
    )
    await instance.start()
    previous_container = getattr(app.state, "container", None)
    app.state.container = instance
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http:
            with pdf_path.open("rb") as stream:
                upload = await http.post(
                    "/api/v1/papers",
                    files={"file": ("paper.pdf", stream, "application/pdf")},
                )
            paper_id = upload.json()["paper"]["id"]

            response = await http.post(f"/api/v1/papers/{paper_id}/reindex")
            paper = await http.get(f"/api/v1/papers/{paper_id}")
            status = await http.get("/api/v1/index/status")

        assert response.status_code == 202
        assert response.json()["status"] == "queued"
        assert paper.json()["index_status"] == "ready"
        assert paper.json()["index_profile"] == status.json()["profile_id"]
        assert status.json()["paper_counts"]["ready"] == 1
        assert status.json()["point_count"] > 0
    finally:
        if previous_container is not None:
            app.state.container = previous_container
        await instance.close()


@pytest.mark.asyncio
async def test_unconfigured_dashscope_upload_returns_503() -> None:
    run_dir = make_run_dir("unconfigured-api")
    settings = Settings(
        app_env="test",
        database_url=f"sqlite+aiosqlite:///{run_dir / 'unconfigured.db'}",
        upload_dir=run_dir / "uploads",
        embedding_mode="dashscope",
        dashscope_api_key=None,
        rerank_mode="dashscope",
        vector_store_mode="memory",
    )
    unconfigured = AppContainer(settings)
    await unconfigured.start()
    previous_container = getattr(app.state, "container", None)
    app.state.container = unconfigured
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http:
            response = await http.post(
                "/api/v1/papers",
                files={"file": ("paper.pdf", b"%PDF-test", "application/pdf")},
            )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "embedding_not_configured"
    finally:
        if previous_container is not None:
            app.state.container = previous_container
        await unconfigured.close()

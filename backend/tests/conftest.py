from pathlib import Path

import pytest
from app.container import AppContainer
from app.core.config import Settings
from app.main import app
from httpx import ASGITransport, AsyncClient
from reportlab.pdfgen import canvas


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "transformer-study.pdf"
    document = canvas.Canvas(str(path))
    document.setTitle("Transformer Study")
    document.drawString(72, 760, "Transformer Study")
    document.drawString(
        72,
        730,
        "Attention layers improve long-range dependency modeling in technical documents.",
    )
    document.drawString(
        72,
        710,
        "The experiment reports accuracy of 92.5 percent on the validation split.",
    )
    document.showPage()
    document.drawString(
        72,
        760,
        "Retrieval augmented generation grounds answers in page-level source evidence.",
    )
    document.drawString(
        72,
        740,
        "Every answer should retain paper identity, page number, and chunk identity.",
    )
    document.save()
    return path


@pytest.fixture
async def container(tmp_path: Path):
    settings = Settings(
        app_env="test",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        upload_dir=tmp_path / "uploads",
        llm_mode="fake",
        vector_store_mode="memory",
        retrieval_score_threshold=0,
    )
    instance = AppContainer(settings)
    await instance.start()
    yield instance
    await instance.close()


@pytest.fixture
async def client(container: AppContainer):
    app.state.container = container
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        yield http

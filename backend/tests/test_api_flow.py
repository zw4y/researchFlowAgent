import asyncio

import pytest


@pytest.mark.asyncio
async def test_upload_rag_and_metrics_flow(client, sample_pdf):
    with sample_pdf.open("rb") as stream:
        upload = await client.post(
            "/api/v1/papers",
            files={"file": ("paper.pdf", stream, "application/pdf")},
        )
    assert upload.status_code == 202
    payload = upload.json()
    paper_id = payload["paper"]["id"]
    job_id = payload["ingestion_job"]["id"]

    for _ in range(20):
        job = await client.get(f"/api/v1/ingestion-jobs/{job_id}")
        if job.json()["status"] in {"completed", "failed"}:
            break
        await asyncio.sleep(0.05)
    assert job.json()["status"] == "completed"

    metrics = (
        "experiment,metric_name,metric_value,unit,split\n"
        "baseline,accuracy,92.5,percent,validation\n"
    )
    imported = await client.post(
        f"/api/v1/papers/{paper_id}/metrics/import",
        files={"file": ("metrics.csv", metrics.encode(), "text/csv")},
    )
    assert imported.status_code == 200
    assert imported.json()["imported"] == 1

    chat = await client.post(
        "/api/v1/chat",
        json={
            "question": "论文中的 attention 结论和准确率指标是什么？",
            "paper_ids": [paper_id],
            "enable_web": False,
        },
    )
    assert chat.status_code == 200, chat.text
    answer = chat.json()
    assert "rag" in answer["routes"]
    assert "metrics" in answer["routes"]
    assert answer["citations"][0]["page"] in {1, 2}
    assert {item["name"] for item in answer["tool_calls"]} == {
        "search_documents",
        "query_metrics",
    }


@pytest.mark.asyncio
async def test_duplicate_upload_is_detected(client, sample_pdf):
    content = sample_pdf.read_bytes()
    first = await client.post(
        "/api/v1/papers",
        files={"file": ("paper.pdf", content, "application/pdf")},
    )
    second = await client.post(
        "/api/v1/papers",
        files={"file": ("copy.pdf", content, "application/pdf")},
    )
    assert first.status_code == 202
    assert second.json()["duplicated"] is True
    assert second.json()["paper"]["id"] == first.json()["paper"]["id"]

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_reports_mock_mode_by_default():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_mode"] == "mock"


def test_analyze_returns_research_disclaimer():
    response = client.post(
        "/analyze",
        json={
            "chromosome": "chr7",
            "position": 140753336,
            "reference": "A",
            "alternate": "T",
            "gene": "BRAF",
            "sequence_context": "ACGTACGTACGTA",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["grch_build"] == "GRCh38"
    assert body["model_mode"] == "mock"
    assert body["risk_label"] in {"likely_benign", "uncertain", "likely_pathogenic"}
    assert 0 <= body["confidence"] <= 1
    assert "research and education only" in body["disclaimer"].lower()
    assert body["input"]["chromosome"] == "7"


def test_rejects_invalid_chromosome():
    response = client.post(
        "/analyze",
        json={
            "chromosome": "25",
            "position": 100,
            "reference": "A",
            "alternate": "C",
        },
    )

    assert response.status_code == 422


def test_rejects_same_reference_and_alternate():
    response = client.post(
        "/analyze",
        json={
            "chromosome": "1",
            "position": 100,
            "reference": "A",
            "alternate": "A",
        },
    )

    assert response.status_code == 422

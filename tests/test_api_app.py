from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ellegon.apps.api_app import create_app
from ellegon.llm.gateway import FakeGateway
from ellegon.sessions.persistence import save_path_for


def _build_client(
    *,
    campaigns_root: Path,
    saves_root: Path,
    output_text: str = "HELLO",
) -> TestClient:
    app = create_app(
        system_prompt="System",
        gateway=FakeGateway(output_text=output_text),
        campaigns_root=campaigns_root,
        saves_root=saves_root,
    )
    return TestClient(app)


def test_health_endpoint(
    temp_campaigns_root: Path, temp_saves_root: Path
) -> None:
    client = _build_client(
        campaigns_root=temp_campaigns_root, saves_root=temp_saves_root
    )
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_campaigns_endpoint_lists_campaigns(
    temp_campaigns_root: Path, temp_saves_root: Path
) -> None:
    client = _build_client(
        campaigns_root=temp_campaigns_root, saves_root=temp_saves_root
    )
    response = client.get("/campaigns")

    assert response.status_code == 200
    assert "test_campaign" in response.json()["campaigns"]


def test_start_session_endpoint_creates_starter(
    temp_campaigns_root: Path, temp_saves_root: Path
) -> None:
    client = _build_client(
        campaigns_root=temp_campaigns_root, saves_root=temp_saves_root, output_text="Starter"
    )
    payload = {
        "campaign_id": "test_campaign",
        "instance_id": "api_session",
        "players": 1,
        "model": "fake-model",
    }

    response = client.post("/sessions/start", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["dm_text"] == "Starter"
    save_path = save_path_for("test_campaign", "api_session", temp_saves_root)
    assert save_path.exists()


def test_turn_endpoint_runs_engine_turn(
    temp_campaigns_root: Path, temp_saves_root: Path
) -> None:
    client = _build_client(
        campaigns_root=temp_campaigns_root, saves_root=temp_saves_root, output_text="Next"
    )
    payload = {
        "campaign_id": "test_campaign",
        "instance_id": "turn_session",
        "players": 1,
        "model": "fake-model",
        "user_text": "Hello",
    }

    response = client.post("/sessions/turn", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["dm_text"] == "Next"
    assert data["turn"] == 1

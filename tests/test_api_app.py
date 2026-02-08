from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ellegon.apps.api_app import create_app
from ellegon.llm.gateway import FakeGateway
from ellegon.speech.stt import FakeTranscriber
from ellegon.speech.tts import FakeTTS
from ellegon.sessions.persistence import save_path_for


def _build_client(
    *,
    campaigns_root: Path,
    saves_root: Path,
    output_text: str = "HELLO",
    tts_audio: bytes = b"AUDIO",
    transcript: str = "transcribed text",
) -> TestClient:
    app = create_app(
        system_prompt="System",
        gateway=FakeGateway(output_text=output_text),
        tts=FakeTTS(output_audio=tts_audio),
        transcriber=FakeTranscriber(output_text=transcript),
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


def test_websocket_turn_streams_tts_chunks(
    temp_campaigns_root: Path, temp_saves_root: Path
) -> None:
    client = _build_client(
        campaigns_root=temp_campaigns_root,
        saves_root=temp_saves_root,
        output_text="Narration",
        tts_audio=b"VOICE",
    )

    with client.websocket_connect("/ws/sessions") as ws:
        ws.receive_json()  # status idle
        ws.receive_json()  # connected message

        ws.send_json(
            {
                "type": "start",
                "campaign_id": "test_campaign",
                "instance_id": "ws_turn",
                "players": 1,
                "model": "fake-model",
            }
        )

        start_events = [ws.receive_json() for _ in range(7)]
        assert [event["type"] for event in start_events] == [
            "status",
            "dm_text",
            "status",
            "tts_start",
            "tts_chunk",
            "tts_end",
            "session",
        ]

        ws.send_json({"type": "turn", "user_text": "Investigate"})

        turn_events = [ws.receive_json() for _ in range(8)]
        assert [event["type"] for event in turn_events] == [
            "status",
            "transcript",
            "dm_text",
            "status",
            "tts_start",
            "tts_chunk",
            "tts_end",
            "session",
        ]
        assert turn_events[2]["text"] == "Narration"


def test_websocket_stt_commit_auto_turn(
    temp_campaigns_root: Path, temp_saves_root: Path
) -> None:
    client = _build_client(
        campaigns_root=temp_campaigns_root,
        saves_root=temp_saves_root,
        output_text="Auto reply",
        tts_audio=b"SPEECH",
        transcript="spoken words",
    )

    with client.websocket_connect("/ws/sessions") as ws:
        ws.receive_json()
        ws.receive_json()

        ws.send_json(
            {
                "type": "start",
                "campaign_id": "test_campaign",
                "instance_id": "ws_stt",
                "players": 1,
            }
        )
        for _ in range(7):
            ws.receive_json()

        ws.send_json({"type": "stt_chunk", "audio_chunk_b64": "AAECAw==", "sample_rate": 16000, "channels": 1})
        buffered = ws.receive_json()
        assert buffered["type"] == "stt_buffered"
        assert buffered["payload"]["bytes"] == 4

        ws.send_json({"type": "stt_commit", "auto_turn": True})

        events = [ws.receive_json() for _ in range(10)]
        assert [event["type"] for event in events] == [
            "status",
            "transcript",
            "status",
            "transcript",
            "dm_text",
            "status",
            "tts_start",
            "tts_chunk",
            "tts_end",
            "session",
        ]
        assert events[1]["text"] == "spoken words"
        assert events[4]["text"] == "Auto reply"

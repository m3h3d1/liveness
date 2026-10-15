import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import server
from tools import eval as replay

CLIP = Path(__file__).resolve().parent / "clips" / "8627673"   # turns right, then left


@pytest.fixture(scope="module")
def client():
    with TestClient(server.app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    monkeypatch.setattr(server, "SAVE_DIR", None)   # ignore SAVE_DIR from .env
    server.sessions.clear()
    server.recent.clear()
    server.candidates.clear()


def start(client, monkeypatch, actions, test=False):
    monkeypatch.setattr(server, "pick_actions", lambda: actions)
    r = client.post("/api/session" + ("?test=1" if test else ""))
    assert r.status_code == 200
    return r.json()["id"]


def upload(client, sid, n=25, replace=None, save=None):
    files = [("frames", (p.name, p.read_bytes(), "image/jpeg")) for p in sorted(CLIP.glob("*.jpg"))[:n]]
    if replace:
        files[0] = ("frames", ("big.jpg", replace, "image/jpeg"))
    times = json.dumps([i * 100 for i in range(len(files))])
    data = {"times": times} | ({"save": save} if save else {})
    return client.post(f"/api/session/{sid}/frames", files=files, data=data)


def test_challenge_done(client, monkeypatch):
    body = upload(client, start(client, monkeypatch, ["turn_right", "turn_left"])).json()
    assert body["decision"] == "PASS", body
    assert body["scores"]["passive"] >= 0.7


def test_pass_offers_photos_and_one_can_be_chosen(client, monkeypatch):
    sid = start(client, monkeypatch, ["turn_right", "turn_left"])
    body = upload(client, sid).json()
    assert body["decision"] == "PASS"
    assert 1 <= len(body["photos"]) <= 3
    assert all(p.startswith("data:image/jpeg;base64,") for p in body["photos"])
    assert client.post(f"/api/session/{sid}/photo", json={"choice": 99}).status_code == 400
    r = client.post(f"/api/session/{sid}/photo", json={"choice": 0})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[0] == img.shape[1] <= 512
    assert client.post(f"/api/session/{sid}/photo", json={"choice": 0}).status_code == 404   # single use


def test_fail_offers_no_photos(client, monkeypatch):
    sid = start(client, monkeypatch, ["turn_left", "turn_right"])
    assert upload(client, sid).json()["photos"] == []
    assert client.post(f"/api/session/{sid}/photo", json={"choice": 0}).status_code == 404


def test_photos_expire(client, monkeypatch):
    sid = start(client, monkeypatch, ["turn_right", "turn_left"])
    upload(client, sid)
    server.candidates[sid]["created"] -= server.PHOTO_TTL + 1
    assert client.post(f"/api/session/{sid}/photo", json={"choice": 0}).status_code == 404


def test_wrong_order(client, monkeypatch):
    body = upload(client, start(client, monkeypatch, ["turn_left", "turn_right"])).json()
    assert body["decision"] == "FAIL"
    assert "CHALLENGE_ORDER" in body["reasons"]


def test_missing_action(client, monkeypatch):
    body = upload(client, start(client, monkeypatch, ["turn_right", "blink"])).json()
    assert "CHALLENGE_MISSING" in body["reasons"]


def test_session_is_single_use(client, monkeypatch):
    sid = start(client, monkeypatch, ["turn_right", "turn_left"])
    assert upload(client, sid).status_code == 200
    assert upload(client, sid).status_code == 404


def test_unknown_session(client):
    assert upload(client, "nope").status_code == 404


def test_expired(client, monkeypatch):
    sid = start(client, monkeypatch, ["turn_right", "turn_left"])
    server.sessions[sid]["created"] -= server.SESSION_TTL + 1
    body = upload(client, sid).json()
    assert (body["decision"], body["reasons"]) == ("FAIL", ["EXPIRED"])


def test_too_few_frames(client, monkeypatch):
    assert upload(client, start(client, monkeypatch, ["blink", "turn_left"]), n=5).status_code == 400


def test_oversized_frame(client, monkeypatch):
    big = b"\xff" * (server.MAX_FRAME_BYTES + 10)
    assert upload(client, start(client, monkeypatch, ["blink", "turn_left"]), replace=big).status_code == 413


def test_not_an_image(client, monkeypatch):
    assert upload(client, start(client, monkeypatch, ["blink", "turn_left"]), replace=b"hello").status_code == 400


def test_rate_limit(client):
    codes = [client.post("/api/session").status_code for _ in range(server.SESSIONS_PER_MIN + 1)]
    assert codes[:-1] == [200] * server.SESSIONS_PER_MIN and codes[-1] == 429


def test_page_health_and_model(client):
    assert client.get("/health").json() == {"status": "ok", "save": False}
    assert client.get("/").status_code == 200
    assert client.get("/test").text == client.get("/").text
    assert len(client.get("/models/face_landmarker.task").content) == 3758596


def test_saved_session_replays_same_result(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "SAVE_DIR", str(tmp_path))
    assert client.get("/health").json()["save"] is True
    body = upload(client, start(client, monkeypatch, ["turn_right", "turn_left"], test=True), save="print").json()
    assert body["saved"] is True
    [(label, _, result)] = replay.run(tmp_path)
    assert label == "print" and result["decision"] == body["decision"] == "PASS"
    assert result["scores"] == body["scores"]


def test_saves_only_with_consent(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "SAVE_DIR", str(tmp_path))
    assert upload(client, start(client, monkeypatch, ["turn_right", "turn_left"], test=True)).json()["saved"] is False
    assert not any(tmp_path.iterdir())


def test_normal_page_never_saves(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "SAVE_DIR", str(tmp_path))
    assert upload(client, start(client, monkeypatch, ["turn_right", "turn_left"]), save="real").json()["saved"] is False
    assert not any(tmp_path.iterdir())


def test_test_page_needs_save_dir(client, monkeypatch):
    sid = start(client, monkeypatch, ["turn_right", "turn_left"], test=True)
    assert upload(client, sid, save="real").json()["saved"] is False


def test_unknown_save_label(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "SAVE_DIR", str(tmp_path))
    assert upload(client, start(client, monkeypatch, ["blink", "turn_left"]), save="../x").status_code == 400
    assert not any(tmp_path.iterdir())


def test_details_explain_result(client, monkeypatch):
    d = upload(client, start(client, monkeypatch, ["turn_right", "blink"])).json()["details"]
    assert d["turn_right_deg"] >= 20 and d["seen"][0] == "turn_right"
    assert {"blink_rise", "face_frames", "brightness", "sharpness"} <= d.keys()

import json
from typing import Any

import pytest

from datapizza.tools.telegram_tool.base import TelegramBotTool


class MockResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


@pytest.fixture
def bot_tool() -> TelegramBotTool:
    return TelegramBotTool(bot_token="TEST_TOKEN", api_base_url="https://example.com/botTEST_TOKEN")


def test_send_message_success(monkeypatch: pytest.MonkeyPatch, bot_tool: TelegramBotTool):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return MockResponse(
            200,
            {
                "ok": True,
                "result": {"message_id": 1, "chat": {"id": "chat123"}, "text": "hello"},
            },
        )

    monkeypatch.setattr("datapizza.tools.telegram_tool.base.requests.post", fake_post)

    response = bot_tool.send_message(chat_id="chat123", text="hello")

    payload = json.loads(response)
    assert payload["message_id"] == 1
    assert payload["chat"]["id"] == "chat123"
    assert payload["text"] == "hello"

    assert captured["url"].endswith("/sendMessage")
    assert captured["json"]["text"] == "hello"
    assert captured["json"]["chat_id"] == "chat123"
    assert captured["timeout"] == 10.0


def test_send_message_api_error(monkeypatch: pytest.MonkeyPatch, bot_tool: TelegramBotTool):
    def fake_post(url, json=None, timeout=None):
        return MockResponse(200, {"ok": False, "description": "chat not found"})

    monkeypatch.setattr("datapizza.tools.telegram_tool.base.requests.post", fake_post)

    response = bot_tool.send_message(chat_id="chat123", text="hello")

    assert response.startswith("Error:")
    assert "chat not found" in response


def test_send_photo_success(monkeypatch: pytest.MonkeyPatch, bot_tool: TelegramBotTool):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return MockResponse(
            200,
            {
                "ok": True,
                "result": {"message_id": 42, "photo": [{"file_id": "abc"}]},
            },
        )

    monkeypatch.setattr("datapizza.tools.telegram_tool.base.requests.post", fake_post)

    response = bot_tool.send_photo(
        chat_id="chat123",
        photo="https://example.com/photo.jpg",
        caption="Snapshot",
    )

    payload = json.loads(response)
    assert payload["message_id"] == 42
    assert captured["url"].endswith("/sendPhoto")
    assert captured["json"]["caption"] == "Snapshot"
    assert captured["json"]["photo"].endswith("photo.jpg")


def test_send_document_api_error(monkeypatch: pytest.MonkeyPatch, bot_tool: TelegramBotTool):
    def fake_post(url, json=None, timeout=None):
        return MockResponse(400, {"ok": False, "description": "unsupported document"})

    monkeypatch.setattr("datapizza.tools.telegram_tool.base.requests.post", fake_post)

    response = bot_tool.send_document(chat_id="chat123", document="file.pdf")
    assert response.startswith("Error:")
    assert "unsupported document" in response


def test_edit_message_text(monkeypatch: pytest.MonkeyPatch, bot_tool: TelegramBotTool):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return MockResponse(
            200,
            {"ok": True, "result": {"message_id": 99, "text": "Updated"}},
        )

    monkeypatch.setattr("datapizza.tools.telegram_tool.base.requests.post", fake_post)

    response = bot_tool.edit_message_text(
        chat_id="chat123",
        message_id=10,
        text="Updated",
        disable_web_page_preview=True,
    )

    payload = json.loads(response)
    assert payload["message_id"] == 99
    assert captured["url"].endswith("/editMessageText")
    assert captured["json"]["disable_web_page_preview"] is True
    assert captured["json"]["message_id"] == 10

"""Telegram bot management tool."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests

from datapizza.tools import tool


class TelegramBotError(Exception):
    """Raised when the Telegram Bot API returns an error."""


@dataclass(slots=True)
class TelegramAPIConfig:
    """Configuration options for the Telegram API client."""

    bot_token: str
    timeout: float
    api_base_url: str


class TelegramBotTool:
    """
    Collection of tools for interacting with the Telegram Bot API.

    This class wraps common Telegram Bot API endpoints so that they can be invoked
    by datapizza agents via the tool interface.
    """

    def __init__(
        self,
        bot_token: str,
        *,
        timeout: float = 10.0,
        api_base_url: str | None = None,
    ):
        """
        Initialize the tool with the given Telegram bot token.

        Args:
            bot_token: Telegram bot token obtained from BotFather.
            timeout: Timeout in seconds for HTTP requests.
            api_base_url: Custom base URL for the Telegram API.
        """
        if not bot_token:
            raise ValueError("bot_token must be provided")

        base_url = api_base_url.rstrip("/") if api_base_url else f"https://api.telegram.org/bot{bot_token}"

        self.config = TelegramAPIConfig(
            bot_token=bot_token,
            timeout=timeout,
            api_base_url=base_url,
        )

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", endpoint, payload)

    def _get(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("GET", endpoint, payload)

    def _request(self, method: str, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.api_base_url}/{endpoint}"
        try:
            if method == "POST":
                response = requests.post(url, json=payload, timeout=self.config.timeout)
            else:
                response = requests.get(url, params=payload, timeout=self.config.timeout)
        except requests.RequestException as exc:
            raise TelegramBotError(f"Network error calling Telegram API: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise TelegramBotError("Invalid JSON response from Telegram API") from exc

        if response.status_code != 200 or not data.get("ok", False):
            description = data.get("description") or response.text
            raise TelegramBotError(f"Telegram API error: {description}")

        return data["result"]

    def _safe_json(self, result: dict[str, Any] | list[Any]) -> str:
        return json.dumps(result, indent=2)

    def _handle_tool_call(self, call: callable, *args, **kwargs) -> str:
        try:
            result = call(*args, **kwargs)
            return self._safe_json(result)
        except TelegramBotError as exc:
            return f"Error: {exc}"

    @tool(name="send_telegram_message", description="Sends a message via Telegram.")
    def send_message(
        self,
        chat_id: str,
        text: str,
        parse_mode: str | None = None,
        disable_web_page_preview: bool = False,
    ) -> str:
        """
        Send a message to a chat.

        Args:
            chat_id: The target chat ID or username.
            text: Message text to send.
            parse_mode: Optional parse mode (e.g., "MarkdownV2", "HTML").
            disable_web_page_preview: Disable link previews in the message.
        """

        def _call():
            payload: dict[str, Any] = {
                "chat_id": chat_id,
                "text": text,
                "disable_web_page_preview": disable_web_page_preview,
            }
            if parse_mode:
                payload["parse_mode"] = parse_mode
            return self._post("sendMessage", payload)

        return self._handle_tool_call(_call)

    @tool(
        name="telegram_send_photo",
        description="Send a photo to a Telegram chat.",
    )
    def send_photo(
        self,
        chat_id: str,
        photo: str,
        caption: str | None = None,
        parse_mode: str | None = None,
    ) -> str:
        """
        Send a photo to a chat.

        Args:
            chat_id: The target chat ID or username.
            photo: File ID, HTTP URL, or file path for the photo.
            caption: Optional caption for the photo.
            parse_mode: Optional parse mode for the caption.
        """

        def _call():
            payload: dict[str, Any] = {
                "chat_id": chat_id,
                "photo": photo,
            }
            if caption:
                payload["caption"] = caption
            if parse_mode:
                payload["parse_mode"] = parse_mode
            return self._post("sendPhoto", payload)

        return self._handle_tool_call(_call)

    @tool(
        name="telegram_send_document",
        description="Send a document to a Telegram chat.",
    )
    def send_document(
        self,
        chat_id: str,
        document: str,
        caption: str | None = None,
        parse_mode: str | None = None,
    ) -> str:
        """
        Send a document to a chat.

        Args:
            chat_id: The target chat ID or username.
            document: File ID, HTTP URL, or file path for the document.
            caption: Optional caption for the document.
            parse_mode: Optional parse mode for the caption.
        """

        def _call():
            payload: dict[str, Any] = {
                "chat_id": chat_id,
                "document": document,
            }
            if caption:
                payload["caption"] = caption
            if parse_mode:
                payload["parse_mode"] = parse_mode
            return self._post("sendDocument", payload)

        return self._handle_tool_call(_call)

    @tool(
        name="telegram_get_me",
        description="Retrieve basic information about the Telegram bot.",
    )
    def get_me(self) -> str:
        """Retrieve basic information about the bot."""

        def _call():
            return self._get("getMe", {})

        return self._handle_tool_call(_call)

    @tool(
        name="telegram_edit_message",
        description="Edit the text of a previously sent Telegram message.",
    )
    def edit_message_text(
        self,
        chat_id: str,
        message_id: int,
        text: str,
        parse_mode: str | None = None,
        disable_web_page_preview: bool = False,
    ) -> str:
        """
        Edit the text of a previously sent message.

        Args:
            chat_id: Identifier for the target chat.
            message_id: Identifier of the message to edit.
            text: New text for the message.
            parse_mode: Optional parse mode for formatting.
            disable_web_page_preview: Disable link previews in the message.
        """

        def _call():
            payload: dict[str, Any] = {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": text,
                "disable_web_page_preview": disable_web_page_preview,
            }
            if parse_mode:
                payload["parse_mode"] = parse_mode
            return self._post("editMessageText", payload)

        return self._handle_tool_call(_call)

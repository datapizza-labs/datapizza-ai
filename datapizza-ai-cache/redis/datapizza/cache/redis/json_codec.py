import json
from enum import Enum
from typing import cast

from datapizza.core.clients.models import ClientResponse

CACHE_SCHEMA_VERSION = 1


class CachePayloadType(str, Enum):
    CLIENT_RESPONSE = "client_response"
    JSON = "json"


def encode_cache_value(obj: object) -> bytes:
    if isinstance(obj, ClientResponse):
        response = cast(ClientResponse, obj)
        envelope = {
            "v": CACHE_SCHEMA_VERSION,
            "t": CachePayloadType.CLIENT_RESPONSE,
            "p": response.to_dict(),
        }
    else:
        envelope = {
            "v": CACHE_SCHEMA_VERSION,
            "t": CachePayloadType.JSON,
            "p": obj,
        }

    return json.dumps(envelope, separators=(",", ":")).encode("utf-8")


def decode_cache_value(raw_value: bytes | str) -> object | None:
    if isinstance(raw_value, str):
        raw_bytes = raw_value.encode("utf-8")
    elif isinstance(raw_value, bytes):
        raw_bytes = raw_value
    else:
        raise ValueError(f"unsupported redis payload type: {type(raw_value)!r}")

    envelope = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(envelope, dict):
        raise ValueError("cache envelope must be a dict")

    if envelope.get("v") != CACHE_SCHEMA_VERSION:
        return None

    payload_type = envelope.get("t")
    payload = envelope.get("p")

    if payload_type == CachePayloadType.CLIENT_RESPONSE:
        if not isinstance(payload, dict):
            return None
        return ClientResponse.from_dict(payload)

    if payload_type == CachePayloadType.JSON:
        return payload

    return None

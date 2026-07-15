from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register


PLUGIN_NAME = "astrbot_plugin_keyword_matcher"
VOLATILE_COMPONENT_KEYS = {"path", "time", "seq"}


@dataclass
class MessageRecord:
    timestamp: float
    message_id: str


def _split_keywords(value: Any) -> list[str]:
    if isinstance(value, str):
        lines = value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        return [line.strip() for line in lines if line.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _prepare_text(value: str, case_sensitive: bool) -> str:
    return value if case_sensitive else value.casefold()


def _get_int_config(
    config: AstrBotConfig | dict,
    key: str,
    default: int,
    minimum: int,
) -> int:
    try:
        value = int(config.get(key, default))
    except (TypeError, ValueError):
        return default
    return max(value, minimum)


def _contains_any(
    text: str,
    keywords: list[str],
    case_sensitive: bool,
) -> tuple[bool, str]:
    prepared_text = _prepare_text(text, case_sensitive)

    for keyword in keywords:
        prepared_keyword = _prepare_text(keyword, case_sensitive)
        if prepared_keyword in prepared_text:
            return True, keyword

    return False, ""


def _render_reply(
    template: str,
    sender_name: str,
    message: str,
    matched_field: str,
    matched_keyword: str,
) -> str:
    replacements = {
        "{name}": sender_name,
        "{message}": message,
        "{field}": matched_field,
        "{keyword}": matched_keyword,
    }

    rendered = template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def _normalize_duplicate_text(text: str, case_sensitive: bool) -> str:
    normalized = " ".join(text.split())
    return normalized if case_sensitive else normalized.casefold()


def _normalize_component_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_component_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
            if val not in (None, "") and str(key) not in VOLATILE_COMPONENT_KEYS
        }
    if isinstance(value, list):
        return [_normalize_component_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_component_value(item) for item in value]
    return value


def _component_to_mapping(component: Any) -> dict[str, Any]:
    if hasattr(component, "toDict"):
        try:
            data = component.toDict()
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    if hasattr(component, "model_dump"):
        try:
            data = component.model_dump()
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    if hasattr(component, "dict"):
        try:
            data = component.dict()
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return {
        "type": str(getattr(component, "type", type(component).__name__)),
        "repr": repr(component),
    }


def _component_fingerprint(component: Any, case_sensitive: bool) -> dict[str, Any]:
    component_type = str(getattr(component, "type", "") or "").lower()

    text = getattr(component, "text", None)
    if component_type.endswith("plain") and isinstance(text, str):
        return {
            "type": "text",
            "text": _normalize_duplicate_text(text, case_sensitive),
        }

    data = _component_to_mapping(component)
    segment_type = str(data.get("type") or component_type or type(component).__name__)
    segment_data = data.get("data", data)

    # Prefer stable media IDs over local temp paths or long transient URLs.
    if isinstance(segment_data, dict):
        if "file" in segment_data and segment_data["file"]:
            segment_data = {"file": segment_data["file"]}
        elif "id" in segment_data and segment_data["id"]:
            segment_data = {"id": segment_data["id"]}
        elif "url" in segment_data and segment_data["url"]:
            segment_data = {"url": segment_data["url"]}

    return {
        "type": segment_type,
        "data": _normalize_component_value(segment_data),
    }


def _message_fingerprint(event: AstrMessageEvent, case_sensitive: bool) -> str:
    components = event.get_messages()
    if components:
        payload: Any = [
            _component_fingerprint(component, case_sensitive)
            for component in components
        ]
    else:
        payload = {
            "text": _normalize_duplicate_text(event.message_str or "", case_sensitive),
        }

    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _message_preview(event: AstrMessageEvent) -> str:
    text = (event.message_str or "").strip()
    if text:
        return text.replace("\n", " ")[:120]
    outline = event.get_message_outline().strip()
    if outline:
        return outline[:120]
    return "<non-text message>"


@register(
    PLUGIN_NAME,
    "Codex",
    "匹配用户名、消息正文和重复刷屏消息，并在命中时执行回复、撤回或拦截。",
    "1.1.0",
)
class KeywordMatcherPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config if config is not None else {}
        self._duplicate_records: dict[tuple[str, str], deque[MessageRecord]] = {}
        self._recalled_message_ids: dict[str, float] = {}

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_all_message(self, event: AstrMessageEvent):
        """监听全部消息，并匹配用户名、消息正文关键字和重复刷屏。"""
        sender_name = event.get_sender_name() or ""
        message = event.message_str or ""

        if await self._handle_duplicate_message(event, sender_name):
            return

        username_keywords = _split_keywords(self.config.get("username_keywords", ""))
        message_keywords = _split_keywords(self.config.get("message_keywords", ""))
        if not username_keywords and not message_keywords:
            return

        case_sensitive = bool(self.config.get("case_sensitive", False))
        match_mode = str(self.config.get("match_mode", "any")).lower()
        reply_text = str(self.config.get("reply_text", "") or "")
        stop_on_match = bool(self.config.get("stop_on_match", False))
        log_match = bool(self.config.get("log_match", True))

        username_hit, username_keyword = _contains_any(
            sender_name,
            username_keywords,
            case_sensitive,
        )
        message_hit, message_keyword = _contains_any(
            message,
            message_keywords,
            case_sensitive,
        )

        checks: list[bool] = []
        if username_keywords:
            checks.append(username_hit)
        if message_keywords:
            checks.append(message_hit)

        matched = all(checks) if match_mode == "all" else any(checks)
        if not matched:
            return

        matched_field = "username" if username_hit else "message"
        matched_keyword = username_keyword or message_keyword

        if log_match:
            preview = message.replace("\n", " ")[:120]
            logger.info(
                f"[{PLUGIN_NAME}] matched field={matched_field} "
                f"keyword={matched_keyword!r} sender={sender_name!r} "
                f"message={preview!r}"
            )

        if reply_text:
            yield event.plain_result(
                _render_reply(
                    reply_text,
                    sender_name,
                    message,
                    matched_field,
                    matched_keyword,
                )
            )

        if stop_on_match:
            event.stop_event()

    async def _handle_duplicate_message(
        self,
        event: AstrMessageEvent,
        sender_name: str,
    ) -> bool:
        if not bool(self.config.get("enable_duplicate_recall", True)):
            return False

        threshold = _get_int_config(
            self.config,
            "duplicate_threshold",
            default=5,
            minimum=2,
        )
        window_seconds = _get_int_config(
            self.config,
            "duplicate_window_seconds",
            default=30,
            minimum=1,
        )
        case_sensitive = bool(self.config.get("duplicate_case_sensitive", False))

        sender_id = event.get_sender_id() or sender_name or "unknown"
        user_key = (
            f"{event.get_platform_id()}:{event.get_session_id()}:{sender_id}"
        )
        fingerprint = _message_fingerprint(event, case_sensitive)
        bucket_key = (user_key, fingerprint)
        now = time.monotonic()

        self._prune_duplicate_cache(now, window_seconds)

        records = self._duplicate_records.setdefault(bucket_key, deque())
        while records and now - records[0].timestamp > window_seconds:
            records.popleft()

        message_id = str(getattr(event.message_obj, "message_id", "") or "")
        records.append(MessageRecord(timestamp=now, message_id=message_id))

        if len(records) < threshold:
            return False

        recall_all = bool(self.config.get("duplicate_recall_all_in_window", True))
        target_records = list(records) if recall_all else [records[-1]]
        target_message_ids = [
            record.message_id
            for record in target_records
            if record.message_id
        ]

        if not target_message_ids:
            logger.warning(
                f"[{PLUGIN_NAME}] duplicate message detected but message_id is empty; "
                f"sender={sender_name!r} preview={_message_preview(event)!r}"
            )
        else:
            await self._recall_messages(
                event,
                target_message_ids,
                reason="duplicate",
            )

        if bool(self.config.get("log_match", True)):
            logger.info(
                f"[{PLUGIN_NAME}] duplicate message detected "
                f"count={len(records)} threshold={threshold} "
                f"window={window_seconds}s sender={sender_name!r} "
                f"preview={_message_preview(event)!r}"
            )

        if bool(self.config.get("duplicate_stop_on_recall", True)):
            event.stop_event()
        return True

    def _prune_duplicate_cache(self, now: float, window_seconds: int) -> None:
        max_age = max(window_seconds * 2, 60)

        empty_keys: list[tuple[str, str]] = []
        for key, records in self._duplicate_records.items():
            while records and now - records[0].timestamp > max_age:
                records.popleft()
            if not records:
                empty_keys.append(key)

        for key in empty_keys:
            self._duplicate_records.pop(key, None)

        expired_recalled_ids = [
            message_id
            for message_id, timestamp in self._recalled_message_ids.items()
            if now - timestamp > max_age
        ]
        for message_id in expired_recalled_ids:
            self._recalled_message_ids.pop(message_id, None)

    async def _recall_messages(
        self,
        event: AstrMessageEvent,
        message_ids: list[str],
        reason: str,
    ) -> None:
        unique_message_ids = list(dict.fromkeys(message_ids))
        for message_id in unique_message_ids:
            recall_key = f"{event.get_platform_id()}:{message_id}"
            if recall_key in self._recalled_message_ids:
                continue

            recalled = await self._recall_message(event, message_id)
            self._recalled_message_ids[recall_key] = time.monotonic()
            if recalled:
                logger.info(
                    f"[{PLUGIN_NAME}] recalled message_id={message_id} "
                    f"reason={reason}"
                )

    async def _recall_message(
        self,
        event: AstrMessageEvent,
        message_id: str,
    ) -> bool:
        if event.get_platform_name() != "aiocqhttp":
            logger.warning(
                f"[{PLUGIN_NAME}] recall is only implemented for aiocqhttp/OneBot; "
                f"platform={event.get_platform_name()!r}"
            )
            return False

        bot = getattr(event, "bot", None)
        api = getattr(bot, "api", None)
        call_action = getattr(api, "call_action", None) or getattr(
            bot,
            "call_action",
            None,
        )
        if call_action is None:
            logger.warning(f"[{PLUGIN_NAME}] aiocqhttp bot has no call_action API")
            return False

        try:
            ret = await call_action("delete_msg", message_id=message_id)
        except Exception as exc:
            logger.warning(
                f"[{PLUGIN_NAME}] failed to recall message_id={message_id}: {exc}"
            )
            return False

        logger.debug(f"[{PLUGIN_NAME}] delete_msg returned: {ret}")
        return True

    async def terminate(self):
        """插件停用或卸载时由 AstrBot 调用。"""

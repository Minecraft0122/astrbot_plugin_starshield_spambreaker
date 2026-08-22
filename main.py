from __future__ import annotations

import hashlib
import json
import time
from threading import Lock
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

try:
    from .spam_detector import SpamDetector
except ImportError:  # AstrBot may load main.py as a standalone module.
    from spam_detector import SpamDetector


PLUGIN_NAME = "astrbot_plugin_keyword_matcher"
PLUGIN_DISPLAY_NAME = "星盾·刷屏终结者"
PLUGIN_VERSION = "1.3.0"
_VOLATILE_COMPONENT_KEYS = {"path", "time", "seq"}


def _value_from_event(event: Any, method_name: str, attribute_name: str = "") -> str:
    value: Any = None
    method = getattr(event, method_name, None)
    if callable(method):
        try:
            value = method()
        except Exception:
            value = None
    if value in (None, "") and attribute_name:
        value = getattr(event, attribute_name, None)
    return str(value or "")


def _config_bool(config: AstrBotConfig | dict, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "yes", "on", "是", "开启"}:
            return True
        if normalized in {"0", "false", "no", "off", "否", "关闭"}:
            return False
    return default


def _message_text(event: AstrMessageEvent) -> str:
    value = getattr(event, "message_str", "")
    if callable(value):
        try:
            value = value()
        except Exception:
            value = ""
    if value:
        return str(value)

    return ""


def _message_id(event: AstrMessageEvent) -> str:
    message_obj = getattr(event, "message_obj", None)
    if isinstance(message_obj, dict):
        value = message_obj.get("message_id", "")
    else:
        value = getattr(message_obj, "message_id", "")
    return str(value or "")


def _sender_key(
    event: AstrMessageEvent,
    platform_id: str,
    session_id: str,
    sender_id: str,
    sender_name: str,
) -> str:
    """Build a per-sender key without merging anonymous users."""

    stable_sender = sender_id
    if not stable_sender:
        message_obj = getattr(event, "message_obj", None)
        containers = [message_obj]
        if isinstance(message_obj, dict):
            containers.append(message_obj.get("sender"))
        else:
            containers.append(getattr(message_obj, "sender", None))

        for container in containers:
            if isinstance(container, dict):
                stable_sender = str(
                    container.get("user_id")
                    or container.get("sender_id")
                    or container.get("id")
                    or ""
                )
            elif container is not None:
                stable_sender = str(
                    getattr(container, "user_id", "")
                    or getattr(container, "sender_id", "")
                    or getattr(container, "id", "")
                    or ""
                )
            if stable_sender:
                break

    stable_sender = stable_sender or sender_name
    if not stable_sender:
        # An adapter without any sender identity cannot safely share state
        # with other users. Isolate this event instead of using one global
        # ``unknown`` bucket that would create false positives.
        stable_sender = f"event:{id(event)}"
    return f"{platform_id}:{session_id}:{stable_sender}"


def _component_mapping(component: Any) -> dict[str, Any]:
    for method_name in ("toDict", "model_dump", "dict"):
        method = getattr(component, method_name, None)
        if callable(method):
            try:
                value = method()
            except Exception:
                continue
            if isinstance(value, dict):
                return value
    return {
        "type": str(getattr(component, "type", type(component).__name__)),
        "repr": repr(component),
    }


def _stable_component_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _stable_component_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if item not in (None, "") and str(key) not in _VOLATILE_COMPONENT_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_stable_component_value(item) for item in value]
    return value


def _media_comparison_key(event: AstrMessageEvent) -> str:
    get_messages = getattr(event, "get_messages", None)
    if not callable(get_messages):
        return ""
    try:
        components = get_messages() or []
    except Exception:
        return ""
    if not components:
        return ""

    payload = []
    for component in components:
        data = _component_mapping(component)
        component_type = str(data.get("type", getattr(component, "type", "")))
        if component_type.lower().endswith(("plain", "text")):
            continue
        component_data = data.get("data", data)
        if isinstance(component_data, dict):
            # Prefer stable IDs over local paths and transient download URLs.
            for stable_key in ("file", "id", "url"):
                if component_data.get(stable_key):
                    component_data = {stable_key: component_data[stable_key]}
                    break
        payload.append(
            {
                "type": component_type,
                "data": _stable_component_value(component_data),
            }
        )

    if not payload:
        return ""

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return "media:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@register(
    PLUGIN_NAME,
    "Codex",
    f"{PLUGIN_DISPLAY_NAME}：拦截连续重复、随机尾缀近重复、单条高重复内容和短时间长消息刷屏。",
    PLUGIN_VERSION,
)
class KeywordMatcherPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config if config is not None else {}
        self.detector = SpamDetector(self.config)
        self._detector_lock = Lock()
        self._recalled_message_ids: dict[str, float] = {}

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_all_message(self, event: AstrMessageEvent):
        """检查所有消息，命中刷屏规则后停止后续处理并尽力撤回。"""

        platform_id = _value_from_event(event, "get_platform_id")
        session_id = _value_from_event(event, "get_session_id")
        sender_id = _value_from_event(event, "get_sender_id")
        sender_name = _value_from_event(event, "get_sender_name")
        sender_key = _sender_key(
            event,
            platform_id,
            session_id,
            sender_id,
            sender_name,
        )
        text = _message_text(event)
        media_key = _media_comparison_key(event)
        message_id = _message_id(event)

        with self._detector_lock:
            decision = self.detector.check(
                sender_key,
                text,
                message_id=message_id,
                comparison_key=media_key or None,
            )
        if not decision.blocked:
            return

        # Stop first: recall is platform-dependent and must not delay blocking
        # downstream plugins or an LLM request.
        event.stop_event()
        logger.info(
            f"[{PLUGIN_NAME}] blocked spam reason={decision.reason} "
            f"sender={sender_name or sender_id or 'unknown'!r} "
            f"platform={platform_id!r} detail={decision.detail}"
        )

        if _config_bool(self.config, "enable_recall", True):
            await self._recall_messages(
                event,
                decision.message_ids,
                reason=decision.reason,
            )

    async def _recall_messages(
        self,
        event: AstrMessageEvent,
        message_ids: tuple[str, ...],
        reason: str,
    ) -> None:
        for message_id in dict.fromkeys(message_ids):
            if not message_id:
                continue
            recall_key = f"{_value_from_event(event, 'get_platform_id')}:{message_id}"
            if recall_key in self._recalled_message_ids:
                continue
            if await self._recall_message(event, message_id):
                self._recalled_message_ids[recall_key] = time.monotonic()
                logger.info(
                    f"[{PLUGIN_NAME}] recalled message_id={message_id} reason={reason}"
                )

        self._prune_recalled_ids()

    async def _recall_message(
        self,
        event: AstrMessageEvent,
        message_id: str,
    ) -> bool:
        platform_name = _value_from_event(event, "get_platform_name")
        if platform_name != "aiocqhttp":
            logger.debug(
                f"[{PLUGIN_NAME}] platform={platform_name!r} has no generic recall API"
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
            await call_action("delete_msg", message_id=message_id)
        except Exception as exc:
            logger.warning(
                f"[{PLUGIN_NAME}] failed to recall message_id={message_id}: {exc}"
            )
            return False
        return True

    def _prune_recalled_ids(self) -> None:
        now = time.monotonic()
        for message_id, timestamp in list(self._recalled_message_ids.items()):
            if now - timestamp > 300:
                self._recalled_message_ids.pop(message_id, None)

    async def terminate(self):
        """插件停用或卸载时由 AstrBot 调用。"""

        self._recalled_message_ids.clear()

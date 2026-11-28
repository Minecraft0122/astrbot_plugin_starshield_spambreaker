from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import time
from threading import Lock
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

try:
    from .spam_detector import DEFAULT_BANNED_WORDS, SpamDetector
except ImportError:  # AstrBot may load main.py as a standalone module.
    from spam_detector import DEFAULT_BANNED_WORDS, SpamDetector


PLUGIN_NAME = "astrbot_plugin_keyword_matcher"
PLUGIN_DISPLAY_NAME = "星盾·刷屏终结者"
PLUGIN_VERSION = "1.10.0"
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
    # WakingCheckStage strips whitespace and removes wake prefixes from
    # event.message_str. Inspect the original top-level Plain chain first.
    get_messages = getattr(event, "get_messages", None)
    try:
        components = get_messages() if callable(get_messages) else []
    except Exception:
        components = []
    parts = []
    for component in components or []:
        data = _component_mapping(component)
        component_type = str(data.get("type", getattr(component, "type", "")))
        if component_type.lower().split(".")[-1] in {"plain", "text"}:
            content = data.get("data", data)
            if isinstance(content, dict):
                parts.append(str(content.get("text", "") or ""))
    if parts:
        return "".join(parts)
    message_obj = getattr(event, "message_obj", None)
    original = (
        message_obj.get("message_str")
        if isinstance(message_obj, dict)
        else getattr(message_obj, "message_str", None)
    )
    if isinstance(original, str):
        return original
    value = getattr(event, "message_str", "")
    getter = getattr(event, "get_message_str", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:
            pass
    if callable(value):
        try:
            value = value()
        except Exception:
            value = ""
    return str(value or "")


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
        if component_type.lower().split(".")[-1] in {
            "plain", "text", "at", "atall", "reply",
        }:
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
    f"{PLUGIN_DISPLAY_NAME}：检测违禁词、空白消息和多种刷屏内容。",
    PLUGIN_VERSION,
)
class KeywordMatcherPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config if config is not None else {}
        self._initialize_keyword_list()
        self.detector = SpamDetector(self.config)
        self._detector_lock = Lock()
        self._recalled_message_ids: dict[str, float] = {}
        self._warned_recall_platforms: set[str] = set()
        logger.info(
            f"[{PLUGIN_NAME}] ready version={PLUGIN_VERSION} "
            f"keywords={len(self.detector.pinyin_detector.words)} "
            f"pinyin={self.detector.pinyin_detector.enabled} "
            f"block_notice={_config_bool(self.config, 'enable_block_notice', False)}"
        )

    async def initialize(self) -> None:
        """Report actual post-load registration and profile selection."""
        try:
            from astrbot.core.star.star import star_map
            from astrbot.core.star.star_handler import EventType, star_handlers_registry

            module_path = type(self).__module__
            metadata = star_map.get(module_path)
            handlers = [
                handler
                for handler in star_handlers_registry.get_handlers_by_module_name(module_path)
                if handler.handler_name == "on_all_message"
            ]
            active = bool(metadata and metadata.activated)
            for handler in handlers:
                callback = handler.handler
                bound = (
                    isinstance(callback, functools.partial)
                    and bool(callback.args)
                    and callback.args[0] is self
                ) or getattr(callback, "__self__", None) is self
                logger.info(
                    f"[{PLUGIN_NAME}] routing registered=True active={active} "
                    f"handler_enabled={handler.enabled} bound={bound} "
                    f"priority={handler.extras_configs.get('priority', 0)} "
                    f"filters={[type(item).__name__ for item in handler.event_filters]} "
                    f"permissions={[str(item.permission_type) for item in handler.event_filters if hasattr(item, 'permission_type')]}"
                )
                if not active or not handler.enabled or not bound:
                    logger.warning(f"[{PLUGIN_NAME}] handler is not ready for message dispatch")
            if not handlers:
                logger.error(f"[{PLUGIN_NAME}] routing registered=False module={module_path!r}")

            manager = getattr(self.context, "astrbot_config_mgr", None)
            profiles = getattr(manager, "confs", None)
            if not isinstance(profiles, dict):
                get_config = getattr(self.context, "get_config", None)
                profiles = {"default": get_config()} if callable(get_config) else {}
            for profile_id, config in profiles.items():
                plugin_set = config.get("plugin_set", ["*"])
                eligible = star_handlers_registry.get_handlers_by_event_type(
                    EventType.AdapterMessageEvent, plugins_name=plugin_set,
                )
                selected = any(handler in eligible for handler in handlers)
                settings = config.get("platform_settings", {})
                logger.info(
                    f"[{PLUGIN_NAME}] routing profile={profile_id!r} selected={selected} "
                    f"whitelist_enabled={settings.get('enable_id_white_list', False)} "
                    f"whitelist_entries={len(settings.get('id_whitelist', []))}"
                )
                if not selected:
                    logger.warning(
                        f"[{PLUGIN_NAME}] profile={profile_id!r} does not dispatch messages "
                        f"to this plugin; select {PLUGIN_NAME!r} in that profile's plugin_set "
                        "and check plugin/handler activation"
                    )
        except Exception as exc:
            # Diagnostics must never prevent an otherwise usable plugin loading.
            logger.warning(f"[{PLUGIN_NAME}] routing diagnostics unavailable: {exc}")

    def _initialize_keyword_list(self) -> None:
        """Persist defaults once so upgrades expose the effective list in WebUI."""
        raw_words = self.config.get("banned_words")
        initialize = _config_bool(self.config, "initialize_banned_words", True)
        if isinstance(raw_words, str):
            words = raw_words.splitlines()
        elif isinstance(raw_words, list):
            words = [str(word) for word in raw_words]
        else:
            words = []
        words = [word.strip() for word in words if word.strip()]
        if initialize:
            # Only replace the exact former default set. A customized list is
            # administrator-owned, even when it contains a broad keyword.
            former_defaults = set(DEFAULT_BANNED_WORDS) | {"垃圾", "滚蛋", "外挂", "作弊"}
            if set(words) == former_defaults:
                words = list(DEFAULT_BANNED_WORDS)
            words = list(dict.fromkeys([*DEFAULT_BANNED_WORDS, *words]))
        if not initialize and raw_words == words:
            return

        previous = dict(self.config)
        self.config["banned_words"] = words
        self.config["initialize_banned_words"] = False
        save_config = getattr(self.config, "save_config", None)
        if callable(save_config):
            try:
                save_config()
            except Exception:
                self.config.clear()
                self.config.update(previous)
                raise
        logger.info(f"[{PLUGIN_NAME}] initialized keyword list ({len(words)} entries)")

    # Match every text shape, including empty/media-only and multiline events.
    # This handler still runs in ProcessStage, after framework-level checks.
    @filter.regex(r"(?s).*", priority=10000)
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
        platform_name = _value_from_event(event, "get_platform_name")
        logger.info(
            f"[{PLUGIN_NAME}] inspect platform={platform_name!r} "
            f"instance={platform_id!r} message_id={message_id!r} "
            f"text_chars={len(text)} has_media={bool(media_key)}"
        )

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
        should_call_llm = getattr(event, "should_call_llm", None)
        if callable(should_call_llm):
            # Despite its name, AstrBot's flag means "disable default LLM".
            should_call_llm(True)
        logger.info(
            f"[{PLUGIN_NAME}] blocked spam reason={decision.reason} "
            f"sender={sender_name or sender_id or 'unknown'!r} "
            f"platform={platform_name!r} instance={platform_id!r} "
            f"message_id={message_id!r} detail={decision.detail}"
        )

        if _config_bool(self.config, "enable_block_notice", False):
            reasons = {
                "banned_keyword": "违禁词",
                "blank_message": "空白消息",
                "excessive_blank_lines": "连续空行过多",
                "exact_duplicate": "重复消息",
                "near_duplicate": "近似重复消息",
                "repeated_content": "内容大量重复",
                "long_message_burst": "长消息刷屏",
            }
            try:
                await asyncio.wait_for(
                    event.send(event.plain_result(
                        f"星盾已拦截本条消息的机器人处理：{reasons.get(decision.reason, '刷屏')}。"
                    )),
                    timeout=5,
                )
            except Exception as exc:
                logger.warning(f"[{PLUGIN_NAME}] failed to send block notice: {exc}")

        if _config_bool(self.config, "enable_recall", True):
            # Historical messages are used only as comparison evidence. Never
            # recall them when the current message is the one being blocked.
            await self._recall_messages(
                event,
                (message_id,) if message_id else (),
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
        if platform_name in {"qq_official", "qq_official_webhook"}:
            return await self._recall_qq_official_message(event, message_id)
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

    async def _recall_qq_official_message(
        self, event: AstrMessageEvent, message_id: str,
    ) -> bool:
        """Recall a member's guild-channel message when official API permits it."""
        message_obj = getattr(event, "message_obj", None)
        raw_message = (
            message_obj.get("raw_message")
            if isinstance(message_obj, dict)
            else getattr(message_obj, "raw_message", None)
        )
        # Group/C2C/DM DELETE routes only recall the bot's own outgoing messages.
        # Never send an incoming user's ID to those routes or reinterpret a
        # group_openid/session_id as a guild channel ID.
        is_channel_message = False
        if raw_message is not None:
            try:
                from botpy.message import Message

                is_channel_message = isinstance(raw_message, Message)
            except ImportError:
                logger.warning(f"[{PLUGIN_NAME}] QQ official recall SDK unavailable")
                return False
        if not is_channel_message:
            platform_name = _value_from_event(event, "get_platform_name")
            warning_key = f"{_value_from_event(event, 'get_platform_id')}:incoming-user-recall"
            if warning_key not in self._warned_recall_platforms:
                self._warned_recall_platforms.add(warning_key)
                logger.warning(
                    f"[{PLUGIN_NAME}] {platform_name}: 已停止 AstrBot 后续处理和默认 LLM 回复；"
                    "QQ 官方普通群、单聊和频道私信的撤回接口仅支持机器人自己发送的消息，"
                    "不能用来撤回本条用户消息。只有文字子频道支持按管理权限撤回成员消息。"
                )
            return False

        channel_id = str(getattr(raw_message, "channel_id", "") or "")
        if not channel_id:
            logger.warning(f"[{PLUGIN_NAME}] QQ guild recall skipped: missing channel_id")
            return False
        api = getattr(getattr(event, "bot", None), "api", None)
        recall_message = getattr(api, "recall_message", None)
        if not callable(recall_message):
            logger.warning(f"[{PLUGIN_NAME}] QQ guild recall API unavailable")
            return False
        try:
            result = await asyncio.wait_for(
                recall_message(channel_id=channel_id, message_id=message_id, hidetip=False),
                timeout=5,
            )
            # botpy raises on HTTP errors and returns None on empty success.
            # Also reject explicit error responses from compatible wrappers.
            if result is False or (
                isinstance(result, dict) and result.get("code") not in (None, 0)
            ):
                logger.warning(
                    f"[{PLUGIN_NAME}] QQ guild recall rejected message_id={message_id}"
                )
                return False
        except asyncio.TimeoutError:
            logger.warning(f"[{PLUGIN_NAME}] QQ guild recall timed out message_id={message_id}")
            return False
        except Exception as exc:
            logger.warning(
                f"[{PLUGIN_NAME}] QQ guild recall failed message_id={message_id}: {exc}; "
                "请检查机器人在频道中的管理权限和接口权限，消息拦截保持生效。"
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

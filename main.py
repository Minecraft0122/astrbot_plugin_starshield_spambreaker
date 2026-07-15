from __future__ import annotations

from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register


PLUGIN_NAME = "astrbot_plugin_keyword_matcher"


def _split_keywords(value: Any) -> list[str]:
    if isinstance(value, str):
        lines = value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        return [line.strip() for line in lines if line.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _prepare_text(value: str, case_sensitive: bool) -> str:
    return value if case_sensitive else value.casefold()


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


@register(
    PLUGIN_NAME,
    "Codex",
    "匹配用户名或消息正文中是否包含指定文字，并在命中时执行回复或拦截。",
    "1.0.0",
)
class KeywordMatcherPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config if config is not None else {}

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_all_message(self, event: AstrMessageEvent):
        """监听全部消息，并匹配用户名或消息正文关键字。"""
        username_keywords = _split_keywords(self.config.get("username_keywords", ""))
        message_keywords = _split_keywords(self.config.get("message_keywords", ""))
        if not username_keywords and not message_keywords:
            return

        case_sensitive = bool(self.config.get("case_sensitive", False))
        match_mode = str(self.config.get("match_mode", "any")).lower()
        reply_text = str(self.config.get("reply_text", "") or "")
        stop_on_match = bool(self.config.get("stop_on_match", False))
        log_match = bool(self.config.get("log_match", True))

        sender_name = event.get_sender_name() or ""
        message = event.message_str or ""

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

    async def terminate(self):
        """插件停用或卸载时由 AstrBot 调用。"""

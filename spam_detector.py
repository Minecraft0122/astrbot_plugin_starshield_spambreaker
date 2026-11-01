"""Pure-Python spam detection for the AstrBot message filter.

The detector intentionally has no AstrBot dependency.  It keeps a small,
per-sender history and returns a structured result so the framework adapter
can decide how to stop or recall a message.
"""

from __future__ import annotations

import time
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping
import re

from pypinyin import Style, pinyin


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
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


def _as_int(
    config: Mapping[str, Any],
    key: str,
    default: int,
    minimum: int,
    maximum: int | None = None,
) -> int:
    try:
        value = int(config.get(key, default))
    except (TypeError, ValueError):
        value = default
    value = max(value, minimum)
    return min(value, maximum) if maximum is not None else value


def _as_float(
    config: Mapping[str, Any],
    key: str,
    default: float,
    minimum: float,
    maximum: float | None = None,
) -> float:
    try:
        value = float(config.get(key, default))
    except (TypeError, ValueError):
        value = default
    value = max(value, minimum)
    return min(value, maximum) if maximum is not None else value


def normalize_text(value: str, *, case_sensitive: bool = False) -> str:
    """Return a comparison form that is resistant to harmless obfuscation."""

    normalized = unicodedata.normalize("NFKC", value)
    cleaned: list[str] = []
    for character in normalized:
        category = unicodedata.category(character)
        # Keep line/tab separators as ordinary spaces; dropping them outright
        # would concatenate words around a newline. Other control/format
        # characters include the invisible separators used for bypasses.
        if character in "\n\r\t":
            cleaned.append(" ")
        elif category.startswith("C"):
            continue
        else:
            cleaned.append(character)

    normalized = " ".join("".join(cleaned).split())
    return normalized if case_sensitive else normalized.casefold()


def _pinyin_forms(value: str, max_combinations: int) -> set[str]:
    """Build bounded, tone-free readings for a configured Chinese keyword."""

    syllables = pinyin(
        value,
        style=Style.NORMAL,
        heteronym=True,
        errors=lambda chars: list(chars),
    )
    forms = {""}
    for options in syllables:
        choices = {option.casefold() for option in options if option}
        if not choices:
            continue
        next_forms: set[str] = set()
        for prefix in sorted(forms):
            for choice in sorted(choices):
                next_forms.add(prefix + choice)
                if len(next_forms) >= max_combinations:
                    break
            if len(next_forms) >= max_combinations:
                break
        forms = next_forms
    return {form for form in forms if form}


class PinyinKeywordDetector:
    """Match configured Chinese keywords by literal text or pinyin spelling."""

    def __init__(self, config: Mapping[str, Any]):
        self.enabled = _as_bool(config.get("enable_pinyin_keywords"), default=True)
        self.max_combinations = _as_int(
            config,
            "pinyin_max_combinations",
            default=32,
            minimum=1,
            maximum=256,
        )
        raw_words = config.get("banned_words", "")
        if isinstance(raw_words, str):
            words = raw_words.splitlines()
        elif isinstance(raw_words, (list, tuple, set)):
            words = [str(word) for word in raw_words]
        else:
            words = []
        self.words = tuple(dict.fromkeys(word.strip() for word in words if word.strip()))
        self.index: dict[str, str] = {}
        self.max_keyword_chars = 1
        if self.enabled:
            for word in self.words:
                self.max_keyword_chars = max(self.max_keyword_chars, len(word))
                for form in _pinyin_forms(word, self.max_combinations):
                    self.index.setdefault(form, word)

    def check(self, text: str) -> str:
        if not self.enabled or not self.index or not text:
            return ""
        normalized = normalize_text(text)
        for word in self.words:
            if normalize_text(word) in normalized:
                return word
        # Compare contiguous alphanumeric runs to avoid matching a short pinyin
        # syllable inside unrelated English words.
        for token in re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", normalized):
            if token.isascii():
                if token in self.index:
                    return self.index[token]
                continue
            max_length = min(len(token), self.max_keyword_chars)
            for start in range(len(token)):
                for length in range(1, max_length + 1):
                    candidate = token[start : start + length]
                    for form in _pinyin_forms(candidate, self.max_combinations):
                        if form in self.index:
                            return self.index[form]
        return ""


@dataclass(frozen=True)
class MessageSnapshot:
    timestamp: float
    text: str
    normalized: str
    message_id: str = ""
    comparison_key: str = ""


@dataclass(frozen=True)
class SpamDecision:
    """A detector result.  ``reason`` is empty when the message is allowed."""

    reason: str = ""
    detail: str = ""
    message_ids: tuple[str, ...] = ()

    @property
    def blocked(self) -> bool:
        return bool(self.reason)


@dataclass
class _SenderState:
    recent: deque[MessageSnapshot] = field(default_factory=deque)
    long_message_times: deque[float] = field(default_factory=deque)
    last_seen: float = 0.0


class SpamDetector:
    """Detect short-window duplicates and excessive repetition in one message."""

    def __init__(self, config: Mapping[str, Any] | None = None):
        self.config: Mapping[str, Any] = config or {}
        self.case_sensitive = _as_bool(
            self.config.get("case_sensitive"),
            default=False,
        )
        self.exact_window_seconds = _as_float(
            self.config,
            "exact_duplicate_window_seconds",
            default=60.0,
            minimum=0.1,
            maximum=86400.0,
        )
        self.near_duplicate_enabled = _as_bool(
            self.config.get("enable_near_duplicate"),
            default=True,
        )
        self.near_duplicate_window_seconds = _as_float(
            self.config,
            "near_duplicate_window_seconds",
            default=60.0,
            minimum=0.1,
            maximum=86400.0,
        )
        self.near_duplicate_min_chars = _as_int(
            self.config,
            "near_duplicate_min_chars",
            default=12,
            minimum=4,
            maximum=10000,
        )
        self.near_duplicate_ratio = _as_float(
            self.config,
            "near_duplicate_prefix_ratio",
            default=0.75,
            minimum=0.5,
            maximum=1.0,
        )
        self.near_duplicate_max_suffix_chars = _as_int(
            self.config,
            "near_duplicate_max_suffix_chars",
            default=64,
            minimum=1,
            maximum=10000,
        )
        self.near_duplicate_history = _as_int(
            self.config,
            "near_duplicate_history",
            default=3,
            minimum=1,
            maximum=10,
        )
        self.repeated_content_enabled = _as_bool(
            self.config.get("enable_repeated_content"),
            default=True,
        )
        self.repeat_min_repetitions = _as_int(
            self.config,
            "repeat_min_repetitions",
            default=4,
            minimum=3,
            maximum=20,
        )
        self.repeat_min_chars = _as_int(
            self.config,
            "repeat_min_chars",
            default=12,
            minimum=4,
            maximum=100000,
        )
        self.repeat_coverage = _as_float(
            self.config,
            "repeat_min_coverage",
            default=0.60,
            minimum=0.3,
            maximum=1.0,
        )
        self.repeat_max_unit_chars = _as_int(
            self.config,
            "repeat_max_unit_chars",
            default=32,
            minimum=1,
            maximum=128,
        )
        self.repeat_scan_chars = _as_int(
            self.config,
            "repeat_scan_chars",
            default=4000,
            minimum=100,
            maximum=20000,
        )
        self.cache_ttl_seconds = _as_float(
            self.config,
            "cache_ttl_seconds",
            default=120.0,
            minimum=1.0,
            maximum=86400.0,
        )
        self.max_recent_messages = _as_int(
            self.config,
            "max_recent_messages",
            default=20,
            minimum=2,
            maximum=100,
        )
        self.max_senders = _as_int(
            self.config,
            "max_senders",
            default=10000,
            minimum=1,
            maximum=1000000,
        )
        self.long_message_enabled = _as_bool(
            self.config.get("enable_long_message_burst"), default=True
        )
        self.long_message_min_chars = _as_int(
            self.config,
            "long_message_min_chars",
            default=300,
            minimum=32,
            maximum=100000,
        )
        self.long_message_max_messages = _as_int(
            self.config,
            "long_message_max_messages",
            default=3,
            minimum=2,
            maximum=100,
        )
        self.long_message_window_seconds = _as_float(
            self.config,
            "long_message_window_seconds",
            default=10.0,
            minimum=0.1,
            maximum=86400.0,
        )
        self.pinyin_detector = PinyinKeywordDetector(self.config)
        self._states: dict[str, _SenderState] = {}

    def check(
        self,
        sender_key: str,
        text: str,
        *,
        message_id: str = "",
        comparison_key: str | None = None,
        timestamp: float | None = None,
    ) -> SpamDecision:
        """Inspect one message and update this sender's bounded history."""

        now = time.monotonic() if timestamp is None else float(timestamp)
        sender_key = str(sender_key or "unknown")
        raw_text = str(text or "")
        normalized = normalize_text(
            raw_text,
            case_sensitive=self.case_sensitive,
        )
        if comparison_key is None:
            comparison_key = normalized
        else:
            external_key = str(comparison_key)
            comparison_key = (
                f"{normalized}\x1f{external_key}" if normalized else external_key
            )
        current = MessageSnapshot(
            timestamp=now,
            text=raw_text,
            normalized=normalized,
            message_id=str(message_id or ""),
            comparison_key=comparison_key,
        )

        state = self._states.setdefault(sender_key, _SenderState())
        state.last_seen = now
        self._prune_state(state, now)

        banned_word = self.pinyin_detector.check(raw_text)
        decision: SpamDecision | None = (
            SpamDecision(
                reason="banned_keyword",
                detail=f"消息命中违禁词或其拼音形式：{banned_word}",
                message_ids=_ids(current),
            )
            if banned_word
            else None
        )

        if decision is None and current.comparison_key:
            duplicate = next(
                (
                    item
                    for item in reversed(state.recent)
                    if self._is_exact_duplicate(current, item)
                ),
                None,
            )
            if duplicate is not None:
                decision = SpamDecision(
                    reason="exact_duplicate",
                    detail="与该用户上一条消息规范化后完全一致",
                    message_ids=_ids(current),
                )
            elif normalized and self.near_duplicate_enabled:
                near_match = self._find_near_duplicate(current, state.recent)
                if near_match is not None:
                    decision = SpamDecision(
                        reason="near_duplicate",
                        detail="与近期消息拥有相同主体，仅追加了短尾部内容",
                        message_ids=_ids(current),
                    )

            if decision is None and normalized and self.repeated_content_enabled:
                repeated = self._find_repeated_segment(normalized)
                if repeated is not None:
                    unit, count, coverage = repeated
                    decision = SpamDecision(
                        reason="repeated_content",
                        detail=(
                            f"单条消息重复片段长度={unit}，连续次数={count}，"
                            f"覆盖率={coverage:.0%}"
                        ),
                        message_ids=_ids(current),
                    )

        if decision is None and self.long_message_enabled:
            while (
                state.long_message_times
                and now - state.long_message_times[0] > self.long_message_window_seconds
            ):
                state.long_message_times.popleft()
            if len(raw_text) >= self.long_message_min_chars:
                if len(state.long_message_times) >= self.long_message_max_messages - 1:
                    decision = SpamDecision(
                        reason="long_message_burst",
                        detail=(
                            f"{self.long_message_max_messages} 条至少 "
                            f"{self.long_message_min_chars} 字符的消息在 "
                            f"{self.long_message_window_seconds:g} 秒内发送"
                        ),
                        message_ids=_ids(current),
                    )
                else:
                    state.long_message_times.append(now)

        # Only accepted messages contribute evidence for subsequent checks.
        if decision is None:
            state.recent.append(current)
            if len(state.recent) > self.max_recent_messages:
                state.recent.popleft()

        self._prune_state(state, now)
        self._trim_senders()
        return decision or SpamDecision()

    def _is_exact_duplicate(
        self,
        current: MessageSnapshot,
        previous: MessageSnapshot | None,
    ) -> bool:
        age = current.timestamp - previous.timestamp if previous else None
        return bool(
            previous
            and age is not None
            and 0 <= age <= self.exact_window_seconds
            and current.comparison_key == previous.comparison_key
        )

    def _find_near_duplicate(
        self,
        current: MessageSnapshot,
        recent: deque[MessageSnapshot],
    ) -> MessageSnapshot | None:
        shorter_text = current.normalized
        if len(shorter_text) < self.near_duplicate_min_chars:
            return None

        candidates = list(recent)[-self.near_duplicate_history :]
        for previous in reversed(candidates):
            age = current.timestamp - previous.timestamp
            if (
                age < 0
                or age > self.near_duplicate_window_seconds
                or not previous.normalized
            ):
                continue
            shorter_length = min(len(current.normalized), len(previous.normalized))
            if shorter_length < self.near_duplicate_min_chars:
                continue

            common_prefix = _common_prefix_length(
                current.normalized,
                previous.normalized,
            )
            suffix_length = (
                max(len(current.normalized), len(previous.normalized)) - common_prefix
            )
            ratio = common_prefix / shorter_length
            if (
                common_prefix >= self.near_duplicate_min_chars
                and ratio >= self.near_duplicate_ratio
                and suffix_length <= self.near_duplicate_max_suffix_chars
                and _looks_like_random_tail(
                    current.normalized[common_prefix:],
                    previous.normalized[common_prefix:],
                )
            ):
                return previous
        return None

    def _find_repeated_segment(self, normalized: str) -> tuple[int, int, float] | None:
        if len(normalized) < self.repeat_min_chars:
            return None

        # Ignore separator-only obfuscation for this rule, while keeping the
        # original normalized form for exact/near duplicate comparisons.
        scan_text = "".join(normalized.split())
        if len(scan_text) > self.repeat_scan_chars:
            # Keep the scan bounded while covering the beginning, middle, and
            # end. A repeated block in the middle must not be able to evade
            # detection merely by padding the message at both ends.
            last_start = len(scan_text) - self.repeat_scan_chars
            starts = {0, last_start // 2, last_start}
            segments = tuple(
                scan_text[start : start + self.repeat_scan_chars]
                for start in sorted(starts)
            )
        else:
            segments = (scan_text,)

        for scan_text in segments:
            result = self._find_repeated_segment_in_text(scan_text)
            if result is not None:
                return result
        return None

    def _find_repeated_segment_in_text(
        self,
        scan_text: str,
    ) -> tuple[int, int, float] | None:
        length = len(scan_text)
        if length < self.repeat_min_chars:
            return None
        max_unit = min(
            self.repeat_max_unit_chars, length // self.repeat_min_repetitions
        )

        for unit_length in range(1, max_unit + 1):
            max_start = length - unit_length * self.repeat_min_repetitions
            for start in range(max_start + 1):
                first_end = start + unit_length
                second_end = first_end + unit_length
                if scan_text[start:first_end] != scan_text[first_end:second_end]:
                    continue

                unit = scan_text[start:first_end]
                count = 2
                while (
                    start + (count + 1) * unit_length <= length
                    and scan_text[
                        start + count * unit_length : start + (count + 1) * unit_length
                    ]
                    == unit
                ):
                    count += 1

                repeated_chars = count * unit_length
                coverage = repeated_chars / length
                if (
                    count >= self.repeat_min_repetitions
                    and repeated_chars >= self.repeat_min_chars
                    and coverage >= self.repeat_coverage
                ):
                    return unit_length, count, coverage
        return None

    def _prune_state(self, state: _SenderState, now: float) -> None:
        while state.recent and now - state.recent[0].timestamp > self.cache_ttl_seconds:
            state.recent.popleft()
        while (
            state.long_message_times
            and now - state.long_message_times[0] > self.long_message_window_seconds
        ):
            state.long_message_times.popleft()

    def _trim_senders(self) -> None:
        if len(self._states) <= self.max_senders:
            return
        excess = len(self._states) - self.max_senders
        for sender_key, _ in sorted(
            self._states.items(),
            key=lambda item: item[1].last_seen,
        )[:excess]:
            self._states.pop(sender_key, None)


def _common_prefix_length(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _looks_like_random_tail(*tails: str) -> bool:
    """Recognize append-only noise without treating normal prose as noise.

    Short ASCII suffixes are inherently ambiguous. They are accepted only when
    the compared messages have different suffixes, which is the characteristic
    pattern of a sender adding a fresh token to every copy. Unicode suffixes
    need more varied characters so ordinary Chinese addenda such as ``谢谢``
    remain allowed.
    """

    non_empty = [tail for tail in tails if tail]
    if not non_empty:
        return False
    different_suffixes = len(set(non_empty)) > 1

    for tail in non_empty:
        if len(tail) > 64 or any(character.isspace() for character in tail):
            return False
        ascii_characters = [character for character in tail if ord(character) < 128]
        ascii_ratio = len(ascii_characters) / len(tail)
        if ascii_ratio >= 0.75:
            if len(tail) <= 3:
                # A one-to-three character ASCII token is accepted only when
                # it changes between messages; otherwise it may be a normal
                # suffix such as a version label.
                if different_suffixes and all(
                    character.isalnum() or character in "-_=+~.!@#$%^&*?/\\|"
                    for character in tail
                ):
                    continue
                return False
            if any(character.isdigit() for character in tail):
                continue
            if any(character in "-_~|=+*/\\" for character in tail):
                continue
            if any(character.isupper() for character in tail) and any(
                character.islower() for character in tail
            ):
                continue
            # An unseparated ASCII suffix is a common random-token bypass.
            if (
                different_suffixes
                and len(tail) >= 4
                and len(set(tail)) / len(tail) >= 0.5
            ):
                continue
            return False

        # Unicode random tails need to be varied. Four or more differing CJK
        # characters are enough when both messages carry different suffixes;
        # a single message needs a longer token to avoid false positives.
        if (
            len(tail) >= (4 if different_suffixes else 6)
            and len(set(tail)) / len(tail) >= 0.8
        ):
            continue
        return False
    return True


def _ids(*snapshots: MessageSnapshot | None) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            snapshot.message_id
            for snapshot in snapshots
            if snapshot is not None and snapshot.message_id
        )
    )

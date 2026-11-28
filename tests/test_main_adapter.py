import asyncio
import importlib
import sys
import types
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch


def _install_astrbot_stubs():
    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    event_module = types.ModuleType("astrbot.api.event")
    star_module = types.ModuleType("astrbot.api.star")

    class Config(dict):
        pass

    class Logger:
        def __getattr__(self, _name):
            return lambda *_args, **_kwargs: None

    class EventMessageType:
        ALL = "all"

    class Filter:
        @staticmethod
        def event_message_type(_message_type, **kwargs):
            def decorate(function):
                function.handler_priority = kwargs.get("priority", 0)
                return function
            return decorate

        @staticmethod
        def regex(_pattern, **kwargs):
            def decorate(function):
                function.handler_priority = kwargs.get("priority", 0)
                return function
            return decorate

    Filter.EventMessageType = EventMessageType

    class AstrMessageEvent:
        pass

    class Star:
        def __init__(self, _context):
            pass

    def register(*_args, **_kwargs):
        return lambda cls: cls

    api.AstrBotConfig = Config
    api.logger = Logger()
    event_module.AstrMessageEvent = AstrMessageEvent
    event_module.filter = Filter
    star_module.Context = object
    star_module.Star = Star
    star_module.register = register

    sys.modules.update(
        {
            "astrbot": astrbot,
            "astrbot.api": api,
            "astrbot.api.event": event_module,
            "astrbot.api.star": star_module,
        }
    )


class FakeBotApi:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    async def call_action(self, action, **kwargs):
        self.calls.append((action, kwargs))
        if self.fail:
            raise RuntimeError("recall failed")
        return {"ok": True}


class FakeGuildMessage:
    def __init__(self, channel_id="channel-1"):
        self.channel_id = channel_id


class FakeGroupMessage:
    def __init__(self, group_openid="raw-group", message_id="current-message"):
        self.group_openid = group_openid
        self.id = message_id


class FakeRoute:
    def __init__(self, method, path, **parameters):
        self.method = method
        self.path = path
        self.parameters = parameters


class FakeOfficialRecallApi:
    def __init__(self, error=None, result=None):
        self.calls = []
        self.error = error
        self.result = result

    async def recall_message(self, channel_id, message_id, hidetip=False):
        self.calls.append((channel_id, message_id, hidetip))
        if self.error:
            raise self.error
        return self.result


class FakeComponent:
    type = "image"

    def __init__(self, file_id):
        self.file_id = file_id

    def toDict(self):
        return {"type": "image", "data": {"file": self.file_id}}


class FakePlainComponent:
    type = "Plain"

    def __init__(self, text):
        self.text = text

    def toDict(self):
        return {"type": "Plain", "text": self.text}


class FakeEvent:
    def __init__(self, text, message_id, platform="aiocqhttp", components=None):
        self.message_str = text
        self.message_obj = {"message_id": message_id}
        self.bot = types.SimpleNamespace(api=FakeBotApi())
        self.platform = platform
        self.components = components or []
        self.stopped = False

    def get_platform_id(self):
        return self.platform

    def get_platform_name(self):
        return self.platform

    def get_session_id(self):
        return "group-1"

    def get_sender_id(self):
        return "user-1"

    def get_sender_name(self):
        return "tester"

    def get_messages(self):
        return self.components

    def stop_event(self):
        self.stopped = True


class AnonymousEvent(FakeEvent):
    def get_sender_id(self):
        return ""

    def get_sender_name(self):
        return ""


class QQOfficialEvent(FakeEvent):
    """Match the normalized object/Plain/At shape of both official adapters."""

    def __init__(self, text, message_id, platform="qq_official", *, plain_only=False):
        super().__init__(text, message_id, platform=platform)
        self.message_obj = types.SimpleNamespace(
            message_id=message_id,
            sender=types.SimpleNamespace(user_id="member-openid", nickname="tester"),
            group_id="group-openid",
        )
        self.components = [types.SimpleNamespace(toDict=lambda: {"type": "At", "qq": "qq_official"}), FakePlainComponent(text)]
        if plain_only:
            self.message_str = ""
        # AstrBot initializes this to False; True prohibits the default LLM.
        self.call_llm = False
        self.notices = []
        self.notice_failure = False

    def get_platform_id(self):
        return "official-instance-1"

    def get_message_str(self):
        return self.message_str

    def should_call_llm(self, enabled):
        self.call_llm = enabled

    def plain_result(self, text):
        return text

    async def send(self, result):
        if self.notice_failure:
            raise RuntimeError("QQ rate limit")
        self.notices.append(result)


class MainAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_astrbot_stubs()
        sys.modules.pop("main", None)
        cls.main = importlib.import_module("main")

    def make_plugin(self, **config):
        return self.main.KeywordMatcherPlugin(object(), config)

    def make_guild_event(self, platform="qq_official", **api_options):
        event = QQOfficialEvent("fuck", "current-message", platform)
        event.message_obj.raw_message = FakeGuildMessage()
        event.bot.api = FakeOfficialRecallApi(**api_options)
        return event

    def run_official_handler(self, plugin, event):
        sdk = types.ModuleType("botpy.message")
        sdk.Message = FakeGuildMessage
        sdk.GroupMessage = FakeGroupMessage
        http = types.ModuleType("botpy.http")
        http.Route = FakeRoute
        with patch.dict(sys.modules, {"botpy.message": sdk, "botpy.http": http}):
            asyncio.run(plugin.on_all_message(event))

    def make_group_event(self, platform="qq_official", result=None, error=None):
        event = QQOfficialEvent("fuck", "current-message", platform)
        event.message_obj.raw_message = FakeGroupMessage()
        calls = []

        async def request(route):
            calls.append(route)
            if error:
                raise error
            return result

        event.bot.api = types.SimpleNamespace(_http=types.SimpleNamespace(request=request))
        return event, calls

    def test_official_group_recalls_raw_incoming_id_with_authenticated_client(self):
        for platform in ("qq_official", "qq_official_webhook"):
            for result in (None, "", {}, {"code": 0}):
                with self.subTest(platform=platform, result=result):
                    plugin = self.make_plugin()
                    event, calls = self.make_group_event(platform, result=result)
                    self.run_official_handler(plugin, event)
                    self.assertTrue(event.stopped)
                    self.assertTrue(event.call_llm)
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0].method, "DELETE")
                    self.assertEqual(calls[0].path, "/v2/groups/{group_openid}/messages/{message_id}")
                    self.assertEqual(calls[0].parameters, {"group_openid": "raw-group", "message_id": "current-message"})
                    self.assertEqual(len(plugin._recalled_message_ids), 1)
                    self.run_official_handler(plugin, event)
                    self.assertEqual(len(calls), 1)

    def test_official_group_error_codes_preserve_blocking_and_success_cache(self):
        for code in (40061001, 40062003, 40064004, 40064009, 40064010, 50065001):
            for as_exception in (False, True):
                with self.subTest(code=code, as_exception=as_exception):
                    plugin = self.make_plugin()
                    options = {"error": RuntimeError(f'{{"code":{code}}}')} if as_exception else {"result": {"code": code}}
                    event, calls = self.make_group_event(**options)
                    self.run_official_handler(plugin, event)
                    self.assertTrue(event.stopped)
                    self.assertTrue(event.call_llm)
                    self.assertEqual(len(plugin._recalled_message_ids), int(code in (40064009, 40064010)))
                    self.assertEqual(len(calls), 1)

    def test_official_group_invalid_raw_identifiers_do_not_send_delete(self):
        for raw in (FakeGroupMessage(""), FakeGroupMessage(message_id=""), FakeGroupMessage(message_id="other-message")):
            with self.subTest(raw=raw):
                plugin = self.make_plugin()
                event, calls = self.make_group_event()
                event.message_obj.raw_message = raw
                self.run_official_handler(plugin, event)
                self.assertTrue(event.stopped)
                self.assertEqual(calls, [])

    def test_official_group_recall_disabled_missing_client_and_timeout(self):
        for mode in ("disabled", "missing-client", "timeout"):
            with self.subTest(mode=mode):
                plugin = self.make_plugin(enable_recall=mode != "disabled")
                event, calls = self.make_group_event(error=asyncio.TimeoutError() if mode == "timeout" else None)
                if mode == "missing-client":
                    event.bot.api = object()
                self.run_official_handler(plugin, event)
                self.assertTrue(event.stopped)
                self.assertTrue(event.call_llm)
                self.assertEqual(len(calls), int(mode == "timeout"))
                self.assertEqual(plugin._recalled_message_ids, {})

    def test_official_guild_recalls_current_message_for_both_transports(self):
        for platform in ("qq_official", "qq_official_webhook"):
            with self.subTest(platform=platform):
                plugin = self.make_plugin()
                event = self.make_guild_event(platform)
                self.run_official_handler(plugin, event)
                self.assertTrue(event.stopped)
                self.assertTrue(event.call_llm)
                self.assertEqual(event.bot.api.calls, [("channel-1", "current-message", False)])
                self.assertEqual(len(plugin._recalled_message_ids), 1)
                self.run_official_handler(plugin, event)
                self.assertEqual(len(event.bot.api.calls), 1)

    def test_official_guild_recall_failure_preserves_blocking_and_allows_retry(self):
        for options in ({"error": PermissionError("403 forbidden")}, {"result": {"code": 304023}}, {"result": False}):
            with self.subTest(options=options):
                plugin = self.make_plugin()
                event = self.make_guild_event(**options)
                self.run_official_handler(plugin, event)
                self.assertTrue(event.stopped)
                self.assertTrue(event.call_llm)
                self.assertEqual(plugin._recalled_message_ids, {})
                event.bot.api.error = None
                event.bot.api.result = None
                self.run_official_handler(plugin, event)
                self.assertEqual(len(event.bot.api.calls), 2)
                self.assertEqual(len(plugin._recalled_message_ids), 1)

    def test_official_incoming_group_and_private_messages_never_use_guild_api(self):
        for source in (
            types.SimpleNamespace(group_openid="group-1"),
            types.SimpleNamespace(author=types.SimpleNamespace(user_openid="user-1")),
            types.SimpleNamespace(guild_id="dm-guild", channel_id="dm-channel"),
            {"channel_id": "unverified-channel"},
        ):
            with self.subTest(source=source):
                plugin = self.make_plugin()
                event = self.make_guild_event()
                event.message_obj.raw_message = source
                self.run_official_handler(plugin, event)
                self.assertTrue(event.stopped)
                self.assertEqual(event.bot.api.calls, [])
                self.assertEqual(plugin._recalled_message_ids, {})

    def test_official_guild_recall_can_be_disabled(self):
        plugin = self.make_plugin(enable_recall=False)
        event = self.make_guild_event()
        self.run_official_handler(plugin, event)
        self.assertTrue(event.stopped)
        self.assertEqual(event.bot.api.calls, [])

    def test_official_guild_missing_api_or_channel_preserves_blocking(self):
        for missing in ("api", "channel"):
            with self.subTest(missing=missing):
                plugin = self.make_plugin()
                event = self.make_guild_event()
                if missing == "api":
                    event.bot.api = object()
                else:
                    event.message_obj.raw_message.channel_id = ""
                self.run_official_handler(plugin, event)
                self.assertTrue(event.stopped)
                self.assertEqual(plugin._recalled_message_ids, {})

    def test_official_guild_recall_timeout_preserves_blocking(self):
        plugin = self.make_plugin()
        event = self.make_guild_event()

        async def timeout(awaitable, timeout):
            self.assertEqual(timeout, 5)
            awaitable.close()
            raise asyncio.TimeoutError

        with patch.object(self.main.asyncio, "wait_for", timeout):
            self.run_official_handler(plugin, event)
        self.assertTrue(event.stopped)
        self.assertTrue(event.call_llm)
        self.assertEqual(plugin._recalled_message_ids, {})

    def test_official_adapters_block_keywords_without_onebot_api(self):
        for platform in ("qq_official", "qq_official_webhook"):
            for plain_only in (False, True):
                with self.subTest(platform=platform, plain_only=plain_only):
                    plugin = self.make_plugin()
                    event = QQOfficialEvent("傻逼", "official-id", platform, plain_only=plain_only)
                    asyncio.run(plugin.on_all_message(event))
                    self.assertTrue(event.stopped)
                    self.assertTrue(event.call_llm)
                    self.assertEqual(event.bot.api.calls, [])
                    self.assertEqual(event.notices, [])

    def test_official_at_component_does_not_hide_blank_message(self):
        plugin = self.make_plugin()
        event = QQOfficialEvent("   ", "blank-id")
        asyncio.run(plugin.on_all_message(event))
        self.assertTrue(event.stopped)
        self.assertTrue(event.call_llm)

    def test_official_block_notice_and_send_failure_preserve_blocking(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                plugin = self.make_plugin(enable_block_notice=True)
                event = QQOfficialEvent("傻逼", "notice-id")
                event.notice_failure = fail
                asyncio.run(plugin.on_all_message(event))
                self.assertTrue(event.stopped)
                self.assertTrue(event.call_llm)
                self.assertEqual(len(event.notices), 0 if fail else 1)
                if not fail:
                    self.assertIn("违禁词", event.notices[0])

    def test_official_normal_message_is_not_stopped(self):
        plugin = self.make_plugin(enable_block_notice=True)
        event = QQOfficialEvent("你好，今天的天气怎么样", "normal-id")
        asyncio.run(plugin.on_all_message(event))
        self.assertFalse(event.stopped)
        self.assertFalse(event.call_llm)
        self.assertEqual(event.notices, [])

    def test_original_plain_keeps_trailing_blank_lines_after_waking_trim(self):
        plugin = self.make_plugin()
        event = QQOfficialEvent("正文" + "\n" * 8, "trailing-lines")
        event.message_str = event.message_str.strip()
        asyncio.run(plugin.on_all_message(event))
        self.assertTrue(event.stopped)

    def test_original_text_is_checked_after_wake_prefix_is_removed(self):
        plugin = self.make_plugin()
        event = QQOfficialEvent("fuck 你好", "wake-prefix")
        event.message_str = "你好"
        asyncio.run(plugin.on_all_message(event))
        self.assertTrue(event.stopped)

    def test_message_object_preserves_raw_text_without_plain_components(self):
        plugin = self.make_plugin()
        event = QQOfficialEvent("正文", "raw-object")
        event.components = []
        event.message_obj.message_str = "正文" + "\n" * 8
        asyncio.run(plugin.on_all_message(event))
        self.assertTrue(event.stopped)

    def test_empty_configured_word_list_does_not_apply_hidden_defaults(self):
        plugin = self.make_plugin(banned_words=[], initialize_banned_words=False)
        event = QQOfficialEvent("fuck", "empty-list")
        asyncio.run(plugin.on_all_message(event))
        self.assertFalse(event.stopped)

    def test_disabling_pinyin_keeps_literal_keyword_detection(self):
        plugin = self.make_plugin(enable_pinyin_keywords=False)
        event = QQOfficialEvent("fuck", "literal-only")
        asyncio.run(plugin.on_all_message(event))
        self.assertTrue(event.stopped)

    def test_detection_precedes_default_priority_reply_handler(self):
        plugin = self.make_plugin()
        event = QQOfficialEvent("傻逼", "priority-id")
        replied = []

        async def reply_handler(event):
            replied.append("reply")

        async def dispatch():
            handlers = [(plugin.on_all_message.handler_priority, plugin.on_all_message), (0, reply_handler)]
            for _, handler in sorted(handlers, key=lambda item: -item[0]):
                await handler(event)
                if event.stopped:
                    break

        asyncio.run(dispatch())
        self.assertTrue(event.stopped)
        self.assertEqual(replied, [])

    def test_keyword_migration_persists_defaults_and_legacy_custom_words(self):
        from spam_detector import DEFAULT_BANNED_WORDS

        class SavedConfig(dict):
            def save_config(self):
                config_path.write_text(json.dumps(self, ensure_ascii=False), encoding="utf-8")

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            for old_value in ([], "", "自定义词\n另一个词", ["自定义词"]):
                with self.subTest(old_value=old_value):
                    config = SavedConfig(banned_words=old_value)
                    self.main.KeywordMatcherPlugin(object(), config)
                    saved = json.loads(config_path.read_text(encoding="utf-8"))
                    self.assertFalse(saved["initialize_banned_words"])
                    self.assertTrue(set(DEFAULT_BANNED_WORDS).issubset(saved["banned_words"]))
                    if old_value:
                        self.assertIn("自定义词", saved["banned_words"])
                    config["banned_words"] = []
                    config.save_config()
                    self.main.KeywordMatcherPlugin(object(), config)
                    self.assertEqual(config["banned_words"], [])
                    self.assertEqual(json.loads(config_path.read_text(encoding="utf-8"))["banned_words"], [])
                    config["initialize_banned_words"] = True
                    self.main.KeywordMatcherPlugin(object(), config)
                    self.assertEqual(config["banned_words"], list(DEFAULT_BANNED_WORDS))

    def test_keyword_migration_preserves_explicit_custom_words(self):
        plugin = self.make_plugin(banned_words=["外挂", "自定义词"])
        self.assertIn("外挂", plugin.config["banned_words"])
        self.assertIn("自定义词", plugin.config["banned_words"])

    def test_keyword_migration_replaces_exact_former_defaults(self):
        from spam_detector import DEFAULT_BANNED_WORDS

        old_words = [*DEFAULT_BANNED_WORDS, "垃圾", "滚蛋", "外挂", "作弊"]
        plugin = self.make_plugin(banned_words=old_words)
        self.assertEqual(plugin.config["banned_words"], list(DEFAULT_BANNED_WORDS))

    def test_keyword_migration_save_failure_preserves_original_config(self):
        class FailingConfig(dict):
            def save_config(self):
                raise OSError("read-only config")

        config = FailingConfig(banned_words="自定义词", enable_recall=False)
        original = dict(config)
        with self.assertRaises(OSError):
            self.main.KeywordMatcherPlugin(object(), config)
        self.assertEqual(config, original)

    def test_legacy_text_converts_without_restoring_deleted_defaults(self):
        plugin = self.make_plugin(banned_words="自定义词\n另一个词", initialize_banned_words=False)
        self.assertEqual(plugin.config["banned_words"], ["自定义词", "另一个词"])

    def test_exact_duplicate_stops_and_recalls_only_new_message(self):
        plugin = self.make_plugin()
        first = FakeEvent("重复消息内容", "m1")
        second = FakeEvent("重复消息内容", "m2")

        asyncio.run(plugin.on_all_message(first))
        asyncio.run(plugin.on_all_message(second))

        self.assertFalse(first.stopped)
        self.assertTrue(second.stopped)
        self.assertEqual(
            second.bot.api.calls,
            [("delete_msg", {"message_id": "m2"})],
        )

    def test_near_duplicate_recalls_only_new_message(self):
        plugin = self.make_plugin()
        base = "服务器什么时候重启以及维护安排请看公告请所有成员提前做好准备"
        first = FakeEvent(base + "-a81k2", "m1")
        second = FakeEvent(base + "-zP93x", "m2")

        asyncio.run(plugin.on_all_message(first))
        asyncio.run(plugin.on_all_message(second))

        self.assertFalse(first.stopped)
        self.assertTrue(second.stopped)
        self.assertEqual(
            second.bot.api.calls,
            [("delete_msg", {"message_id": "m2"})],
        )

    def test_repeated_content_recalls_only_current_message(self):
        plugin = self.make_plugin()
        event = FakeEvent("神经病" * 12, "m1")

        asyncio.run(plugin.on_all_message(event))

        self.assertTrue(event.stopped)
        self.assertEqual(
            event.bot.api.calls,
            [("delete_msg", {"message_id": "m1"})],
        )

    def test_configured_pinyin_keyword_stops_message(self):
        plugin = self.make_plugin(banned_words="傻逼")
        event = FakeEvent("shabi", "m1")

        asyncio.run(plugin.on_all_message(event))

        self.assertTrue(event.stopped)
        self.assertEqual(
            event.bot.api.calls,
            [("delete_msg", {"message_id": "m1"})],
        )

    def test_blank_message_stops_and_recalls_current_message(self):
        plugin = self.make_plugin()
        event = FakeEvent(" \n\t", "m1")

        asyncio.run(plugin.on_all_message(event))

        self.assertTrue(event.stopped)
        self.assertEqual(
            event.bot.api.calls,
            [("delete_msg", {"message_id": "m1"})],
        )

    def test_non_onebot_stops_without_recall_error(self):
        plugin = self.make_plugin()
        first = FakeEvent("重复消息内容", "m1", platform="telegram")
        second = FakeEvent("重复消息内容", "m2", platform="telegram")

        asyncio.run(plugin.on_all_message(first))
        asyncio.run(plugin.on_all_message(second))

        self.assertTrue(second.stopped)
        self.assertEqual(second.bot.api.calls, [])

    def test_recall_failure_does_not_make_handler_fail(self):
        plugin = self.make_plugin()
        first = FakeEvent("重复消息内容", "m1")
        second = FakeEvent("重复消息内容", "m2")
        second.bot.api.fail = True

        asyncio.run(plugin.on_all_message(first))
        asyncio.run(plugin.on_all_message(second))
        self.assertTrue(second.stopped)

    def test_identical_media_is_compared_by_stable_component_key(self):
        plugin = self.make_plugin()
        first = FakeEvent("", "m1", components=[FakeComponent("image-1")])
        second = FakeEvent("", "m2", components=[FakeComponent("image-1")])
        different = FakeEvent("", "m3", components=[FakeComponent("image-2")])

        asyncio.run(plugin.on_all_message(first))
        asyncio.run(plugin.on_all_message(second))
        asyncio.run(plugin.on_all_message(different))

        self.assertTrue(second.stopped)
        self.assertFalse(different.stopped)

    def test_plain_component_uses_normalized_text_not_raw_component(self):
        plugin = self.make_plugin()
        first = FakeEvent(
            "  重复消息\u200b ",
            "m1",
            components=[FakePlainComponent("  重复消息\u200b ")],
        )
        second = FakeEvent(
            "重复消息",
            "m2",
            components=[FakePlainComponent("重复消息")],
        )

        asyncio.run(plugin.on_all_message(first))
        asyncio.run(plugin.on_all_message(second))
        self.assertTrue(second.stopped)

    def test_anonymous_events_do_not_share_sender_state(self):
        plugin = self.make_plugin()
        first = AnonymousEvent("重复消息内容", "m1")
        second = AnonymousEvent("重复消息内容", "m2")

        asyncio.run(plugin.on_all_message(first))
        asyncio.run(plugin.on_all_message(second))

        self.assertFalse(second.stopped)


if __name__ == "__main__":
    unittest.main()

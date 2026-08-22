import asyncio
import importlib
import sys
import types
import unittest


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
        def event_message_type(_message_type):
            return lambda function: function

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


class MainAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_astrbot_stubs()
        sys.modules.pop("main", None)
        cls.main = importlib.import_module("main")

    def make_plugin(self, **config):
        return self.main.KeywordMatcherPlugin(object(), config)

    def test_exact_duplicate_stops_and_recalls_both_messages(self):
        plugin = self.make_plugin()
        first = FakeEvent("重复消息内容", "m1")
        second = FakeEvent("重复消息内容", "m2")

        asyncio.run(plugin.on_all_message(first))
        asyncio.run(plugin.on_all_message(second))

        self.assertFalse(first.stopped)
        self.assertTrue(second.stopped)
        self.assertEqual(
            second.bot.api.calls,
            [
                ("delete_msg", {"message_id": "m2"}),
                ("delete_msg", {"message_id": "m1"}),
            ],
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


if __name__ == "__main__":
    unittest.main()

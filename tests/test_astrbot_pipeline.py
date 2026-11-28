"""Run upstream pipeline code without installing AstrBot's external services.

Set ASTRBOT_SOURCE to a checkout (validated against tag v4.28.2). Imports are
replaced with local dependencies; method bodies and decorators run unchanged.
This checks pipeline semantics, not live QQ delivery or platform API access.
"""

from __future__ import annotations

import ast
import asyncio
import enum
import functools
import inspect
import os
import re
import traceback
import types
import typing
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

from tests.test_main_adapter import QQOfficialEvent, _install_astrbot_stubs


class RemoveImports(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None


def load_source(root, relative_path, namespace):
    path = root / relative_path
    tree = RemoveImports().visit(ast.parse(path.read_text(encoding="utf-8-sig")))
    tree.body.insert(
        0,
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)


@unittest.skipUnless(
    os.environ.get("ASTRBOT_SOURCE"), "Set ASTRBOT_SOURCE for upstream pipeline tests"
)
class AstrBotPipelineTests(unittest.TestCase):
    def setUp(self):
        import importlib
        import sys

        _install_astrbot_stubs()
        sys.modules.pop("main", None)
        self.main = importlib.import_module("main")
        self.plugin = self.main.KeywordMatcherPlugin(object(), {})
        self.ns = dict(vars(typing))
        self.ns.update(
            enum=enum,
            dataclass=dataclass,
            field=field,
            re=re,
            inspect=inspect,
            traceback=traceback,
            logger=types.SimpleNamespace(
                debug=lambda *a, **k: None,
                info=lambda *a, **k: None,
                warning=lambda *a, **k: None,
                error=lambda *a, **k: None,
            ),
            star_map={},
            HandlerFilter=object,
            Stage=object,
            register_stage=lambda cls: cls,
            MessageEventResult=type("MessageEventResult", (), {}),
            CommandResult=type("CommandResult", (), {}),
        )
        root = Path(os.environ["ASTRBOT_SOURCE"])
        for file in ("star/star_handler.py", "star/filter/regex.py"):
            load_source(root, "astrbot/core/" + file, self.ns)

        # Load only registration functions needed by @filter.regex, avoiding
        # unrelated agent/tool initialization while retaining actual bodies.
        path = root / "astrbot/core/star/register/star_handler.py"
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        tree.body = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name
            in {
                "get_handler_full_name",
                "get_handler_or_create",
                "register_regex",
            }
        ]
        tree.body.insert(
            0,
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
        )
        exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), self.ns)
        handler = type(self.plugin).on_all_message
        plugin_tree = ast.parse(
            Path(self.main.__file__).read_text(encoding="utf-8-sig")
        )
        plugin_class = next(
            n
            for n in plugin_tree.body
            if isinstance(n, ast.ClassDef) and n.name == "KeywordMatcherPlugin"
        )
        handler_node = next(
            n
            for n in plugin_class.body
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "on_all_message"
        )
        self.ns["filter"] = types.SimpleNamespace(regex=self.ns["register_regex"])
        decorator = eval(
            compile(
                ast.Expression(handler_node.decorator_list[0]),
                self.main.__file__,
                "eval",
            ),
            self.ns,
        )
        decorator(handler)
        md = self.ns["star_handlers_registry"]._handlers[0]
        self.ns["star_map"][md.handler_module_path] = types.SimpleNamespace(
            name=self.main.PLUGIN_NAME,
            activated=True,
            reserved=False,
        )
        # Register the unbound method as import-time decorators do, then run
        # the loader's real binding loop rather than supplying a bound method.
        manager_path = root / "astrbot/core/star/star_manager.py"
        manager_tree = ast.parse(manager_path.read_text(encoding="utf-8-sig"))
        binding_loop = next(
            node for node in ast.walk(manager_tree)
            if isinstance(node, ast.For)
            and isinstance(node.iter, ast.Name)
            and node.iter.id == "related_handlers"
        )
        self.bind_code = compile(
            ast.fix_missing_locations(ast.Module(body=[binding_loop], type_ignores=[])),
            str(manager_path), "exec",
        )
        self.ns.update(
            functools=functools, related_handlers=[md], plugin_disabled=False,
            metadata=types.SimpleNamespace(star_cls=self.plugin),
        )
        exec(self.bind_code, self.ns)

        self.session_plugin_config = {}

        async def get_async(**kwargs):
            return self.session_plugin_config

        self.ns["sp"] = types.SimpleNamespace(get_async=get_async)
        load_source(root, "astrbot/core/star/session_plugin_manager.py", self.ns)
        for name in (
            "At",
            "AtAll",
            "Reply",
            "PermissionTypeFilter",
            "CommandGroupFilter",
        ):
            self.ns[name] = type(name, (), {})
        self.ns["MessageType"] = types.SimpleNamespace(GROUP_MESSAGE="group")
        load_source(root, "astrbot/core/pipeline/waking_check/stage.py", self.ns)
        load_source(root, "astrbot/core/pipeline/context_utils.py", self.ns)
        load_source(
            root, "astrbot/core/pipeline/process_stage/method/star_request.py", self.ns
        )
        load_source(root, "astrbot/core/pipeline/process_stage/stage.py", self.ns)

        # Execute upstream should_call_llm, rather than reinterpreting its name.
        path = root / "astrbot/core/platform/astr_message_event.py"
        event_class = next(
            n
            for n in ast.parse(path.read_text(encoding="utf-8-sig")).body
            if isinstance(n, ast.ClassDef) and n.name == "AstrMessageEvent"
        )
        method = next(
            n
            for n in event_class.body
            if isinstance(n, ast.FunctionDef) and n.name == "should_call_llm"
        )
        tree = ast.Module(body=[method], type_ignores=[])
        exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), self.ns)

    def event(self, text, platform="qq_official"):
        event = QQOfficialEvent(text, "upstream-id", platform)
        event.unified_msg_origin = platform + ":GroupMessage:group-openid"
        event.should_call_llm = types.MethodType(self.ns["should_call_llm"], event)
        event.message_obj.type = "group"
        event._extras = {}
        event.get_extra = lambda key=None, default=None: (
            event._extras.get(key, default) if key else event._extras
        )
        event.set_extra = event._extras.__setitem__
        event.get_self_id = lambda: "bot"
        event.is_private_chat = lambda: False
        event.is_stopped = lambda: event.stopped
        event.clear_result = lambda: None
        event.get_result = lambda: None
        event._has_send_oper = False
        event.is_at_or_wake_command = False
        event.is_wake = False
        return event

    async def dispatch(self, event, plugin_set=None):
        cfg = {
            "platform_settings": {},
            "admins_id": [],
            "wake_prefix": [],
            "provider_settings": {"enable": True},
        }
        if plugin_set is not None:
            cfg["plugin_set"] = plugin_set
        ctx = types.SimpleNamespace(astrbot_config=cfg)
        waking = self.ns["WakingCheckStage"]()
        waking.ctx = ctx
        waking.unique_session = False
        waking.ignore_bot_self_message = False
        waking.friend_message_needs_wake_prefix = True
        waking.ignore_at_all = False
        waking.disable_builtin_commands = False
        waking._umo_auto_name_recorder = types.SimpleNamespace(schedule=lambda e: None)
        await waking.process(event)
        if event.stopped:
            return
        star = self.ns["StarRequestSubStage"]()
        async for _ in star.process(event):
            pass

    def test_official_events_activate_and_block_through_upstream_pipeline(self):
        for platform in ("qq_official", "qq_official_webhook"):
            for text in ("fuck", "f\nu\nc\nk", " \n\t", "正文" + "\n" * 8):
                with self.subTest(platform=platform, text=text):
                    event = self.event(text, platform)
                    asyncio.run(self.dispatch(event))
                    self.assertTrue(event.stopped)
                    self.assertTrue(event.call_llm)
                    self.assertEqual(len(event.get_extra("activated_handlers")), 1)

    def test_normal_group_message_does_not_force_default_llm(self):
        event = self.event("你好世界")
        asyncio.run(self.dispatch(event))
        self.assertFalse(event.stopped)
        self.assertFalse(event.is_at_or_wake_command)
        self.assertFalse(event.call_llm)

    def test_plugin_whitelist_exclusion_explains_missing_inspect(self):
        event = self.event("fuck")
        asyncio.run(self.dispatch(event, plugin_set=[]))
        self.assertTrue(event.stopped)  # Framework rejects the unwoken event.
        self.assertEqual(event.get_extra("activated_handlers"), [])
        self.assertFalse(event.call_llm)  # Plugin was never executed.

    def test_session_disabled_plugin_is_filtered_before_execution(self):
        event = self.event("fuck")
        self.session_plugin_config[event.unified_msg_origin] = {
            "disabled_plugins": [self.main.PLUGIN_NAME],
        }
        asyncio.run(self.dispatch(event))
        self.assertEqual(event.get_extra("activated_handlers"), [])
        self.assertFalse(event.call_llm)

    def test_actual_registry_priority_blocks_later_reply_handler(self):
        replied = []

        async def later_handler(event):
            replied.append(event)

        self.ns["register_regex"](r".*", priority=0)(later_handler)
        md = self.ns["star_handlers_registry"]._handlers[-1]
        self.ns["star_map"][md.handler_module_path] = types.SimpleNamespace(
            name="other-plugin",
            activated=True,
            reserved=False,
        )
        event = self.event("fuck")
        asyncio.run(self.dispatch(event))
        self.assertTrue(event.stopped)
        self.assertTrue(event.call_llm)
        self.assertEqual(replied, [])

    def test_repeated_loader_binding_uses_current_instance_once(self):
        replacement = self.main.KeywordMatcherPlugin(object(), {})
        self.ns["metadata"].star_cls = replacement
        exec(self.bind_code, self.ns)
        exec(self.bind_code, self.ns)
        callback = self.ns["related_handlers"][0].handler
        self.assertEqual(callback.args, (replacement,))
        event = self.event("fuck")
        asyncio.run(self.dispatch(event))
        self.assertTrue(event.call_llm)

    def test_startup_reports_excluded_profile_without_changing_selection(self):
        self.check_startup_report([], "selected=False")

    def test_startup_reports_selected_profile_and_bound_handler(self):
        self.check_startup_report(["*"], "selected=True")

    def check_startup_report(self, selection, expected):
        import sys

        messages = []
        config = {"plugin_set": selection}
        self.plugin.context = types.SimpleNamespace(get_config=lambda: config)
        star_module = types.ModuleType("astrbot.core.star.star")
        star_module.star_map = self.ns["star_map"]
        handler_module = types.ModuleType("astrbot.core.star.star_handler")
        handler_module.EventType = self.ns["EventType"]
        handler_module.star_handlers_registry = self.ns["star_handlers_registry"]
        with patch.dict(sys.modules, {
            "astrbot.core.star.star": star_module,
            "astrbot.core.star.star_handler": handler_module,
        }), patch.object(self.main, "logger", types.SimpleNamespace(
            info=messages.append, warning=messages.append, error=messages.append,
        )):
            asyncio.run(self.plugin.initialize())
        self.assertTrue(any(expected in message for message in messages), messages)
        self.assertTrue(any("bound=True" in message for message in messages), messages)
        self.assertEqual(config["plugin_set"], selection)

    def test_default_llm_remains_disabled_even_if_stop_result_is_cleared(self):
        event = self.event("fuck")
        asyncio.run(self.dispatch(event))
        event.stopped = False
        event.is_at_or_wake_command = True
        event.set_extra("activated_handlers", [])
        calls = []

        async def agent_process(e):
            calls.append(e)
            yield None

        stage = self.ns["ProcessStage"]()
        stage.ctx = types.SimpleNamespace(
            astrbot_config={"provider_settings": {"enable": True}}
        )
        stage.agent_sub_stage = types.SimpleNamespace(process=agent_process)

        async def run():
            async for _ in stage.process(event):
                pass

        asyncio.run(run())
        self.assertEqual(calls, [])

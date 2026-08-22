import unittest

from spam_detector import SpamDetector, normalize_text


class SpamDetectorTests(unittest.TestCase):
    def make_detector(self, **overrides):
        config = {
            "near_duplicate_min_chars": 12,
            "near_duplicate_prefix_ratio": 0.65,
            "near_duplicate_max_suffix_chars": 48,
            "long_message_chars": 100,
            "long_burst_count": 3,
            "long_burst_window_seconds": 10,
            "long_burst_max_gap_seconds": 3,
        }
        config.update(overrides)
        return SpamDetector(config)

    @staticmethod
    def long_text(seed):
        return "".join(f"{seed:02d}-{index:03d}-payload;" for index in range(20))

    def test_normalization_removes_invisible_formatting(self):
        self.assertEqual(
            normalize_text("  ＡＢＣ\u200b\n123  "),
            "abc 123",
        )

    def test_second_exact_message_is_blocked(self):
        detector = self.make_detector()
        self.assertFalse(
            detector.check("group:user", "请注意这条消息", timestamp=0).blocked
        )
        decision = detector.check("group:user", "请注意这条消息", timestamp=1)
        self.assertTrue(decision.blocked)
        self.assertEqual(decision.reason, "exact_duplicate")

    def test_non_consecutive_duplicate_is_not_exact_duplicate(self):
        detector = self.make_detector()
        detector.check("group:user", "第一条消息", timestamp=0)
        detector.check("group:user", "第二条消息", timestamp=1)
        decision = detector.check("group:user", "第一条消息", timestamp=2)
        self.assertFalse(decision.blocked)

    def test_random_suffix_is_near_duplicate(self):
        detector = self.make_detector()
        base = "服务器什么时候重启以及维护安排请看公告请所有成员提前做好准备"
        detector.check("group:user", base + "-a81k2", timestamp=0)
        decision = detector.check("group:user", base + "-zP93x", timestamp=1)
        self.assertTrue(decision.blocked)
        self.assertEqual(decision.reason, "near_duplicate")

    def test_lowercase_random_suffix_is_near_duplicate(self):
        detector = self.make_detector(near_duplicate_min_chars=8)
        detector.check("group:user", "请大家在群里保持安静-aqwer", timestamp=0)
        decision = detector.check(
            "group:user", "请大家在群里保持安静-zmnbv", timestamp=1
        )
        self.assertTrue(decision.blocked)

    def test_short_changing_ascii_suffix_is_near_duplicate(self):
        detector = self.make_detector()
        base = "请所有成员按照公告安排完成本次维护准备工作"
        detector.check("group:user", base + "a", timestamp=0)
        decision = detector.check("group:user", base + "b", timestamp=1)
        self.assertTrue(decision.blocked)
        self.assertEqual(decision.reason, "near_duplicate")

    def test_short_changing_unicode_suffix_is_near_duplicate(self):
        detector = self.make_detector()
        base = "请所有成员按照公告安排完成本次维护准备工作"
        detector.check("group:user", base + "鰯鱟麤龘", timestamp=0)
        decision = detector.check("group:user", base + "麤龘鰯鱟", timestamp=1)
        self.assertTrue(decision.blocked)
        self.assertEqual(decision.reason, "near_duplicate")

    def test_long_random_suffix_is_near_duplicate(self):
        detector = self.make_detector()
        base = "服务器什么时候重启以及维护安排请看公告请所有成员提前做好准备" * 3
        detector.check("group:user", base + "-a1B2" * 8, timestamp=0)
        decision = detector.check("group:user", base + "-z9X8" * 8, timestamp=1)
        self.assertTrue(decision.blocked)
        self.assertEqual(decision.reason, "near_duplicate")

    def test_varied_unicode_suffix_is_near_duplicate(self):
        detector = self.make_detector(near_duplicate_min_chars=8)
        detector.check("group:user", "请大家在群里保持安静-鰯鱟麤龘龗", timestamp=0)
        decision = detector.check(
            "group:user", "请大家在群里保持安静-麤龘龗鰯鱟", timestamp=1
        )
        self.assertTrue(decision.blocked)

    def test_short_messages_are_not_near_duplicate(self):
        detector = self.make_detector()
        detector.check("group:user", "ok-1", timestamp=0)
        decision = detector.check("group:user", "ok-2", timestamp=1)
        self.assertFalse(decision.blocked)

    def test_short_body_with_random_suffix_is_near_duplicate(self):
        detector = self.make_detector(
            near_duplicate_min_chars=8,
            near_duplicate_prefix_ratio=0.65,
        )
        detector.check("group:user", "请大家在群里保持安静-a81k2", timestamp=0)
        decision = detector.check(
            "group:user", "请大家在群里保持安静-zP93x", timestamp=1
        )
        self.assertTrue(decision.blocked)
        self.assertEqual(decision.reason, "near_duplicate")

    def test_meaningfully_different_tail_is_not_near_duplicate(self):
        detector = self.make_detector()
        detector.check(
            "group:user", "服务器什么时候重启以及维护安排请看公告", timestamp=0
        )
        decision = detector.check(
            "group:user", "服务器什么时候维护以及重启安排请看公告", timestamp=1
        )
        self.assertFalse(decision.blocked)

    def test_meaningful_chinese_suffix_is_not_near_duplicate(self):
        detector = self.make_detector()
        detector.check("group:user", "请大家注意群规并保持文明发言", timestamp=0)
        decision = detector.check(
            "group:user", "请大家注意群规并保持文明发言谢谢", timestamp=1
        )
        self.assertFalse(decision.blocked)

    def test_repeated_phrase_is_blocked(self):
        detector = self.make_detector()
        decision = detector.check("group:user", "神经病" * 12, timestamp=0)
        self.assertTrue(decision.blocked)
        self.assertEqual(decision.reason, "repeated_content")

    def test_repeated_phrase_with_prefix_is_blocked(self):
        detector = self.make_detector()
        decision = detector.check("group:user", "警告：" + "神经病" * 10, timestamp=0)
        self.assertTrue(decision.blocked)
        self.assertEqual(decision.reason, "repeated_content")

    def test_repeated_phrase_with_spaces_is_blocked(self):
        detector = self.make_detector()
        decision = detector.check("group:user", "神经病 " * 10, timestamp=0)
        self.assertTrue(decision.blocked)
        self.assertEqual(decision.reason, "repeated_content")

    def test_repeated_phrase_in_middle_of_long_message_is_blocked(self):
        detector = self.make_detector(repeat_scan_chars=1000)
        prefix = "".join(chr(0x4E00 + index) for index in range(300))
        suffix = "".join(chr(0x5200 + index) for index in range(300))
        decision = detector.check(
            "group:user",
            prefix + ("神经病" * 300) + suffix,
            timestamp=0,
        )
        self.assertTrue(decision.blocked)
        self.assertEqual(decision.reason, "repeated_content")

    def test_small_repetition_is_not_blocked(self):
        detector = self.make_detector()
        decision = detector.check("group:user", "哈哈哈哈", timestamp=0)
        self.assertFalse(decision.blocked)

    def test_three_rapid_long_messages_are_blocked(self):
        detector = self.make_detector()
        long_texts = [self.long_text(seed) for seed in range(3)]
        self.assertFalse(
            detector.check("group:user", long_texts[0], timestamp=0).blocked
        )
        self.assertFalse(
            detector.check("group:user", long_texts[1], timestamp=1).blocked
        )
        decision = detector.check("group:user", long_texts[2], timestamp=2)
        self.assertTrue(decision.blocked)
        self.assertEqual(decision.reason, "long_message_burst")

    def test_slow_long_messages_are_not_a_burst(self):
        detector = self.make_detector()
        for timestamp, text in enumerate(
            (self.long_text(0), self.long_text(1), self.long_text(2))
        ):
            decision = detector.check("group:user", text, timestamp=timestamp * 4)
            self.assertFalse(decision.blocked)

    def test_sender_state_is_isolated(self):
        detector = self.make_detector()
        detector.check("group:a", "相同消息", timestamp=0)
        self.assertFalse(detector.check("group:b", "相同消息", timestamp=1).blocked)

    def test_cache_expires(self):
        detector = self.make_detector(
            exact_duplicate_window_seconds=2, cache_ttl_seconds=2
        )
        detector.check("group:user", "相同消息", timestamp=0)
        decision = detector.check("group:user", "相同消息", timestamp=3)
        self.assertFalse(decision.blocked)


if __name__ == "__main__":
    unittest.main()

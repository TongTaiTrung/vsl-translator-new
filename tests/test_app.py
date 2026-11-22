import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from src.app.realtime import RealtimeTranslator
from src.app.sentence import (SentenceComposer, build_system_prompt, clean_completion, format_labels,
                              load_api_key, naive_sentence)
from src.app.voting import VoteConfig, WordVoter


class FakeClient:
    def __init__(self, reply="Xin chào.", error=None, gate=None):
        self.reply, self.error, self.gate, self.calls = reply, error, gate, []

    def chat(self, messages, **_kwargs):
        self.calls.append(messages)
        if self.gate is not None:
            self.gate.wait(2)
        if self.error:
            raise self.error
        return self.reply


class FakeSession:
    """Phát lại danh sách (status, label); label=None nghĩa là dự đoán bị từ chối."""

    def __init__(self, script):
        self.script = list(script)
        self.prediction = None
        self.resets = 0
        self.closed = False

    def update_frame(self, _frame, now=None):
        status, label = self.script.pop(0)
        if status in ("predicting", "ready"):
            self.prediction = {"label": label}
        return {"status": status, "prediction": self.prediction, "stable_label": None,
                "hand_present": status != "idle", "frames": 0, "duration": 0.0}

    def reset(self):
        self.resets += 1
        self.prediction = None

    def close(self):
        self.closed = True


class WordVoterTests(unittest.TestCase):
    def test_commits_after_five_consecutive_votes(self):
        voter = WordVoter(VoteConfig(min_consecutive=5))
        self.assertEqual([voter.vote("toi") for _ in range(4)], [None] * 4)
        self.assertEqual(voter.vote("toi"), "toi")
        self.assertEqual(voter.sequence, ["toi"])

    def test_interruption_or_other_label_restarts_count(self):
        voter = WordVoter(VoteConfig(min_consecutive=3))
        for label in ["a", "a", None, "a", "a", "b", "a", "a"]:
            self.assertIsNone(voter.vote(label))
        self.assertEqual(voter.vote("a"), "a")

    def test_held_sign_is_not_repeated_until_hand_released(self):
        voter = WordVoter(VoteConfig(min_consecutive=2))
        committed = [voter.vote("toi") for _ in range(8)]
        self.assertEqual(committed.count("toi"), 1)
        voter.interrupt()                      # cửa sổ đang gom lại: vẫn chặn
        self.assertEqual([voter.vote("toi") for _ in range(3)].count("toi"), 0)
        voter.release()                        # tay rời khung: cho phép lặp
        self.assertEqual([voter.vote("toi") for _ in range(2)][-1], "toi")
        self.assertEqual(voter.sequence, ["toi", "toi"])

    def test_undo_and_max_words(self):
        voter = WordVoter(VoteConfig(min_consecutive=1, max_words=2))
        for label in ["a", "b", "c"]:
            voter.vote(label)
        self.assertEqual(voter.sequence, ["b", "c"])
        self.assertEqual(voter.undo(), "c")
        self.assertEqual(voter.sequence, ["b"])


class SentenceTests(unittest.TestCase):
    def test_letters_are_grouped_for_prompt_and_fallback(self):
        words = ["xin chào", "toi", "ten_la", "t", "r", "u", "n", "g"]
        self.assertEqual(format_labels(words), "xin chào | toi | ten_la | [đánh vần: T-R-U-N-G]")
        self.assertEqual(naive_sentence(words), "Xin chào tôi tên là Trung")

    def test_tone_marks_join_spelling_group(self):
        words = ["m", "i", "n", "h", "b", "i", "dấu nặng", "n", "g", "u"]
        self.assertEqual(format_labels(words), "[đánh vần: M-I-N-H-B-I-(dấu nặng)-N-G-U]")
        self.assertEqual(naive_sentence(words), "Minh bị ngu")
        self.assertEqual(naive_sentence(["xin chào", "toi", *words]), "Xin chào tôi Minh bị ngu")
        self.assertIn("Minh bị ngu", build_system_prompt())

    def test_vowel_marks_join_spelling_group(self):
        words = ["xin chào", "toi", "ten_la", "t", "h", "o", "dấu ơ,ư"]
        self.assertEqual(format_labels(words), "xin chào | toi | ten_la | [đánh vần: T-H-O-(dấu ơ,ư)]")
        self.assertEqual(naive_sentence(words), "Xin chào tôi tên là Thơ")
        self.assertIn("Kết quả: Xin chào, tôi tên là Thơ.", build_system_prompt())
        self.assertEqual(naive_sentence(["c", "o", "dấu â,ô,ê", "n", "g"]), "Công")
        self.assertEqual(naive_sentence(["h", "o", "n", "dấu ơ,ư"]), "Hơn")
        self.assertEqual(naive_sentence(["t", "h", "o", "dấu ơ,ư", "dấu sắc"]), "Thớ")
        self.assertEqual(naive_sentence(["t", "h", "o", "dấu sắc", "dấu ơ,ư"]), "Thớ")
        self.assertEqual(naive_sentence(["n", "g", "u", "dấu ơ,ư", "o", "i", "dấu huyền"]), "Người")

    def test_misspelled_sac_label_is_a_tone(self):
        words = ["t", "r", "o", "n", "g", "dấu nặng", "t", "u", "a", "dấu â,ô,ê", "t", "dấu sắt"]
        self.assertEqual(format_labels(words), "[đánh vần: T-R-O-N-G-(dấu nặng)-T-U-A-(dấu â,ô,ê)-T-(dấu sắc)]")
        self.assertEqual(naive_sentence(words), "Trọng tuất")

    def test_fallback_splits_syllables_and_places_tone(self):
        self.assertEqual(naive_sentence(["h", "o", "a", "dấu huyền"]), "Hòa")
        self.assertEqual(naive_sentence(["t", "o", "a", "n", "dấu sắc", "h", "o", "c", "dấu nặng"]), "Toán học")
        self.assertEqual(naive_sentence(list("anhem")), "Anh em")
        self.assertEqual(naive_sentence(list("xyz")), "Xyz")

    def test_system_prompt_contains_glossary(self):
        prompt = build_system_prompt({"toi": "tôi"})
        self.assertIn('"toi" = "tôi"', prompt)
        self.assertIn("đánh vần", prompt)

    def test_clean_completion_strips_thinking_and_prefix(self):
        self.assertEqual(clean_completion('<think>hmm</think>\nKết quả: "Tôi tên là Trung."'), "Tôi tên là Trung.")

    def test_compose_uses_fallback_on_error(self):
        composer = SentenceComposer(FakeClient(error=RuntimeError("down")), background=False)
        composer.submit(["toi"])
        state = composer.state()
        self.assertEqual((state["sentence"], state["status"]), ("Tôi", "error"))

    def test_empty_sequence_does_not_call_api(self):
        client = FakeClient()
        composer = SentenceComposer(client, background=False)
        composer.submit([])
        self.assertEqual(client.calls, [])
        self.assertEqual(composer.state()["sentence"], "")

    def test_background_worker_keeps_only_latest_request(self):
        gate = threading.Event()
        client = FakeClient(reply="Câu mới.", gate=gate)
        done = threading.Event()
        composer = SentenceComposer(
            client, on_update=lambda s: done.set() if s["words"] == ["c"] else None)
        try:
            composer.submit(["a"])
            composer.submit(["b"])
            composer.submit(["c"])
            gate.set()
            self.assertTrue(done.wait(2))
            self.assertEqual(composer.state()["words"], ["c"])
            self.assertLessEqual(len(client.calls), 2)   # "b" bị gộp, không gửi
        finally:
            composer.close()

    def test_load_api_key_from_env_file(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / ".env"
            path.write_text('GROQ_API_KEY="abc"\n', encoding="utf-8")
            self.assertEqual(load_api_key(env_file=path), "abc")


class RealtimeTranslatorTests(unittest.TestCase):
    def test_commit_resets_window_and_sends_sequence_to_composer(self):
        script = [("collecting", None)] + [("predicting", "toi")] * 5 + [("collecting", None)] \
            + [("predicting", "toi")] * 5 + [("idle", None)] + [("predicting", "cảm ơn")] * 5
        session = FakeSession(script)
        client = FakeClient(reply="Tôi cảm ơn.")
        translator = RealtimeTranslator(None, session=session, vote_config=VoteConfig(min_consecutive=5),
                                        composer=SentenceComposer(client, background=False))
        committed = [translator.process(None)["committed"] for _ in range(len(script))]
        self.assertEqual([c for c in committed if c], ["toi", "cảm ơn"])
        self.assertEqual(translator.words, ["toi", "cảm ơn"])
        self.assertEqual(session.resets, 2)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(translator.composer.state()["sentence"], "Tôi cảm ơn.")
        translator.undo()
        self.assertEqual(translator.words, ["toi"])
        translator.close()
        self.assertTrue(session.closed)


if __name__ == "__main__":
    unittest.main()

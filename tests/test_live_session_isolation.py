import threading
import tempfile
import time
import unittest
from pathlib import Path

from scripts.live_workflow_server import LiveSessionManager


def _blocking_target(*, _generation: int, _session_root: Path, ready, release) -> None:
    ready.set()
    release.wait(timeout=10)


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class LiveSessionIsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_reset_while_worker_alive_keeps_new_session_root(self) -> None:
        manager = LiveSessionManager(live_root=self.root / "current")
        initial_root = manager.live_root

        ready = threading.Event()
        release = threading.Event()
        manager._start_worker(_blocking_target, ready=ready, release=release)
        self.assertTrue(ready.wait(timeout=5))
        old_worker = manager._worker
        self.assertIsNotNone(old_worker)
        self.assertTrue(old_worker.is_alive())

        manager.reset(preserve_staged_data=True)
        new_root = manager.live_root
        self.assertNotEqual(new_root, initial_root)
        self.assertTrue(new_root.parent.name == "sessions")
        self.assertIsNone(manager._worker)

        release.set()
        self.assertTrue(_wait_for(lambda: not old_worker.is_alive()))
        self.assertFalse(initial_root.exists())
        self.assertTrue(new_root.exists())

    def test_second_worker_is_rejected_while_first_is_running(self) -> None:
        manager = LiveSessionManager(live_root=self.root / "current")

        ready = threading.Event()
        release = threading.Event()
        manager._start_worker(_blocking_target, ready=ready, release=release)
        self.assertTrue(ready.wait(timeout=5))

        with self.assertRaises(RuntimeError):
            manager._start_worker(_blocking_target, ready=threading.Event(), release=threading.Event())

        old_worker = manager._worker
        release.set()
        old_worker.join(timeout=5)
        self.assertFalse(old_worker.is_alive())

    def test_second_worker_is_rejected_while_first_is_starting(self) -> None:
        manager = LiveSessionManager(live_root=self.root / "current")

        original_start = threading.Thread.start
        start_blocked = threading.Event()
        start_release = threading.Event()
        pause_state = {"pause_next": True}

        def pausing_start(self_thread):
            if pause_state["pause_next"] and not getattr(self_thread, "_ignore_pause", False):
                pause_state["pause_next"] = False
                start_blocked.set()
                deadline = time.time() + 5
                while time.time() < deadline and not start_release.is_set():
                    time.sleep(0.02)
            return original_start(self_thread)

        threading.Thread.start = pausing_start
        second_calls: list[tuple[str, Exception | None]] = []
        second_done = threading.Event()

        def call_second() -> None:
            try:
                manager._start_worker(
                    _blocking_target,
                    ready=threading.Event(),
                    release=threading.Event(),
                )
            except Exception as exc:  # noqa: BLE001 - test asserts the raised type
                second_calls.append(("raised", exc))
            else:
                second_calls.append(("accepted", None))
            finally:
                second_done.set()

        first_ready = threading.Event()
        first_release = threading.Event()
        first_run_result: list[Exception | None] = []

        def call_first() -> None:
            try:
                manager._start_worker(
                    _blocking_target,
                    ready=first_ready,
                    release=first_release,
                )
            except Exception as exc:  # noqa: BLE001 - test asserts no exception
                first_run_result.append(exc)

        try:
            first_thread = threading.Thread(target=call_first)
            first_thread._ignore_pause = True
            first_thread.start()
            self.assertTrue(_wait_for(start_blocked.is_set, timeout=5))
            self.assertEqual(first_run_result, [])

            second_thread = threading.Thread(target=call_second)
            second_thread._ignore_pause = True
            second_thread.start()
            time.sleep(0.2)
            self.assertFalse(second_done.is_set())

            start_release.set()
            first_thread.join(timeout=5)
            second_thread.join(timeout=5)
            self.assertTrue(second_done.is_set())
            self.assertEqual(len(second_calls), 1)
            self.assertEqual(second_calls[0][0], "raised")
            self.assertIsInstance(second_calls[0][1], RuntimeError)
        finally:
            threading.Thread.start = original_start
            start_release.set()
            first_release.set()
            first_worker = manager._worker
            if first_worker is not None:
                first_worker.join(timeout=5)

    def test_consecutive_resets_stay_in_same_sessions_directory(self) -> None:
        manager = LiveSessionManager(live_root=self.root / "current")
        session_base = self.root / "sessions"

        roots = []
        for _ in range(3):
            manager.reset(preserve_staged_data=True)
            roots.append(manager.live_root)

        self.assertEqual(len(set(roots)), 3)
        for root in roots:
            self.assertEqual(root.parent, session_base)
            self.assertNotIn("sessions", root.relative_to(session_base).parts)
        self.assertTrue(manager.live_root.exists())

    def test_manager_recovers_active_session_after_restart(self) -> None:
        manager = LiveSessionManager(live_root=self.root / "current")
        manager.reset(preserve_staged_data=True)

        manager._status.update(
            {
                "status": "running",
                "stage": "result_analysis",
                "question": "宇宙线日影南北偏移能否用于太阳风速度预测",
                "currentRound": 2,
            }
        )
        manager._write_status()
        session_root = manager.live_root

        restarted = LiveSessionManager(live_root=self.root / "current")
        self.assertEqual(restarted.live_root, session_root)
        self.assertEqual(restarted._status.get("question"), "宇宙线日影南北偏移能否用于太阳风速度预测")
        self.assertEqual(restarted._status.get("currentRound"), 2)


if __name__ == "__main__":
    unittest.main()

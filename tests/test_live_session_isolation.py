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


if __name__ == "__main__":
    unittest.main()

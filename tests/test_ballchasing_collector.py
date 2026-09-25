from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ingestion.ballchasing.collect_metadata import API_ROOT, ApiResponse, Collector
from ingestion.common.replay_state import atomic_write_json, load_json, parse_utc


class FakeClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += timedelta(seconds=seconds)


def response(url: str, payload: object, status: int = 200, headers: dict[str, str] | None = None) -> ApiResponse:
    body = json.dumps(payload).encode("utf-8")
    return ApiResponse(url, status, headers or {"Content-Type": "application/json"}, body, payload)


class BallchasingCollectorTests(unittest.TestCase):
    def _catalog(self, path: Path) -> None:
        atomic_write_json(
            path,
            {
                "schema_version": 1,
                "groups": [{"group_id": "event", "event": "Verified Event", "enabled": True}],
            },
        )

    def _collector(self, root: Path, clock: FakeClock, fetch, logs: list[str] | None = None) -> Collector:
        return Collector(
            root / "catalog.json",
            root / "metadata_state.json",
            root / "debug",
            root / "api_pacing.json",
            "secret-token",
            fetch=fetch,
            sleep=clock.sleep,
            now=clock.now,
            log=(logs.append if logs is not None else None),
        )

    def test_recurses_groups_deduplicates_replays_and_persists_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._catalog(root / "catalog.json")
            clock = FakeClock()
            calls: list[tuple[str, dict[str, str], datetime]] = []

            def fetch(url: str, headers: dict[str, str]) -> ApiResponse:
                calls.append((url, headers, clock.now()))
                if url == f"{API_ROOT}/groups/event":
                    return response(url, {"id": "event", "name": "Verified Event"})
                if url == f"{API_ROOT}/groups?group=event&count=200":
                    return response(url, {"list": [{"id": "series"}]})
                if url == f"{API_ROOT}/replays?group=event&count=200":
                    return response(
                        url,
                        {
                            "list": [{"id": "replay-1", "replay_title": "first"}],
                            "next": "https://ballchasing.com/api/replays?group=event&after=page-2",
                        },
                    )
                if url == "https://ballchasing.com/api/replays?group=event&after=page-2":
                    return response(url, {"list": [{"id": "replay-1", "replay_title": "duplicate"}]})
                if url == f"{API_ROOT}/groups/series":
                    return response(url, {"id": "series", "name": "Series"})
                if url == f"{API_ROOT}/groups?group=series&count=200":
                    return response(url, {"list": []})
                if url == f"{API_ROOT}/replays?group=series&count=200":
                    return response(url, {"list": [{"id": "replay-1", "replay_title": "same replay"}]})
                if url == f"{API_ROOT}/replays/replay-1":
                    return response(url, {"id": "replay-1", "title": "detailed"})
                self.fail(f"unexpected URL {url}")

            logs: list[str] = []
            result = self._collector(root, clock, fetch, logs).run(max_requests=20)
            self.assertEqual(result["pending"], 0)
            self.assertEqual(result["failed"], 0)
            state = load_json(root / "metadata_state.json", {})
            self.assertEqual(set(state["groups"]), {"event", "series"})
            self.assertEqual(state["replays"]["replay-1"]["group_ids"], ["event", "series"])
            self.assertEqual(state["replays"]["replay-1"]["metadata"]["title"], "detailed")
            self.assertTrue(any((root / "debug").iterdir()))
            self.assertTrue(all(headers["Authorization"] == "secret-token" for _, headers, _ in calls))
            self.assertTrue(all("secret-token" not in line for line in logs))
            call_times = [call[2] for call in calls]
            self.assertEqual(call_times[0], datetime(2026, 1, 1, tzinfo=timezone.utc))
            self.assertTrue(all((later - earlier).total_seconds() >= 10 for earlier, later in zip(call_times, call_times[1:])))

    def test_max_requests_and_second_collector_share_the_pacing_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._catalog(root / "catalog.json")
            clock = FakeClock()
            call_times: list[datetime] = []

            def fetch(url: str, headers: dict[str, str]) -> ApiResponse:
                call_times.append(clock.now())
                if url == f"{API_ROOT}/groups/event":
                    return response(url, {"id": "event"})
                if url == f"{API_ROOT}/groups?group=event&count=200":
                    return response(url, {"list": []})
                if url == f"{API_ROOT}/replays?group=event&count=200":
                    return response(url, {"list": []})
                self.fail(f"unexpected URL {url}")

            first = self._collector(root, clock, fetch).run(max_requests=1)
            self.assertEqual(first["attempted"], 1)
            second = self._collector(root, clock, fetch).run(max_requests=1)
            self.assertEqual(second["attempted"], 1)
            self.assertEqual((call_times[1] - call_times[0]).total_seconds(), 10)
            self.assertEqual(clock.sleeps, [10.0])

    def test_interruption_recovers_task_but_honors_reserved_slot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._catalog(root / "catalog.json")
            clock = FakeClock()

            def interrupted(url: str, headers: dict[str, str]) -> ApiResponse:
                raise KeyboardInterrupt()

            with self.assertRaises(KeyboardInterrupt):
                self._collector(root, clock, interrupted).run(max_requests=1)
            state = load_json(root / "metadata_state.json", {})
            task = next(task for task in state["tasks"].values() if task["kind"] == "group_detail")
            self.assertEqual(task["status"], "in_progress")
            pacing = load_json(root / "api_pacing.json", {})
            self.assertEqual((parse_utc(pacing["next_allowed_at"]) - clock.now()).total_seconds(), 10)

            def recovered(url: str, headers: dict[str, str]) -> ApiResponse:
                return response(url, {"id": "event"})

            result = self._collector(root, clock, recovered).run(max_requests=1)
            self.assertEqual(result["completed"], 1)
            self.assertEqual(clock.sleeps, [10.0])
            state = load_json(root / "metadata_state.json", {})
            resumed = next(task for task in state["tasks"].values() if task["kind"] == "group_detail")
            self.assertEqual(task["task_id"], resumed["task_id"])
            self.assertEqual(resumed["status"], "complete")

    def test_rate_limit_extends_pacing_and_leaves_task_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._catalog(root / "catalog.json")
            clock = FakeClock()

            def limited(url: str, headers: dict[str, str]) -> ApiResponse:
                return response(url, {"error": "slow down"}, 429, {"Retry-After": "30"})

            result = self._collector(root, clock, limited).run(max_requests=1)
            self.assertEqual(result["retried"], 1)
            state = load_json(root / "metadata_state.json", {})
            task = next(task for task in state["tasks"].values() if task["kind"] == "group_detail")
            self.assertEqual(task["status"], "retry")
            pacing = load_json(root / "api_pacing.json", {})
            self.assertEqual((parse_utc(pacing["next_allowed_at"]) - clock.now()).total_seconds(), 60)


if __name__ == "__main__":
    unittest.main()

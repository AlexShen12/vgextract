#!/usr/bin/env python3
"""Collect replay URLs for the live OpenDota professional-match snapshot.

The script is deliberately conservative: it makes one match-detail request every
60 seconds, persists state before and after every request, and retries failures
for the same match after another full minute.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from replay_state import SCHEMA_VERSION, atomic_write_json, exclusive_lock, load_json, parse_utc


API_ROOT = "https://api.opendota.com/api"
PRO_MATCHES_URL = f"{API_ROOT}/proMatches"
REQUEST_INTERVAL_SECONDS = 60


def empty_manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "endpoint": PRO_MATCHES_URL,
            "last_snapshot_at": None,
            "last_snapshot_count": 0,
            "last_snapshot_match_ids": [],
        },
        "request_pacing": {"next_detail_attempt_at": None},
        "next_discovery_order": 1,
        "matches": {},
    }


def fetch_json(url: str, timeout: float = 30.0) -> Any:
    """Fetch a JSON document, raising for HTTP, transport, and JSON errors."""
    request = Request(url, headers={"User-Agent": "opendota-replay-collector/1.0"})
    with urlopen(request, timeout=timeout) as response:
        payload = response.read()
    return json.loads(payload.decode("utf-8"))


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported or missing manifest schema_version")
    if not isinstance(manifest.get("matches"), dict):
        raise ValueError("manifest matches must be an object")
    if not isinstance(manifest.get("source"), dict):
        raise ValueError("manifest source must be an object")
    if not isinstance(manifest.get("request_pacing"), dict):
        raise ValueError("manifest request_pacing must be an object")


def _match_ids(snapshot: Any) -> list[int]:
    if not isinstance(snapshot, list):
        raise ValueError("/proMatches response must be a JSON array")
    result: list[int] = []
    seen: set[int] = set()
    for item in snapshot:
        if not isinstance(item, dict):
            raise ValueError("/proMatches item must be an object")
        match_id = item.get("match_id")
        if isinstance(match_id, bool) or not isinstance(match_id, int) or match_id <= 0:
            raise ValueError("/proMatches item has an invalid match_id")
        if match_id not in seen:
            result.append(match_id)
            seen.add(match_id)
    return result


class Collector:
    """Stateful collector with injectable timing/network functions for testing."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        fetch: Callable[[str], Any] = fetch_json,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] | None = None,
        interval_seconds: int = REQUEST_INTERVAL_SECONDS,
    ) -> None:
        self.manifest_path = manifest_path
        self.fetch = fetch
        self.sleep = sleep
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.interval_seconds = interval_seconds
        self.manifest: dict[str, Any] = {}

    def _timestamp(self) -> str:
        return self.now().astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def _save(self) -> None:
        atomic_write_json(self.manifest_path, self.manifest)

    def _wait_for_detail_slot(self) -> None:
        value = self.manifest["request_pacing"].get("next_detail_attempt_at")
        if not value:
            return
        deadline = parse_utc(value)
        while True:
            remaining = (deadline - self.now().astimezone(timezone.utc)).total_seconds()
            if remaining <= 0:
                return
            # A single long sleep remains interruptible with Ctrl-C, and avoids a busy loop.
            self.sleep(remaining)

    def _record_attempt_start(self) -> None:
        now = self.now().astimezone(timezone.utc)
        next_attempt = now + timedelta(seconds=self.interval_seconds)
        self.manifest["request_pacing"] = {
            "last_detail_attempt_at": now.isoformat().replace("+00:00", "Z"),
            "next_detail_attempt_at": next_attempt.isoformat().replace("+00:00", "Z"),
        }
        # Save before networking: a crash after this point still cannot burst on restart.
        self._save()

    def _add_snapshot(self, snapshot: Any) -> list[int]:
        match_ids = _match_ids(snapshot)
        timestamp = self._timestamp()
        source = self.manifest["source"]
        source.update(
            {
                "endpoint": PRO_MATCHES_URL,
                "last_snapshot_at": timestamp,
                "last_snapshot_count": len(match_ids),
                "last_snapshot_match_ids": match_ids,
            }
        )
        matches = self.manifest["matches"]
        for match_id in match_ids:
            key = str(match_id)
            if key in matches:
                matches[key]["last_seen_at"] = timestamp
                continue
            discovery_order = self.manifest["next_discovery_order"]
            self.manifest["next_discovery_order"] += 1
            matches[key] = {
                "match_id": match_id,
                "first_seen_at": timestamp,
                "last_seen_at": timestamp,
                "discovery_order": discovery_order,
                "lookup": {
                    "status": "pending",
                    "attempts": 0,
                    "replay_url": None,
                    "last_attempt_at": None,
                    "last_error": None,
                    "next_attempt_at": None,
                },
            }
        self._save()
        return match_ids

    def _pending_records(self) -> list[dict[str, Any]]:
        records = [
            record
            for record in self.manifest["matches"].values()
            if record.get("lookup", {}).get("status") in {"pending", "error"}
        ]
        return sorted(records, key=lambda record: record.get("discovery_order", 0))

    def _lookup_one(self, record: dict[str, Any]) -> bool:
        """Try once.  Return True when the record has reached a terminal state."""
        self._wait_for_detail_slot()
        self._record_attempt_start()
        lookup = record["lookup"]
        lookup["attempts"] += 1
        lookup["last_attempt_at"] = self._timestamp()
        lookup["next_attempt_at"] = self.manifest["request_pacing"]["next_detail_attempt_at"]
        match_id = record["match_id"]
        try:
            response = self.fetch(f"{API_ROOT}/matches/{match_id}")
            if not isinstance(response, dict):
                raise ValueError("match-detail response must be an object")
            if response.get("match_id") != match_id:
                raise ValueError("match-detail response has a missing or mismatched match_id")
            if "replay_url" not in response:
                raise ValueError("match-detail response is missing replay_url")
            replay_url = response["replay_url"]
            if replay_url is None:
                lookup.update(
                    {
                        "status": "unavailable",
                        "replay_url": None,
                        "last_error": None,
                        "next_attempt_at": None,
                    }
                )
            elif isinstance(replay_url, str) and replay_url:
                lookup.update(
                    {
                        "status": "resolved",
                        "replay_url": replay_url,
                        "last_error": None,
                        "next_attempt_at": None,
                    }
                )
            else:
                raise ValueError("match-detail replay_url must be a non-empty string or null")
        except Exception as exc:
            lookup.update(
                {
                    "status": "error",
                    "last_error": f"{type(exc).__name__}: {exc}",
                    "next_attempt_at": self.manifest["request_pacing"]["next_detail_attempt_at"],
                }
            )
            self._save()
            return False
        self._save()
        return True

    def run(self) -> dict[str, int]:
        self.manifest = load_json(self.manifest_path, empty_manifest())
        _validate_manifest(self.manifest)
        snapshot = self.fetch(PRO_MATCHES_URL)
        match_ids = self._add_snapshot(snapshot)

        terminal = 0
        for record in self._pending_records():
            while not self._lookup_one(record):
                # The next pass waits for the persisted one-minute retry deadline.
                pass
            terminal += 1
        return {"snapshot_count": len(match_ids), "processed_to_terminal": terminal}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/pro_replays.json"),
        help="durable collector manifest (default: data/pro_replays.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        with exclusive_lock(args.manifest.with_suffix(args.manifest.suffix + ".lock")):
            result = Collector(args.manifest).run()
    except KeyboardInterrupt:
        print("Interrupted; saved state can be resumed by rerunning this command.", file=sys.stderr)
        return 130
    except (HTTPError, URLError, OSError, ValueError, RuntimeError) as exc:
        print(f"Collector failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        "Collected pro-match snapshot "
        f"({result['snapshot_count']} IDs); {result['processed_to_terminal']} pending records reached a terminal state."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Collect replay URLs for the live OpenDota professional-match snapshot.

The script is deliberately conservative: it makes one match-detail request every
60 seconds, persists state before and after every request, and retries failures
for the same match after another full minute. Every API response is also saved
to a debug artifact so cluster-side failures can be diagnosed after the job.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from replay_state import (
    SCHEMA_VERSION,
    atomic_write_bytes,
    atomic_write_json,
    exclusive_lock,
    load_json,
    parse_utc,
)


API_ROOT = "https://api.opendota.com/api"
PRO_MATCHES_URL = f"{API_ROOT}/proMatches"
REQUEST_INTERVAL_SECONDS = 60
BODY_PREVIEW_LIMIT = 4_000


@dataclass
class ApiResponse:
    url: str
    status: int | None
    headers: dict[str, str]
    body: bytes
    parsed: Any = None
    parse_error: str | None = None
    request_error: str | None = None

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", self.headers.get("Content-Type", ""))

    @property
    def body_preview(self) -> str:
        return self.body[:BODY_PREVIEW_LIMIT].decode("utf-8", errors="replace")


class ApiTransportError(RuntimeError):
    """A request failed before an HTTP response body could be obtained."""


def _parse_body(url: str, status: int | None, headers: dict[str, str], body: bytes) -> ApiResponse:
    response = ApiResponse(url=url, status=status, headers=headers, body=body)
    try:
        response.parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        response.parse_error = f"{type(exc).__name__}: {exc}"
    return response


def fetch_api_response(url: str, timeout: float = 30.0) -> ApiResponse:
    """Fetch a JSON response while retaining status, headers, and raw bytes."""
    request = Request(url, headers={"User-Agent": "opendota-replay-collector/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
            headers = dict(response.headers.items())
            status = getattr(response, "status", None) or response.getcode()
            return _parse_body(url, status, headers, body)
    except HTTPError as exc:
        try:
            body = exc.read()
        except OSError:
            body = b""
        headers = dict(exc.headers.items()) if exc.headers else {}
        response = _parse_body(url, exc.code, headers, body)
        response.request_error = f"HTTP {exc.code}: {exc.reason}"
        return response
    except (OSError, URLError, TimeoutError) as exc:
        raise ApiTransportError(f"{type(exc).__name__}: {exc}") from exc


def fetch_json(url: str, timeout: float = 30.0) -> Any:
    """Compatibility helper returning only parsed JSON."""
    response = fetch_api_response(url, timeout=timeout)
    if response.request_error:
        raise ValueError(response.request_error)
    if response.status is None or not 200 <= response.status < 300:
        raise ValueError(f"unexpected HTTP status {response.status}")
    if response.parse_error:
        raise ValueError(response.parse_error)
    return response.parsed


def empty_manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "endpoint": PRO_MATCHES_URL,
            "last_snapshot_at": None,
            "last_snapshot_count": 0,
            "last_snapshot_match_ids": [],
            "last_response_artifact": None,
        },
        "request_pacing": {"next_detail_attempt_at": None},
        "next_discovery_order": 1,
        "matches": {},
    }


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


def _response_summary(response: ApiResponse) -> dict[str, Any]:
    parsed = response.parsed if isinstance(response.parsed, dict) else {}
    replay_url = parsed.get("replay_url") if isinstance(parsed, dict) else None
    return {
        "status": response.status,
        "content_type": response.content_type,
        "bytes": len(response.body),
        "match_id": parsed.get("match_id") if isinstance(parsed, dict) else None,
        "replay_url": replay_url,
        "replay_url_type": type(replay_url).__name__ if "replay_url" in parsed else "missing",
        "parse_error": response.parse_error,
        "request_error": response.request_error,
        "body_preview": response.body_preview,
    }


class Collector:
    """Stateful collector with injectable timing/network functions for testing."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        debug_dir: Path | None = None,
        fetch: Callable[[str], Any] = fetch_api_response,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] | None = None,
        interval_seconds: int = REQUEST_INTERVAL_SECONDS,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.debug_dir = debug_dir or manifest_path.parent / "api_debug"
        self.fetch = fetch
        self.sleep = sleep
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.interval_seconds = interval_seconds
        self.log = log or (lambda message: print(message, file=sys.stderr, flush=True))
        self.manifest: dict[str, Any] = {}

    def _timestamp(self) -> str:
        return self.now().astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def _save(self) -> None:
        atomic_write_json(self.manifest_path, self.manifest)

    def _persist_response(
        self, response: ApiResponse, *, label: str, attempt: int | None = None
    ) -> tuple[Path, dict[str, Any]]:
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_attempt_{attempt}" if attempt is not None else ""
        artifact = self.debug_dir / f"{label}{suffix}.json"
        atomic_write_bytes(artifact, response.body)
        summary = _response_summary(response)
        summary["artifact"] = str(artifact)
        self.log(
            f"[OpenDota API] GET {response.url} status={response.status} "
            f"bytes={len(response.body)} content_type={response.content_type or '<missing>'} "
            f"artifact={artifact}"
        )
        self.log(
            f"[OpenDota API] response match_id={summary['match_id']!r} "
            f"replay_url_type={summary['replay_url_type']} "
            f"replay_url={summary['replay_url']!r}"
        )
        if response.request_error or response.parse_error:
            self.log(
                f"[OpenDota API] response diagnostics: "
                f"request_error={response.request_error!r} parse_error={response.parse_error!r} "
                f"body_preview={response.body_preview!r}"
            )
        return artifact, summary

    def _wait_for_detail_slot(self) -> None:
        value = self.manifest["request_pacing"].get("next_detail_attempt_at")
        if not value:
            return
        deadline = parse_utc(value)
        while True:
            remaining = (deadline - self.now().astimezone(timezone.utc)).total_seconds()
            if remaining <= 0:
                return
            self.sleep(remaining)

    def _record_attempt_start(self) -> None:
        now = self.now().astimezone(timezone.utc)
        next_attempt = now + timedelta(seconds=self.interval_seconds)
        self.manifest["request_pacing"] = {
            "last_detail_attempt_at": now.isoformat().replace("+00:00", "Z"),
            "next_detail_attempt_at": next_attempt.isoformat().replace("+00:00", "Z"),
        }
        self._save()

    def _prepare_response(
        self, response: Any, *, label: str, attempt: int | None = None
    ) -> tuple[Any, dict[str, Any] | None]:
        if isinstance(response, ApiResponse):
            artifact, summary = self._persist_response(response, label=label, attempt=attempt)
            summary["artifact"] = str(artifact)
            return response.parsed, summary
        # Injected test fetchers may return already-parsed data.
        return response, None

    def _add_snapshot(self, snapshot: Any, response_summary: dict[str, Any] | None) -> list[int]:
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
        if response_summary:
            source["last_response"] = response_summary
            source["last_response_artifact"] = response_summary["artifact"]

        matches = self.manifest["matches"]
        for match_id in match_ids:
            key = str(match_id)
            if key in matches:
                record = matches[key]
                record["last_seen_at"] = timestamp
                lookup = record.get("lookup", {})
                # Null replay URLs can be transient. A later rediscovery makes
                # the entry eligible for another detail lookup.
                if lookup.get("status") == "unavailable":
                    lookup.update(
                        {
                            "status": "pending",
                            "last_error": "rediscovered in a later proMatches snapshot",
                            "next_attempt_at": None,
                        }
                    )
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
                    "last_response": None,
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
        self._wait_for_detail_slot()
        self._record_attempt_start()
        lookup = record["lookup"]
        lookup["attempts"] += 1
        lookup["last_attempt_at"] = self._timestamp()
        lookup["next_attempt_at"] = self.manifest["request_pacing"]["next_detail_attempt_at"]
        lookup["last_response"] = None
        match_id = record["match_id"]
        try:
            response = self.fetch(f"{API_ROOT}/matches/{match_id}")
            payload, response_summary = self._prepare_response(
                response, label=f"match_{match_id}", attempt=lookup["attempts"]
            )
            if response_summary:
                lookup["last_response"] = response_summary
                status = response_summary["status"]
                if response_summary["request_error"] or status is None or not 200 <= status < 300:
                    raise ValueError(response_summary["request_error"] or f"unexpected HTTP status {status}")
                if response_summary["parse_error"]:
                    raise ValueError(response_summary["parse_error"])
            if not isinstance(payload, dict):
                raise ValueError("match-detail response must be a JSON object")
            if payload.get("match_id") != match_id:
                raise ValueError(
                    f"match-detail response has match_id={payload.get('match_id')!r}; expected {match_id}"
                )
            if "replay_url" not in payload:
                raise ValueError("match-detail response is missing replay_url")
            replay_url = payload["replay_url"]
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
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_response = lookup.get("last_response") or {}
            lookup.update(
                {
                    "status": "error",
                    "last_error": f"{type(exc).__name__}: {exc}",
                    "next_attempt_at": self.manifest["request_pacing"]["next_detail_attempt_at"],
                }
            )
            self.log(
                f"[OpenDota API] match_id={match_id} lookup failed: "
                f"{lookup['last_error']} body_preview="
                f"{last_response.get('body_preview', '<unavailable>')!r}"
            )
            self._save()
            return False
        self._save()
        return True

    def run(self) -> dict[str, int]:
        self.manifest = load_json(self.manifest_path, empty_manifest())
        _validate_manifest(self.manifest)

        source_response = self.fetch(PRO_MATCHES_URL)
        source_payload, source_summary = self._prepare_response(
            source_response, label="pro_matches"
        )
        if source_summary:
            status = source_summary["status"]
            if source_summary["request_error"] or status is None or not 200 <= status < 300:
                raise ValueError(source_summary["request_error"] or f"unexpected HTTP status {status}")
            if source_summary["parse_error"]:
                raise ValueError(source_summary["parse_error"])
        match_ids = self._add_snapshot(source_payload, source_summary)

        terminal = 0
        for record in self._pending_records():
            while not self._lookup_one(record):
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
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="directory for raw API response artifacts (default: <manifest-dir>/api_debug)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        with exclusive_lock(args.manifest.with_suffix(args.manifest.suffix + ".lock")):
            result = Collector(args.manifest, debug_dir=args.debug_dir).run()
    except KeyboardInterrupt:
        print("Interrupted; saved state can be resumed by rerunning this command.", file=sys.stderr)
        return 130
    except (ApiTransportError, HTTPError, URLError, OSError, ValueError, RuntimeError) as exc:
        print(f"Collector failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        "Collected pro-match snapshot "
        f"({result['snapshot_count']} IDs); {result['processed_to_terminal']} pending records reached a terminal state."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

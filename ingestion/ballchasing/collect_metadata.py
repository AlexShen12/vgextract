#!/usr/bin/env python3
"""Collect curated Ballchasing event metadata with durable checkpoints.

The collector intentionally makes one metadata API request every ten seconds.
It walks every child group of each catalog seed, indexes direct replays for each
group, and fetches detailed group/replay records. Rerunning the command resumes
the persisted task queue without repeating completed work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from ingestion.common.replay_state import atomic_write_bytes, atomic_write_json, exclusive_lock, load_json, parse_utc


API_ROOT = "https://ballchasing.com/api"
API_HOST = "ballchasing.com"
SCHEMA_VERSION = 1
REQUEST_INTERVAL_SECONDS = 10
RETRY_BASE_SECONDS = 60
BODY_PREVIEW_LIMIT = 4_000


class ConfigurationError(RuntimeError):
    """The API token or catalog prevents collection from continuing."""


class ApiTransportError(RuntimeError):
    """A request did not yield an HTTP response."""


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
    def body_preview(self) -> str:
        return self.body[:BODY_PREVIEW_LIMIT].decode("utf-8", errors="replace")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_response(url: str, status: int | None, headers: dict[str, str], body: bytes) -> ApiResponse:
    response = ApiResponse(url=url, status=status, headers=headers, body=body)
    try:
        response.parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        response.parse_error = f"{type(exc).__name__}: {exc}"
    return response


def fetch_api_response(url: str, headers: dict[str, str], timeout: float = 30.0) -> ApiResponse:
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return parse_response(
                url,
                getattr(response, "status", None) or response.getcode(),
                dict(response.headers.items()),
                response.read(),
            )
    except HTTPError as exc:
        try:
            body = exc.read()
        except OSError:
            body = b""
        response = parse_response(url, exc.code, dict(exc.headers.items()) if exc.headers else {}, body)
        response.request_error = f"HTTP {exc.code}: {exc.reason}"
        return response
    except (OSError, URLError, TimeoutError) as exc:
        raise ApiTransportError(f"{type(exc).__name__}: {exc}") from exc


def empty_state() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "next_task_order": 1,
        "source": {"catalog": None, "last_catalog_sync_at": None},
        "groups": {},
        "replays": {},
        "tasks": {},
    }


def empty_pacing_state() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "last_reserved_at": None, "next_allowed_at": None}


def _validate_state(state: dict[str, Any]) -> None:
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported or missing Ballchasing state schema_version")
    for key in ("groups", "replays", "tasks", "source"):
        if not isinstance(state.get(key), dict):
            raise ValueError(f"Ballchasing state {key} must be an object")


def _validate_pacing(state: dict[str, Any]) -> None:
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported or missing Ballchasing pacing schema_version")


def load_catalog(path: Path) -> list[dict[str, str]]:
    catalog = load_json(path, {})
    if catalog.get("schema_version") != SCHEMA_VERSION or not isinstance(catalog.get("groups"), list):
        raise ValueError("catalog requires schema_version=1 and a groups array")
    seeds: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in catalog["groups"]:
        if not isinstance(entry, dict) or entry.get("enabled", True) is False:
            continue
        group_id, event = entry.get("group_id"), entry.get("event")
        if not isinstance(group_id, str) or not group_id.strip() or group_id == "...":
            raise ValueError("every enabled catalog group requires a concrete non-empty group_id")
        if not isinstance(event, str) or not event.strip():
            raise ValueError(f"catalog group {group_id!r} requires a non-empty event")
        if group_id not in seen:
            seeds.append({"group_id": group_id, "event": event})
            seen.add(group_id)
    return seeds


def _safe_next_url(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("API next field must be a URL string or null")
    parsed = urlsplit(value)
    if parsed.scheme != "https" or parsed.hostname != API_HOST or not parsed.path.startswith("/api/"):
        raise ValueError("API next URL is outside the Ballchasing API origin")
    return value


def _response_summary(response: ApiResponse, artifact: Path) -> dict[str, Any]:
    return {
        "url": response.url,
        "status": response.status,
        "bytes": len(response.body),
        "artifact": str(artifact),
        "parse_error": response.parse_error,
        "request_error": response.request_error,
        "body_preview": response.body_preview,
    }


class PacingGate:
    """Persisted, cross-process reservation of Ballchasing request slots."""

    def __init__(self, path: Path, interval_seconds: int, now: Callable[[], datetime]) -> None:
        self.path = path
        self.interval_seconds = interval_seconds
        self.now = now

    def reserve_or_wait(self) -> float:
        with exclusive_lock(self.path.with_suffix(self.path.suffix + ".lock")):
            state = load_json(self.path, empty_pacing_state())
            _validate_pacing(state)
            current = self.now().astimezone(timezone.utc)
            deadline = state.get("next_allowed_at")
            if deadline:
                seconds = (parse_utc(deadline) - current).total_seconds()
                if seconds > 0:
                    return seconds
            state["last_reserved_at"] = isoformat(current)
            state["next_allowed_at"] = isoformat(current + timedelta(seconds=self.interval_seconds))
            atomic_write_json(self.path, state)
            return 0.0

    def extend_until(self, deadline: datetime) -> None:
        with exclusive_lock(self.path.with_suffix(self.path.suffix + ".lock")):
            state = load_json(self.path, empty_pacing_state())
            _validate_pacing(state)
            existing = state.get("next_allowed_at")
            existing_time = parse_utc(existing) if existing else None
            if existing_time is None or deadline > existing_time:
                state["next_allowed_at"] = isoformat(deadline)
                atomic_write_json(self.path, state)


class Collector:
    def __init__(
        self,
        catalog_path: Path,
        state_path: Path,
        debug_dir: Path,
        pacing_path: Path,
        token: str,
        *,
        fetch: Callable[[str, dict[str, str]], ApiResponse] = fetch_api_response,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = utc_now,
        interval_seconds: int = REQUEST_INTERVAL_SECONDS,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.catalog_path = catalog_path
        self.state_path = state_path
        self.debug_dir = debug_dir
        self.token = token
        self.fetch = fetch
        self.sleep = sleep
        self.now = now
        self.log = log or (lambda message: print(message, file=sys.stderr, flush=True))
        self.gate = PacingGate(pacing_path, interval_seconds, now)
        self.state: dict[str, Any] = {}

    def _save(self) -> None:
        atomic_write_json(self.state_path, self.state)

    def _timestamp(self) -> str:
        return isoformat(self.now())

    def _task_id(self, kind: str, url: str, target_id: str) -> str:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        return f"{kind}:{target_id}:{digest}"

    def _enqueue(self, kind: str, url: str, target_id: str) -> None:
        task_id = self._task_id(kind, url, target_id)
        if task_id in self.state["tasks"]:
            return
        order = self.state["next_task_order"]
        self.state["next_task_order"] = order + 1
        self.state["tasks"][task_id] = {
            "task_id": task_id,
            "kind": kind,
            "target_id": target_id,
            "url": url,
            "status": "pending",
            "attempts": 0,
            "order": order,
            "last_attempt_at": None,
            "next_attempt_at": None,
            "last_error": None,
            "last_response": None,
        }

    def _upsert_group(self, group_id: str, *, parent_id: str | None, event: str | None, seed: bool) -> None:
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("group response contains an invalid id")
        record = self.state["groups"].setdefault(
            group_id,
            {"group_id": group_id, "parent_ids": [], "events": [], "seed": False, "metadata": None},
        )
        if parent_id and parent_id not in record["parent_ids"]:
            record["parent_ids"].append(parent_id)
        if event and event not in record["events"]:
            record["events"].append(event)
        record["seed"] = bool(record["seed"] or seed)
        self._enqueue("group_detail", f"{API_ROOT}/groups/{group_id}", group_id)
        self._enqueue("group_children", f"{API_ROOT}/groups?group={group_id}&count=200", group_id)
        self._enqueue("group_replays", f"{API_ROOT}/replays?group={group_id}&count=200", group_id)

    def _upsert_replay(self, replay_id: str, group_id: str, summary: dict[str, Any]) -> None:
        if not isinstance(replay_id, str) or not replay_id:
            raise ValueError("replay response contains an invalid id")
        record = self.state["replays"].setdefault(
            replay_id,
            {"replay_id": replay_id, "group_ids": [], "summaries": {}, "metadata": None},
        )
        if group_id not in record["group_ids"]:
            record["group_ids"].append(group_id)
        record["summaries"][group_id] = summary
        self._enqueue("replay_detail", f"{API_ROOT}/replays/{replay_id}", replay_id)

    def _recover_interrupted(self) -> None:
        changed = False
        for task in self.state["tasks"].values():
            if task.get("status") == "in_progress":
                task["status"] = "retry"
                task["last_error"] = "interrupted before a response was persisted"
                task["next_attempt_at"] = None
                changed = True
        if changed:
            self._save()

    def _next_task(self) -> dict[str, Any] | None:
        current = self.now().astimezone(timezone.utc)
        candidates = []
        for task in self.state["tasks"].values():
            if task.get("status") not in {"pending", "retry"}:
                continue
            deadline = task.get("next_attempt_at")
            if deadline and parse_utc(deadline) > current:
                continue
            candidates.append(task)
        return min(candidates, key=lambda item: item["order"]) if candidates else None

    def _persist_response(self, task: dict[str, Any], response: ApiResponse) -> dict[str, Any]:
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        artifact = self.debug_dir / f"{task['task_id'].replace(':', '_')}_attempt_{task['attempts']}.json"
        atomic_write_bytes(artifact, response.body)
        summary = _response_summary(response, artifact)
        self.log(
            f"[Ballchasing API] GET {response.url} status={response.status} bytes={len(response.body)} artifact={artifact}"
        )
        return summary

    def _retry_task(self, task: dict[str, Any], error: str, delay_seconds: float) -> None:
        task.update(
            {
                "status": "retry",
                "last_error": error,
                "next_attempt_at": isoformat(self.now() + timedelta(seconds=delay_seconds)),
            }
        )
        self._save()

    def _complete_task(self, task: dict[str, Any]) -> None:
        task.update({"status": "complete", "last_error": None, "next_attempt_at": None})
        self._save()

    def _handle_payload(self, task: dict[str, Any], payload: Any) -> None:
        kind, target_id = task["kind"], task["target_id"]
        if kind == "group_detail":
            if not isinstance(payload, dict) or payload.get("id") != target_id:
                raise ValueError("group-detail response does not match its requested group")
            self.state["groups"][target_id]["metadata"] = payload
        elif kind == "group_children":
            if not isinstance(payload, dict) or not isinstance(payload.get("list"), list):
                raise ValueError("group-list response requires an object with a list")
            event = next(iter(self.state["groups"][target_id]["events"]), None)
            for child in payload["list"]:
                if not isinstance(child, dict):
                    raise ValueError("group-list entry must be an object")
                self._upsert_group(child.get("id"), parent_id=target_id, event=event, seed=False)
            next_url = _safe_next_url(payload.get("next"))
            if next_url:
                self._enqueue("group_children", next_url, target_id)
        elif kind == "group_replays":
            if not isinstance(payload, dict) or not isinstance(payload.get("list"), list):
                raise ValueError("replay-list response requires an object with a list")
            for replay in payload["list"]:
                if not isinstance(replay, dict):
                    raise ValueError("replay-list entry must be an object")
                self._upsert_replay(replay.get("id"), target_id, replay)
            next_url = _safe_next_url(payload.get("next"))
            if next_url:
                self._enqueue("group_replays", next_url, target_id)
        elif kind == "replay_detail":
            if not isinstance(payload, dict) or payload.get("id") != target_id:
                raise ValueError("replay-detail response does not match its requested replay")
            self.state["replays"][target_id]["metadata"] = payload
        else:
            raise ValueError(f"unsupported task kind {kind!r}")

    def _attempt(self, task: dict[str, Any]) -> str:
        while True:
            wait_seconds = self.gate.reserve_or_wait()
            if wait_seconds <= 0:
                break
            self.sleep(wait_seconds)
        task["status"] = "in_progress"
        task["attempts"] += 1
        task["last_attempt_at"] = self._timestamp()
        task["next_attempt_at"] = None
        self._save()
        try:
            response = self.fetch(task["url"], {"Authorization": self.token, "User-Agent": "ballchasing-event-collector/1.0"})
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            delay = RETRY_BASE_SECONDS * (2 ** min(task["attempts"] - 1, 4))
            self._retry_task(task, f"{type(exc).__name__}: {exc}", delay)
            return "retry"

        task["last_response"] = self._persist_response(task, response)
        status = response.status
        if status in {401, 403}:
            task.update({"status": "failed", "last_error": f"HTTP {status}: invalid or unauthorized API token"})
            self._save()
            raise ConfigurationError(task["last_error"])
        if status == 429:
            retry_after = response.headers.get("Retry-After") or response.headers.get("retry-after")
            try:
                delay = max(float(retry_after), RETRY_BASE_SECONDS) if retry_after else RETRY_BASE_SECONDS
            except ValueError:
                delay = RETRY_BASE_SECONDS
            self.gate.extend_until(self.now() + timedelta(seconds=delay))
            self._retry_task(task, "HTTP 429: rate limited", delay)
            return "retry"
        if status is None or status >= 500:
            delay = RETRY_BASE_SECONDS * (2 ** min(task["attempts"] - 1, 4))
            self._retry_task(task, response.request_error or f"unexpected HTTP status {status}", delay)
            return "retry"
        if status == 404:
            task.update({"status": "unavailable", "last_error": "HTTP 404: resource unavailable", "next_attempt_at": None})
            self._save()
            return "failed"
        if not 200 <= status < 300:
            task.update({"status": "failed", "last_error": f"unexpected HTTP status {status}", "next_attempt_at": None})
            self._save()
            return "failed"
        if response.parse_error:
            delay = RETRY_BASE_SECONDS * (2 ** min(task["attempts"] - 1, 4))
            self._retry_task(task, response.parse_error, delay)
            return "retry"
        try:
            self._handle_payload(task, response.parsed)
        except Exception as exc:
            task.update({"status": "failed", "last_error": f"{type(exc).__name__}: {exc}", "next_attempt_at": None})
            self._save()
            return "failed"
        self._complete_task(task)
        return "complete"

    def run(self, max_requests: int | None = None) -> dict[str, int]:
        self.state = load_json(self.state_path, empty_state())
        _validate_state(self.state)
        self._recover_interrupted()
        seeds = load_catalog(self.catalog_path)
        if not seeds:
            raise ConfigurationError(
                f"catalog {self.catalog_path} has no enabled groups; "
                "add at least one verified Ballchasing group_id before submitting the job"
            )
        self.state["source"].update({"catalog": str(self.catalog_path), "last_catalog_sync_at": self._timestamp()})
        for seed in seeds:
            self._upsert_group(seed["group_id"], parent_id=None, event=seed["event"], seed=True)
        self._save()

        attempted = completed = retried = failed = 0
        while max_requests is None or attempted < max_requests:
            task = self._next_task()
            if task is None:
                break
            outcome = self._attempt(task)
            attempted += 1
            completed += outcome == "complete"
            retried += outcome == "retry"
            failed += outcome == "failed"
        pending = sum(1 for task in self.state["tasks"].values() if task["status"] in {"pending", "retry", "in_progress"})
        return {"attempted": attempted, "completed": completed, "retried": retried, "failed": failed, "pending": pending}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path(__file__).with_name("professional_groups.json"))
    parser.add_argument("--state", type=Path, default=Path("data/ballchasing/metadata_state.json"))
    parser.add_argument("--debug-dir", type=Path, default=Path("data/ballchasing/api_debug"))
    parser.add_argument("--pacing-state", type=Path, default=Path("data/ballchasing/api_pacing.json"))
    parser.add_argument("--max-requests", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_requests is not None and args.max_requests < 0:
        print("--max-requests must be zero or greater", file=sys.stderr)
        return 2
    token = os.environ.get("BALLCHASING_API_TOKEN")
    if not token:
        print("BALLCHASING_API_TOKEN is required", file=sys.stderr)
        return 2
    try:
        with exclusive_lock(args.state.with_suffix(args.state.suffix + ".lock")):
            result = Collector(args.catalog, args.state, args.debug_dir, args.pacing_state, token).run(args.max_requests)
    except KeyboardInterrupt:
        print("Interrupted; durable state can be resumed by rerunning this command.", file=sys.stderr)
        return 130
    except (ConfigurationError, ApiTransportError, OSError, RuntimeError, ValueError) as exc:
        print(f"Collector failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("Ballchasing metadata: " + ", ".join(f"{key}={value}" for key, value in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

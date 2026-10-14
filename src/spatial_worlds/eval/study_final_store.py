"""Small append-only store for the active formal collection.

The historical ledger is reused without altering or dropping any legacy table.
Only two compact state tables are added for the final run.  Every provider call
is an ``attempt`` row; the first usable model outcome for a job is the only one
that may be scored.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class FormalRunError(ValueError):
    """The run cannot safely make another provider call."""


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def digest(value: Any) -> str:
    import hashlib

    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def microdollars(value: float) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("cost must be finite")
    if value < 0:
        raise ValueError("cost must be nonnegative")
    return math.ceil(value * 1_000_000)


def job_key(job: Mapping[str, Any]) -> str:
    return digest([job["condition_id"], job["instance_id"], job["prompt_sha256"]])


def observed_cost(result: Mapping[str, Any], pricing: Mapping[str, Any]) -> int | None:
    if result.get("usage_complete") is not True:
        return None
    usage = result.get("usage") or {}
    values = [usage.get("input_tokens"), usage.get("output_tokens")]
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
        return None
    assert isinstance(values[0], int)
    assert isinstance(values[1], int)
    return math.ceil(
        values[0] * float(pricing["input_per_million"])
        + values[1] * float(pricing["output_per_million"])
    )


class FinalStore:
    """SQLite state with one active writer per campaign and atomic shared transactions."""

    def __init__(self, path: Path, campaign_id: str, *, readonly: bool = False) -> None:
        self.path = path
        self.campaign_id = campaign_id
        self.readonly = readonly
        self.lock = threading.RLock()
        self.lock_file = None
        self.legacy_lock_file = None
        if readonly:
            self.db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        else:
            import fcntl

            legacy_lock = path.with_suffix(".lock")
            campaign_lock = path.with_name(f"{path.stem}.{digest(campaign_id)[:16]}.lock")
            if any(item.is_symlink() for item in (path, legacy_lock, campaign_lock)):
                raise FormalRunError("refusing a symlinked ledger or lock")
            path.parent.mkdir(parents=True, exist_ok=True)
            # Older runners take an exclusive lock here. Shared holders may run
            # different campaigns, but never overlap an older global-lock runner.
            self.legacy_lock_file = legacy_lock.open("a+")
            try:
                fcntl.flock(self.legacy_lock_file.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                self.lock_file = campaign_lock.open("a+")
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                if self.lock_file is not None:
                    self.lock_file.close()
                self.legacy_lock_file.close()
                raise FormalRunError(
                    "another formal writer owns this campaign or legacy ledger"
                ) from error
            self.db = sqlite3.connect(path, check_same_thread=False, timeout=30)
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY,
                    scope TEXT NOT NULL,
                    job_key TEXT NOT NULL,
                    model TEXT NOT NULL,
                    number INTEGER NOT NULL,
                    started TEXT NOT NULL,
                    reservation INTEGER NOT NULL,
                    accounted INTEGER NOT NULL,
                    known_upper INTEGER,
                    job TEXT NOT NULL,
                    result TEXT,
                    account_group TEXT,
                    UNIQUE(job_key, number)
                );
                CREATE TABLE IF NOT EXISTS formal_account_groups (
                    name TEXT PRIMARY KEY,
                    baseline_spend REAL NOT NULL,
                    max_budget REAL NOT NULL,
                    budget_reset_at TEXT,
                    buffer REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY,
                    attempt_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    event TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    document TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS formal_final_runs (
                    campaign_id TEXT PRIMARY KEY,
                    manifest_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    pause_reason TEXT,
                    started_at TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS formal_final_models (
                    campaign_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    pause_reason TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(campaign_id, model)
                );
                """
            )
            columns = {row[1] for row in self.db.execute("PRAGMA table_info(attempts)")}
            if "account_group" not in columns:
                self.db.execute("ALTER TABLE attempts ADD COLUMN account_group TEXT")
        self.db.row_factory = sqlite3.Row

    def __enter__(self) -> FinalStore:
        return self

    def __exit__(self, *_: Any) -> None:
        self.db.close()
        if self.lock_file is not None:
            self.lock_file.close()
        if self.legacy_lock_file is not None:
            self.legacy_lock_file.close()

    def _audit(self, action: str, subject: str, document: Mapping[str, Any]) -> None:
        self.db.execute(
            "INSERT INTO audit(timestamp,action,subject,document) VALUES (?,?,?,?)",
            (timestamp(), action, subject, json.dumps(dict(document), ensure_ascii=False)),
        )

    def metadata(self, key: str) -> Any | None:
        with self.lock:
            row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return None if row is None else json.loads(row["value"])

    def ensure_run(self, manifest_sha256: str, models: Iterable[str]) -> None:
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute(
                "SELECT * FROM formal_final_runs WHERE campaign_id=?", (self.campaign_id,)
            ).fetchone()
            now = timestamp()
            if row is None:
                self.db.execute(
                    "INSERT INTO formal_final_runs VALUES (?,?,?,?,?,?)",
                    (self.campaign_id, manifest_sha256, "ready", None, None, now),
                )
                self.db.executemany(
                    "INSERT INTO formal_final_models VALUES (?,?,?,?,?)",
                    [(self.campaign_id, model, "active", None, now) for model in models],
                )
                self._audit(
                    "formal_final_created",
                    self.campaign_id,
                    {"manifest_sha256": manifest_sha256},
                )
            elif row["manifest_sha256"] != manifest_sha256:
                raise FormalRunError("saved final manifest differs from the current manifest")

    def ensure_account_group(
        self, name: str, account: Mapping[str, Any], buffer_usd: float
    ) -> None:
        """Freeze a pre-expansion account baseline shared by concurrent campaigns."""
        if not name or not math.isfinite(buffer_usd) or buffer_usd < 0:
            raise FormalRunError("invalid shared account group")
        spend = float(account["spend_usd"])
        maximum = float(account["max_budget_usd"])
        reset_at = account.get("budget_reset_at")
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute(
                "SELECT * FROM formal_account_groups WHERE name=?", (name,)
            ).fetchone()
            if row is None:
                self.db.execute(
                    "INSERT INTO formal_account_groups VALUES (?,?,?,?,?)",
                    (name, spend, maximum, reset_at, buffer_usd),
                )
            elif row["budget_reset_at"] != reset_at or row["buffer"] != buffer_usd:
                raise FormalRunError("shared account reset or buffer changed")
            elif maximum > row["max_budget"]:
                self.db.execute(
                    "UPDATE formal_account_groups SET max_budget=? WHERE name=?",
                    (maximum, name),
                )

    def run_state(self) -> dict[str, Any] | None:
        with self.lock:
            try:
                row = self.db.execute(
                    "SELECT * FROM formal_final_runs WHERE campaign_id=?", (self.campaign_id,)
                ).fetchone()
            except sqlite3.OperationalError:
                return None
        return None if row is None else dict(row)

    def model_states(self) -> dict[str, dict[str, Any]]:
        with self.lock:
            try:
                rows = list(
                    self.db.execute(
                        "SELECT * FROM formal_final_models WHERE campaign_id=?",
                        (self.campaign_id,),
                    )
                )
            except sqlite3.OperationalError:
                return {}
        return {row["model"]: dict(row) for row in rows}

    def set_running(self, models: Iterable[str]) -> None:
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.run_state()
            if row is None:
                raise FormalRunError("final run has not been initialized")
            if row["status"] == "complete":
                return
            now = timestamp()
            self.db.execute(
                "UPDATE formal_final_runs SET status='running',pause_reason=NULL,"
                "started_at=COALESCE(started_at,?),updated_at=? WHERE campaign_id=?",
                (now, now, self.campaign_id),
            )
            self.db.executemany(
                "UPDATE formal_final_models SET status='active',pause_reason=NULL,updated_at=? "
                "WHERE campaign_id=? AND model=? AND status!='complete'",
                [(now, self.campaign_id, model) for model in models],
            )
            self._audit("formal_final_resumed", self.campaign_id, {})

    def set_run_state(self, status: str, reason: str | None = None) -> None:
        if status not in {"ready", "running", "paused", "complete"}:
            raise ValueError("unknown final run state")
        with self.lock, self.db:
            now = timestamp()
            self.db.execute(
                "UPDATE formal_final_runs SET status=?,pause_reason=?,updated_at=? "
                "WHERE campaign_id=?",
                (status, reason, now, self.campaign_id),
            )
            self._audit(
                "formal_final_state", self.campaign_id, {"status": status, "reason": reason}
            )

    def set_model_state(self, model: str, status: str, reason: str | None = None) -> None:
        if status not in {"active", "paused", "complete"}:
            raise ValueError("unknown final model state")
        with self.lock, self.db:
            now = timestamp()
            self.db.execute(
                "UPDATE formal_final_models SET status=?,pause_reason=?,updated_at=? "
                "WHERE campaign_id=? AND model=?",
                (status, reason, now, self.campaign_id, model),
            )
            self._audit("formal_final_model_state", model, {"status": status, "reason": reason})

    def paused(self, model: str) -> str | None:
        with self.lock:
            try:
                row = self.db.execute(
                    "SELECT status,pause_reason FROM formal_final_models "
                    "WHERE campaign_id=? AND model=?",
                    (self.campaign_id, model),
                ).fetchone()
            except sqlite3.OperationalError:
                return None
        return row["pause_reason"] if row is not None and row["status"] == "paused" else None

    def all_attempts(self, scope: str) -> list[dict[str, Any]]:
        with self.lock:
            rows = list(
                self.db.execute("SELECT * FROM attempts WHERE scope=? ORDER BY id", (scope,))
            )
        return [dict(row) for row in rows]

    def history(self, job: Mapping[str, Any]) -> list[dict[str, Any]]:
        with self.lock:
            rows = list(
                self.db.execute(
                    "SELECT * FROM attempts WHERE job_key=? ORDER BY number,id", (job_key(job),)
                )
            )
        return [dict(row) for row in rows]

    def inflight(self, scope: str | None = None) -> list[dict[str, Any]]:
        with self.lock:
            if scope is None:
                rows = list(
                    self.db.execute("SELECT * FROM attempts WHERE result IS NULL ORDER BY id")
                )
            else:
                rows = list(
                    self.db.execute(
                        "SELECT * FROM attempts WHERE scope=? AND result IS NULL ORDER BY id",
                        (scope,),
                    )
                )
        return [dict(row) for row in rows]

    def locked_result(self, attempt_id: int) -> dict[str, Any] | None:
        with self.lock:
            rows = list(
                self.db.execute(
                    "SELECT event FROM events WHERE attempt_id=? ORDER BY id", (attempt_id,)
                )
            )
        locked = None
        usage = None
        trace = None
        for row in rows:
            event = json.loads(row["event"])
            if event.get("kind") == "answer_locked":
                locked = dict(event["value"])
            elif event.get("kind") == "usage" and event.get("final"):
                usage = dict(event.get("value") or {})
            elif event.get("kind") == "trace_summary":
                trace = dict(event.get("value") or {})
        if locked is None:
            return None
        if usage is not None:
            locked["usage"] = usage
            locked["usage_complete"] = all(
                isinstance(usage.get(key), int) and not isinstance(usage.get(key), bool)
                for key in ("input_tokens", "output_tokens")
            )
        if trace is not None:
            locked["trace"] = trace
            locked["trace_error"] = None
        return locked

    def begin_batch(
        self,
        scope: str,
        jobs: list[dict[str, Any]],
        reservations: list[int],
        *,
        formal_scopes: Iterable[str],
        formal_cap: int,
        per_model_concurrency: int,
        total_concurrency: int,
        shared_account: Mapping[str, Any] | None = None,
        live_account: Mapping[str, Any] | None = None,
    ) -> list[int]:
        formal_scopes = tuple(formal_scopes)
        if (
            not jobs
            or len(jobs) != len(reservations)
            or not isinstance(per_model_concurrency, int)
            or not isinstance(total_concurrency, int)
            or isinstance(per_model_concurrency, bool)
            or isinstance(total_concurrency, bool)
            or per_model_concurrency < 1
            or total_concurrency < per_model_concurrency
            or len(jobs) > total_concurrency
        ):
            raise FormalRunError("invalid final dispatch batch")
        if len({job_key(job) for job in jobs}) != len(jobs):
            raise FormalRunError("duplicate job in final dispatch batch")
        if any(value <= 0 for value in reservations):
            raise FormalRunError("invalid request reservation")
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            run = self.run_state()
            if run is None or run["status"] != "running":
                raise FormalRunError("final run is not active")
            inflight = self.inflight(scope)
            if len(inflight) + len(jobs) > total_concurrency:
                raise FormalRunError("final dispatch exceeds total concurrency")
            # Limits are campaign-local; identity collisions remain global.
            active_keys = {row["job_key"] for row in self.inflight()}
            if active_keys & {job_key(job) for job in jobs}:
                raise FormalRunError("formal job is already in flight")
            active_by_model = Counter(row["model"] for row in inflight)
            active_by_model.update(job["model"] for job in jobs)
            # A capacity fallback can lower the cap while earlier high-stage
            # calls are still draining. Those calls may temporarily exceed
            # the new cap, but must not prevent dispatch for another model.
            if any(
                active_by_model[model] > per_model_concurrency
                for model in {job["model"] for job in jobs}
            ):
                raise FormalRunError("final dispatch exceeds per-model concurrency")
            states = self.model_states()
            if any(states.get(job["model"], {}).get("status") != "active" for job in jobs):
                raise FormalRunError("dispatch batch contains an inactive model")
            placeholders = ",".join("?" for _ in formal_scopes)
            accounted = self.db.execute(
                f"SELECT COALESCE(SUM(accounted),0) FROM attempts WHERE scope IN ({placeholders})",
                tuple(formal_scopes),
            ).fetchone()[0]
            if accounted + sum(reservations) > formal_cap:
                raise FormalRunError(
                    f"formal {formal_cap / 1_000_000:g} USD ceiling "
                    "cannot fit the next dispatch batch"
                )
            group_name = None
            if shared_account is not None:
                if live_account is None:
                    raise FormalRunError("shared account requires a live account snapshot")
                group_name = str(shared_account["group"])
                group = self.db.execute(
                    "SELECT * FROM formal_account_groups WHERE name=?", (group_name,)
                ).fetchone()
                if group is None or group["budget_reset_at"] != live_account.get(
                    "budget_reset_at"
                ):
                    raise FormalRunError("shared account baseline is missing or reset")
                spent = self.db.execute(
                    "SELECT COALESCE(SUM(accounted),0) FROM attempts WHERE account_group=?",
                    (group_name,),
                ).fetchone()[0]
                ceiling = max(float(group["max_budget"]), float(live_account["max_budget_usd"]))
                projected = max(
                    float(live_account["spend_usd"]),
                    float(group["baseline_spend"]) + spent / 1_000_000,
                ) + sum(reservations) / 1_000_000
                if projected + float(group["buffer"]) > ceiling + 1e-9:
                    raise FormalRunError(
                        "shared account ceiling cannot fit the next dispatch batch"
                    )
            now = timestamp()
            ids = []
            for job, reservation in zip(jobs, reservations, strict=True):
                key = job_key(job)
                number = self.db.execute(
                    "SELECT COALESCE(MAX(number),0)+1 FROM attempts WHERE job_key=?", (key,)
                ).fetchone()[0]
                cursor = self.db.execute(
                    "INSERT INTO attempts(scope,job_key,model,number,started,reservation,"
                    "accounted,job,account_group) VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        scope,
                        key,
                        job["model"],
                        number,
                        now,
                        reservation,
                        reservation,
                        json.dumps(job, ensure_ascii=False),
                        group_name,
                    ),
                )
                assert cursor.lastrowid is not None
                ids.append(cursor.lastrowid)
            self._audit(
                "formal_final_batch_started",
                self.campaign_id,
                {
                    "attempt_ids": ids,
                    "models": [job["model"] for job in jobs],
                    "per_model_concurrency": per_model_concurrency,
                    "total_concurrency": total_concurrency,
                },
            )
            return ids

    def record_dispatch(self, document: Mapping[str, Any]) -> None:
        with self.lock, self.db:
            self._audit("formal_final_dispatch", self.campaign_id, document)

    def event(self, attempt_id: int, event: Mapping[str, Any]) -> None:
        if event.get("kind") not in {
            "answer_locked",
            "usage",
            "trace_summary",
            "trace_error",
            "error",
            "transport_error",
            "protocol_error",
            "cancelled",
        }:
            return
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO events(attempt_id,timestamp,event) VALUES (?,?,?)",
                (attempt_id, timestamp(), json.dumps(dict(event), ensure_ascii=False)),
            )

    def finish(self, attempt_id: int, result: Mapping[str, Any]) -> None:
        with self.lock, self.db:
            row = self.db.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone()
            if row is None or row["result"] is not None:
                raise FormalRunError("attempt is missing or already finalized")
            job = json.loads(row["job"])
            pricing = job["collection"]["pricing"]
            known = observed_cost(result, pricing)
            self.db.execute(
                "UPDATE attempts SET result=?,known_upper=?,accounted=? WHERE id=?",
                (
                    json.dumps(dict(result), ensure_ascii=False),
                    known,
                    row["reservation"] if known is None else known,
                    attempt_id,
                ),
            )
            self._audit(
                "formal_final_attempt_finished",
                str(attempt_id),
                {
                    "model": row["model"],
                    "known_cost_usd": None if known is None else known / 1_000_000,
                    "answer_eligible": bool(result.get("answer_eligible")),
                    "finish_reason": result.get("finish_reason"),
                },
            )

    def accounting(self, scopes: Iterable[str]) -> dict[str, Any]:
        values = tuple(scopes)
        placeholders = ",".join("?" for _ in values)
        with self.lock:
            row = self.db.execute(
                f"SELECT COUNT(*) n,COALESCE(SUM(accounted),0) accounted,"
                f"COALESCE(SUM(known_upper),0) known,"
                f"COALESCE(SUM(CASE WHEN known_upper IS NULL THEN accounted ELSE 0 END),0) unknown "
                f"FROM attempts WHERE scope IN ({placeholders})",
                values,
            ).fetchone()
        assert row is not None
        return {
            "attempts": row["n"],
            "accounted_upper_usd": row["accounted"] / 1_000_000,
            "known_usage_cost_upper_usd": row["known"] / 1_000_000,
            "unknown_or_inflight_reserved_usd": row["unknown"] / 1_000_000,
        }

    def record_account(self, account: Mapping[str, Any]) -> None:
        with self.lock, self.db:
            self._audit("formal_final_account", self.campaign_id, account)

    def latest_account(self) -> dict[str, Any] | None:
        with self.lock:
            try:
                row = self.db.execute(
                    "SELECT document FROM audit WHERE action='formal_final_account' AND subject=? "
                    "ORDER BY id DESC LIMIT 1",
                    (self.campaign_id,),
                ).fetchone()
            except sqlite3.OperationalError:
                return None
        return None if row is None else json.loads(row["document"])

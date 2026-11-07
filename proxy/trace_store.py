"""Best-effort SQLite writer on a bounded queue; no disk I/O on inference path."""
import asyncio
from contextlib import closing
import json
import logging
from pathlib import Path
import queue
import sqlite3
import threading
import time

log = logging.getLogger(__name__)


class TraceStore:
    def __init__(self, path: Path):
        self.path = path
        self.queue = queue.Queue(maxsize=2048)
        self.dropped = 0
        self.errors = 0
        self.thread = None
        self.lock = threading.Lock()
        self.config = None
        self.generation = 0
        self.request_generation = 0

    def configure(self, config):
        if not config.traceEnabled:
            self.generation += 1
        if not config.traceAllRequests:
            self.request_generation += 1
        self.config = config
        self.submit("configure", {}, config)

    def submit(self, kind, record, config):
        if kind != "configure" and not config.traceEnabled:
            return
        if self.config is None:
            self.config = config
        if self.thread is None:
            with self.lock:
                if self.thread is None:
                    self.thread = threading.Thread(target=self._worker, daemon=True)
                    self.thread.start()
        try:
            self.queue.put_nowait((kind, record, config, self.generation))
        except queue.Full:
            self.dropped += 1

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=.1)
        # Reserve half of the quota for SQLite's rollback journal. DELETE mode
        # avoids an independently growing WAL while dashboard readers are active.
        db.execute("PRAGMA journal_mode=DELETE")
        db.execute("PRAGMA auto_vacuum=FULL")
        db.executescript('''
            CREATE TABLE IF NOT EXISTS tools (
                session TEXT, id TEXT, timestamp REAL, record TEXT,
                PRIMARY KEY(session, id));
            CREATE TABLE IF NOT EXISTS loops (
                id TEXT PRIMARY KEY, timestamp REAL, record TEXT);
            CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER);
            CREATE TABLE IF NOT EXISTS requests (
                id TEXT PRIMARY KEY, timestamp REAL, record TEXT);
            CREATE INDEX IF NOT EXISTS tools_time ON tools(timestamp);
            CREATE INDEX IF NOT EXISTS loops_time ON loops(timestamp);
            CREATE INDEX IF NOT EXISTS requests_time ON requests(timestamp);
        ''')
        db.commit()
        if db.execute("PRAGMA auto_vacuum").fetchone()[0] != 1:
            db.execute("VACUUM")
        return db

    def disk_bytes(self):
        total = 0
        for suffix in ("", "-journal", "-wal", "-shm"):
            try:
                total += Path(str(self.path) + suffix).stat().st_size
            except FileNotFoundError:
                pass
        return total

    def _limit(self, db, config):
        # Space for journal page headers and filesystem-independent SQLite data.
        budget = (config.maxDiskMB * 1024 * 1024 - 65536) // 2
        page_size = db.execute("PRAGMA page_size").fetchone()[0]
        pages = max(16, budget // page_size)
        self._prune(db, config)
        db.commit()
        # Shrink pre-existing databases and immediately apply a lowered quota.
        while db.execute("PRAGMA page_count").fetchone()[0] > pages:
            if not self._evict(db):
                break
        db.execute(f"PRAGMA max_page_count={pages}")

    @staticmethod
    def _evict(db):
        oldest = []
        for table in ("tools", "loops", "requests"):
            row = db.execute(f"SELECT rowid, timestamp FROM {table} ORDER BY timestamp LIMIT 1").fetchone()
            if row:
                oldest.append((row[1], table, row[0]))
        if not oldest:
            return False
        _, table, rowid = min(oldest)
        db.execute(f"DELETE FROM {table} WHERE rowid=?", (rowid,))
        db.commit()
        return True

    def _write(self, db, kind, record, config):
        if kind == "tool":
            if config.storeToolHistory:
                existing = db.execute("SELECT record FROM tools WHERE session=? AND id=?",
                                      (record["session"], record["id"])).fetchone()
                merged = {**json.loads(existing[0]), **record} if existing else record
                if not config.traceContent:
                    merged = {k: v for k, v in merged.items() if k not in {
                        "arguments_content", "result_content", "arguments_preview", "result_preview"}}
                db.execute("INSERT OR REPLACE INTO tools VALUES (?, ?, ?, ?)",
                           (record["session"], record["id"], record["timestamp"], json.dumps(merged)))
                increment = not existing
            else:
                increment = record.get("new", False)
            if increment:
                self._count(db, "tool_calls")
        else:
            table = "requests" if kind == "request" else "loops"
            exists = db.execute(f"SELECT 1 FROM {table} WHERE id=?", (record["id"],)).fetchone()
            db.execute(f"INSERT OR REPLACE INTO {table} VALUES (?, ?, ?)",
                       (record["id"], record["timestamp"], json.dumps(record)))
            if not exists:
                self._count(db, table)

    def _worker(self):
        db = None
        config = None
        applied_limits = None
        while True:
            try:
                item = self.queue.get(timeout=30)
            except queue.Empty:
                if db is not None and config is not None:
                    try:
                        self._prune(db, config)
                        db.commit()
                    except sqlite3.Error:
                        self.errors += 1
                        db.close()
                        db = None
                continue
            try:
                if item is None:
                    return
                kind, record, queued_config, generation = item
                config = self.config or queued_config
                if db is None:
                    db = self._connect()
                    applied_limits = None
                limits = (config.maxDiskMB, config.retentionDays, config.maxTraceRows)
                if limits != applied_limits:
                    self._limit(db, config)
                    applied_limits = limits
                if kind == "configure":
                    continue
                if not config.traceEnabled or generation != self.generation:
                    continue
                while True:
                    try:
                        self._write(db, kind, record, config)
                        break
                    except sqlite3.OperationalError as exc:
                        db.rollback()
                        if getattr(exc, "sqlite_errorcode", None) != sqlite3.SQLITE_FULL:
                            raise
                        # A single record that cannot fit is dropped; inference
                        # always continues. Otherwise discard the oldest first.
                        if len(json.dumps(record).encode()) > (config.maxDiskMB * 1024 * 1024 - 65536) // 2 - 65536:
                            self.dropped += 1
                            break
                        if not self._evict(db):
                            self.dropped += 1
                            break
                # Enforce the row cap on every write; expire old content also
                # while inference is idle, using the queue timeout above.
                self._prune(db, config)
                db.commit()
            except Exception:
                self.errors += 1
                if self.errors == 1:
                    log.warning("Traces SQLite indisponibles ; requêtes maintenues")
                if db is not None:
                    db.close()
                    db = None
            finally:
                self.queue.task_done()
                if item is None and db is not None:
                    db.close()

    @staticmethod
    def _prune(db, config):
        for table in ("tools", "loops", "requests"):
            db.execute(f"DELETE FROM {table} WHERE timestamp < ?", (time.time() - config.retentionDays * 86400,))
            db.execute(f"DELETE FROM {table} WHERE rowid IN (SELECT rowid FROM {table} ORDER BY timestamp DESC LIMIT -1 OFFSET ?)", (config.maxTraceRows,))

    @staticmethod
    def _count(db, name):
        db.execute("INSERT INTO counters VALUES (?, 1) ON CONFLICT(name) DO UPDATE SET value=value+1", (name,))

    def read(self, limit=30, event_id=None, kind="loops"):
        if not self.path.exists():
            return {"tool_calls": 0, "loops": 0, "requests": 0, "recent": []}
        try:
            with closing(sqlite3.connect(self.path.absolute().as_uri() + "?mode=ro", uri=True, timeout=.1)) as db:
                counters = dict(db.execute("SELECT name, value FROM counters"))
                if event_id:
                    if event_id.startswith("tool:"):
                        if event_id.count(":") != 2:
                            return None
                        _, session, call_id = event_id.split(":", 2)
                        row = db.execute("SELECT record FROM tools WHERE session=? AND id=?", (session, call_id)).fetchone()
                    else:
                        row = db.execute("SELECT record FROM loops WHERE id=?", (event_id,)).fetchone()
                        if row is None:
                            row = db.execute("SELECT record FROM requests WHERE id=?", (event_id,)).fetchone()
                    return json.loads(row[0]) if row else None
                recent = []
                for table in ("tools", "loops", "requests"):
                    if kind not in {"all", table}:
                        continue
                    for row in db.execute(f"SELECT record FROM {table} ORDER BY timestamp DESC LIMIT ?", (limit,)):
                        item = json.loads(row[0])
                        if table == "tools":
                            item.update(id="tool:" + item["session"] + ":" + item["id"], kind="tool", tools=[item["name"]])
                        recent.append({k: item[k] for k in ("id", "timestamp", "model", "tools", "kind", "intervention", "status", "path") if k in item})
                return {"tool_calls": counters.get("tool_calls", 0), "loops": counters.get("loops", 0),
                        "requests": counters.get("requests", 0),
                        "recent": sorted(recent, key=lambda r: r["timestamp"], reverse=True)[:limit]}
        except sqlite3.Error:
            return {"tool_calls": 0, "loops": 0, "requests": 0, "recent": [], "unavailable": True}

    async def close(self):
        if self.thread is not None:
            await asyncio.to_thread(self.queue.join)
            self.queue.put_nowait(None)
            await asyncio.to_thread(self.thread.join)
            self.thread = None

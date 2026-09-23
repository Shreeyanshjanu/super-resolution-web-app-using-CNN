from __future__ import annotations
import json
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

from backend.services.storage import object_directory, remove_object

LOG = logging.getLogger(__name__)
ACTIVE = ("preparing", "queued", "processing")


class QueueFullError(ValueError):
    pass


class JobManager:
    """SQLite job records and a bounded, single-model executor."""

    def __init__(self, srm_service, jobs_directory="outputs/app/jobs", max_pending=8):
        self.srm_service = srm_service
        self.jobs_directory = Path(jobs_directory).resolve()
        self.jobs_directory.mkdir(parents=True, exist_ok=True)
        self.database = self.jobs_directory.parent / "jobs.sqlite3"
        self.max_pending = max_pending
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ldsrs2")
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, status TEXT NOT NULL, payload TEXT NOT NULL)")

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database, timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def recover(self):
        """Queued work resumes; an interrupted inference is recorded as failed."""
        for job in self.list_jobs():
            if job["status"] in ("processing", "preparing"):
                self.update_job(job["job_id"], status="failed", completed_at=self._now(),
                                error="Server stopped during this job. Select the area and submit again.",
                                message="Interrupted by server restart.")
            elif job["status"] == "queued":
                self.enqueue(job["job_id"])

    def create_job(self, filename, **details):
        with self._lock, self._connect() as db:
            count = db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('preparing','queued','processing')").fetchone()[0]
            if count >= self.max_pending:
                raise QueueFullError("The CPU queue is full. Wait for a job to finish.")
            identifier = uuid4().hex
            directory = object_directory(self.jobs_directory, identifier)
            directory.mkdir()
            job = dict(job_id=identifier, filename=filename, status="preparing", progress=0,
                       patches_completed=0, message="Preparing selected area.", created_at=self._now(),
                       completed_at=None, output_path=None, error=None, **details)
            db.execute("INSERT INTO jobs VALUES (?,?,?)", (identifier, job["status"], json.dumps(job)))
        return identifier

    def get_job(self, job_id):
        with self._connect() as db:
            row = db.execute("SELECT payload FROM jobs WHERE id=?", (job_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def list_jobs(self):
        with self._connect() as db:
            rows = db.execute("SELECT payload FROM jobs ORDER BY rowid DESC").fetchall()
        return [json.loads(row[0]) for row in rows]

    def update_job(self, job_id, **changes):
        with self._lock, self._connect() as db:
            row = db.execute("SELECT payload FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                return
            job = json.loads(row[0])
            job.update(changes)
            db.execute("UPDATE jobs SET status=?,payload=? WHERE id=?", (job["status"], json.dumps(job), job_id))

    def enqueue(self, job_id):
        self._executor.submit(self.run_job, job_id)

    def run_job(self, job_id):
        job = self.get_job(job_id)
        if job is None or job["status"] != "queued":
            return
        self.update_job(job_id, status="processing", progress=5, started_at=self._now(),
                        message=f"Running {job['plan']['patches']} selected-area patches.")
        try:
            def progress(done, total):
                self.update_job(job_id, patches_completed=done, progress=5 + int(90 * done / total),
                                message=f"Completed patch {done} of {total}.")
            result = self.srm_service.process(
                input_path=Path(job["input_path"]),
                job_directory=object_directory(self.jobs_directory, job_id),
                plan=job["plan"], progress_callback=progress,
                sampling_steps=job.get("sampling_steps", 20),
            )
            self.update_job(job_id, status="completed", progress=100, completed_at=self._now(),
                            message="Selected-area super-resolution completed.", **result)
        except Exception as exc:
            LOG.exception("SR job %s failed", job_id)
            self.update_job(job_id, status="failed", error=str(exc), completed_at=self._now(),
                            message="Super-resolution failed.")

    def delete_job(self, job_id):
        with self._lock:
            job = self.get_job(job_id)
            if job is None:
                return False
            if job["status"] in ACTIVE:
                raise ValueError("An active job cannot be deleted.")
            remove_object(self.jobs_directory, job_id)
            with self._connect() as db:
                db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            return True

    def shutdown(self):
        # Pending records remain queued and are restored next startup.
        self._executor.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

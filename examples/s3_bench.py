"""Crash test for examples/s3_ckpt_job.py against an S3 API.

Default: moto server on 127.0.0.1:5005. Start it first, in its own terminal:  python -m moto.server -p 5005
Real AWS:  S3_ENDPOINT=aws AWS_PROFILE=<profile> python examples/s3_bench.py
Other S3 API:  S3_ENDPOINT=http://host:port python examples/s3_bench.py

Every PutObject is a crash point: the job is killed just before that request leaves the process. Each run uses its
own bucket (name ck-<RUN>-<n>), deleted after its row. S3_BENCH_KEEP=1 keeps them and prints RUN. Cleanup deletes only
buckets whose names this run generated. Verdicts come from ckpt_chaos.invariants.classify, the same judge as the other
targets. Crash runs that do not exit 137, timeouts, and resumes that fail without NoSuchKey are reported as INVALID
instead of being classified.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ckpt_chaos.invariants import ResumeOutcome, classify  # noqa: E402

CRASH_RC = 137  # os._exit(137) in the job
_ep = os.environ.get("S3_ENDPOINT", "http://127.0.0.1:5005")
ENDPOINT = None if _ep == "aws" else _ep  # S3_ENDPOINT=aws means real AWS
REGION = os.environ.get("AWS_REGION") or boto3.Session().region_name or "us-east-1"
KEEP = os.environ.get("S3_BENCH_KEEP") == "1"
JOB = [sys.executable, str(ROOT / "examples" / "s3_ckpt_job.py")]

# dummy keys only for a local or non-AWS endpoint with no credentials configured (moto accepts any key)
host = urlparse(ENDPOINT).hostname if ENDPOINT else ""
no_creds = not any(os.environ.get(v) for v in ("AWS_ACCESS_KEY_ID", "AWS_PROFILE", "AWS_DEFAULT_PROFILE"))
if host and "amazonaws.com" not in host and no_creds:
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "x")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "x")

RUN = uuid.uuid4().hex[:8]  # fresh bucket names every invocation: a leftover checkpoint would resume past the crash point
print(f"endpoint: {ENDPOINT or 'real AWS'}  region: {REGION}  run: {RUN}", flush=True)
s3 = boto3.client("s3", endpoint_url=ENDPOINT, region_name=REGION,
                  config=Config(retries={"max_attempts": 5, "mode": "standard"}))  # retries apply to this script's own calls
work = Path(tempfile.mkdtemp(prefix="s3bench_"))


def _text(x) -> str:
    if isinstance(x, bytes):
        return x.decode(errors="replace")
    return x or ""


def run(bucket: str, result: Path, crash_at: int = 0):
    """Run the job. Returns (returncode or None on timeout, stdout, stderr)."""
    env = dict(os.environ, S3_ENDPOINT=ENDPOINT or "aws")
    env.pop("CKPT_S3_CRASH_AT", None)
    if crash_at:
        env["CKPT_S3_CRASH_AT"] = str(crash_at)
    try:
        p = subprocess.run(JOB + ["--bucket", bucket, "--result", str(result)], env=env,
                           capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired as e:
        return None, _text(e.stdout), _text(e.stderr) or "timeout"
    return p.returncode, p.stdout, p.stderr


def list_keys(bucket: str) -> list[str]:
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        keys += [o["Key"] for o in page.get("Contents", [])]
    return keys


def complete_prefixes(bucket: str) -> set[int]:
    keys = list_keys(bucket)
    keyset = set(keys)
    done = set()
    for k in keys:
        m = re.fullmatch(r"ckpt-(\d+)/meta\.json", k)
        if m and {f"ckpt-{m.group(1)}/shard-0.bin", f"ckpt-{m.group(1)}/shard-1.bin"} <= keyset:
            done.add(int(m.group(1)))
    return done


def cleanup(bucket: str) -> None:
    """Delete one bucket this run named. Warns instead of raising, so one bad bucket does not stop the bench."""
    if not bucket.startswith(f"ck-{RUN}-"):
        print(f"WARNING: not deleting {bucket}: not named by this run")
        return
    try:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
            objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if objs:
                resp = s3.delete_objects(Bucket=bucket, Delete={"Objects": objs, "Quiet": True})
                if resp.get("Errors"):
                    raise RuntimeError(f"{len(resp['Errors'])} objects not deleted, first: {resp['Errors'][0]}")
        s3.delete_bucket(Bucket=bucket)
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchBucket":
            return  # the run never created it
        print(f"WARNING: cleanup of {bucket} failed: {e}")
    except RuntimeError as e:
        print(f"WARNING: cleanup of {bucket} failed: {e}")


# fault-free reference: the job prints its PutObject count, and the final state is the expected result
ref_bucket = f"ck-{RUN}-ref"
ref_out = work / "ref.json"
try:
    rc_ref, ref_stdout, ref_stderr = run(ref_bucket, ref_out)
    if rc_ref != 0:
        raise SystemExit(f"reference run failed (rc={rc_ref})\n{ref_stderr[-2000:]}\n{ref_stdout}")
    m = re.search(r"^puts (\d+)$", ref_stdout, re.M)
    if not m:
        raise SystemExit(f"reference run printed no PutObject count:\n{ref_stdout}")
    total_puts = int(m.group(1))
    ref_state = json.loads(ref_out.read_text())
finally:
    if not KEEP:
        cleanup(ref_bucket)
print(f"reference: {total_puts} PutObject calls, final step {ref_state['step']}")

rows = []
invalid = []
try:
    for n in range(1, total_puts + 1):
        bucket = f"ck-{RUN}-{n}"
        result = work / f"r{n}.json"
        try:
            rc_train, _, err_train = run(bucket, result, crash_at=n)
            if rc_train != CRASH_RC:
                invalid.append(n)
                print(f"put {n:2d}  INVALID: crash run exited rc={rc_train}\n{err_train[-2000:]}")
                continue
            newest = max(complete_prefixes(bucket), default=0)
            rc_resume, out, err_resume = run(bucket, result)
            if rc_resume is None or (rc_resume != 0 and "NoSuchKey" not in err_resume):
                invalid.append(n)
                print(f"put {n:2d}  INVALID: resume failed without NoSuchKey (rc={rc_resume})\n{err_resume[-2000:]}")
                continue
            m = re.search(r"resumed from step (\d+)", out)
            resumed = int(m.group(1)) if m else None
            raised = None if rc_resume == 0 else f"ExitCode{rc_resume}"
            diff = None
            if raised is None:
                got = json.loads(result.read_text())
                diff = 0.0 if got == ref_state else 1.0
            verdict = classify(ResumeOutcome(raised, resumed, newest, diff))
            rows.append((n, rc_train, newest, resumed, raised, verdict))
            print(f"put {n:2d}  crashed(rc={rc_train})  newest ok {newest:2d}  resumed {resumed}  {raised or '':12s}  {verdict}")
        finally:
            if not KEEP:
                cleanup(bucket)
finally:
    if not KEEP:
        shutil.rmtree(work, ignore_errors=True)

if invalid:
    print(f"INVALID rows (not classified): {invalid}")
counts = {}
for r in rows:
    counts[r[5]] = counts.get(r[5], 0) + 1
print("totals:", counts)
if invalid:
    raise SystemExit(1)

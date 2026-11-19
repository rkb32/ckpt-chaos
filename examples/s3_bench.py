"""Crash test for examples/s3_ckpt_job.py against a real S3 API (moto server on 127.0.0.1:5005).

Start the server first, in its own terminal:  python -m moto.server -p 5005
Then run:                                       python examples/s3_bench.py

Every PutObject is a crash point: the job is killed just before that request leaves the process. Each run uses its
own bucket. The verdicts come from ckpt_chaos.invariants.classify, the same judge as the other targets.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import boto3
from botocore.config import Config

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ckpt_chaos.invariants import ResumeOutcome, classify  # noqa: E402

ENDPOINT = os.environ.get("S3_ENDPOINT", "http://127.0.0.1:5005")
JOB = [sys.executable, str(ROOT / "examples" / "s3_ckpt_job.py")]
os.environ.setdefault("AWS_ACCESS_KEY_ID", "x")  # moto accepts any key; real AWS users keep their own
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "x")
s3 = boto3.client("s3", endpoint_url=ENDPOINT, region_name="us-east-1",
                  config=Config(retries={"max_attempts": 0}))
work = Path(tempfile.mkdtemp(prefix="s3bench_"))


def run(bucket: str, result: Path, crash_at: int = 0):
    env = dict(os.environ, S3_ENDPOINT=ENDPOINT)
    env.pop("CKPT_S3_CRASH_AT", None)
    if crash_at:
        env["CKPT_S3_CRASH_AT"] = str(crash_at)
    p = subprocess.run(JOB + ["--bucket", bucket, "--result", str(result)], env=env,
                       capture_output=True, text=True, timeout=300)
    return p.returncode, p.stdout


def complete_prefixes(bucket: str) -> set[int]:
    keys = [o["Key"] for o in s3.list_objects_v2(Bucket=bucket).get("Contents", [])]
    done = set()
    for k in keys:
        m = re.fullmatch(r"ckpt-(\d+)/meta\.json", k)
        if m and {f"ckpt-{m.group(1)}/shard-0.bin", f"ckpt-{m.group(1)}/shard-1.bin"} <= set(keys):
            done.add(int(m.group(1)))
    return done


# fault-free reference: count the PutObject calls and keep the final state
RUN = uuid.uuid4().hex[:8]  # fresh buckets every invocation: a leftover checkpoint would resume past the crash point
ref_bucket = f"ck-{RUN}-ref"
ref_out = work / "ref.json"
_, _ = run(ref_bucket, ref_out)
ref_state = json.loads(ref_out.read_text())
total_puts = len(s3.list_objects_v2(Bucket=ref_bucket).get("Contents", []))
print(f"reference: {total_puts} objects, final step {ref_state['step']}")

rows = []
for n in range(1, total_puts + 1):
    bucket = f"ck-{RUN}-{n}"
    result = work / f"r{n}.json"
    rc_train, _ = run(bucket, result, crash_at=n)
    newest = max(complete_prefixes(bucket), default=0)
    rc_resume, out = run(bucket, result)
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

counts = {}
for r in rows:
    counts[r[5]] = counts.get(r[5], 0) + 1
print("totals:", counts)

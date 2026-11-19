"""A checkpoint job that writes to a real S3 API (moto server, or MinIO / AWS with the same API).

Each checkpoint is three objects under ckpt-N/: shard-0.bin, shard-1.bin, meta.json (metadata last, no commit
marker). Resume lists the prefixes, picks the highest N with no completeness check, and reads its meta.json.

CKPT_S3_CRASH_AT=n kills the process (os._exit, no cleanup) just before the n-th PutObject call leaves the client:
that is the network-level kill point. Unset, the job runs to the end.
"""
import argparse
import json
import os
import re

import boto3
from botocore.config import Config

ap = argparse.ArgumentParser()
ap.add_argument("--bucket", required=True)
ap.add_argument("--endpoint", default=os.environ.get("S3_ENDPOINT", "http://127.0.0.1:5005"))
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--every", type=int, default=5)
ap.add_argument("--result", required=True)
a = ap.parse_args()

s3 = boto3.client("s3", endpoint_url=a.endpoint, aws_access_key_id="x", aws_secret_access_key="x",
                  region_name="us-east-1", config=Config(retries={"max_attempts": 0}))
try:
    s3.create_bucket(Bucket=a.bucket)
except Exception:  # noqa: BLE001 - already exists
    pass

CRASH_AT = int(os.environ.get("CKPT_S3_CRASH_AT", "0"))
puts = {"n": 0}


def count_put(params, **_):
    puts["n"] += 1
    if puts["n"] == CRASH_AT:
        os._exit(137)  # the request never leaves the process


s3.meta.events.register("before-call.s3.PutObject", count_put)

# list the prefixes: the step is the number in ckpt-N/
resp = s3.list_objects_v2(Bucket=a.bucket, Prefix="ckpt-", Delimiter="/")
steps = sorted(int(re.fullmatch(r"ckpt-(\d+)/", p["Prefix"]).group(1)) for p in resp.get("CommonPrefixes", []))
state = {"step": 0, "acc": 0}
if steps:
    body = s3.get_object(Bucket=a.bucket, Key=f"ckpt-{steps[-1]}/meta.json")["Body"].read()
    state = json.loads(body)
print(f"resumed from step {state['step']}", flush=True)

while state["step"] < a.steps:
    state["step"] += 1
    state["acc"] = (state["acc"] * 31 + state["step"]) % 1000003
    if state["step"] % a.every == 0:
        prefix = f"ckpt-{state['step']}"
        s3.put_object(Bucket=a.bucket, Key=f"{prefix}/shard-0.bin", Body=b"x" * 4096)
        s3.put_object(Bucket=a.bucket, Key=f"{prefix}/shard-1.bin", Body=b"y" * 4096)
        s3.put_object(Bucket=a.bucket, Key=f"{prefix}/meta.json", Body=json.dumps(state).encode())
        print(f"saved step {state['step']}", flush=True)

with open(a.result, "w") as f:
    json.dump(state, f)

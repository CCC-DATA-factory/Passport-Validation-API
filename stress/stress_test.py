#!/usr/bin/env python3
"""
stress_test.py

Usage examples:
  python stress_test.py            # runs default test (mode=poisson)
  python stress_test.py --mode ramp --duration 600 --total 2000

What it does:
- Sends requests to API_URL uploading two passport images per request ('passport1', 'passport2').
- Supports modes: uniform, poisson, ramp, spike, burst, constant.
- Controls concurrency (max concurrent requests).
- Logs each request to logs/ and appends a CSV summary.
"""

import argparse
import asyncio
import random
import time
import json
import csv
from pathlib import Path
from datetime import datetime, timedelta
import httpx
import statistics

# -------------------------
# CONFIG (tweak these)
# -------------------------
API_URL = "http://localhost:8000/api/v1/validate?include_debug=false"   # change to your endpoint
IMAGES_DIR = Path("passports")                        # put many passport images here
LOG_DIR = Path("logs")
CSV_SUMMARY = LOG_DIR / "summary.csv"

# defaults (override via CLI)
DEFAULT_TOTAL = 1800
DEFAULT_DURATION = 1 * 3600   # seconds (3 hours)
DEFAULT_CONCURRENCY = 100
DEFAULT_USERS = 500           # simulated user ids
DEFAULT_TIMEOUT = 60.0        # per-request timeout seconds

# -------------------------
# Schedule generators
# -------------------------
def uniform_schedule(total, duration):
    # uniform offsets across duration in seconds
    return sorted(random.uniform(0, duration) for _ in range(total))

def poisson_schedule(total, duration):
    # Approximate Poisson arrivals: inter-arrival ~ exponential with mean = duration/total
    if total <= 0:
        return []
    lam = total / duration
    offsets = []
    t = 0.0
    for _ in range(total):
        # exponential inter-arrival: -ln(U)/lam
        inter = random.expovariate(lam)
        t += inter
        offsets.append(min(t, duration))
    # if we generated fewer due to rounding, pad with randoms
    return sorted(offsets[:total])

def ramp_schedule(total, duration, ramp_up_fraction=0.2):
    # slowly ramping arrivals then steady: first ramp_up_fraction of duration we increase rate linearly
    ramp_time = duration * ramp_up_fraction
    remaining_time = duration - ramp_time
    ramp_count = int(total * ramp_up_fraction)
    remainder = total - ramp_count
    # generate ramp_count offsets clustered in ramp_time (denser at end)
    ramp_offsets = sorted((ramp_time * ((i / max(1, ramp_count-1))**1.8)) for i in range(ramp_count))
    steady_offsets = sorted(ramp_time + random.uniform(0, remaining_time) for _ in range(remainder))
    return sorted(ramp_offsets + steady_offsets)

def spike_schedule(total, duration, spike_fraction=0.1, spike_intensity=10):
    # A big spike in the middle: spike_fraction of duration contains spike_intensity * normal rate
    spike_time = duration * 0.5
    spike_window = duration * 0.05
    offsets = []
    for _ in range(total):
        if random.random() < spike_fraction:
            offsets.append(spike_time + random.uniform(-spike_window/2, spike_window/2))
        else:
            offsets.append(random.uniform(0, duration))
    return sorted(offsets)

def burst_schedule(total, duration, bursts=10, burst_size=50):
    # create bursts at random times
    offsets = []
    for _ in range(bursts):
        center = random.uniform(0, duration)
        for _ in range(burst_size):
            offsets.append(max(0, min(duration, random.gauss(center, 2.0))))
    # fill rest uniform
    while len(offsets) < total:
        offsets.append(random.uniform(0, duration))
    return sorted(offsets[:total])

# -------------------------
# Runner
# -------------------------
async def call_api(client: httpx.AsyncClient, img: bytes, timeout: float, include_debug: bool = False):
    r = {
        "started_at": time.time(),
        "duration_sec": None,
        "http_status": None,
        "response": None,
        "error": None,
    }
    start = time.time()
    files = [
        ("file", ("passport.jpg", img, "image/jpeg")),
    ]
    params = {"include_debug": "true"} if include_debug else {}
    try:
        resp = await client.post(API_URL, params=params, files=files, timeout=timeout)
        r["http_status"] = resp.status_code
        r["duration_sec"] = round(time.time() - start, 3)
        rtext = resp.text
        if resp.status_code == 200:
            try:
                r["response"] = resp.json()
            except Exception:
                r["response"] = rtext
        else:
            r["error"] = f"HTTP {resp.status_code}: {rtext[:400]}"
    except Exception as e:
        r["duration_sec"] = round(time.time() - start, 3)
        r["error"] = f"{type(e).__name__}: {e}"
    return r

async def handle_interaction(client, interaction_id, user_id, offset, imgs, sem, timeout):
    await asyncio.sleep(offset)
    img_path = random.choice(imgs)

    with open(img_path, "rb") as f:
        img = f.read()

    async with sem:
        # pass include_debug=False (or True if you want server debug info)
        result = await call_api(client, img, timeout, include_debug=False)

    # annotate result
    result.update({
        "interaction_id": interaction_id,
        "user_id": user_id,
        "scheduled_offset": offset,
        "timestamp_iso": datetime.utcnow().isoformat() + "Z",
        "img": str(Path(img_path).name),
    })

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    filename = LOG_DIR / f"req_{interaction_id}_user{user_id}.json"
    with open(filename, "w") as jf:
        json.dump(result, jf, indent=2)
    print(f"Logged {filename.name}: status={result.get('http_status')} dur={result.get('duration_sec')}s err={result.get('error')}")
    return result


async def run_test(mode, total, duration, concurrency, users, timeout):
    imgs = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.jpeg")) + list(IMAGES_DIR.glob("*.png"))
    assert imgs, f"No images in {IMAGES_DIR}, put passport images there."

    # choose schedule
    if mode == "uniform":
        schedule = uniform_schedule(total, duration)
    elif mode == "poisson":
        schedule = poisson_schedule(total, duration)
    elif mode == "ramp":
        schedule = ramp_schedule(total, duration)
    elif mode == "spike":
        schedule = spike_schedule(total, duration)
    elif mode == "burst":
        schedule = burst_schedule(total, duration)
    elif mode == "constant":
        # evenly spaced
        schedule = [i * (duration / total) for i in range(total)]
    else:
        raise ValueError("Unknown mode")

    users_list = [random.randint(1, users) for _ in schedule]
    sem = asyncio.Semaphore(concurrency)
    results = []
    async with httpx.AsyncClient() as client:
        tasks = []
        start_time = time.time()
        for idx, (offset, uid) in enumerate(zip(schedule, users_list), start=1):
            tasks.append(asyncio.create_task(handle_interaction(client, idx, uid, offset, imgs, sem, timeout)))
        # run tasks and gather in streaming style to avoid holding all results in memory too long
        for coro in asyncio.as_completed(tasks):
            res = await coro
            results.append(res)
            # optional: print progress
            if len(results) % 50 == 0:
                now = time.time()
                elapsed = now - start_time
                print(f"[{len(results)}/{total}] elapsed {elapsed:.1f}s last_status={res.get('http_status')} dur={res.get('duration_sec')}s")
    return results

# -------------------------
# Simple analysis (summary)
# -------------------------
def analyze_results(results):
    durations = [r["duration_sec"] for r in results if r.get("duration_sec") is not None]
    successes = [r for r in results if r.get("http_status") == 200]
    failures = [r for r in results if r.get("http_status") != 200 or r.get("error")]

    summary = {
        "total_requests": len(results),
        "successful": len(successes),
        "failed": len(failures),
    }
    if durations:
        summary.update({
            "avg_ms": round(statistics.mean(durations)*1000, 2),
            "median_ms": round(statistics.median(durations)*1000, 2),
            "p95_ms": round(statistics.quantiles(durations, n=100)[94]*1000, 2),
            "p99_ms": round(statistics.quantiles(durations, n=100)[98]*1000, 2),
            "min_ms": round(min(durations)*1000, 2),
            "max_ms": round(max(durations)*1000, 2),
        })
    return summary

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="poisson", choices=["uniform","poisson","ramp","spike","burst","constant"])
    p.add_argument("--total", type=int, default=DEFAULT_TOTAL)
    p.add_argument("--duration", type=int, default=DEFAULT_DURATION, help="Duration of simulated day in seconds")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--users", type=int, default=DEFAULT_USERS)
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    return p.parse_args()

def write_csv_summary(summary, args):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    header = ["timestamp","mode","total","duration","concurrency","users","successful","failed","avg_ms","median_ms","p95_ms","p99_ms","min_ms","max_ms"]
    row = [
        datetime.utcnow().isoformat()+"Z",
        args.mode,
        args.total,
        args.duration,
        args.concurrency,
        args.users,
        summary.get("successful",0),
        summary.get("failed",0),
        summary.get("avg_ms",""),
        summary.get("median_ms",""),
        summary.get("p95_ms",""),
        summary.get("p99_ms",""),
        summary.get("min_ms",""),
        summary.get("max_ms",""),
    ]
    new_file = not CSV_SUMMARY.exists()
    with open(CSV_SUMMARY, "a", newline="") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(header)
        writer.writerow(row)

if __name__ == "__main__":
    args = parse_args()
    print(f"Starting stress test: mode={args.mode} total={args.total} duration={args.duration}s concurrency={args.concurrency}")
    results = asyncio.run(run_test(args.mode, args.total, args.duration, args.concurrency, args.users, args.timeout))
    summary = analyze_results(results)
    print("Test summary:", json.dumps(summary, indent=2))
    write_csv_summary(summary, args)
    print(f"Per-request logs written to {LOG_DIR}. Summary appended to {CSV_SUMMARY}.")

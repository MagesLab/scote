#!/usr/bin/env python
"""Portable replacement for `/usr/bin/time -v` (absent in some containers): run a command, print wall time and the
children's peak RSS to stderr and, with --json, append a JSON line. Usage: timed_run.py [--json FILE] -- cmd args..."""
import argparse, json, resource, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--json", default=None)
ap.add_argument("cmd", nargs=argparse.REMAINDER)
a = ap.parse_args()
cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
t0 = time.perf_counter()
rc = subprocess.call(cmd)
wall = time.perf_counter() - t0
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
rec = dict(cmd=cmd, wall_s=wall, user_s=ru.ru_utime, sys_s=ru.ru_stime, maxrss_MB=ru.ru_maxrss / 1024, rc=rc)
print(f"timed_run: wall {wall:.2f} s, user {ru.ru_utime:.1f} s, sys {ru.ru_stime:.1f} s, maxRSS {ru.ru_maxrss/1024:.0f} MB, rc={rc}", file=sys.stderr)
if a.json:
    with open(a.json, "a") as f:
        f.write(json.dumps(rec) + "\n")
sys.exit(rc)

"""Offline deterministic forecast for pumpfun-top5-forecast-01.

This entry uses a rank-persistence kernel calibrated on the supplied rehearsal:
most probability stays near each observed rank, with a conservative unnamed
share for coins that move or launch after the history cut. Market-cap sizes use
a narrow log-space distribution around the latest observed rank cap.
"""
import json, math, os

INPUT = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
with open(os.path.join(INPUT, "task.json"), encoding="utf-8") as f:
    task = json.load(f)
marks = []
with open(os.path.join(INPUT, "history.ndjson"), encoding="utf-8") as f:
    for line in f:
        if line.strip():
            m = json.loads(line)
            if m.get("ok") and m.get("rows"):
                marks.append(m)
if not marks:
    raise RuntimeError("history has no usable marks")

latest = {r["mint"]: r for r in marks[-1]["rows"]}
rows = sorted(latest.values(), key=lambda r: int(r["rank"]))[: int(task.get("chart_size", 100))]
places = int(task["places"])
edges = task["size"]["edges"]

# Best dev-rehearsal parameters: rank kernel, downward movement penalty, and
# explicit mass. The zero momentum term is intentional: the short rehearsal
# showed momentum extrapolation degraded the log score.
MASS = 0.85
KERNEL_SD = 0.8
DOWN_PEN = 1.6
SIZE_SD = 0.20
MICRO = 1_000_000

def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def size_bin_prob(mean, sd, i):
    lo = -float("inf") if i == 0 else edges[i - 1]
    hi = float("inf") if i == len(edges) else edges[i]
    a = 0.0 if lo == -float("inf") else normal_cdf((lo - mean) / sd)
    b = 1.0 if hi == float("inf") else normal_cdf((hi - mean) / sd)
    return max(0.0, b - a)

probabilities = []
sizes = []
for place in range(1, places + 1):
    weights = []
    for r in rows:
        rank = int(r["rank"])
        d = rank - place
        dist = abs(d) * (DOWN_PEN if d > 0 else 1.0)
        weights.append(math.exp(-dist / KERNEL_SD))
    total = sum(weights)
    for r, w in zip(rows, weights):
        p = math.floor((w / total) * MASS * MICRO) / MICRO
        if p > 0:
            probabilities.append({"mint": r["mint"], "place": place, "p": p})

# Distinct best guess: rank persistence, with a deterministic assignment.
top5 = []
taken = set()
for place in range(1, places + 1):
    candidates = [r for r in rows if r["mint"] not in taken]
    best = min(candidates, key=lambda r: (abs(int(r["rank"]) - place), int(r["rank"]), r["mint"]))
    top5.append(best["mint"])
    taken.add(best["mint"])

for place, mint in enumerate(top5, 1):
    r = latest[mint]
    mean = math.log10(max(1, int(r["market_cap_lamports"]))) - 9
    for i in range(len(edges) + 1):
        p = math.floor(size_bin_prob(mean, SIZE_SD, i) * MICRO) / MICRO
        if p > 0:
            sizes.append({"place": place, "bin": i, "p": p})

os.makedirs(OUTPUT, exist_ok=True)
with open(os.path.join(OUTPUT, task["output"]), "w", encoding="utf-8") as f:
    json.dump({"probabilities": probabilities, "top5": top5, "sizes": sizes}, f, separators=(",", ":"))
    f.write("\n")
print("forecast written")

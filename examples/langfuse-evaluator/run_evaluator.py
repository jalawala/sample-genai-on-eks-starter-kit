"""Read evaluator_config.json and score recent Langfuse traces.

Usage:
  python3 run_evaluator.py [path/to/evaluator_config.json]

Reads the rubric from config, picks recent traces matching the filter,
calls the judge LLM via the LiteLLM gateway, and POSTs scores back.
No Langfuse SDK or RAGAS required — uses only urllib.
"""
import urllib.request, urllib.error, json, ssl, base64, sys, random, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

CFG_PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "evaluator_config.json")
cfg = json.loads(CFG_PATH.read_text())
print(f"Loaded config: {cfg['evaluator_name']}  (judge={cfg['judge_model']})")

LF_BASE = "https://langfuse.dev.cloud-tech-consulting.com"
LF_AUTH = "Basic " + base64.b64encode(b"lf_pk_1234567890:lf_sk_1234567890").decode()
GW_AUTH = "Bearer sk-1234"
ctx = ssl._create_unverified_context()

def lf_get(path):
    req = urllib.request.Request(f"{LF_BASE}{path}", headers={"Authorization": LF_AUTH})
    with urllib.request.urlopen(req, context=ctx, timeout=20) as r:
        return json.loads(r.read())

def lf_post(path, body):
    req = urllib.request.Request(f"{LF_BASE}{path}", method="POST",
        data=json.dumps(body).encode(),
        headers={"Authorization": LF_AUTH, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, context=ctx, timeout=20) as r:
        return r.status

def judge(user_q, ai_a):
    rubric = cfg["rubric_prompt"].replace("{{input}}", str(user_q)[:1500]) \
                                 .replace("{{output}}", str(ai_a)[:1500])
    body = {"model": cfg["judge_model"],
            "messages":[{"role":"user","content":rubric}],
            "max_tokens": 200, "temperature": 0.0}
    req = urllib.request.Request(cfg["judge_gateway"], method="POST",
        data=json.dumps(body).encode(),
        headers={"Authorization": GW_AUTH, "Content-Type":"application/json"})
    with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
        reply = json.loads(r.read())["choices"][0]["message"]["content"].strip()
    if reply.startswith("```"):
        reply = reply.split("```")[1]
        if reply.startswith("json"): reply = reply[4:]
    return json.loads(reply.strip())

# Fetch traces within filter window
since_min = cfg["filter"].get("since_minutes", 60)
from_ts = (datetime.now(timezone.utc) - timedelta(minutes=since_min)).replace(microsecond=0).strftime('%Y-%m-%dT%H:%M:%SZ')
import urllib.parse
q = f"/api/public/traces?limit=50&orderBy=timestamp.desc&fromTimestamp={urllib.parse.quote(from_ts)}"
traces = [t for t in lf_get(q).get("data", []) if t.get("input") and t.get("output")]
print(f"Found {len(traces)} traces in last {since_min} min with input+output")

# Apply sample_rate
sampled = [t for t in traces if random.random() < cfg["sample_rate"]]
if len(sampled) == 0 and len(traces) > 0:
    sampled = [traces[0]]   # always score at least 1 in demo mode
print(f"Sampled {len(sampled)} for evaluation (rate={cfg['sample_rate']})\n")

ok = errs = 0
for t in sampled:
    try:
        v = judge(t["input"], t["output"])
        score = max(cfg["score_range"][0], min(cfg["score_range"][1], int(v["score"])))
        lf_post("/api/public/scores", {
            "name": cfg["score_name"],
            "value": float(score),
            "dataType": cfg["score_data_type"],
            "traceId": t["id"],
            "comment": v.get("reasoning","")[:500]
        })
        print(f"  ✓ trace={t['id'][:8]}  score={score}  — {v.get('reasoning','')[:90]}")
        ok += 1
    except urllib.error.HTTPError as e:
        print(f"  ✗ trace={t['id'][:8]}  HTTP {e.code}: {e.read().decode()[:100]}")
        errs += 1
    except Exception as e:
        print(f"  ✗ trace={t['id'][:8]}  {type(e).__name__}: {str(e)[:100]}")
        errs += 1

print(f"\n✅ Scored {ok} traces, {errs} errors. View: {LF_BASE}/")

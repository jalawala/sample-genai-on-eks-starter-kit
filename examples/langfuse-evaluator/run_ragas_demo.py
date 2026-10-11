"""RAGAS 4-metric demo → upload scores to Langfuse.

Builds a tiny RAG dataset (3 rows), computes answer_relevancy, faithfulness,
context_recall, context_precision using Bedrock Claude as the RAGAS judge LLM,
and POSTs each metric as a score on a Langfuse trace.
"""
import urllib.request, json, ssl, base64, os, sys, ssl
from pathlib import Path

LF_BASE = "https://langfuse.dev.cloud-tech-consulting.com"
LF_AUTH = "Basic " + base64.b64encode(b"lf_pk_1234567890:lf_sk_1234567890").decode()
ctx = ssl._create_unverified_context()

def lf_post(path, body):
    req = urllib.request.Request(f"{LF_BASE}{path}", method="POST",
        data=json.dumps(body).encode(),
        headers={"Authorization": LF_AUTH, "Content-Type":"application/json"})
    with urllib.request.urlopen(req, context=ctx, timeout=20) as r:
        return r.status, (json.loads(r.read()) if r.status < 300 else None)

# Build a tiny RAG dataset: 3 questions with answers, context, ground truth
dataset = [
    {
        "question":  "What is Amazon EKS?",
        "answer":    "Amazon EKS is a managed Kubernetes service on AWS.",
        "contexts":  ["Amazon EKS is a managed Kubernetes service that runs Kubernetes without needing to install and manage your own control plane."],
        "ground_truth": "Amazon EKS is a managed Kubernetes service."
    },
    {
        "question":  "What is RBI FREE-AI?",
        "answer":    "A framework from the Reserve Bank of India for responsible AI in finance, released August 2025.",
        "contexts":  ["In August 2025, RBI released FREE-AI, a Framework for Responsible and Ethical Enablement of AI in the financial sector."],
        "ground_truth": "FREE-AI is RBI's 2025 framework for responsible AI in Indian financial sector."
    },
    {
        "question":  "Is the sky blue?",
        "answer":    "Yes, the sky appears blue due to Rayleigh scattering of sunlight.",
        "contexts":  ["Unrelated: Kubernetes manages containers."],   # DELIBERATELY wrong context — to show faithfulness/context metrics dropping
        "ground_truth": "Yes, the sky is blue due to Rayleigh scattering."
    },
]

# Create one Langfuse trace per dataset row so RAGAS scores attach
trace_ids = []
for i, row in enumerate(dataset):
    # Build an ingestion event
    import uuid
    trace_id = str(uuid.uuid4())
    obs_id   = str(uuid.uuid4())
    batch = [{
        "id": str(uuid.uuid4()),
        "type": "trace-create",
        "body": {
            "id": trace_id,
            "name": f"ragas-demo-{i+1}",
            "input":  row["question"],
            "output": row["answer"],
            "tags":   ["ragas-demo","workshop"],
            "metadata": {"contexts": row["contexts"], "ground_truth": row["ground_truth"]}
        }
    }]
    s, _ = lf_post("/api/public/ingestion", {"batch": batch})
    print(f"  created trace {trace_id[:8]} for row {i+1}")
    trace_ids.append(trace_id)

print(f"\n✅ {len(trace_ids)} RAGAS-demo traces created in Langfuse")

# Run RAGAS
try:
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness, context_recall, context_precision
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from datasets import Dataset
except ImportError as e:
    print(f"\n⚠ RAGAS not installed: {e}")
    print("  Install with:  pip install ragas datasets langchain-openai")
    print("  Then re-run this script.")
    sys.exit(1)

# Configure RAGAS to use our LiteLLM gateway as its judge (OpenAI-compatible)
os.environ["OPENAI_API_KEY"]  = "sk-1234"
os.environ["OPENAI_API_BASE"] = "https://litellm.dev.cloud-tech-consulting.com/v1"

import httpx
http_client = httpx.Client(verify=False)
judge_llm  = LangchainLLMWrapper(ChatOpenAI(
    model="bedrock/claude-4.5-sonnet", temperature=0, http_client=http_client))
judge_emb  = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
    model="tei/qwen3-embedding-06b-bf16-cpu", http_client=http_client))

ds = Dataset.from_list([
    {"question":   r["question"],
     "answer":     r["answer"],
     "contexts":   r["contexts"],
     "ground_truth": r["ground_truth"]}
    for r in dataset])

print("\nRunning RAGAS evaluate() — this may take 30-60s...")
result = evaluate(ds,
    metrics=[answer_relevancy, faithfulness, context_recall, context_precision],
    llm=judge_llm, embeddings=judge_emb)

print("\nRAGAS scores per row:")
scores_df = result.to_pandas()
print(scores_df[["answer_relevancy","faithfulness","context_recall","context_precision"]].to_string())

# POST each metric as a Langfuse score on its corresponding trace
print("\nUploading scores to Langfuse...")
for i, trace_id in enumerate(trace_ids):
    row = scores_df.iloc[i]
    for metric in ["answer_relevancy","faithfulness","context_recall","context_precision"]:
        val = float(row[metric])
        lf_post("/api/public/scores", {
            "name": f"ragas-{metric}",
            "value": val,
            "dataType": "NUMERIC",
            "traceId": trace_id,
            "comment": f"RAGAS v0.x  on row {i+1}"
        })
        print(f"    trace={trace_id[:8]}  {metric}={val:.3f}")

print("\n✅ RAGAS demo complete. View in Langfuse UI → tag=ragas-demo")

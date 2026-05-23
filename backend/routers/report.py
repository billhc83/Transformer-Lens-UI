from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import json

router = APIRouter(prefix="/api/report", tags=["report"])

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:14b"

SYSTEM_PROMPT = (
    "You are a mechanistic interpretability expert AI tasked with synthesizing "
    "TransformerLens analysis findings into a structured markdown report. "
    "Generate a report with three sections: Executive Summary (concise overview), "
    "Key Findings per Tool (bullet points detailing results from each analysis tool), "
    "and Conclusions (interpretive insights about the circuit's behavior). "
    "Maintain technical accuracy, clarity, and coherence, adhering to best practices "
    "in mechanistic interpretability."
)


class Finding(BaseModel):
    page: str
    timestamp: int
    headline: str
    data: dict


class ReportRequest(BaseModel):
    model_name: str = "unknown"
    findings: list[Finding]


PAGE_LABELS = {
    "token-inspector": "Token Inspector",
    "forward-pass": "Forward Pass",
    "activation-browser": "Activation Browser",
    "attention-viz": "Attention Viz",
    "logit-lens": "Logit Lens",
    "attribution": "Attribution Analyzer",
    "patching-lab": "Patching Lab",
    "circuit-analyzer": "Circuit Analyzer",
    "hook-lab": "Hook Lab",
    "generation-studio": "Generation Studio",
    "normalization-probe": "Normalization Probe",
}


def format_findings(model_name: str, findings: list[Finding]) -> str:
    lines = [
        f"Model: {model_name}",
        f"Session findings ({len(findings)} total):\n",
    ]
    for f in findings:
        label = PAGE_LABELS.get(f.page, f.page)
        lines.append(f"## {label}")
        lines.append(f"Finding: {f.headline}")
        data_str = json.dumps(f.data, indent=2)[:600]
        lines.append(f"Data:\n{data_str}\n")
    lines.append(
        "\nSynthesize these findings into a structured markdown report with: "
        "## Executive Summary, ## Key Findings (one sub-section per tool used), "
        "and ## Conclusions."
    )
    return "\n".join(lines)


async def stream_report(prompt: str):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "think": False,
        "options": {"temperature": 0.25},
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        async with client.stream("POST", OLLAMA_URL, json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue


@router.post("/generate")
async def generate_report(req: ReportRequest):
    prompt = format_findings(req.model_name, req.findings)
    return StreamingResponse(stream_report(prompt), media_type="text/plain; charset=utf-8")

"""HTTP 服务层: 提供 /api/assess 接口与静态演示页面."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import llm
from .engine import DISCLAIMER, assess, load_kb

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
KB = load_kb()

app = FastAPI(
    title="CDS-Hint",
    version="0.1.0",
    description="临床辅助诊断提示原型（演示用途，不构成诊断或治疗建议）",
)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class Vitals(BaseModel):
    temp: float | None = Field(None, description="体温 ℃")
    hr: float | None = Field(None, description="心率 次/分")
    rr: float | None = Field(None, description="呼吸 次/分")
    sbp: float | None = Field(None, description="收缩压 mmHg")
    dbp: float | None = Field(None, description="舒张压 mmHg")
    sbp_diff: float | None = Field(None, description="双上肢收缩压差 mmHg")
    spo2: float | None = Field(None, description="血氧饱和度 %")


class CaseIn(BaseModel):
    complaint: str = Field("", description="主诉自由文本", max_length=500)
    symptoms: list[str] = Field(default_factory=list, description="症状关键词")
    history: list[str] = Field(default_factory=list, description="既往史/用药/手术等")
    age: int | None = Field(None, ge=0, le=120, description="年龄")
    sex: str | None = Field(None, description="male / female")
    vitals: Vitals = Field(default_factory=Vitals)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "kb_version": KB.get("version"), "llm": llm.enabled()}


@app.get("/api/presentations")
def presentations() -> dict:
    return {
        "kb_version": KB.get("version"),
        "items": [
            {"id": p["id"], "name": p["name"], "differentials": len(p.get("differentials", []))}
            for p in KB.get("presentations", [])
        ],
        "disclaimer": DISCLAIMER,
    }


@app.post("/api/assess")
def api_assess(case: CaseIn) -> dict:
    payload = case.model_dump()
    payload["vitals"] = {k: v for k, v in payload["vitals"].items() if v is not None}
    result = assess(payload, KB)
    result["narrative"] = llm.explain(result)
    return result

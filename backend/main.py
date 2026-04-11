"""
AutoChunker Platform — FastAPI Backend
Provides 5 REST endpoints that orchestrate the 7-stage pipeline.
All heavy work is executed in a background thread so the HTTP server
stays responsive; progress is polled via GET /status/{job_id}.
"""

import io
import csv
import json
import uuid
import threading
import traceback
from typing import Any, Dict, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

# ── Pipeline stages ───────────────────────────────────────────────────────
from pipeline.s1_profiler import profile_document
from pipeline.s2_chunkers import run_all_chunkers, select_best_strategy
from pipeline.s3_entropy import refine_boundaries, get_jsd_series
from pipeline.s4_boundary import filter_boundaries
from pipeline.s5_graph import enrich_graph, build_entity_graph_data
from pipeline.s6_embedding import embed_chunks
from pipeline.s7_rl import run_rl_loop

# ── In-memory stores ──────────────────────────────────────────────────────
doc_store: Dict[str, Dict] = {}   # document_id → {filename, content, …}
job_store: Dict[str, Dict] = {}   # job_id      → {status, stage, progress, …}

# ── Default pipeline configuration ───────────────────────────────────────
DEFAULT_CONFIG: Dict[str, Any] = {
    "n_min": 100,
    "n_max": 500,
    "tau_jsd_low": 0.15,
    "tau_jsd_high": 0.45,
    "tau_sem": 0.75,
    "max_iterations": 10,
    "alpha": 0.4,
    "beta": 0.4,
    "lambda": 0.2,
    "embedding_model": "all-MiniLM-L6-v2",
}

# ─────────────────────────────────────────────────────────────────────────
app = FastAPI(title="AutoChunker Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: parse uploaded file to plain text
# ═══════════════════════════════════════════════════════════════════════════

def _parse_file(filename: str, content: bytes) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"

    if ext == "pdf":
        return _parse_pdf(content)
    # TXT, MD, code files — decode as UTF-8 with fallback
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1", errors="replace")


def _parse_pdf(content: bytes) -> str:
    # Try pdfplumber first, then PyPDF2
    try:
        import pdfplumber  # noqa: PLC0415
        pages: List[str] = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n\n".join(pages)
    except Exception:
        pass

    try:
        import PyPDF2  # noqa: PLC0415
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        pages = [
            reader.pages[i].extract_text() or ""
            for i in range(len(reader.pages))
        ]
        return "\n\n".join(pages)
    except Exception:
        pass

    return content.decode("utf-8", errors="replace")


# ═══════════════════════════════════════════════════════════════════════════
# Helper: update job progress
# ═══════════════════════════════════════════════════════════════════════════

def _update_job(job_id: str, **kwargs) -> None:
    if job_id in job_store:
        job_store[job_id].update(kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline runner (executed in a background thread)
# ═══════════════════════════════════════════════════════════════════════════

def _run_pipeline(job_id: str, doc_id: str, user_config: Dict[str, Any]) -> None:
    try:
        doc = doc_store.get(doc_id)
        if not doc:
            _update_job(job_id, status="error", error="Document not found.")
            return

        text: str = doc["content"]
        config = {**DEFAULT_CONFIG, **user_config}

        # ── S1: Profile ───────────────────────────────────────────────────
        _update_job(
            job_id,
            stage="S1",
            progress=5,
            message="Profiling document…",
        )
        doc_profile = profile_document(text, config)
        # Blend suggested params into config (user overrides take precedence)
        suggested = doc_profile.get("suggested_config", {})
        for k, v in suggested.items():
            if k not in user_config:
                config[k] = v

        _update_job(job_id, stage_details={"s1": doc_profile})

        # ── S2: Parallel chunkers ─────────────────────────────────────────
        _update_job(
            job_id,
            stage="S2",
            progress=15,
            message="Running parallel chunkers…",
        )
        all_chunks_map = run_all_chunkers(text, doc_profile["type"], config)
        initial_chunks = select_best_strategy(
            all_chunks_map, doc_profile["type"], config
        )

        if not initial_chunks:
            # Ultra-short document: treat the whole text as one chunk
            initial_chunks = [
                {"text": text, "start": 0, "end": len(text), "method": "fallback"}
            ]

        _update_job(
            job_id,
            stage_details={
                **job_store[job_id].get("stage_details", {}),
                "s2": {
                    "strategies": {
                        k: len(v) for k, v in all_chunks_map.items()
                    },
                    "selected_count": len(initial_chunks),
                },
            },
        )

        # ── S3: Entropy refinement ────────────────────────────────────────
        _update_job(
            job_id,
            stage="S3",
            progress=28,
            message="Computing entropy boundaries…",
        )
        refined = refine_boundaries(initial_chunks, config)
        jsd_series = get_jsd_series(refined)

        _update_job(
            job_id,
            stage_details={
                **job_store[job_id].get("stage_details", {}),
                "s3": {"jsd_series": jsd_series, "chunk_count": len(refined)},
            },
        )

        # ── S4: Boundary quality filter ───────────────────────────────────
        _update_job(
            job_id,
            stage="S4",
            progress=42,
            message="Filtering boundaries…",
        )
        filtered = filter_boundaries(refined, doc_profile["type"], [], config)

        _update_job(
            job_id,
            stage_details={
                **job_store[job_id].get("stage_details", {}),
                "s4": {"chunk_count": len(filtered)},
            },
        )

        # ── S5: Graph enrichment ──────────────────────────────────────────
        _update_job(
            job_id,
            stage="S5",
            progress=55,
            message="Building entity graph…",
        )
        enriched = enrich_graph(filtered, [], config)
        graph_data = build_entity_graph_data(enriched)

        _update_job(
            job_id,
            stage_details={
                **job_store[job_id].get("stage_details", {}),
                "s5": {"entity_graph": graph_data},
            },
        )

        # ── S6: Contextual embedding ──────────────────────────────────────
        _update_job(
            job_id,
            stage="S6",
            progress=68,
            message="Generating embeddings…",
        )
        embedded, embeddings = embed_chunks(
            enriched,
            text,
            doc_profile,
            config["embedding_model"],
            config,
        )

        _update_job(
            job_id,
            stage_details={
                **job_store[job_id].get("stage_details", {}),
                "s6": {
                    "embedding_dim": len(embeddings[0]) if embeddings else 0,
                    "model": config["embedding_model"],
                },
            },
        )

        # ── S7: RL reward calibration ─────────────────────────────────────
        _update_job(
            job_id,
            stage="S7",
            progress=78,
            message="Running RL calibration loop…",
        )
        best_chunks, reward_history, final_config = run_rl_loop(
            text, doc_profile, embedded, config
        )

        _update_job(
            job_id,
            stage_details={
                **job_store[job_id].get("stage_details", {}),
                "s7": {
                    "reward_history": reward_history,
                    "iterations": len(reward_history),
                    "final_config": final_config,
                },
            },
        )

        # ── Final scoring ─────────────────────────────────────────────────
        _update_job(
            job_id,
            stage="DONE",
            progress=95,
            message="Finalising results…",
        )
        final_chunks = _finalise_chunks(best_chunks)

        mean_score = (
            sum(c.get("chunk_score", 0.0) for c in final_chunks) / len(final_chunks)
            if final_chunks
            else 0.0
        )

        results = {
            "document_id": doc_id,
            "job_id": job_id,
            "doc_profile": doc_profile,
            "chunks": final_chunks,
            "summary": {
                "doc_type": doc_profile.get("type"),
                "domain": doc_profile.get("domain"),
                "length_bucket": doc_profile.get("length_bucket"),
                "token_count": doc_profile.get("token_count"),
                "chunk_count": len(final_chunks),
                "mean_chunk_score": round(mean_score, 4),
                "rl_iterations": len(reward_history),
                "final_reward": reward_history[-1] if reward_history else 0.0,
            },
            "stage_details": job_store[job_id].get("stage_details", {}),
            "reward_history": reward_history,
        }

        _update_job(
            job_id,
            status="complete",
            stage="DONE",
            progress=100,
            message="Pipeline complete.",
            results=results,
        )

    except Exception as exc:
        _update_job(
            job_id,
            status="error",
            message=str(exc),
            error=traceback.format_exc(),
        )


def _finalise_chunks(chunks: list) -> list:
    """Assign a composite chunk_score and clean up non-serialisable fields."""
    out = []
    for i, c in enumerate(chunks):
        chunk = dict(c)

        jsd = float(chunk.get("jsd_score", 0.0))
        boundary = float(chunk.get("boundary_score", 0.5))
        icc = float(chunk.get("icc", 0.5))

        # Composite chunk quality score (higher = better)
        # Low JSD = similar to neighbours (less coherent boundary) → penalise
        # High boundary_score = similar to neighbours → penalise (should differ)
        chunk_score = float(
            0.4 * (1.0 - boundary)        # boundary distinctiveness
            + 0.3 * icc                    # internal cohesion
            + 0.3 * min(jsd * 2.0, 1.0)   # JSD strength (rescaled)
        )
        chunk["chunk_index"] = i
        chunk["chunk_score"] = round(chunk_score, 4)
        chunk["token_count"] = len(chunk.get("text", "").split())

        # Drop raw embedding vectors from results (too large)
        chunk.pop("embedding", None)
        chunk.pop("graph_vector", None)

        out.append(chunk)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> JSONResponse:
    """Accept a file upload and store its parsed text content."""
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:  # 50 MB guard
        raise HTTPException(413, "File too large (max 50 MB).")

    text = _parse_file(file.filename or "upload.txt", content)
    if not text.strip():
        raise HTTPException(400, "Could not extract text from the uploaded file.")

    doc_id = str(uuid.uuid4())
    doc_store[doc_id] = {
        "filename": file.filename,
        "content": text,
        "size": len(content),
        "content_type": file.content_type or "application/octet-stream",
    }

    return JSONResponse(
        {
            "document_id": doc_id,
            "filename": file.filename,
            "char_count": len(text),
            "token_count": len(text.split()),
        }
    )


@app.post("/run/{document_id}")
async def run_pipeline(
    document_id: str,
    config: str = Form(default="{}"),
) -> JSONResponse:
    """Trigger the full pipeline for a previously uploaded document."""
    if document_id not in doc_store:
        raise HTTPException(404, "Document not found.")

    try:
        user_config = json.loads(config)
    except json.JSONDecodeError:
        user_config = {}

    job_id = str(uuid.uuid4())
    job_store[job_id] = {
        "status": "running",
        "stage": "QUEUED",
        "progress": 0,
        "message": "Job queued…",
        "doc_id": document_id,
        "results": None,
        "error": None,
        "stage_details": {},
        "reward_history": [],
    }

    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, document_id, user_config),
        daemon=True,
    )
    thread.start()

    return JSONResponse({"job_id": job_id, "status": "running"})


@app.get("/status/{job_id}")
async def get_status(job_id: str) -> JSONResponse:
    """Return the current pipeline stage and progress percentage."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")

    return JSONResponse(
        {
            "job_id": job_id,
            "status": job["status"],
            "stage": job["stage"],
            "progress": job["progress"],
            "message": job.get("message", ""),
        }
    )


@app.get("/results/{job_id}")
async def get_results(job_id: str) -> JSONResponse:
    """Return the full chunk results once the pipeline is complete."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")

    if job["status"] == "error":
        raise HTTPException(500, f"Pipeline error: {job.get('message', 'Unknown error')}")

    if job["status"] != "complete":
        raise HTTPException(202, "Pipeline still running.")

    return JSONResponse(job["results"])


@app.get("/export/{job_id}/{fmt}")
async def export_results(job_id: str, fmt: str) -> Response:
    """Download results as json / csv / markdown."""
    job = job_store.get(job_id)
    if not job or job["status"] != "complete":
        raise HTTPException(404, "Results not available.")

    results = job["results"]
    chunks = results.get("chunks", [])

    if fmt == "json":
        payload = json.dumps(results, indent=2, ensure_ascii=False)
        return Response(
            content=payload,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="chunks_{job_id[:8]}.json"'},
        )

    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            ["index", "token_count", "chunk_score", "jsd_score",
             "boundary_score", "boundary_type", "entities", "text_preview"]
        )
        for c in chunks:
            ents = ", ".join(e.get("text", "") for e in c.get("entities", []))
            preview = c.get("text", "")[:120].replace("\n", " ")
            writer.writerow(
                [
                    c.get("chunk_index", ""),
                    c.get("token_count", ""),
                    c.get("chunk_score", ""),
                    c.get("jsd_score", ""),
                    c.get("boundary_score", ""),
                    c.get("boundary_type", ""),
                    ents,
                    preview,
                ]
            )
        return Response(
            content=buf.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="chunks_{job_id[:8]}.csv"'},
        )

    if fmt == "markdown":
        lines = []
        summary = results.get("summary", {})
        lines.append(f"# AutoChunker Results\n")
        lines.append(f"- Document type: {summary.get('doc_type')}")
        lines.append(f"- Domain: {summary.get('domain')}")
        lines.append(f"- Chunks: {summary.get('chunk_count')}")
        lines.append(f"- Mean score: {summary.get('mean_chunk_score')}\n")
        for c in chunks:
            ents = ", ".join(e.get("text", "") for e in c.get("entities", []))
            lines.append("---")
            lines.append(
                f"<!-- chunk_index: {c.get('chunk_index')} | "
                f"tokens: {c.get('token_count')} | "
                f"score: {c.get('chunk_score')} | "
                f"entities: {ents} -->"
            )
            lines.append(c.get("text", ""))
            lines.append("")
        return Response(
            content="\n".join(lines),
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="chunks_{job_id[:8]}.md"'},
        )

    raise HTTPException(400, f"Unsupported format '{fmt}'. Use json, csv, or markdown.")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})

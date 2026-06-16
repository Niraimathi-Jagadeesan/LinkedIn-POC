"""
LinkedIn AI Content Agent — Web UI
====================================
FastAPI backend with Server-Sent Events (SSE) for real-time pipeline progress.

Run:
    python ui_app.py
Then open http://localhost:8080
"""

import asyncio
import json
import queue
import threading
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langgraph.graph import END, StateGraph

from main import (
    retrieve_context,
    analyze_tone,
    generate_content,
    create_visual,
    publish,
    _route_visual,
    _route_review,
    AgentState,
)
from config.settings import get_settings
from linkedin.auth import get_access_token
from linkedin.api_client import LinkedInAPIClient
from linkedin.mock_client import MockLinkedInAPIClient, MOCK_PERSON_URN

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="LinkedIn AI Content Agent")

# Serve generated output files (images, flyers, carousels) at /outputs/*
_outputs_dir = Path("./data/outputs")
_outputs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(_outputs_dir)), name="outputs")

# ── Session store ─────────────────────────────────────────────────────────────

# session_id → {"event_queue", "review_event", "review_queue"}
_sessions: dict = {}

# ── Request / Response models ─────────────────────────────────────────────────

class StartRequest(BaseModel):
    topic: str
    user_context: str
    post_format: str
    carousel_image_mode: str = "shared"   # "shared" | "per_slide"


class ReviewRequest(BaseModel):
    action: str  # "approve" | "regenerate" | "cancel"


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def _node_wrapper(fn, step_id: str, step_label: str, eq: queue.Queue):
    """Wrap a pipeline node function to emit step_start / step_done events."""
    def wrapper(state: AgentState) -> AgentState:
        eq.put({"type": "step_start", "step": step_id, "label": step_label})
        result = fn(state)
        eq.put({"type": "step_done", "step": step_id})
        return result
    return wrapper


def _build_ui_graph(eq: queue.Queue, review_event: threading.Event, review_queue: queue.Queue):
    """Build the LangGraph pipeline wired to the UI session queues."""

    def ui_human_review(state: AgentState) -> AgentState:
        # Notify UI about the generated content and that we need a decision
        eq.put({"type": "step_start", "step": "review", "label": "Awaiting your review"})

        # Compute preview URL for any generated visual.
        # Append a millisecond timestamp so the browser never serves a cached
        # file from a previous generation when the same filename is reused.
        def _to_url(path: str | None) -> str | None:
            if not path:
                return None
            p = Path(path)
            if p.is_relative_to(Path("./data/outputs")) or "data/outputs" in str(p):
                return f"/outputs/{p.name}?v={int(time.time() * 1000)}"
            return None

        eq.put({
            "type": "review",
            "generated_text": state.get("generated_text", ""),
            "hashtags": state.get("hashtags", []),
            "post_format": state.get("post_format", ""),
            "image_url": _to_url(state.get("image_path")),
            "flyer_url": _to_url(state.get("flyer_path")),
            "carousel_url": _to_url(state.get("carousel_path")),
        })

        # Block until the user clicks Approve / Regenerate / Cancel
        review_event.wait()
        review_event.clear()
        action = review_queue.get()

        eq.put({"type": "step_done", "step": "review"})

        if action == "approve":
            return {**state, "review_action": "approve", "approved": True}
        if action == "regenerate":
            return {**state, "review_action": "regenerate", "approved": False}
        return {**state, "review_action": "cancel", "approved": False}

    wf = StateGraph(AgentState)

    wf.add_node("retrieve_context", _node_wrapper(retrieve_context, "retrieve", "Fetching LinkedIn history", eq))
    wf.add_node("analyze_tone",     _node_wrapper(analyze_tone,     "tone",     "Analyzing writing tone",    eq))
    wf.add_node("generate_content", _node_wrapper(generate_content, "generate", "Generating AI content",     eq))
    wf.add_node("create_visual",    _node_wrapper(create_visual,    "visual",   "Creating visual",           eq))
    wf.add_node("human_review",     ui_human_review)
    wf.add_node("publish",          _node_wrapper(publish,          "publish",  "Publishing to LinkedIn",    eq))

    wf.set_entry_point("retrieve_context")
    wf.add_edge("retrieve_context", "analyze_tone")
    wf.add_edge("analyze_tone", "generate_content")
    wf.add_conditional_edges(
        "generate_content",
        _route_visual,
        {"create_visual": "create_visual", "human_review": "human_review"},
    )
    wf.add_edge("create_visual", "human_review")
    wf.add_conditional_edges(
        "human_review",
        _route_review,
        {"publish": "publish", "generate_content": "generate_content", END: END},
    )
    wf.add_edge("publish", END)

    return wf.compile()


def _run_pipeline(session_id: str, topic: str, user_context: str, post_format: str,
                  carousel_image_mode: str = "shared"):
    """Runs the full pipeline in a background thread."""
    session = _sessions.get(session_id)
    if not session:
        return

    eq: queue.Queue    = session["event_queue"]
    re: threading.Event = session["review_event"]
    rq: queue.Queue    = session["review_queue"]

    try:
        # Clear settings cache so .env is re-read
        get_settings.cache_clear()
        settings = get_settings()

        # Resolve person URN
        if settings.mock_linkedin:
            person_urn = MOCK_PERSON_URN
        else:
            token = get_access_token()
            api_client = LinkedInAPIClient(token)
            profile = api_client.get_profile()
            person_urn = f"urn:li:person:{profile['sub']}"

        initial: AgentState = {
            "topic": topic,
            "user_context": user_context,
            "post_format": post_format,
            "person_urn": person_urn,
            "carousel_image_mode": carousel_image_mode,
            "past_posts_context": "",
            "last_post_text": "",
            "tone_profile": "",
            "generated_text": "",
            "hashtags": [],
            "image_prompt": "",
            "hook_line": None,
            "flyer_headline": None,
            "flyer_subtitle": None,
            "slides": None,
            "image_path": None,
            "flyer_path": None,
            "carousel_path": None,
            "review_action": None,
            "approved": False,
            "post_result": None,
            "error": None,
        }

        graph = _build_ui_graph(eq, re, rq)
        final = graph.invoke(initial)

        if final.get("approved") and final.get("post_result"):
            eq.put({
                "type": "complete",
                "success": True,
                "message": "Your post is now live on LinkedIn!",
                "post_id": final["post_result"].get("id", ""),
            })
        elif final.get("review_action") == "cancel":
            eq.put({"type": "complete", "success": False, "message": "Session ended — no post was published."})
        else:
            eq.put({"type": "complete", "success": False, "message": "Done."})

    except Exception as exc:
        eq.put({"type": "error", "message": str(exc)})
    finally:
        eq.put({"type": "done"})


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/api/start")
async def start_pipeline(req: StartRequest):
    if not req.topic.strip():
        return JSONResponse({"error": "Topic cannot be empty."}, status_code=400)
    if req.post_format not in ("text", "image", "flyer", "carousel"):
        return JSONResponse({"error": "Invalid post format."}, status_code=400)

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "event_queue": queue.Queue(),
        "review_event": threading.Event(),
        "review_queue": queue.Queue(),
    }

    threading.Thread(
        target=_run_pipeline,
        args=(session_id, req.topic.strip(), req.user_context.strip(),
              req.post_format, req.carousel_image_mode),
        daemon=True,
    ).start()

    return {"session_id": session_id}


@app.get("/api/stream/{session_id}")
async def stream_events(session_id: str, request: Request):
    if session_id not in _sessions:
        return JSONResponse({"error": "Session not found."}, status_code=404)

    eq: queue.Queue = _sessions[session_id]["event_queue"]

    async def generator():
        while True:
            if await request.is_disconnected():
                break
            try:
                event = eq.get_nowait()
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "done":
                    break
            except queue.Empty:
                await asyncio.sleep(0.1)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/review/{session_id}")
async def submit_review(session_id: str, req: ReviewRequest):
    if session_id not in _sessions:
        return JSONResponse({"error": "Session not found."}, status_code=404)
    if req.action not in ("approve", "regenerate", "cancel"):
        return JSONResponse({"error": "Invalid action."}, status_code=400)

    session = _sessions[session_id]
    session["review_queue"].put(req.action)
    session["review_event"].set()
    return {"ok": True}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting LinkedIn AI Content Agent UI at http://localhost:8080")
    uvicorn.run("ui_app:app", host="0.0.0.0", port=8080, reload=False)

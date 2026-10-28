"""Cockpit server: serves the manual and the cockpit, and streams steered generations as server-sent events.

    .venv/bin/python server.py            # then open http://127.0.0.1:7860

The page is up at once; the model loads in the background (about 35 seconds) and /api/status says when it is ready.
"""

import asyncio
import base64
import json
import threading
import time
from dataclasses import asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import personas as persona_files
from steer import FIELD_BINS, MODEL_ID, Intervention, Steerer, field_bytes, load_hub_adapter, probe_list

ROOT = Path(__file__).parent
STATIC = ROOT / "static"
MAX_ROWS = 6

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC), name="static")
steerer: Steerer | None = None
personas: list[dict] = []
loading = {"started": time.monotonic(), "error": None}
stop = threading.Event()


@app.middleware("http")
async def revalidate(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache"  # edits to static/ show up on the next reload
    return response


class InterventionIn(BaseModel):
    layer: int
    neuron: int
    dose: float = Field(ge=-persona_files.MAX_DOSE, le=persona_files.MAX_DOSE)
    scale: float | None = Field(default=None, gt=0)


class RowIn(BaseModel):
    label: str
    interventions: list[InterventionIn] = []


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class GenerateIn(BaseModel):
    messages: list[Message] = Field(min_length=1)
    rows: list[RowIn] = Field(min_length=1, max_length=MAX_ROWS)
    max_new_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.8, gt=0, le=1)
    top_k: int = Field(default=20, ge=0)
    repetition_penalty: float = Field(default=1.1, gt=0, le=2)
    seed: int | None = None
    focus_row: int | None = Field(default=None, ge=0, description="Row drawn in the probe field; the last by default")


class PersonaIn(BaseModel):
    name: str
    effect: str
    interventions: list[InterventionIn]
    sweet: list[float] | None = None
    author: str | None = None


def load_personas() -> list[dict]:
    """Persona files; those backed by a Hub adapter take their doses and scales from it."""
    loaded = persona_files.load_all()
    for persona in loaded:
        if "hub" in persona:
            persona["interventions"] = [asdict(iv) for iv in load_hub_adapter(persona["hub"])]
    return loaded


def load_model():
    global steerer
    try:
        steerer = Steerer()
    except Exception as error:  # reported by /api/status instead of dying in a background thread
        loading["error"] = f"{type(error).__name__}: {error}"
        print(f"Loading the model failed: {loading['error']}", flush=True)
        return
    print(f"Model ready after {time.monotonic() - loading['started']:.0f} s", flush=True)


def model() -> Steerer:
    if steerer is None:
        raise HTTPException(503, loading["error"] or "The model is still loading.")
    return steerer


@app.get("/")
def manual():
    return FileResponse(STATIC / "index.html")


@app.get("/cockpit")
def cockpit():
    return FileResponse(STATIC / "cockpit.html")


@app.get("/api/status")
def status():
    return {
        "model": MODEL_ID,
        "ready": steerer is not None,
        "error": loading["error"],
        "elapsed": round(time.monotonic() - loading["started"]),
        "personas": len(personas),
    }


@app.get("/api/info")
def info():
    loaded = model()
    return {
        "model": MODEL_ID,
        "n_layers": loaded.n_layers,
        "n_neurons": loaded.n_neurons,
        "max_rows": MAX_ROWS,
        "groups": persona_files.GROUPS,
        "personas": personas,
        "field_bins": FIELD_BINS,
        "field_idle": base64.b64encode(field_bytes(loaded.field_idle)).decode(),
    }


@app.post("/api/personas", status_code=201)
def save_persona(request: PersonaIn):
    """Save the electrodes as a community persona in personas/, ready to commit and share."""
    persona = {
        "name": request.name.strip(),
        "group": "community",
        "effect": request.effect.strip(),
        "interventions": [iv.model_dump(exclude_none=True) for iv in request.interventions],
        "source": "Made in the neurollm cockpit",
    }
    if request.sweet:
        persona["sweet"] = request.sweet
    if request.author and request.author.strip():
        persona["author"] = request.author.strip()
    try:
        saved = persona_files.save(persona)
    except FileExistsError as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    personas.append(saved)
    personas.sort(key=persona_files.sort_key)
    return {"persona": saved, "file": f"personas/{saved['id']}.json", "personas": personas}


@app.post("/api/stop")
def stop_generation():
    stop.set()
    return {"stopping": steerer is not None and steerer.lock.locked()}


@app.post("/api/generate")
async def generate(request: GenerateIn):
    loaded = model()
    if loaded.lock.locked():
        raise HTTPException(409, "A generation is already running. Stop it or wait for it to finish.")
    try:
        rows = [[Intervention(**iv.model_dump()) for iv in row.interventions] for row in request.rows]
        for row in rows:
            for iv in row:
                loaded.direction(iv.layer, iv.neuron)  # validates the coordinates before streaming starts
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    focus_row = len(rows) - 1 if request.focus_row is None else request.focus_row
    if focus_row >= len(rows):
        raise HTTPException(422, f"focus_row {focus_row} is past the last of {len(rows)} rows.")

    loop = asyncio.get_running_loop()
    events: asyncio.Queue = asyncio.Queue()
    stop.clear()

    def on_step(pieces, probe, field):
        event = {"pieces": pieces, "probe": probe, "field": base64.b64encode(field).decode()}
        loop.call_soon_threadsafe(events.put_nowait, event)

    def run():
        try:
            texts = loaded.generate(
                [m.model_dump() for m in request.messages],
                rows,
                on_step=on_step,
                stop=stop,
                focus_row=focus_row,
                **request.model_dump(exclude={"messages", "rows", "focus_row"}),
            )
            event = {"done": True, "texts": texts, "stopped": stop.is_set()}
        except Exception as error:  # surfaced in the UI rather than killing the stream silently
            event = {"error": f"{type(error).__name__}: {error}"}
        loop.call_soon_threadsafe(events.put_nowait, event)

    async def stream():
        yield f"data: {json.dumps({'probes': probe_list(rows), 'focus_row': focus_row})}\n\n"
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        try:
            while True:
                event = await events.get()
                yield f"data: {json.dumps(event)}\n\n"
                if "done" in event or "error" in event:
                    break
        finally:
            stop.set()  # the client went away, or we are finished

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    personas = load_personas()
    threading.Thread(target=load_model, daemon=True).start()
    print(f"Open http://{args.host}:{args.port} ({len(personas)} personas; {MODEL_ID} loads in the background)", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

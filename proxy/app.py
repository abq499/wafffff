# proxy/app.py
import os
import json
import time
import pathlib
from httpx import AsyncClient
from fastapi import FastAPI, Request, Response
import asyncio
from collections import defaultdict

# Config giu nguyen
TARGET = os.getenv("TARGET_HOST", "http://localhost:8080")
MODEL = os.getenv("MODEL_API", "http://localhost:8001/score")
THRESHOLD = float(os.getenv("THRESHOLD", "0.8"))
LOG_PATH = os.getenv("REQUEST_LOG_PATH", "/data/requests.jsonl")

# --- CAU HINH LAYER 1 (DDoS Detection) ---
# Bai bao section 3.1: Loc DDoS o lop 1 truoc khi vao lop 2
DDOS_WINDOW = 10  # Cua so thoi gian (giay)
DDOS_LIMIT = 50   # Gioi han request trong cua so thoi gian
request_counters = defaultdict(list)

if LOG_PATH == "/data/requests.jsonl" and not pathlib.Path("/data").exists():
    local_data = pathlib.Path(__file__).parent.parent / "data"
    local_data.mkdir(parents=True, exist_ok=True)
    LOG_PATH = str(local_data / "requests.jsonl")

pathlib.Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

app = FastAPI()

async def call_model(payload: dict):
    async with AsyncClient(timeout=10.0) as client:
        try:
            r = await client.post(MODEL, json=payload)
            return r.json()
        except Exception as e:
            return {"error": str(e), "score": 0.0, "info": "model-error"}

async def write_log(entry: dict):
    def _sync_write(e, p):
        try:
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        except Exception as ex:
            pass
    await asyncio.to_thread(_sync_write, entry, LOG_PATH)

@app.api_route("/{path:path}", methods=["GET","POST","PUT","DELETE","PATCH","OPTIONS"])
async def proxy(path: str, request: Request):
    client_ip = request.client.host
    current_time = time.time()
    
    # --- LAYER 1: DDoS Detection (Simulation) ---
    # Lam sach bo dem cu
    request_counters[client_ip] = [t for t in request_counters[client_ip] if t > current_time - DDOS_WINDOW]
    request_counters[client_ip].append(current_time)
    
    # Kiem tra neu vuot qua gioi han (High Rate Traffic) -> Block ngay
    if len(request_counters[client_ip]) > DDOS_LIMIT:
        entry = {
            "ts": int(current_time),
            "method": request.method,
            "path": "/" + path,
            "query": str(request.query_params),
            "body": "",
            "model_score": 1.0, 
            "model_info": "DDoS-Layer-1",
            "response_status": 429,
            "action": "blocked-ddos"
        }
        await write_log(entry)
        return Response(content="Blocked by WAF Layer 1 (DDoS Detection)", status_code=429)

    # --- LAYER 2: SQLi/XSS Detection (Deep Learning) ---
    body_bytes = await request.body()
    body_text = body_bytes.decode('utf-8', errors='ignore')
    payload_for_model = {
        "method": request.method,
        "path": "/" + path,
        "headers": dict(request.headers),
        "body": body_text,
        "query": str(request.query_params)
    }

    # Goi LSTM Model
    model_task = asyncio.create_task(call_model(payload_for_model))

    # Forward request toi WebGoat
    async with AsyncClient(follow_redirects=False) as client:
        target_url = f"{TARGET}/{path}"
        try:
            forwarded = await client.request(
                request.method,
                target_url,
                headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
                content=body_bytes,
                params=request.query_params
            )
        except Exception as e:
            return Response(f"Error connecting to WebGoat: {e}", status_code=502)

    model_res = await model_task
    score = model_res.get("score", 0.0)
    info = model_res.get("info", "")

    headers = dict(forwarded.headers)
    headers["X-WAF-Model-Score"] = str(score)
    headers["X-WAF-Model-Info"] = str(info)
    
    if "location" in headers:
        try:
            target_host_port = TARGET.replace("http://", "").replace("https://", "")
            proxy_host = "localhost:8010" 
            headers["location"] = headers["location"].replace(target_host_port, proxy_host)
        except: pass

    entry = {
        "ts": int(time.time()),
        "method": request.method,
        "path": "/" + path,
        "query": payload_for_model["query"],
        "body": payload_for_model["body"],
        "model_score": score,
        "model_info": info,
        "response_status": forwarded.status_code
    }

    # Block neu vuot qua nguong (Threshold) cua LSTM
    if score >= THRESHOLD:
        entry["action"] = "blocked-threshold"
        await write_log(entry)
        return Response(content=f"Blocked by DL-WAF Layer 2 (Score={score})", status_code=403, headers=headers)

    entry["action"] = "allowed"
    await write_log(entry)

    return Response(content=forwarded.content, status_code=forwarded.status_code, headers=headers)

if __name__ == "__main__":
    import uvicorn
    # Port giu nguyen 8010 nhu yeu cau
    uvicorn.run("app:app", host="0.0.0.0", port=8010, reload=True)
# proxy/app.py
import os
import json
import time
import pathlib
from httpx import AsyncClient
from fastapi import FastAPI, Request, Response
import asyncio

# Config (env vars)
TARGET = os.getenv("TARGET_HOST", "http://localhost:8080")
MODEL = os.getenv("MODEL_API", "http://localhost:8001/score")
THRESHOLD = float(os.getenv("THRESHOLD", "0.8"))

# Log file path (inside container it should be /data/requests.jsonl if you mount ./data:/data)
LOG_PATH = os.getenv("REQUEST_LOG_PATH", "/data/requests.jsonl")  # default for docker-compose
# If running locally (not docker) and ./data doesn't exist, fall back to local ./data/requests.jsonl
if LOG_PATH == "/data/requests.jsonl" and not pathlib.Path("/data").exists():
    local_data = pathlib.Path(__file__).parent.parent / "data"
    local_data.mkdir(parents=True, exist_ok=True)
    LOG_PATH = str(local_data / "requests.jsonl")

# Ensure directory exists
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
    """Write log entry (dict) as JSON line. Run in thread to avoid blocking event loop."""
    def _sync_write(e, p):
        try:
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        except Exception as ex:
            print("log write error:", ex)
    await asyncio.to_thread(_sync_write, entry, LOG_PATH)

@app.api_route("/{path:path}", methods=["GET","POST","PUT","DELETE","PATCH","OPTIONS"])
async def proxy(path: str, request: Request):
    # read body and prepare payload for model
    body_bytes = await request.body()
    body_text = body_bytes.decode('utf-8', errors='ignore')
    payload_for_model = {
        "method": request.method,
        "path": "/" + path,
        "headers": dict(request.headers),
        "body": body_text,
        "query": str(request.query_params)
    }

    # call model asynchronously (non-blocking)
    model_task = asyncio.create_task(call_model(payload_for_model))

    # forward original request to target (do not follow redirects here)
    async with AsyncClient(follow_redirects=False) as client:
        target_url = f"{TARGET}/{path}"
        forwarded = await client.request(
            request.method,
            target_url,
            headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
            content=body_bytes,
            params=request.query_params
        )

    # await model result
    model_res = await model_task
    score = model_res.get("score", 0.0) if isinstance(model_res, dict) else 0.0
    info = model_res.get("info", "") if isinstance(model_res, dict) else ""

    # prepare headers to return (copy from forwarded)
    headers = dict(forwarded.headers)
    headers["X-WAF-Model-Score"] = str(score)
    headers["X-WAF-Model-Info"] = str(info)

    # rewrite Location header so redirects keep using proxy port (basic replacement)
    if "location" in headers:
        try:
            # replace occurrences of the target host:port with proxy host:port
            target_host_port = TARGET.replace("http://", "").replace("https://", "")
            # common case: location contains exact http://localhost:8080/...
            # replace target host->proxy host (assume client uses localhost:PORT where proxy runs)
            proxy_host = "localhost:8010"
            headers["location"] = headers["location"].replace(target_host_port, proxy_host)
        except Exception:
            pass

    # build log entry (timestamp, request, model result, action)
    entry = {
        "ts": int(time.time()),
        "method": request.method,
        "path": payload_for_model["path"],
        "query": payload_for_model["query"],
        "body": payload_for_model["body"],
        "model_score": score,
        "model_info": info,
        "response_status": forwarded.status_code
    }

    # immediate-block for high-confidence rule matches (e.g., sqli-regex, xss-regex, lfi-regex)
    try:
        if isinstance(info, str) and any(tag in info for tag in ("sqli-regex","xss-regex","lfi-regex")):
            entry["action"] = "blocked-immediate-rule"
            await write_log(entry)
            return Response(content=f"Blocked by DL-WAF (rule match: {info})", status_code=403, headers={
                "X-WAF-Model-Score": str(score),
                "X-WAF-Model-Info": str(info)
            })
    except Exception:
        # proceed if something unexpected happens
        pass

    # detection-only / threshold block
    if score >= THRESHOLD:
        entry["action"] = "blocked-threshold"
        await write_log(entry)
        return Response(content=f"Blocked by DL-WAF (score={score})", status_code=403, headers={
            "X-WAF-Model-Score": str(score),
            "X-WAF-Model-Info": str(info)
        })

    # otherwise log and forward response content/status/headers to client
    entry["action"] = "allowed"
    await write_log(entry)

    return Response(content=forwarded.content, status_code=forwarded.status_code, headers=headers)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8010, reload=True)

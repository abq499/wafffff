#model_api/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import urllib.parse
import re
import random

app = FastAPI()

class Req(BaseModel):
    method: str
    path: str
    headers: dict
    body: str
    query: str

# strong regex patterns for SQLi / XSS / Local File Inclusion etc.
SQLI_PATTERNS = [
    re.compile(r"(?i)(\bunion\b.*\bselect\b)"),            # union select
    re.compile(r"(?i)(\b(select|update|delete|insert)\b.*\bfrom\b)"),
    re.compile(r"(?i)(\b'or\b|\b' or\b|\b\"or\b).*\=|\b1=1\b"),  # OR '1'='1' etc
    re.compile(r"(?i)(--|#\s*$)"),                          # SQL comment
    re.compile(r"(?i)(\bexec\b|\bexecute\b|\bbenchmark\b)"),
]

XSS_PATTERNS = [
    re.compile(r"(?i)<script\b"),
    re.compile(r"(?i)on\w+\s*="),  # onmouseover=...
    re.compile(r"(?i)javascript:"),
]

LFI_PATTERNS = [
    re.compile(r"(\.\./)+"),  # ../.. patterns
    re.compile(r"(?i)(etc/passwd|boot.ini)")
]

def naive_score(req: Req):
    # normalize / decode
    s = (req.path + " " + req.query + " " + (req.body or "")).lower()
    try:
        s = urllib.parse.unquote_plus(s)
    except Exception:
        pass

    score = 0.0
    info = []

    # strong patterns -> big jump
    for p in SQLI_PATTERNS:
        if p.search(s):
            score += 0.6
            info.append("sqli-regex")
    for p in XSS_PATTERNS:
        if p.search(s):
            score += 0.6
            info.append("xss-regex")
    for p in LFI_PATTERNS:
        if p.search(s):
            score += 0.6
            info.append("lfi-regex")

    # weaker keywords
    weak_keywords = ["select ","union ","wget ","curl ","base64","drop ","insert ","update ","/etc/","<script","alert("]
    for k in weak_keywords:
        if k in s:
            score += 0.08
            info.append("kw:"+k.strip())

    # small noise / normalize score
    score = min(1.0, score + random.random()*0.05)
    if not info:
        info = ["none"]
    return {"score": round(score,3), "info": ",".join(info)}

@app.post("/score")
def score(req: Req):
    # use naive_score (regex+keywords) for now
    r = naive_score(req)
    return r

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)

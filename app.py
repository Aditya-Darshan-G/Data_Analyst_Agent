import os
import networkx as nx
import re
import json
import base64
import tempfile
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # kept as requested
import io
import os
import re
import json
import base64
import tempfile
import subprocess
import logging
from io import BytesIO
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi import FastAPI
from dotenv import load_dotenv

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Optional image conversion
try:
    from PIL import Image
    PIL_READY = True
except Exception:
    PIL_READY = False

# Optional PDF libs (use if available)
try:
    import pdfplumber
    PDFPLUMBER_READY = True
except Exception:
    PDFPLUMBER_READY = False

try:
    from pypdf import PdfReader
    PYPDF_READY = True
except Exception:
    PYPDF_READY = False

# NEW: optional pdfminer text extractor
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    PDFMINER_READY = True
except Exception:
    PDFMINER_READY = False

import zlib  # for raw fallback on Flate streams

# NEW: DB support
import sqlite3

# LangChain / LLM imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="TDS Data Analyst Agent")

# ---- CORS (needed if prof site calls from browser) ----
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- AI Pipe / OpenAI-compatible setup ----
def get_ai_pipe_token() -> str:
    """
    Pick the first defined OPENAI_API_KEY_* from env.
    Falls back to OPENAI_API_KEY if present.
    """
    for i in range(1, 6):
        k = os.getenv(f"OPENAI_API_KEY_{i}")
        if k:
            return k
    return os.getenv("OPENAI_API_KEY")

OPENAI_API_KEY = get_ai_pipe_token()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://aipipe.org/openai/v1")
if not OPENAI_API_KEY:
    raise RuntimeError("No OPENAI_API_KEY or OPENAI_API_KEY_{i} found in environment.")

# LangChain OpenAI chat model pointing at AI Pipe
chat_model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),  # <- default to gpt-5-mini
    temperature=0,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    timeout=int(os.getenv("LLM_TIMEOUT_SECONDS", "240")),
)

LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 240))


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML interface"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found</h1><p>Please ensure index.html is in the same directory as app.py</p>", status_code=404)


def parse_keys_and_types(raw_qs: str):
    """
    Parses the key/type section from the questions file.
    Returns:
        ordered_keys: list of keys in order
        cast_lookup: dict key -> casting function
    """
    import re
    keytype_pattern = r"-\s*`([^`]+)`\s*:\s*(\w+)"
    keytype_matches = re.findall(keytype_pattern, raw_qs)
    cast_map = {
        "number": float,
        "string": str,
        "integer": int,
        "int": int,
        "float": float
    }
    cast_lookup = {key: cast_map.get(t.lower(), str) for key, t in keytype_matches}
    ordered_keys = [k for k, _ in keytype_matches]
    return ordered_keys, cast_lookup


# -----------------------------
# PDF helpers (updated)
# -----------------------------

def _pdf_fallback_text_from_bytes(pdf_bytes: bytes) -> str:
    """
    Very rough, pure-Python fallback:
    - Try to locate Flate-compressed streams and decompress with zlib.
    - If nothing decompresses, extract long printable sequences from raw bytes.
    """
    text_chunks = []

    # Try Flate streams (stream ... endstream)
    try:
        raw = pdf_bytes
        # find all streams
        for m in re.finditer(rb"stream\s*([\s\S]*?)\s*endstream", raw, flags=re.MULTILINE):
            block = m.group(1)
            # common: streams begin with newline; try zlib
            for wbits in (15, -15):
                try:
                    dec = zlib.decompress(block, wbits)
                    try:
                        text_chunks.append(dec.decode("utf-8", "ignore"))
                    except Exception:
                        text_chunks.append(dec.decode("latin-1", "ignore"))
                    break
                except Exception:
                    continue
    except Exception:
        pass

    combined = "\n".join([t for t in text_chunks if t and t.strip()])

    if not combined:
        # As last resort, pull visible ASCII-ish sequences directly from bytes
        try:
            ascii_like = re.findall(rb"[ -~\t\r\n]{5,}", pdf_bytes)  # printable
            combined = "\n".join([a.decode("latin-1", "ignore") for a in ascii_like])
        except Exception:
            combined = ""

    return combined.strip()


def extract_pdf_to_dataframe(pdf_bytes: bytes) -> (pd.DataFrame, str):
    """
    Attempt to extract tables and/or text from a PDF.
    Return (df, info_text). df is:
      - the largest detected table (if any), or
      - a single-column DataFrame with 'text' if only text found.
    info_text is a brief diagnostic string.
    """
    # 1) pdfplumber: tables + text (try empty password if encrypted)
    if PDFPLUMBER_READY:
        try:
            tables = []
            texts = []
            # pdfplumber will raise on password; try with and without empty password
            try:
                with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                    pages_iter = pdf.pages
                    for page in pages_iter:
                        try:
                            t = page.extract_tables() or []
                            for tbl in t:
                                if not tbl:
                                    continue
                                tbl = [row for row in tbl if any(cell is not None and str(cell).strip() for cell in row)]
                                if not tbl:
                                    continue
                                header = tbl[0]
                                body = tbl[1:] if len(tbl) > 1 else []
                                if not header or len(set([str(h).strip() for h in header if h is not None])) != len(header):
                                    cols = [f"col_{i}" for i in range(len(header or []))]
                                    df_tbl = pd.DataFrame(body, columns=cols if body and len(body[0]) == len(cols) else None)
                                else:
                                    df_tbl = pd.DataFrame(body, columns=[str(h).strip() if h is not None else "" for h in header])
                                if not df_tbl.empty:
                                    df_tbl = df_tbl.dropna(how="all", axis=1)
                                if not df_tbl.empty:
                                    tables.append(df_tbl)
                        except Exception:
                            pass
                        try:
                            txt = page.extract_text() or ""
                            if txt.strip():
                                texts.append(txt)
                        except Exception:
                            pass
            except Exception:
                # try again with empty password
                try:
                    with pdfplumber.open(BytesIO(pdf_bytes), password="") as pdf:
                        for page in pdf.pages:
                            try:
                                t = page.extract_tables() or []
                                for tbl in t:
                                    if not tbl:
                                        continue
                                    tbl = [row for row in tbl if any(cell is not None and str(cell).strip() for cell in row)]
                                    if not tbl:
                                        continue
                                    header = tbl[0]
                                    body = tbl[1:] if len(tbl) > 1 else []
                                    if not header or len(set([str(h).strip() for h in header if h is not None])) != len(header):
                                        cols = [f"col_{i}" for i in range(len(header or []))]
                                        df_tbl = pd.DataFrame(body, columns=cols if body and len(body[0]) == len(cols) else None)
                                    else:
                                        df_tbl = pd.DataFrame(body, columns=[str(h).strip() if h is not None else "" for h in header])
                                    if not df_tbl.empty:
                                        df_tbl = df_tbl.dropna(how="all", axis=1)
                                    if not df_tbl.empty:
                                        tables.append(df_tbl)
                            except Exception:
                                pass
                            try:
                                txt = page.extract_text() or ""
                                if txt.strip():
                                    texts.append(txt)
                            except Exception:
                                pass
                except Exception as e2:
                    log.warning("pdfplumber failed (even with empty password): %s", e2)

            if tables:
                best = max(tables, key=lambda d: (len(d) * max(1, len(d.columns))))
                return best.reset_index(drop=True), "pdfplumber:table"
            if texts:
                text_joined = "\n".join(texts).strip()
                if text_joined:
                    return pd.DataFrame({"text": [text_joined]}), "pdfplumber:text"
        except Exception as e:
            log.warning("pdfplumber failed: %s", e)

    # 2) pypdf text-only fallback (try decrypt with empty password if encrypted)
    if PYPDF_READY:
        try:
            reader = PdfReader(BytesIO(pdf_bytes))
            if getattr(reader, "is_encrypted", False):
                try:
                    reader.decrypt("")  # try empty password
                except Exception:
                    pass
            pages_text = []
            for p in reader.pages:
                try:
                    t = p.extract_text() or ""
                    if t.strip():
                        pages_text.append(t)
                except Exception:
                    continue
            if pages_text:
                return pd.DataFrame({"text": ["\n".join(pages_text).strip()]}), "pypdf:text"
        except Exception as e:
            log.warning("pypdf failed: %s", e)

    # 2.5) NEW: pdfminer text-only fallback
    if PDFMINER_READY:
        try:
            txt = pdfminer_extract_text(BytesIO(pdf_bytes)) or ""
            if txt.strip():
                return pd.DataFrame({"text": [txt.strip()]}), "pdfminer:text"
        except Exception as e:
            log.warning("pdfminer failed: %s", e)

    # 3) pure-Python raw fallback
    fallback_text = _pdf_fallback_text_from_bytes(pdf_bytes)
    if fallback_text:
        return pd.DataFrame({"text": [fallback_text]}), "raw:flate-or-ascii"

    # nothing found
    raise HTTPException(400, "No extractable content found in PDF (scanned images or unsupported PDF).")


# -----------------------------
# Tools
# -----------------------------

@tool
def scrape_url_to_dataframe(target_url: str) -> Dict[str, Any]:
    """
    Fetch a URL and return data as a DataFrame (supports HTML tables, CSV, Excel, Parquet, JSON, PDF, and plain text).
    Always returns {"status": "success", "data": [...], "columns": [...]} if fetch works.
    """
    print(f"Scraping URL: {target_url}")
    try:
        from io import BytesIO, StringIO
        from bs4 import BeautifulSoup

        req_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/138.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.google.com/",
        }

        r = requests.get(target_url, headers=req_headers, timeout=20)
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "").lower()

        frame = None

        # --- PDF ---
        if "application/pdf" in content_type or target_url.lower().endswith(".pdf"):
            df_pdf, _info = extract_pdf_to_dataframe(r.content)
            frame = df_pdf

        # --- CSV ---
        elif "text/csv" in content_type or target_url.lower().endswith(".csv"):
            frame = pd.read_csv(BytesIO(r.content))

        # --- Excel ---
        elif any(target_url.lower().endswith(ext) for ext in (".xls", ".xlsx")) or "spreadsheetml" in content_type:
            frame = pd.read_excel(BytesIO(r.content))

        # --- Parquet ---
        elif target_url.lower().endswith(".parquet"):
            frame = pd.read_parquet(BytesIO(r.content))

        # --- JSON ---
        elif "application/json" in content_type or target_url.lower().endswith(".json"):
            try:
                data = r.json()
                frame = pd.json_normalize(data)
            except Exception:
                frame = pd.DataFrame([{"text": r.text}])

        # --- HTML / Fallback ---
        elif "text/html" in content_type or re.search(r'/wiki/|\.org|\.com', target_url, re.IGNORECASE):
            html_text = r.text
            # Try HTML tables first
            try:
                html_tables = pd.read_html(StringIO(html_text), flavor="bs4")
                if html_tables:
                    frame = html_tables[0]
            except ValueError:
                pass

            # If no table found, fallback to plain text
            if frame is None:
                soup_obj = BeautifulSoup(html_text, "html.parser")
                page_text = soup_obj.get_text(separator="\n", strip=True)
                frame = pd.DataFrame({"text": [page_text]})

        # --- Unknown type fallback ---
        else:
            frame = pd.DataFrame({"text": [r.text]})

        # --- Normalize columns ---
        frame.columns = frame.columns.map(str).str.replace(r'\[.*\]', '', regex=True).str.strip()

        return {
            "status": "success",
            "data": frame.to_dict(orient="records"),
            "columns": frame.columns.tolist()
        }

    except Exception as err:
        return {"status": "error", "message": str(err)}


# -----------------------------
# Utilities for executing code safely
# -----------------------------
def clean_llm_output(llm_text: str) -> Dict:
    """
    Extract JSON object from LLM output robustly.
    Returns dict or {"error": "..."}
    """
    try:
        if not llm_text:
            return {"error": "Empty LLM output"}
        # remove triple-fence markers if present
        content = re.sub(r"^```(?:json)?\s*", "", llm_text.strip())
        content = re.sub(r"\s*```$", "", content)
        # find outermost JSON object by scanning for balanced braces
        lbrace = content.find("{")
        rbrace = content.rfind("}")
        if lbrace == -1 or rbrace == -1 or rbrace <= lbrace:
            return {"error": "No JSON object found in LLM output", "raw": content}
        json_candidate = content[lbrace:rbrace+1]
        try:
            return json.loads(json_candidate)
        except Exception as e:
            # fallback: try last balanced pair scanning backwards
            for i in range(rbrace, lbrace, -1):
                cand = content[lbrace:i+1]
                try:
                    return json.loads(cand)
                except Exception:
                    continue
            return {"error": f"JSON parsing failed: {str(e)}", "raw": json_candidate}
    except Exception as e:
        return {"error": str(e)}

# --- helper: convert bytes -> data URI for multimodal LLM ---
def _bytes_to_data_uri(img_bytes: bytes, mime="image/png") -> str:
    b64 = base64.b64encode(img_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"

# ---------- NEW: normalization & numeric coercion helpers ----------
def _normalize_columns_inplace(df: pd.DataFrame):
    df.columns = (
        pd.Index(df.columns)
        .map(str)
        .map(lambda s: s.replace("\xa0", " ").strip())
        .map(lambda s: re.sub(r"\s+", "_", s.lower()))
        .map(lambda s: re.sub(r"[^0-9a-z_]", "", s))
    )

def _coerce_numeric_columns_inplace(df: pd.DataFrame):
    for c in list(df.columns):
        if df[c].dtype == object:
            s = df[c].astype(str)
            s2 = s.str.replace(r"[,$%]", "", regex=True).str.replace(r"\s", "", regex=True)
            coerced = pd.to_numeric(s2, errors="coerce")
            if coerced.notna().sum() >= max(3, int(0.5 * len(df))):
                df[c] = coerced

def run_vision_extraction(questions_text: str, image_uris: List[str]) -> Dict:
    """
    Send images + instructions to a vision-capable LLM and expect JSON-only output.
    No Python OCR: the model must visually read the images.
    """
    system_msg = (
        "You are a precise data extraction assistant. "
        "You will be given one or more images (e.g., scanned bank passbooks) and a set of questions. "
        "Answer ONLY by visually inspecting the images. "
        "DO NOT suggest Python OCR libraries like pytesseract; they are unavailable. "
        "Return strictly valid JSON as the final answer. No extra text."
    )

    # Build a multimodal human message: text + images
    msg_content = [{"type": "text", "text": questions_text}]
    for uri in image_uris:
        msg_content.append({"type": "image_url", "image_url": {"url": uri}})

    # Call the configured LLM
    resp = chat_model.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=msg_content),
    ])
    text = resp.content if hasattr(resp, "content") else str(resp)
    parsed = clean_llm_output(text)
    return parsed


SCRAPE_FUNC = r'''
from typing import Dict, Any
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5
        )
        response.raise_for_status()
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "data": [],
            "columns": []
        }

    soup = BeautifulSoup(response.text, "html.parser")
    tables = pd.read_html(response.text)

    if tables:
        df = tables[0]  # Take first table
        df.columns = [str(c).strip() for c in df.columns]
        
        # Ensure all columns are unique and string
        df.columns = [str(col) for col in df.columns]

        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": list(df.columns)
        }
    else:
        # Fallback to plain text
        text_data = soup.get_text(separator="\n", strip=True)

        # Try to detect possible "keys" from text like Runtime, Genre, etc.
        detected_cols = set(re.findall(r"\b[A-Z][a-zA-Z ]{2,15}\b", text_data))
        df = pd.DataFrame([{}])  # start empty
        for col in detected_cols:
            df[col] = None

        if df.empty:
            df["text"] = [text_data]

        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": list(df.columns)
        }
'''


def write_and_run_temp_python(code: str, injected_pickle: str = None, timeout: int = 60) -> Dict[str, Any]:
    """
    Write a temp python file which:
      - provides a safe environment (imports)
      - loads df/from pickle if provided into df and data variables
      - defines a robust plot_to_base64() helper that ensures < 100kB (attempts resizing/conversion)
      - executes the user code (which should populate `results` dict)
      - prints json.dumps({"status":"success","result":results})
    Returns dict with parsed JSON or error details.
    """
    # create file content
    bootstrap_lines = [
        "import json, sys, gc, types, re, math",
        "import pandas as pd, numpy as np",
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "from io import BytesIO",
        "import base64",
        # Avoid hard seaborn dependency in the sandbox
        "try:\n    import seaborn as sns\nexcept Exception:\n    sns = None",
    ]
    if PIL_READY:
        bootstrap_lines.append("from PIL import Image")
    # NEW: loader that can handle dict payloads (tables) or a single df and expose helpers/short names
    if injected_pickle:
        bootstrap_lines.append(f"_payload = pd.read_pickle(r'''{injected_pickle}''')\n")
        bootstrap_lines.append("tables = {}\n")
        bootstrap_lines.append("df = None\n")
        bootstrap_lines.append("data = {}\n")
        bootstrap_lines.append(
            "def _sanitize(name):\n"
            "    name = re.sub(r'[^0-9a-zA-Z_]+', '_', str(name))\n"
            "    if name and name[0].isdigit():\n"
            "        name = '_' + name\n"
            "    return name\n"
        )
        # helpers
        bootstrap_lines.append(
            "def find_column(df, candidates):\n"
            "    cols = [c for c in df.columns]\n"
            "    lower = {c.lower(): c for c in cols}\n"
            "    for cand in candidates:\n"
            "        k = str(cand).lower().strip()\n"
            "        if k in lower:\n"
            "            return lower[k]\n"
            "    # fuzzy contains\n"
            "    for cand in candidates:\n"
            "        k = str(cand).lower().strip()\n"
            "        for c in cols:\n"
            "            if k in c.lower():\n"
            "                return c\n"
            "    return None\n"
        )
        bootstrap_lines.append(
            "def smart_to_numeric(s):\n"
            "    s = s.astype(str).str.replace(r'[,$%]', '', regex=True).str.replace(r'\\s','', regex=True)\n"
            "    return pd.to_numeric(s, errors='coerce')\n"
        )
        bootstrap_lines.append(
            "def coerce_numeric_columns(df):\n"
            "    for c in list(df.columns):\n"
            "        if df[c].dtype == object:\n"
            "            coerced = smart_to_numeric(df[c])\n"
            "            if coerced.notna().sum() >= max(3, int(0.5*len(df))):\n"
            "                df[c] = coerced\n"
        )
        bootstrap_lines.append(
            "def safe_scatter_with_trend(df, x_candidates, y_candidates, title=''):\n"
            "    xcol = find_column(df, x_candidates)\n"
            "    ycol = find_column(df, y_candidates)\n"
            "    if not xcol or not ycol:\n"
            "        plt.figure(); plt.text(0.5,0.5,'Columns not found',ha='center');\n"
            "        return\n"
            "    coerce_numeric_columns(df)\n"
            "    x = df[xcol]\n"
            "    y = df[ycol]\n"
            "    mask = x.notna() & y.notna()\n"
            "    x = x[mask]; y = y[mask]\n"
            "    plt.figure()\n"
            "    if len(x) < 2:\n"
            "        plt.text(0.5,0.5,'Not enough data to plot',ha='center'); plt.xlabel(xcol); plt.ylabel(ycol); plt.title(title); return\n"
            "    plt.scatter(x, y)\n"
            "    # trend line\n"
            "    try:\n"
            "        coeffs = np.polyfit(x, y, 1)\n"
            "        xx = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 50)\n"
            "        yy = coeffs[0]*xx + coeffs[1]\n"
            "        plt.plot(xx, yy, 'r:', linewidth=2)\n"
            "    except Exception:\n"
            "        pass\n"
            "    plt.xlabel(xcol); plt.ylabel(ycol); plt.title(title)\n"
        )
        bootstrap_lines.append(
            "def get_table(name):\n"
            "    # smart lookup in tables by exact key, by short key, or fuzzy\n"
            "    if name in tables: return tables[name]\n"
            "    key = name.lower().strip()\n"
            "    # exact on lower\n"
            "    for k in tables:\n"
            "        if k.lower() == key: return tables[k]\n"
            "    # contains\n"
            "    for k in tables:\n"
            "        if key in k.lower(): return tables[k]\n"
            "    return None\n"
        )
        bootstrap_lines.append(
            "if isinstance(_payload, dict):\n"
            "    df = _payload.get('__df__')\n"
            "    tables = _payload.get('__tables__', {}) or {}\n"
            "    # expose convenience variables: full key and short key\n"
            "    for _tname, _tdf in list(tables.items()):\n"
            "        try:\n"
            "            globals()[f'{_sanitize(_tname)}_df'] = _tdf\n"
            "            _short = _tname.split('::')[-1]\n"
            "            globals()[f'{_sanitize(_short)}_df'] = _tdf\n"
            "        except Exception:\n"
            "            pass\n"
            "    if df is None and tables:\n"
            "        df = max(tables.values(), key=lambda d: (len(d) * max(1, len(d.columns))))\n"
            "else:\n"
            "    df = _payload\n"
        )
        bootstrap_lines.append("if isinstance(df, pd.DataFrame):\n    data = df.to_dict(orient='records')\nelse:\n    data = {}\n")
        bootstrap_lines.append("all_tables = list(tables.keys())\n")
    else:
        # ensure data exists so user code that references data won't break
        bootstrap_lines.append("tables = {}\n")
        bootstrap_lines.append("df = None\n")
        bootstrap_lines.append("data = {}\n")
        bootstrap_lines.append("def find_column(df, candidates):\n    return None\n")
        bootstrap_lines.append("def smart_to_numeric(s):\n    return pd.to_numeric(s, errors='coerce')\n")
        bootstrap_lines.append("def coerce_numeric_columns(df):\n    pass\n")
        bootstrap_lines.append("def safe_scatter_with_trend(df, x_candidates, y_candidates, title=''):\n    plt.figure(); plt.text(0.5,0.5,'No data',ha='center')\n")
        bootstrap_lines.append("def get_table(name):\n    return None\n")

    # plot_to_base64 helper that tries to reduce size under 100_000 bytes
    plot_helper = r'''
def plot_to_base64(max_bytes=100000):
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_bytes = buf.getvalue()
    if len(img_bytes) <= max_bytes:
        return base64.b64encode(img_bytes).decode('ascii')
    # try decreasing dpi/figure size iteratively
    for dpi in [80, 60, 50, 40, 30]:
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        b = buf.getvalue()
        if len(b) <= max_bytes:
            return base64.b64encode(b).decode('ascii')
    # if Pillow available, try convert to WEBP which is typically smaller
    try:
        from PIL import Image
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=40)
        buf.seek(0)
        im = Image.open(buf)
        out_buf = BytesIO()
        im.save(out_buf, format='WEBP', quality=80, method=6)
        out_buf.seek(0)
        ob = out_buf.getvalue()
        if len(ob) <= max_bytes:
            return base64.b64encode(ob).decode('ascii')
        # try lower quality
        out_buf = BytesIO()
        im.save(out_buf, format='WEBP', quality=60, method=6)
        out_buf.seek(0)
        ob = out_buf.getvalue()
        if len(ob) <= max_bytes:
            return base64.b64encode(ob).decode('ascii')
    except Exception:
        pass
    # as last resort return downsized PNG even if > max_bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=20)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('ascii')
'''

    # Build the code to write
    script_buf = []
    script_buf.extend(bootstrap_lines)
    script_buf.append(plot_helper)
    script_buf.append(SCRAPE_FUNC)
    script_buf.append("\nresults = {}\n")
    script_buf.append(code)
    # ensure results printed as json
    script_buf.append("\nprint(json.dumps({'status':'success','result':results}, default=str), flush=True)\n")

    tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8')
    tmpfile.write("\n".join(script_buf))
    tmpfile.flush()
    tmp_pathname = tmpfile.name
    tmpfile.close()

    try:
        proc = subprocess.run([sys.executable, tmp_pathname],
                              capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            # collect stderr and stdout for debugging
            return {"status": "error", "message": proc.stderr.strip() or proc.stdout.strip()}
        # parse stdout as json
        stdout_text = proc.stdout.strip()
        try:
            parsed_json = json.loads(stdout_text)
            return parsed_json
        except Exception as e:
            return {"status": "error", "message": f"Could not parse JSON output: {str(e)}", "raw": stdout_text}
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Execution timed out"}
    finally:
        try:
            os.unlink(tmp_pathname)
            if injected_pickle and os.path.exists(injected_pickle):
                os.unlink(injected_pickle)
        except Exception:
            pass


# -----------------------------
# LLM agent setup
# -----------------------------
# Use the ChatOpenAI `chat_model` defined above.

# Tools list for agent (LangChain tool decorator returns metadata for the LLM)
tools = [scrape_url_to_dataframe]  # we only expose scraping as a tool; agent will still produce code

# Prompt: instruct agent to call the tool and output JSON only
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a full-stack autonomous data analyst agent.

You will receive:
- A set of **rules** for this request (these rules may differ depending on whether a dataset is uploaded or not)
- One or more **questions**
- An optional **dataset preview**

You must:
1. Follow the provided rules exactly.
2. Return only a valid JSON object — no extra commentary or formatting.
3. The JSON must contain:
   - "questions": [ list of original question strings exactly as provided ]
   - "code": "..." (Python code that creates a dict called `results` with each question string as a key and its computed answer as the value)
4. Your Python code will run in a sandbox with:
   - pandas, numpy, matplotlib available
   - A helper function `plot_to_base64(max_bytes=100000)` for generating base64-encoded images under 100KB.
   - Helpers available: `find_column`, `smart_to_numeric`, `coerce_numeric_columns`, `safe_scatter_with_trend`, and `get_table`.
5. When returning plots, prefer `safe_scatter_with_trend(...)` for scatter + dotted red regression when appropriate.
6. Make sure all variables are defined before use, and the code can run without any undefined references.
7) If image files are provided, DO NOT call any OCR libraries in Python. Rely on the model’s visual inspection to extract text from images.
8) If a database was uploaded, access its tables via the dict `tables` (pandas DataFrames keyed by table name). Convenience variables like `<table_name>_df` are also available (both the sanitized full name and the short name after '::'). The largest table is bound to `df` for quick inspection.
9) If data is insufficient for a numeric answer or a plot, still populate an informative string in `results` explaining why (e.g., "no matching rows" or "columns not found").
"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

tool_agent = create_tool_calling_agent(
    llm=chat_model,
    tools=[scrape_url_to_dataframe],  # let the agent call tools if it wants; we will also pre-process scrapes
    prompt=agent_prompt
)

agent_runner = AgentExecutor(
    agent=tool_agent,
    tools=[scrape_url_to_dataframe],
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    handle_parsing_errors=True,
    return_intermediate_steps=False
)


# -----------------------------
# Runner: orchestrates agent -> pre-scrape inject -> execute
# -----------------------------
def run_agent_safely(llm_input: str) -> Dict:
    """
    1. Run the agent_executor.invoke to get LLM output
    2. Extract JSON, get 'code' and 'questions'
    3. Detect scrape_url_to_dataframe("...") calls in code, run them here, pickle df and inject before exec
    4. Execute the code in a temp file and return results mapping questions -> answers
    """
    try:
        log.info("Invoking agent with input length=%d", len(llm_input or ""))
        agent_reply = agent_runner.invoke({"input": llm_input}, {"timeout": LLM_TIMEOUT_SECONDS})
        agent_text = agent_reply.get("output") or agent_reply.get("final_output") or agent_reply.get("text") or ""
        log.info("Agent raw_out bytes=%d", len(agent_text.encode("utf-8")) if agent_text else 0)
        if not agent_text:
            return {"error": f"Agent returned no output. Full response: {agent_reply}"}

        parsed_out = clean_llm_output(agent_text)
        if "error" in parsed_out:
            return parsed_out

        if not isinstance(parsed_out, dict) or "code" not in parsed_out or "questions" not in parsed_out:
            return {"error": f"Invalid agent response format: {parsed_out}"}

        gen_code = parsed_out["code"]
        question_list: List[str] = parsed_out["questions"]

        # Detect scrape calls; find all URLs used in scrape_url_to_dataframe("URL")
        url_list = re.findall(r"scrape_url_to_dataframe\(\s*['\"](.*?)['\"]\s*\)", gen_code)
        pickle_file = None
        if url_list:
            # For now support only the first URL (agent may code multiple scrapes; you can extend this)
            first_url = url_list[0]
            scrape_out = scrape_url_to_dataframe(first_url)
            if scrape_out.get("status") != "success":
                return {"error": f"Scrape tool failed: {scrape_out.get('message')}"}
            # create df and pickle it
            frame = pd.DataFrame(scrape_out["data"])
            tmp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            tmp_pkl.close()
            frame.to_pickle(tmp_pkl.name)
            pickle_file = tmp_pkl.name

        # Execute code in temp python script
        exec_out = write_and_run_temp_python(gen_code, injected_pickle=pickle_file, timeout=LLM_TIMEOUT_SECONDS)
        if exec_out.get("status") != "success":
            return {"error": f"Execution failed: {exec_out.get('message', exec_out)}", "raw": exec_out.get("raw")}

        # exec_out['result'] should be results dict
        result_map = exec_out.get("result", {})
        # Map to original questions (they asked to use exact question strings)
        final_map = {}
        for qtext in question_list:
            final_map[qtext] = result_map.get(qtext, "Answer not found")
        return final_map

    except Exception as err:
        log.exception("run_agent_safely failed")
        return {"error": str(err)}


from fastapi import Request

# --- Accept POST on "/", "/api", and "/api/" (prevents 405 on prof portal) ---
@app.post("/")
@app.post("/api")
@app.post("/api/")
async def analyze_data(request: Request):
    try:
        formdata = await request.form()
        qfile = None
        # collect all non-.txt uploads
        other_files: List[UploadFile] = []

        # Collect image data URIs for LLM vision
        image_data_uris = []
        image_filenames = []

        for key, v in formdata.items():
            if hasattr(v, "filename") and v.filename:
                fname_lower = v.filename.lower()
                if fname_lower.endswith(".txt") and qfile is None:
                    qfile = v
                else:
                    other_files.append(v)

        if not qfile:
            raise HTTPException(400, "Missing questions file (.txt)")

        qs_text = (await qfile.read()).decode("utf-8")
        ordered_keys, cast_lookup = parse_keys_and_types(qs_text)

        # prefetch any URLs mentioned in questions.txt and treat as dataset
        url_regex = r"https?://[^\s)]+"
        urls_in_q = re.findall(url_regex, qs_text) or []

        # collect all tabular frames and a tables dict (for DBs etc.)
        tabular_frames: List[pd.DataFrame] = []
        tables_dict: Dict[str, pd.DataFrame] = {}

        def _add_frame(df: pd.DataFrame, source: str):
            if isinstance(df, pd.DataFrame):
                df = df.copy()
                _normalize_columns_inplace(df)
                _coerce_numeric_columns_inplace(df)
                df["source_file"] = source
                tabular_frames.append(df)

        # Load each uploaded file (multiple supported)
        for up in other_files:
            data_filename = up.filename
            file_bytes = await up.read()
            fl = data_filename.lower()
            bio = BytesIO(file_bytes)

            if fl.endswith(".csv"):
                try:
                    frame = pd.read_csv(bio)
                except Exception:
                    bio.seek(0)
                    frame = pd.read_csv(bio, encoding_errors="ignore")
                _add_frame(frame, data_filename)

            elif fl.endswith((".xlsx", ".xls")):
                try:
                    xls = pd.ExcelFile(bio)
                    for sheet in xls.sheet_names:
                        df_sheet = xls.parse(sheet)
                        _add_frame(df_sheet, f"{data_filename}::{sheet}")
                        # also register per-sheet in tables
                        key = f"{os.path.basename(data_filename)}::{sheet}"
                        tdf = df_sheet.copy()
                        _normalize_columns_inplace(tdf); _coerce_numeric_columns_inplace(tdf)
                        tables_dict[key] = tdf
                        short = sheet
                        short_key = short.lower()
                        if short_key not in tables_dict:
                            tables_dict[short] = tdf
                except Exception:
                    bio.seek(0)
                    frame = pd.read_excel(bio)
                    _add_frame(frame, data_filename)

            elif fl.endswith(".parquet"):
                frame = pd.read_parquet(bio)
                _add_frame(frame, data_filename)

            elif fl.endswith(".json"):
                try:
                    frame = pd.read_json(bio)
                    if isinstance(frame, dict):
                        frame = pd.json_normalize(frame)
                except ValueError:
                    frame = pd.DataFrame(json.loads(file_bytes.decode("utf-8")))
                _add_frame(frame, data_filename)

            elif fl.endswith(".pdf"):
                frame, _info = extract_pdf_to_dataframe(file_bytes)
                _add_frame(frame, data_filename)

            elif fl.endswith((".png", ".jpg", ".jpeg", ".webp")):
                # Keep data URI for vision
                try:
                    mime = "image/png" if fl.endswith(".png") else "image/jpeg"
                    image_data_uris.append(_bytes_to_data_uri(file_bytes, mime=mime))
                    image_filenames.append(data_filename)
                except Exception as e:
                    log.warning("Image -> data URI failed for %s: %s", data_filename, e)

            elif fl.endswith((".db", ".sqlite", ".sqlite3")):
                try:
                    # write to temp file to let sqlite connect
                    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(fl)[1], delete=False) as tmpdb:
                        tmpdb.write(file_bytes)
                        tmpdb_path = tmpdb.name
                    conn = sqlite3.connect(tmpdb_path)
                    cur = conn.cursor()
                    tbls = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                    tbl_names = [t[0] for t in tbls if t and t[0]]
                    for tname in tbl_names:
                        df_tbl = pd.read_sql_query(f"SELECT * FROM [{tname}]", conn)
                        _normalize_columns_inplace(df_tbl); _coerce_numeric_columns_inplace(df_tbl)
                        long_key = f"{os.path.basename(data_filename)}::{tname}"
                        tables_dict[long_key] = df_tbl
                        short = tname
                        # also expose short key (lower) if unique
                        if short not in tables_dict:
                            tables_dict[short] = df_tbl
                        _add_frame(df_tbl, long_key)
                    conn.close()
                except Exception as e:
                    log.warning("DB load failed for %s: %s", data_filename, e)
                finally:
                    try:
                        if 'tmpdb_path' in locals() and os.path.exists(tmpdb_path):
                            os.unlink(tmpdb_path)
                    except Exception:
                        pass
            else:
                log.info("Skipping unsupported file type: %s", data_filename)

        # Prefetch tables from any URLs in the questions and stitch them
        for u in urls_in_q:
            try:
                tool_resp = scrape_url_to_dataframe(u)
                if tool_resp.get("status") == "success":
                    df_url = pd.DataFrame(tool_resp["data"])
                    _normalize_columns_inplace(df_url); _coerce_numeric_columns_inplace(df_url)
                    _add_frame(df_url, u)
                else:
                    log.warning("Prefetch failed for %s: %s", u, tool_resp.get("message"))
            except Exception as e:
                log.warning("Prefetch exception for %s: %s", u, e)

        # If we have any tabular frames, merge into one df
        merged_df = None
        if tabular_frames:
            try:
                merged_df = pd.concat(tabular_frames, ignore_index=True, sort=False)
            except Exception:
                tabular_frames2 = []
                for dfi in tabular_frames:
                    dfx = dfi.copy()
                    dfx.columns = dfx.columns.map(str)
                    tabular_frames2.append(dfx)
                merged_df = pd.concat(tabular_frames2, ignore_index=True, sort=False)

        # Build payload to inject into the sandbox: combined df + tables dict
        payload_obj = {"__df__": merged_df, "__tables__": tables_dict}

        pickle_file = None
        df_head_text = ""
        has_dataset = bool(merged_df is not None)

        # Pickle for injection (even if df is None, we pass dict)
        temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        temp_pkl.close()
        pd.to_pickle(payload_obj, temp_pkl.name)
        pickle_file = temp_pkl.name

        if isinstance(merged_df, pd.DataFrame):
            # small per-table preview for prompt
            table_summaries = []
            for k, tdf in list(tables_dict.items())[:12]:  # cap summary
                try:
                    cols = ", ".join(map(str, list(tdf.columns)[:10]))
                    table_summaries.append(f"- {k}: cols=[{cols}] rows={len(tdf)}")
                except Exception:
                    continue
            tables_preview_text = "Available tables:\n" + ("\n".join(table_summaries) if table_summaries else "None")
            df_head_text = (
                f"\n\nThe stitched dataset has {len(merged_df)} rows and {len(merged_df.columns)} columns.\n"
                f"Columns: {', '.join(merged_df.columns.astype(str))}\n"
                f"{tables_preview_text}\n"
                f"First rows (stitched df):\n{merged_df.head(5).to_markdown(index=False)}\n"
            )
        else:
            tables_preview_text = "No stitched df. " + ("Available tables:\n" + "\n".join([f"- {k}" for k in list(tables_dict.keys())[:12]]) if tables_dict else "No tables.")

        # Build rules based on data presence
        if has_dataset or tables_dict:
            rules_text = (
                "Rules:\n"
                "1) You have access to a pandas DataFrame called `df` (stitched from all provided tabular sources and URLs) and its dictionary form `data`.\n"
                "2) A dict `tables` provides per-table DataFrames (e.g., from .db and multi-sheet Excel). Convenience variables `<table_name>_df` are also available (both full and short names).\n"
                "3) DO NOT call scrape_url_to_dataframe() or fetch any external data.\n"
                "4) Use only the uploaded/stitched dataset(s) for answering questions.\n"
                "5) Produce a final JSON object with keys:\n"
                '   - "questions": [ ... original question strings ... ]\n'
                '   - "code": "..."  (Python code that fills `results` with exact question strings as keys)\n'
                "6) For plots: use plot_to_base64() helper to return base64 image data under 100kB. Prefer safe_scatter_with_trend when plotting scatter.\n"
                "7) If a requested chart would be empty due to no data, still generate a figure that explains why (e.g., annotate message) so evaluators don't see a blank plot.\n"
            )
        else:
            rules_text = (
                "Rules:\n"
                "1) If you need web data, CALL scrape_url_to_dataframe(url).\n"
                "2) Produce a final JSON object with keys:\n"
                '   - "questions": [ ... original question strings ... ]\n'
                '   - "code": "..."  (Python code that fills `results` with exact question strings as keys)\n'
                "3) For plots: use plot_to_base64() helper to return base64 image data under 100kB.\n"
            )

        prompt_text = (
            f"{rules_text}\nQuestions:\n{qs_text}\n"
            f"{df_head_text if df_head_text else tables_preview_text}\n"
            "Respond with the JSON object only."
        )

        # If images are present, route to vision-based extraction (no Python OCR)
        if image_data_uris:
            prompt_with_images = (
                prompt_text
                + "\nAdditional rules for images:\n"
                  "- You have access to the uploaded images directly in this message.\n"
                  "- Do NOT attempt to use Python OCR libraries (not available).\n"
                  "- Extract text by visually inspecting the images.\n"
                  "- Return only JSON in the requested shape."
            )
            vision_result = run_vision_extraction(prompt_with_images, image_data_uris)
            if "error" in vision_result:
                raise HTTPException(500, detail=f"Vision extraction failed: {vision_result['error']}")
            return JSONResponse(content=vision_result)

        # Run agent
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(run_agent_safely_unified, prompt_text, pickle_file)
            try:
                result_payload = future.result(timeout=LLM_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                raise HTTPException(408, "Processing timeout")

        if "error" in result_payload:
            raise HTTPException(500, detail=result_payload["error"])

        # Post-process key mapping & type casting
        if ordered_keys and cast_lookup:
            projected = {}
            for i, qtxt in enumerate(result_payload.keys()):
                if i < len(ordered_keys):
                    k = ordered_keys[i]
                    to_type = cast_lookup.get(k, str)
                    try:
                        v = result_payload[qtxt]
                        if isinstance(v, str) and v.startswith("data:image/"):
                            # Remove data URI prefix
                            v = v.split(",", 1)[1] if "," in v else v
                        projected[k] = to_type(v) if v not in (None, "") else v
                    except Exception:
                        projected[k] = result_payload[qtxt]
            result_payload = projected

        return JSONResponse(content=result_payload)

    except HTTPException as he:
        raise he
    except Exception as e:
        log.exception("analyze_data failed")
        raise HTTPException(500, detail=str(e))


def run_agent_safely_unified(llm_input: str, pickle_path: str = None) -> Dict:
    """
    Runs the LLM agent and executes code.
    - Retries up to 3 times if agent returns no output.
    - If pickle_path is provided, injects that DataFrame directly.
    - If no pickle_path, falls back to scraping when needed.
    """
    try:
        attempt_limit = 3
        agent_text = ""
        for attempt in range(1, attempt_limit + 1):
            log.info("Agent attempt %d/%d", attempt, attempt_limit)
            agent_reply = agent_runner.invoke({"input": llm_input}, {"timeout": LLM_TIMEOUT_SECONDS})
            agent_text = agent_reply.get("output") or agent_reply.get("final_output") or agent_reply.get("text") or ""
            if agent_text:
                break
        if not agent_text:
            return {"error": f"Agent returned no output after {attempt_limit} attempts"}

        parsed_out = clean_llm_output(agent_text)
        if "error" in parsed_out:
            return parsed_out

        if "code" not in parsed_out or "questions" not in parsed_out:
            return {"error": f"Invalid agent response: {parsed_out}"}

        gen_code = parsed_out["code"]
        question_list = parsed_out["questions"]

        if pickle_path is None:
            url_list = re.findall(r"scrape_url_to_dataframe\(\s*['\"](.*?)['\"]\s*\)", gen_code)
            if url_list:
                target = url_list[0]
                scrape_out = scrape_url_to_dataframe(target)
                if scrape_out.get("status") != "success":
                    return {"error": f"Scrape tool failed: {scrape_out.get('message')}"}
                frame = pd.DataFrame(scrape_out["data"])
                tmp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
                tmp_pkl.close()
                frame.to_pickle(tmp_pkl.name)
                pickle_path = tmp_pkl.name

        exec_out = write_and_run_temp_python(gen_code, injected_pickle=pickle_path, timeout=LLM_TIMEOUT_SECONDS)
        if exec_out.get("status") != "success":
            return {"error": f"Execution failed: {exec_out.get('message')}", "raw": exec_out.get("raw")}

        result_map = exec_out.get("result", {})
        return {q: result_map.get(q, "Answer not found") for q in question_list}

    except Exception as e:
        log.exception("run_agent_safely_unified failed")
        return {"error": str(e)}


from fastapi.responses import FileResponse, Response
import base64, os

# 1×1 transparent PNG fallback (if favicon.ico file not present)
_FAVICON_FALLBACK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3n+9QAAAAASUVORK5CYII="
)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """
    Serve favicon.ico if present in the working directory.
    Otherwise return a tiny transparent PNG to avoid 404s.
    """
    path = "favicon.ico"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/x-icon")
    return Response(content=_FAVICON_FALLBACK_PNG, media_type="image/png")

# Health/info — allow both /api and /api/ to avoid redirects
@app.get("/api", include_in_schema=False)
@app.get("/api/", include_in_schema=False)
async def analyze_get_info():
    """Health/info endpoint. Use POST / (or /api) for actual analysis."""
    return JSONResponse({
        "ok": True,
        "message": "Server is running. Use POST / or /api with 'questions_file' and optional uploads.",
    })


# -----------------------------
# System Diagnostics
# -----------------------------
# ---- Add these imports near other imports at top of app.py ----
import asyncio
import httpx
import importlib.metadata
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from datetime import datetime, timedelta
import socket
import platform
import psutil
import shutil
import tempfile
import os
import time 
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse    

# ---- Configuration for diagnostics (tweak as needed) ----
DIAG_NETWORK_TARGETS = {
    "Google AI": "https://generativelanguage.googleapis.com",
    "AISTUDIO": "https://aistudio.google.com/",
    "OpenAI": "https://api.openai.com",
    "AI Pipe (OpenAI)": "https://aipipe.org/openai/v1/models",
    "GitHub": "https://api.github.com",
}
DIAG_LLM_KEY_TIMEOUT = 30  # seconds per key/model simple ping test (sync checks run in threadpool)
DIAG_PARALLELISM = 6       # how many thread workers for sync checks
RUN_LONGER_CHECKS = False  # Playwright/duckdb tests run only if true (they can be slow)

# helper: iso timestamp
def _now_iso():
    return datetime.utcnow().isoformat() + "Z"

# helper: run sync func in threadpool and return result / exception info
_executor = ThreadPoolExecutor(max_workers=DIAG_PARALLELISM)
async def run_in_thread(fn, *a, timeout=30, **kw):
    loop = asyncio.get_running_loop()
    try:
        task = loop.run_in_executor(_executor, partial(fn, *a, **kw))
        return await asyncio.wait_for(task, timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError("timeout")
    except Exception as e:
        # re-raise for caller to capture stacktrace easily
        raise

# ---- Diagnostic check functions (safely return dicts) ----
def _env_check(required=None):
    required = required or []
    state = {}
    for k in required:
        state[k] = {"present": bool(os.getenv(k)), "masked": (os.getenv(k)[:4] + "..." + os.getenv(k)[-4:]) if os.getenv(k) else None}
    # Also include simple helpful values
    state["GOOGLE_MODEL"] = os.getenv("GOOGLE_MODEL")
    state["LLM_TIMEOUT_SECONDS"] = os.getenv("LLM_TIMEOUT_SECONDS")
    state["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")
    return state

def _system_info():
    info = {
        "host": socket.gethostname(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "memory_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
    }
    # disk free for app dir and tmp
    try:
        _cwd = os.getcwd()
        info["cwd_free_gb"] = round(shutil.disk_usage(_cwd).free / 1024**3, 2)
    except Exception:
        info["cwd_free_gb"] = None
    try:
        info["tmp_free_gb"] = round(shutil.disk_usage(tempfile.gettempdir()).free / 1024**3, 2)
    except Exception:
        info["tmp_free_gb"] = None
    # GPU quick probe (if torch installed)
    try:
        import torch
        info["torch_installed"] = True
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch_installed"] = False
        info["cuda_available"] = False
    return info

def _temp_write_test():
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, f"diag_test_{int(time.time())}.tmp")
    with open(path, "w") as f:
        f.write("ok")
    ok = os.path.exists(path)
    os.remove(path)
    return {"tmp_dir": tmp, "write_ok": ok}

def _app_write_test():
    # try writing into current working directory
    cwd = os.getcwd()
    path = os.path.join(cwd, f"diag_test_{int(time.time())}.tmp")
    with open(path, "w") as f:
        f.write("ok")
    ok = os.path.exists(path)
    os.remove(path)
    return {"cwd": cwd, "write_ok": ok}

def _pandas_pipeline_test():
    import pandas as _pd
    df = _pd.DataFrame({"x":[1,2,3], "y":[4,5,6]})
    df["z"] = df["x"] * df["y"]
    agg = df["z"].sum()
    return {"rows": df.shape[0], "cols": df.shape[1], "z_sum": int(agg)}

def _installed_packages_sample():
    # return top 20 installed package names + versions
    try:
        out = []
        for dist in importlib.metadata.distributions():
            try:
                out.append(f"{dist.metadata['Name']}=={dist.version}")
            except Exception:
                try:
                    out.append(f"{dist.metadata['Name']}")
                except Exception:
                    continue
        return {"sample_packages": sorted(out)[:20]}
    except Exception as e:
        return {"error": str(e)}

def _network_probe_sync(url, timeout=30):
    # synchronous network probe for threadpool use
    try:
        r = requests.head(url, timeout=timeout)
        return {"ok": True, "status_code": r.status_code, "latency_ms": int(r.elapsed.total_seconds()*1000)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---- LLM key/model light test (AI Pipe/OpenAI) ----
def _test_openai_aipipe_models(api_key: str, base_url: str):
    try:
        url = base_url.rstrip("/") + "/models"
        r = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
        return {"ok": r.ok, "status_code": r.status_code}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---- Async wrappers that call the sync checks in threadpool ----
async def check_network():
    coros = []
    for name, url in DIAG_NETWORK_TARGETS.items():
        coros.append(run_in_thread(_network_probe_sync, url, timeout=30))
    results = await asyncio.gather(*[asyncio.create_task(c) for c in coros], return_exceptions=True)
    out = {}
    for (name, _), res in zip(DIAG_NETWORK_TARGETS.items(), results):
        if isinstance(res, Exception):
            out[name] = {"ok": False, "error": str(res)}
        else:
            out[name] = res
    return out

async def check_llm_keys_models():
    """Check AI Pipe/OpenAI list-models endpoint with provided token."""
    if not OPENAI_API_KEY:
        return {"warning": "no OPENAI_API_KEY configured"}
    return await run_in_thread(_test_openai_aipipe_models, OPENAI_API_KEY, OPENAI_BASE_URL, timeout=10)

# ---- Optional slow heavy checks (DuckDB, Playwright) ----
async def check_duckdb():
    try:
        import duckdb
        def duck_check():
            conn = duckdb.connect(":memory:")
            conn.execute("SELECT 1")
            conn.close()
            return {"duckdb": True}
        return await run_in_thread(duck_check, timeout=30)
    except Exception as e:
        return {"duckdb_error": str(e)}

async def check_playwright():
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            b = await p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = await b.new_page()
            await page.goto("about:blank")
            ua = await page.evaluate("() => navigator.userAgent")
            await b.close()
            return {"playwright_ok": True, "ua": ua[:200]}
    except Exception as e:
        return {"playwright_error": str(e)}

# ---- Final /diagnose route (concurrent) ----
from fastapi import Query

@app.get("/summary")
async def diagnose(full: bool = Query(False, description="If true, run extended checks (duckdb/playwright)")):
    started = datetime.utcnow()
    report = {
        "status": "ok",
        "server_time": _now_iso(),
        "summary": {},
        "checks": {},
        "elapsed_seconds": None
    }

    # prepare tasks
    tasks = {
        "env": run_in_thread(_env_check, ["GOOGLE_API_KEY", "GOOGLE_MODEL", "LLM_TIMEOUT_SECONDS", "OPENAI_BASE_URL"], timeout=3),
        "system": run_in_thread(_system_info, timeout=30),
        "tmp_write": run_in_thread(_temp_write_test, timeout=30),
        "cwd_write": run_in_thread(_app_write_test, timeout=30),
        "pandas": run_in_thread(_pandas_pipeline_test, timeout=30),
        "packages": run_in_thread(_installed_packages_sample, timeout=50),
        "network": asyncio.create_task(check_network()),
        "llm_keys_models": asyncio.create_task(check_llm_keys_models())
    }

    if full or RUN_LONGER_CHECKS:
        tasks["duckdb"] = asyncio.create_task(check_duckdb())
        tasks["playwright"] = asyncio.create_task(check_playwright())

    # run all concurrently, collect results
    results = {}
    for name, coro in tasks.items():
        try:
            res = await coro
            results[name] = {"status": "ok", "result": res}
        except TimeoutError:
            results[name] = {"status": "timeout", "error": "check timed out"}
        except Exception as e:
            results[name] = {"status": "error", "error": str(e), "trace": traceback.format_exc()}

    report["checks"] = results

    # quick summary flags
    failed = [k for k, v in results.items() if v.get("status") != "ok"]
    if failed:
        report["status"] = "warning"
        report["summary"]["failed_checks"] = failed
    else:
        report["status"] = "ok"
        report["summary"]["failed_checks"] = []

    report["elapsed_seconds"] = (datetime.utcnow() - started).total_seconds()
    return report


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))

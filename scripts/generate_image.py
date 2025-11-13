from getpass import getpass
import os
os.environ["GOOGLE_API_KEY"] = getpass("Paste NEW GOOGLE_API_KEY (hidden): ").strip()

OUT_DIR        = "/projects/bfir/ssourav/desi_png_dr1"  # fresh output folder
SURVEY         = "main"         # 'main' or 'sv1'/'sv2'/'sv3'
PROGRAM        = "dark"        
NSIDE          = 64             # DR1 uses NSIDE=64 Healpix grouping
HPIX_COUNT     = 40             # how many random healpix pixels to sample
MAX_PER_PIXEL  = 150            # max targets per healpix file
SEED           = 42
MAKE_CAPTIONS  = False         
MODEL_NAME     = "gemini-2.0-flash"
QPS            = 0.5          
RETRIES        = 6

import sys, subprocess
def _pip(p): subprocess.run([sys.executable, "-m", "pip", "install", "-qU", *p.split()], check=False)
for mod, pipname in {
    "requests":"requests", "numpy":"numpy", "matplotlib":"matplotlib", "PIL":"Pillow",
    "astropy.io":"astropy", "tqdm":"tqdm", "pyarrow":"pyarrow", "pandas":"pandas",
    "google.genai":"google-genai" if MAKE_CAPTIONS else "google-genai",  # harmless if unused
}.items():
    try: __import__(mod.split('.')[0])
    except Exception: _pip(pipname)


import os, io, time, json, random, math, glob
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from astropy.io import fits
from tqdm import tqdm
import pyarrow as pa, pyarrow.parquet as pq
import pandas as pd


from google import genai
from google.genai import errors


# DR1 path pattern: .../spectro/redux/iron/healpix/SURVEY/PROGRAM/PIXGROUP/PIXNUM/{coadd,redrock}-SURVEY-PROGRAM-PIXNUM.fits
BASES = [
    "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/healpix",
    "https://users.flatironinstitute.org/~apricewhelan/data/surveys/DESI/DR1/healpix",
]

# output directories
os.makedirs(OUT_DIR, exist_ok=True)
PNG_DIR      = os.path.join(OUT_DIR, "png")
MANIFEST_JSON= os.path.join(OUT_DIR, "manifest.jsonl")
MANIFEST_PARQ= os.path.join(OUT_DIR, "manifest.parquet")
os.makedirs(PNG_DIR, exist_ok=True)
# start fresh manifest
if os.path.exists(MANIFEST_JSON): os.remove(MANIFEST_JSON)


def healpix_path(hpix: int) -> str:
    grp = hpix // 100
    return f"{SURVEY}/{PROGRAM}/{grp}/{hpix}"

def head_ok(url: str, timeout=20) -> bool:
    try:
        r = requests.head(url, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False

def coadd_url(hpix: int):
    rel = healpix_path(hpix)
    fn  = f"coadd-{SURVEY}-{PROGRAM}-{hpix}.fits"
    for base in BASES:
        url = f"{base}/{rel}/{fn}"
        if head_ok(url): return url
    return None

def redrock_url(hpix: int):
    rel = healpix_path(hpix)
    fn  = f"redrock-{SURVEY}-{PROGRAM}-{hpix}.fits"
    for base in BASES:
        url = f"{base}/{rel}/{fn}"
        if head_ok(url): return url
    return None

def get_bytes(url, tries=3, sleep=2):
    for a in range(tries):
        try:
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            return r.content
        except Exception:
            if a == tries-1: raise
            time.sleep(sleep*(a+1))

def pick_indices(n, cap):
    if n <= cap: return np.arange(n, dtype=int)
    return np.array(sorted(random.sample(range(n), cap)), dtype=int)

# plotting
def render_png(bw, bf, rw, rf, zw, zf, title=None) -> bytes:
    fig, ax = plt.subplots(figsize=(5.4, 2.8), dpi=140)
    ax.grid(True, lw=0.3, alpha=0.35)

    def plot_arm(w, f):
        if w is None or f is None: return
        m = np.isfinite(w) & np.isfinite(f)
        if m.sum() < 5: return
        med = np.nanmedian(f[m]); mad = np.nanmedian(np.abs(f[m]-med)) + 1e-9
        y = np.clip(f, med - 8*mad, med + 12*mad)
        ax.plot(w, y, lw=0.6)

    plot_arm(bw, bf); plot_arm(rw, rf); plot_arm(zw, zf)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Flux (arb.)")
    if title: ax.set_title(title, fontsize=9)
    fig.tight_layout(pad=0.25)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    return buf.getvalue()


def make_prompt(meta: dict) -> str:
    base = ("Describe the key features in this DESI 1D spectrum image: "
            "overall continuum, strongest emission/absorption lines, and any notable breaks. "
            "Write 3–6 concise scientific sentences. No chit-chat.")
    extra = f" TargetID={meta.get('TARGETID','?')}, Program={meta.get('PROGRAM','?')}."
    if meta.get("SPECTYPE"): extra += f" Pipeline class: {meta['SPECTYPE']}."
    if meta.get("Z") is not None: extra += f" Pipeline redshift z≈{meta['Z']:.3f}."
    return base + extra

_last = 0.0
def rate_limit(qps: float):
    global _last
    interval = 1.0 / max(qps, 1e-6)
    now = time.time()
    wait = (_last + interval) - now
    if wait > 0: time.sleep(wait)
    _last = time.time()

def caption_png(png: bytes, prompt: str, client) -> str:
    # very conservative retries
    for attempt in range(1, RETRIES+1):
        try:
            rate_limit(QPS)
            img = Image.open(io.BytesIO(png))
            resp = client.models.generate_content(model=MODEL_NAME, contents=[prompt, img])
            return (resp.text or "").strip()
        except errors.APIError as e:
            code = getattr(e, "code", None)
            back = min(60, 2**attempt)  # exponential backoff
            print(f"[retry {attempt}/{RETRIES}] API {code}: backing off {back:.1f}s")
            time.sleep(back)
        except Exception as e:
            return f"[error] {type(e).__name__}: {str(e)[:300]}"
    return "[error] exceeded retries"

# init LLM client; stays unused if MAKE_CAPTIONS=False ----
client = None
if MAKE_CAPTIONS:
    if not os.environ.get("GOOGLE_API_KEY", "").strip():
        raise RuntimeError("Set GOOGLE_API_KEY before enabling captions.")
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# main
random.seed(SEED)
TOTAL_PIXELS = 12 * (NSIDE**2)  # 49152 for NSIDE=64
hpix_list = random.sample(range(TOTAL_PIXELS), HPIX_COUNT)

rows = []
for hp in hpix_list:
    url_coadd = coadd_url(hp)
    if not url_coadd:
        print(f"[skip] hpix={hp}: no {SURVEY}/{PROGRAM} coadd")
        continue

    print(f"[fetch] hpix={hp} → {url_coadd}")
    try:
        coadd_bytes = get_bytes(url_coadd)
    except Exception as e:
        print(f"[warn] coadd download failed: {e}")
        continue

    # optional redrock with Z/SPECTYPE
    url_red = redrock_url(hp)
    red = None
    if url_red:
        try:
            rb = get_bytes(url_red)
            with fits.open(io.BytesIO(rb), memmap=False) as rr:
                if "REDSHIFTS" in rr: red = rr["REDSHIFTS"].data
        except Exception as e:
            print(f"[warn] redrock fetch failed hpix={hp}: {e}")

    with fits.open(io.BytesIO(coadd_bytes), memmap=False) as hdus:
        # coadd data model: B/R/Z wavelength+flux + FIBERMAP (per DR1 docs)
        bw = hdus["B_WAVELENGTH"].data if "B_WAVELENGTH" in hdus else None
        bf = hdus["B_FLUX"].data        if "B_FLUX" in hdus        else None
        rw = hdus["R_WAVELENGTH"].data if "R_WAVELENGTH" in hdus else None
        rf = hdus["R_FLUX"].data        if "R_FLUX" in hdus        else None
        zw = hdus["Z_WAVELENGTH"].data if "Z_WAVELENGTH" in hdus else None
        zf = hdus["Z_FLUX"].data        if "Z_FLUX" in hdus        else None

        fibermap = hdus["FIBERMAP"].data
        nobj = int(fibermap.shape[0])
        if bf is None or rf is None or zf is None or nobj == 0:
            print(f"[warn] hpix={hp}: missing arrays; skipping")
            continue

        sel = pick_indices(nobj, MAX_PER_PIXEL)
        print(f"[info] hpix={hp}: {nobj} spectra; processing {len(sel)}")

        for ii in sel:
            meta = {
                "RELEASE": "DESI-DR1",
                "SPECPROD": "iron",
                "SURVEY": SURVEY,
                "PROGRAM": PROGRAM,
                "HEALPIX": int(hp),
                "TARGETID": int(fibermap["TARGETID"][ii]),
                "TARGET_RA": float(fibermap["TARGET_RA"][ii]),
                "TARGET_DEC": float(fibermap["TARGET_DEC"][ii]),
            }
            if red is not None and ii < len(red):
                try: meta["Z"] = float(red["Z"][ii])
                except Exception: pass
                try: meta["SPECTYPE"] = str(red["SPECTYPE"][ii])
                except Exception: pass

            # per-target arrays
            b_w, b_f = (bw, bf[ii]) if bw is not None else (None, None)
            r_w, r_f = (rw, rf[ii]) if rw is not None else (None, None)
            z_w, z_f = (zw, zf[ii]) if zw is not None else (None, None)

            # render PNG and save
            png_bytes = render_png(b_w, b_f, r_w, r_f, z_w, z_f,
                                   title=f"HPX {hp} TID {meta['TARGETID']}")
            png_name = f"hpx{hp}_idx{ii}_tid{meta['TARGETID']}.png"
            png_path = os.path.join(PNG_DIR, png_name)
            with open(png_path, "wb") as fp: fp.write(png_bytes)

            # optional caption
            caption = None
            if MAKE_CAPTIONS:
                prompt  = make_prompt(meta)
                caption = caption_png(png_bytes, prompt, client)

            # manifest row (stringify dicts; Parquet-friendly)
            rows.append({
                "png_path": png_path,
                "caption": caption if caption is not None else "",
                "meta_json": json.dumps(meta, ensure_ascii=False),
                "coadd_url": url_coadd,
                "redrock_url": url_red or "",
                "healpix": int(hp),
                "index_in_file": int(ii),
                "model": MODEL_NAME if MAKE_CAPTIONS else "",
            })

# write JSONL
with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[done] JSONL → {MANIFEST_JSON}  | rows={len(rows)}")

# write Parquet 
schema = pa.schema([
    pa.field("png_path", pa.string()),
    pa.field("caption", pa.string()),
    pa.field("meta_json", pa.string()),
    pa.field("coadd_url", pa.string()),
    pa.field("redrock_url", pa.string()),
    pa.field("healpix", pa.int32()),
    pa.field("index_in_file", pa.int32()),
    pa.field("model", pa.string()),
])
pq.write_table(pa.Table.from_pylist(rows, schema=schema), MANIFEST_PARQ)
print(f"[done] Parquet → {MANIFEST_PARQ}")


for r in rows[:3]:
    print({"png": os.path.basename(r["png_path"]),
           "caption": (r["caption"][:100]+"...") if r["caption"] else "",
           "meta": r["meta_json"][:90]+"..."})

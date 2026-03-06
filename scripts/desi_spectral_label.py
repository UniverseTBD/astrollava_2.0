
# !pip install huggingface_hub --quiet

HF_TOKEN    = "hf_TOKEN_HERE"          # paste HF write token
SPACE_NAME  = "desi-spectra-labeler"        # name for the Space


from huggingface_hub import HfApi, create_repo, upload_file
import tempfile, os

api = HfApi(token=HF_TOKEN)
username = api.whoami()["name"]
repo_id = f"{username}/{SPACE_NAME}"

print(f"Deploying to: https://huggingface.co/spaces/{repo_id}")
print()

# Create the Space
try:
    create_repo(
        repo_id=repo_id,
        token=HF_TOKEN,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        private=False,      # set True if you want it private
    )
    print(f"✓ Space created (or already exists): {repo_id}")
except Exception as e:
    print(f"Space creation: {e}")

# Write app.py 
APP_CODE = r'''
"""
DESI Spectra Quality Labeling Tool
Hosted on HuggingFace Spaces — runs 24/7, no compute node needed.
Data streams from HuggingFace REST API — instant access to any of 100k spectra.
"""

import os
import json
import time
import warnings
import csv
import io
import tempfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests as req
import gradio as gr

warnings.filterwarnings("ignore")


# CONFIG

NUM_SPECTRA = 100_000
LABEL_DIR = "/tmp/desi_labels"           # persistent within Space runtime
LABEL_FILE = os.path.join(LABEL_DIR, "desi_labels.csv")
os.makedirs(LABEL_DIR, exist_ok=True)

HF_ROWS_API = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=MultimodalUniverse/desi"
    "&config=default"
    "&split=train"
)

# Rest-frame spectral lines (Angstroms)
LINES = {
    "Lyα": 1216, "CIV": 1549, "CIII]": 1909, "MgII": 2798,
    "[OII]": 3727, "CaK": 3934, "CaH": 3969,
    "Hδ": 4102, "Hγ": 4340, "Hβ": 4861,
    "[OIII]4959": 4959, "[OIII]5007": 5007,
    "MgI": 5175, "NaD": 5893,
    "Hα": 6563, "[NII]": 6584, "[SII]6717": 6717, "[SII]6731": 6731,
}



# DATA ACCESS — HF REST API (instant random access)

_prefetch = {}

@lru_cache(maxsize=512)
def _fetch_cached(idx):
    url = f"{HF_ROWS_API}&offset={idx}&length=1"
    try:
        r = req.get(url, timeout=30)
        r.raise_for_status()
        rows = r.json().get("rows", [])
        if rows:
            return rows[0]["row"]
    except Exception as e:
        print(f"  [WARN] fetch {idx}: {e}")
    return None


def fetch_batch(idx, n=10):
    """Pre-warm cache around idx."""
    global _prefetch
    start = max(0, idx - 2)
    length = min(n, NUM_SPECTRA - start)
    url = f"{HF_ROWS_API}&offset={start}&length={length}"
    try:
        r = req.get(url, timeout=45)
        r.raise_for_status()
        for obj in r.json().get("rows", []):
            _prefetch[obj["row_idx"]] = obj["row"]
    except:
        pass


def get_row(idx):
    if idx in _prefetch:
        return _prefetch.pop(idx)
    return _fetch_cached(idx)


def extract(row):
    spec = row.get("spectrum", {})
    if isinstance(spec, dict):
        flux = np.array(spec.get("flux", []), dtype=np.float64)
        lam = np.array(
            spec.get("lambda", spec.get("lambda_", spec.get("wavelength", []))),
            dtype=np.float64,
        )
        ivar = np.array(spec.get("ivar", []), dtype=np.float64)
        mask = np.array(spec.get("mask", np.zeros(len(flux))), dtype=np.float64)
    else:
        flux = np.array(spec if isinstance(spec, list) else [], dtype=np.float64)
        lam, ivar, mask = np.array([]), np.array([]), np.array([])

    if len(lam) == 0 or np.all(lam == 0):
        lam = np.linspace(3600, 9800, max(len(flux), 7781))
    if len(ivar) == 0:
        ivar = np.ones_like(flux)
    if len(mask) == 0:
        mask = np.zeros_like(flux)
    return flux, lam, ivar, mask



# LABELS — CSV file with download support

def load_labels():
    if os.path.exists(LABEL_FILE):
        return pd.read_csv(LABEL_FILE)
    return pd.DataFrame(columns=[
        "spectrum_idx", "object_id", "redshift", "label",
        "quality_score", "notes", "labeler", "timestamp"
    ])


def save_label(idx, oid, z, label, score, notes, labeler):
    df = load_labels()
    df = df[df["spectrum_idx"] != int(idx)]
    row = pd.DataFrame([{
        "spectrum_idx": int(idx), "object_id": str(oid),
        "redshift": float(z), "label": label,
        "quality_score": int(score), "notes": notes,
        "labeler": labeler,
        "timestamp": datetime.now().isoformat(),
    }])
    df = pd.concat([df, row], ignore_index=True)
    df.to_csv(LABEL_FILE, index=False)
    return df


def get_existing(idx):
    df = load_labels()
    m = df[df["spectrum_idx"] == int(idx)]
    if len(m):
        r = m.iloc[-1]
        return r["label"], int(r["quality_score"]), str(r.get("notes", "")), str(r.get("labeler", ""))
    return None, 5, "", ""


def next_unlabeled(cur):
    df = load_labels()
    done = set(df["spectrum_idx"].values)
    for i in range(cur + 1, NUM_SPECTRA):
        if i not in done:
            return i
    for i in range(0, cur):
        if i not in done:
            return i
    return cur


# PLOTTING

def plot_main(idx, lines=True, ivar_panel=False, mask_show=True,
              smooth=0, z_ov=None):
    row = get_row(idx)
    if row is None:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.text(0.5, 0.5, f"Could not load spectrum {idx}",
                transform=ax.transAxes, ha="center", fontsize=14)
        return fig

    flux, lam, ivar, mask = extract(row)
    if len(flux) == 0:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.text(0.5, 0.5, "Empty flux array", transform=ax.transAxes,
                ha="center", fontsize=14)
        return fig

    z = float(row.get("Z", 0))
    zerr = float(row.get("ZERR", 0))
    oid = str(row.get("object_id", "?"))
    zwarn = row.get("ZWARN", False)
    z_lines = z_ov if z_ov is not None else z

    if ivar_panel:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(15, 7),
            height_ratios=[3, 1], sharex=True, gridspec_kw={"hspace": 0.05})
    else:
        fig, ax = plt.subplots(figsize=(15, 5.5))

    good = np.isfinite(flux) & np.isfinite(lam)
    if mask_show and len(mask) == len(flux):
        good &= mask == 0

    fp = flux.copy()
    if smooth > 1:
        fp = np.convolve(flux, np.ones(smooth) / smooth, mode="same")
        ax.plot(lam[good], flux[good], color="#ccc", alpha=0.35, lw=0.3,
                rasterized=True, label="Raw")

    ax.plot(lam[good], fp[good], color="#1f77b4", lw=0.55, alpha=0.92,
            rasterized=True,
            label="Flux" if smooth <= 1 else f"Smooth (w={smooth})")

    if mask_show:
        bad = np.isfinite(flux) & np.isfinite(lam) & (mask != 0)
        if np.sum(bad) > 0:
            ax.scatter(lam[bad], flux[bad], c="red", s=1, alpha=0.3,
                      label="Masked", zorder=0)

    ax.axhline(0, color="gray", ls="-", alpha=0.3, lw=0.5)

    if lines:
        cols = plt.cm.tab20(np.linspace(0, 1, len(LINES)))
        for i, (name, rl) in enumerate(LINES.items()):
            ol = rl * (1 + z_lines)
            if lam.min() <= ol <= lam.max():
                ax.axvline(ol, color=cols[i], alpha=0.4, ls="--", lw=0.7)
                ax.text(ol + 5, 0.97, name, rotation=90, fontsize=5.5,
                        color=cols[i], ha="left", va="top",
                        transform=ax.get_xaxis_transform(), alpha=0.7)

    title = f"DESI #{idx}  •  {oid}  •  z={z:.6f}±{zerr:.6f}"
    if zwarn:
        title += "  •  ⚠ ZWARN"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Flux", fontsize=11)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
    ax.grid(True, alpha=0.12)

    if np.sum(good) > 100:
        p1, p99 = np.nanpercentile(fp[good], [1, 99])
        m = (p99 - p1) * 0.3
        ax.set_ylim(p1 - m, p99 + m)

    vs = good & (ivar > 0)
    if np.sum(vs) > 10:
        snr = np.nanmedian(np.abs(flux[vs]) * np.sqrt(ivar[vs]))
        ax.text(0.02, 0.95, f"S/N ≈ {snr:.1f}", transform=ax.transAxes,
                fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))

    if ivar_panel:
        ax2.plot(lam[good], ivar[good], color="#2ca02c", lw=0.4, alpha=0.8,
                 rasterized=True)
        ax2.set_ylabel("Inv. Var.", fontsize=10)
        ax2.set_xlabel("Wavelength (Å)", fontsize=11)
        ax2.grid(True, alpha=0.12)
    else:
        ax.set_xlabel("Wavelength (Å)", fontsize=11)

    plt.tight_layout()
    return fig


def plot_zoom(idx, wmin, wmax, smooth=0, z_ov=None):
    row = get_row(idx)
    if row is None:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return fig

    flux, lam, ivar, mask = extract(row)
    z = float(row.get("Z", 0))
    zl = z_ov if z_ov is not None else z

    rng = (lam >= wmin) & (lam <= wmax)
    good = (mask == 0) & np.isfinite(flux) & np.isfinite(lam) & rng

    fig, ax = plt.subplots(figsize=(14, 4))
    if np.sum(good) < 2:
        ax.text(0.5, 0.5, "No data in range", transform=ax.transAxes, ha="center")
        return fig

    fp = flux.copy()
    if smooth > 1:
        fp = np.convolve(flux, np.ones(smooth) / smooth, mode="same")
        ax.plot(lam[good], flux[good], color="#ccc", alpha=0.3, lw=0.3, rasterized=True)

    ax.plot(lam[good], fp[good], color="#1f77b4", lw=0.8, rasterized=True)

    pi = good & (ivar > 0)
    if np.sum(pi) > 2:
        sig = np.zeros_like(flux)
        sig[pi] = 1.0 / np.sqrt(ivar[pi])
        ax.fill_between(lam[good], fp[good] - sig[good], fp[good] + sig[good],
                        alpha=0.15, color="#1f77b4", label="±1σ")

    for name, rl in LINES.items():
        ol = rl * (1 + zl)
        if wmin <= ol <= wmax:
            ax.axvline(ol, color="red", alpha=0.45, ls="--", lw=0.8)
            ax.text(ol + 3, 0.95, name, rotation=90, fontsize=7, color="red",
                    ha="left", va="top", transform=ax.get_xaxis_transform())

    ax.axhline(0, color="gray", ls="-", alpha=0.3)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Flux")
    ax.set_title(f"Zoom: {wmin:.0f}–{wmax:.0f} Å")
    ax.set_xlim(wmin, wmax)
    if np.sum(good) > 10:
        p1, p99 = np.nanpercentile(fp[good], [1, 99])
        m = (p99 - p1) * 0.3
        ax.set_ylim(p1 - m, p99 + m)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.12)
    plt.tight_layout()
    return fig



# GRADIO APP

def build():
    with gr.Blocks(
        title="DESI Spectra Labeler",
        theme=gr.themes.Soft(),
        css="""
        .good-btn{background:#28a745!important;color:#fff!important;font-weight:bold!important}
        .bad-btn{background:#dc3545!important;color:#fff!important;font-weight:bold!important}
        .unc-btn{background:#ffc107!important;color:#000!important;font-weight:bold!important}
        .skip-btn{background:#6c757d!important;color:#fff!important}
        """
    ) as app:
        cur = gr.State(0)

        gr.Markdown(
            "# 🔭 DESI Spectra Quality Labeler\n"
            "Label DESI EDR SV3 spectra. Data streams from HuggingFace — "
            "any spectrum loads in ~1-2 sec.  \n"
            "**This app runs 24/7.** Share this URL with collaborators."
        )

        # Nav
        with gr.Row():
            idx_in = gr.Number(label="Spectrum # (0–99999)", value=0,
                               precision=0, minimum=0, maximum=99999)
            btn_go = gr.Button("Go ➜", size="sm", variant="primary")
            btn_prev = gr.Button("◀", size="sm")
            btn_next = gr.Button("▶", size="sm")
            btn_unl = gr.Button("⏭ Unlabeled", size="sm", variant="secondary")
            btn_rnd = gr.Button("🎲", size="sm")

        with gr.Row():
            meta = gr.Markdown("*Loading…*")
        with gr.Row():
            lbl_md = gr.Markdown("")

        # Plot,controls
        with gr.Row():
            with gr.Column(scale=4):
                main_plt = gr.Plot(label="Spectrum")
            with gr.Column(scale=1):
                gr.Markdown("#### Display")
                c_lines = gr.Checkbox(label="Lines", value=True)
                c_ivar = gr.Checkbox(label="Inv. var.", value=False)
                c_mask = gr.Checkbox(label="Mask", value=True)
                s_smooth = gr.Slider(label="Smooth", minimum=0, maximum=51,
                                     step=2, value=0)
                n_zov = gr.Number(label="Override z", value=None, precision=6,
                                  info="Blank=catalog z")
                btn_ref = gr.Button("🔄", size="sm")

        # Zoom
        with gr.Accordion("🔍 Zoom", open=False):
            with gr.Row():
                n_wmin = gr.Number(label="λ_min", value=3600, precision=0)
                n_wmax = gr.Number(label="λ_max", value=9800, precision=0)
                s_zsmooth = gr.Slider(label="Smooth", minimum=0, maximum=31,
                                      step=2, value=0)
                btn_zm = gr.Button("Zoom ➜", size="sm")
            with gr.Row():
                btn_ha = gr.Button("Hα", size="sm")
                btn_o3 = gr.Button("[OIII]+Hβ", size="sm")
                btn_o2 = gr.Button("[OII]", size="sm")
                btn_la = gr.Button("Lyα", size="sm")
                btn_bl = gr.Button("Blue", size="sm")
                btn_rd = gr.Button("Red", size="sm")
                btn_fl = gr.Button("Full", size="sm")
            zoom_plt = gr.Plot(label="Zoomed")

        # Label 
        gr.Markdown("---\n### 🏷️ Label")
        with gr.Row():
            s_qual = gr.Slider(label="Quality (1–10)", minimum=1, maximum=10,
                               step=1, value=5)
        with gr.Row():
            t_notes = gr.Textbox(label="Notes", placeholder="emission lines, noise, sky residuals…", lines=2)
        with gr.Row():
            t_labeler = gr.Textbox(label="Your name / initials",
                                   placeholder="e.g. SS", lines=1, max_lines=1)
        with gr.Row():
            bg = gr.Button("✅ Good", variant="primary", elem_classes=["good-btn"], size="lg")
            bb = gr.Button("❌ Bad", variant="stop", elem_classes=["bad-btn"], size="lg")
            bu = gr.Button("❓ Uncertain", elem_classes=["unc-btn"], size="lg")
            bs = gr.Button("⏩ Skip", elem_classes=["skip-btn"], size="lg")

        # Progress
        with gr.Accordion("📊 Progress & Export", open=False):
            prog = gr.Markdown("*Click refresh*")
            btn_prog = gr.Button("Refresh", size="sm")
            btn_dl = gr.Button("📥 Download CSV", size="sm")
            csv_out = gr.File(label="CSV", visible=False)
            gr.Markdown(
                "💡 **Tip:** Download the CSV periodically as a backup. "
                "Labels persist as long as the Space is running, but may reset "
                "if the Space rebuilds."
            )

        # Callbacks
        dins = [c_lines, c_ivar, c_mask, s_smooth, n_zov]

        def _disp(idx, li, iv, mk, sm, zo):
            idx = int(idx)
            fetch_batch(idx, 10)
            row = get_row(idx)
            if row:
                z = float(row.get("Z", 0)); zerr = float(row.get("ZERR", 0))
                oid = str(row.get("object_id", "?"))
                zwarn = row.get("ZWARN", False)
                ebv = float(row.get("EBV", 0))
                mt = (f"**ID:** `{oid}` │ **z={z:.6f}±{zerr:.6f}** │ "
                      f"**E(B-V)={ebv:.4f}** │ "
                      f"**ZWARN:** {'⚠️' if zwarn else '✅'}")
            else:
                mt = f"⚠ Could not load #{idx}"

            el, es, en, elb = get_existing(idx)
            if el:
                lb = f"🔖 **Labeled:** {el} (score {es}) by {elb} — _{en}_"
            else:
                lb = "_Not yet labeled_"

            zv = zo if (zo is not None and zo != 0) else None
            fig = plot_main(idx, lines=li, ivar_panel=iv, mask_show=mk,
                           smooth=int(sm), z_ov=zv)
            return fig, mt, lb, idx, es if el else 5, en if el else ""

        dout = [main_plt, meta, lbl_md, cur, s_qual, t_notes]

        def go_to(i, *a): return _disp(max(0, min(int(i), 99999)), *a)
        def go_p(i, *a):
            n = max(0, int(i)-1); return (n, *_disp(n, *a))
        def go_n(i, *a):
            n = min(99999, int(i)+1); return (n, *_disp(n, *a))
        def go_u(i, *a):
            n = next_unlabeled(int(i)); return (n, *_disp(n, *a))
        def go_r(*a):
            n = int(np.random.randint(0, NUM_SPECTRA)); return (n, *_disp(n, *a))

        def submit(idx, lbl, q, notes, labeler, *da):
            idx = int(idx)
            row = get_row(idx)
            oid = str(row.get("object_id", "?")) if row else "?"
            z = float(row.get("Z", 0)) if row else 0
            save_label(idx, oid, z, lbl, int(q), notes, labeler or "anon")
            nxt = next_unlabeled(idx)
            return (nxt, *_disp(nxt, *da), 5, "")

        def do_zoom(i, wn, wx, zs, zo):
            zv = zo if (zo and zo != 0) else None
            return plot_zoom(int(i), float(wn), float(wx), int(zs), zv)

        def preset(cen, hw):
            def fn(i, zo, zs):
                row = get_row(int(i))
                z = float(row.get("Z", 0)) if row else 0
                zu = zo if (zo and zo != 0) else z
                c = cen * (1 + zu)
                wn, wx = c - hw, c + hw
                return plot_zoom(int(i), wn, wx, int(zs),
                                 zo if (zo and zo != 0) else None), wn, wx
            return fn

        def progress():
            df = load_labels()
            n = len(df)
            if not n: return "No labels yet."
            g = len(df[df.label=="Good"]); b = len(df[df.label=="Bad"])
            u = len(df[df.label=="Uncertain"]); av = df.quality_score.mean()
            labelers = df["labeler"].nunique()
            return (f"**Labeled:** {n}/{NUM_SPECTRA} ({n/NUM_SPECTRA*100:.2f}%)\n\n"
                    f"| Good | Bad | Uncertain | Avg Score | Labelers |\n"
                    f"|------|-----|-----------|-----------|----------|\n"
                    f"| {g} | {b} | {u} | {av:.1f} | {labelers} |")

        def dl_csv():
            if os.path.exists(LABEL_FILE):
                return gr.File(value=LABEL_FILE, visible=True)
            return gr.File(visible=False)

        # Wire
        nav = [idx_in] + dout
        btn_go.click(go_to, [idx_in]+dins, dout)
        idx_in.submit(go_to, [idx_in]+dins, dout)
        btn_prev.click(go_p, [cur]+dins, nav)
        btn_next.click(go_n, [cur]+dins, nav)
        btn_unl.click(go_u, [cur]+dins, nav)
        btn_rnd.click(go_r, dins, nav)
        btn_ref.click(lambda i, *a: _disp(i, *a), [cur]+dins, dout)

        lo = [idx_in] + dout + [s_qual, t_notes]
        for b, l in [(bg,"Good"), (bb,"Bad"), (bu,"Uncertain")]:
            b.click(lambda i,q,n,lb,*da,_l=l: submit(i,_l,q,n,lb,*da),
                    [cur, s_qual, t_notes, t_labeler]+dins, lo)
        bs.click(go_n, [cur]+dins, nav)

        btn_zm.click(do_zoom, [cur, n_wmin, n_wmax, s_zsmooth, n_zov], [zoom_plt])
        for b, c, h in [(btn_ha,6563,400),(btn_o3,4960,400),(btn_o2,3727,300),
                         (btn_la,1216,200),(btn_bl,4200,600),(btn_rd,8200,1600),
                         (btn_fl,6700,3200)]:
            b.click(preset(c, h), [cur, n_zov, s_zsmooth], [zoom_plt, n_wmin, n_wmax])

        btn_prog.click(progress, outputs=[prog])
        btn_dl.click(dl_csv, outputs=[csv_out])
        app.load(go_to, [idx_in]+dins, dout)

    return app


# Launch
if __name__ == "__main__":
    app = build()
    app.launch()
'''

# Write requirements.txt
REQUIREMENTS = """numpy
pandas
matplotlib
requests
gradio
"""

#  Upload files to the Space
with tempfile.TemporaryDirectory() as tmp:
    # app.py
    app_path = os.path.join(tmp, "app.py")
    with open(app_path, "w") as f:
        f.write(APP_CODE)

    # requirements.txt
    req_path = os.path.join(tmp, "requirements.txt")
    with open(req_path, "w") as f:
        f.write(REQUIREMENTS)

    print("Uploading app.py...")
    upload_file(
        path_or_fileobj=app_path,
        path_in_repo="app.py",
        repo_id=repo_id,
        repo_type="space",
        token=HF_TOKEN,
    )

    print("Uploading requirements.txt...")
    upload_file(
        path_or_fileobj=req_path,
        path_in_repo="requirements.txt",
        repo_id=repo_id,
        repo_type="space",
        token=HF_TOKEN,
    )

print()
print("=" * 60)
print("DEPLOYED SUCCESSFULLY!")
print("=" * 60)
print()
print(f" app is live at:")
print(f"  https://huggingface.co/spaces/{repo_id}")
print()
print(f" app URL (for public):")
print(f" https://{username}-{SPACE_NAME}.hf.space")
print()

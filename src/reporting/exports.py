"""CSV / JSON / PDF exports of an analysis result."""
from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.risk.risk_engine import LEVEL_COLORS  # noqa: E402
from src.utils.io import _default  # noqa: E402


def timeline_csv(res) -> bytes:
    return res.timeline.to_csv(index=False).encode("utf-8")


def result_json(res, extra: dict | None = None) -> bytes:
    obj = {"generated": datetime.now().isoformat(timespec="seconds"), "summary": res.summary(),
           "alarms_s": res.alarms, "timeline": res.timeline.drop(columns=["explanation"]).to_dict("records"),
           "assessments": [a.__dict__ for a in res.assessments if a.level != "NORMAL"], **(extra or {})}
    return json.dumps(obj, indent=2, default=_default, ensure_ascii=False).encode("utf-8")


def _timeline_png(res) -> bytes:
    tl = res.timeline
    fig, ax = plt.subplots(figsize=(8, 2.8), dpi=130)
    ax.plot(tl.time_s, tl.p_mean, color="#2563eb", lw=1.8, label="P(collision)")
    ax.fill_between(tl.time_s, (tl.p_mean - tl.p_std).clip(0), (tl.p_mean + tl.p_std).clip(upper=1),
                    color="#2563eb", alpha=0.15, label="±1 sd (MC)")
    ax.plot(tl.time_s, tl.risk_score, color="#ef4444", lw=1.2, ls="--", label="risk score")
    for a in res.alarms:
        ax.axvline(a, color="#f97316", lw=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _latin(s: str) -> str:
    return s.encode("latin-1", "replace").decode("latin-1")


def result_pdf(res, model_meta: dict | None = None) -> bytes:
    from fpdf import FPDF
    s = res.summary()
    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "SafeDrive AI - Collision Risk Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, _latin(f"Generated {datetime.now():%Y-%m-%d %H:%M} | video: {s['video']}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    rows = [("Duration analysed", f"{s['duration_s']:.1f} s ({s['frames_analysed']} frames @10 Hz)"),
            ("Highest risk level", s["max_level"]), ("Max P(collision)", f"{s['max_probability']:.3f}"),
            ("Max risk score", f"{s['max_risk_score']:.3f}"),
            ("First alarm", "none" if s["first_alarm_s"] is None else f"{s['first_alarm_s']:.1f} s"),
            ("Alarm onsets", str(s["n_alarms"])), ("Tracks", str(s["tracking"].get("n_tracks"))),
            ("Feature extraction", f"{s['timings'].get('extract_fps') or 0:.1f} frames/s")]
    for k, v in rows:
        pdf.cell(60, 6, _latin(k))
        pdf.cell(0, 6, _latin(str(v)), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.image(io.BytesIO(_timeline_png(res)), w=180)
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Elevated-risk moments", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    shown = 0
    for a in res.assessments:
        if a.level in ("WARNING", "HIGH RISK", "UNCERTAIN") and shown < 25:
            pdf.multi_cell(0, 4.5, _latin(f"t={a.timestamp:5.1f}s  {a.explanation}"), new_x="LMARGIN", new_y="NEXT")
            shown += 1
    if not shown:
        pdf.cell(0, 6, "No WARNING / HIGH RISK / UNCERTAIN moments.", new_x="LMARGIN", new_y="NEXT")
    if model_meta:
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 8)
        pdf.multi_cell(0, 4, _latin(f"Model: {model_meta.get('description', '')} | window {model_meta.get('window_s')} s | "
                                    f"temperature {model_meta.get('temperature', 1):.2f}. Image-plane motion cues are "
                                    "apparent motion, not physical speed. Research prototype - not a safety device."))
    return bytes(pdf.output())


def save_all(res, out_dir: str | Path, model_meta: dict | None = None) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(res.video_path).stem or "video"
    paths = {"csv": out_dir / f"{stem}_timeline.csv", "json": out_dir / f"{stem}_result.json",
             "pdf": out_dir / f"{stem}_report.pdf"}
    paths["csv"].write_bytes(timeline_csv(res))
    paths["json"].write_bytes(result_json(res))
    paths["pdf"].write_bytes(result_pdf(res, model_meta))
    return paths


__all__ = ["timeline_csv", "result_json", "result_pdf", "save_all", "LEVEL_COLORS"]

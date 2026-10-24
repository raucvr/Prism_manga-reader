"""
Summarise a finished job: per-panel QA scores, attempts, timings and an estimated cost.
"att" and "retried" count the image attempts of each panel's latest render (QA retries); user redraws are counted apart.

usage: .venv/bin/python backend/tools/job_report.py output/<job_id> [--contact-sheet out.jpg]

Prices (BytePlus ModelArk, Oct 2026): Seedream 5.0 Pro $0.045 per output image ≤ 2.61 MP ($0.09 above),
$0.003 per reference image from the 2nd one; text per model (services.pricing).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.pricing import IMAGE_LARGE, IMAGE_REF, IMAGE_SMALL, usage_usd  # noqa: E402

PRICE_IMAGE_SMALL = IMAGE_SMALL
PRICE_IMAGE_LARGE = IMAGE_LARGE
PRICE_REF = IMAGE_REF
PRICE_TEXT_IN = 0.50 / 1_000_000    # older jobs without per-model token counts
PRICE_TEXT_OUT = 3.00 / 1_000_000


def load(job_dir: Path) -> tuple[dict, list[dict]]:
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    prompts = []
    p = job_dir / "prompts.jsonl"
    if p.exists():
        prompts = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return job, prompts


def estimate_cost(job: dict, refs_per_image: int = 3) -> dict:
    """Images at Seedream's price; text per model when the job counted tokens per model ("text_by_model": the writer,
    checker and vision model may differ), else every token at Seed 2.0 Pro's price (older jobs)."""
    cost = job.get("cost", {})
    images = int(cost.get("images_generated", 0))
    large = any((p.get("width") or 0) * (p.get("height") or 0) > 2_610_000 for p in job.get("panels", []))
    image_usd = images * (PRICE_IMAGE_LARGE if large else PRICE_IMAGE_SMALL) + images * max(0, refs_per_image - 1) * PRICE_REF
    by_model = cost.get("text_by_model") or {}
    if by_model:
        text_usd = usage_usd(by_model)
    else:
        text_usd = int(cost.get("text_prompt_tokens", 0)) * PRICE_TEXT_IN + int(cost.get("text_completion_tokens", 0)) * PRICE_TEXT_OUT
    out = {"images": images, "image_usd": round(image_usd, 3), "text_usd": round(text_usd, 3), "total_usd": round(image_usd + text_usd, 3)}
    if by_model:
        out["text_usd_by_model"] = {m: round(usage_usd({m: u}), 3) for m, u in by_model.items()}
    return out


def render_attempts(panel: dict) -> int:
    """Image attempts of the panel's latest render (1 = passed QA first time). Older job.json files only have the
    cumulative `attempts`, which also counts user redraws."""
    value = panel.get("render_attempts")
    return int(panel.get("attempts") or 0) if value is None else int(value)


def report(job_dir: Path) -> str:
    job, prompts = load(job_dir)
    secs = {tuple(p["panels"]): p.get("seconds") for p in prompts}
    renders = Counter(n for p in prompts for n in p.get("panels", []))   # one prompts.jsonl line per published picture
    lines = [f"Job {job['job_id']}  status={job['status']}  {job.get('stage_message', '')}",
             f"Title: {job.get('title')}  | language={job.get('language')} theme={job.get('theme')} layout={job.get('layout')} params={job.get('params')}",
             f"Elapsed: {job.get('cost', {}).get('elapsed_seconds')} s  | cost counters: {job.get('cost')}",
             f"Estimated cost: {estimate_cost(job)}", ""]
    lines.append(f"{'n':>3} {'chapter':<9} {'status':<8} {'att':>3} {'score':>5} {'text':>6} {'secs':>6}  issues / title")
    total_exact = total_text = 0
    for p in job.get("panels", []):
        qa = p.get("qa") or {}
        total_exact += int(qa.get("text_exact", 0) or 0)
        total_text += int(qa.get("text_total", 0) or 0)
        s = next((v for k, v in secs.items() if p["n"] in k), None)
        issues = "; ".join(qa.get("issues", [])[:2])[:90]
        score = str(qa.get("score", "-")) if qa.get("available", True) else "n/a"   # "available" is absent in older jobs
        lines.append(f"{p['n']:>3} {p.get('chapter', ''):<9} {p['status']:<8} {render_attempts(p):>3} {score:>5} {str(qa.get('text_exact', '-')) + '/' + str(qa.get('text_total', '-')):>6} {str(s or '-'):>6}  {issues or p.get('title', '')}")
    panels = job.get("panels", [])
    done = sum(1 for p in panels if p["status"] == "done")
    retried = sum(1 for p in panels if render_attempts(p) > 1)
    redrawn = sum(1 for p in panels if renders[p["n"]] > 1)
    lines += ["", f"Panels done: {done}/{len(panels)}  retried: {retried}  redrawn by the user: {redrawn}  text exact: {total_exact}/{total_text}"
              + (f" ({100 * total_exact / total_text:.1f}%)" if total_text else "")]
    cov = job.get("coverage") or {}   # absent in jobs made before the key-point coverage
    if cov:
        lines.append("Key points covered: " + "  ".join(f"{k} {v[0]}/{v[1]}" for k, v in cov.items() if isinstance(v, list) and len(v) == 2))
    return "\n".join(lines)


def contact_sheet(job_dir: Path, out: Path, columns: int = 4, width: int = 320) -> None:
    from PIL import Image

    job, _ = load(job_dir)
    tiles = []
    for p in sorted(job.get("panels", []), key=lambda x: x["n"]):
        if not p.get("image_url"):
            continue
        im = Image.open(job_dir / Path(p["image_url"]).name).convert("RGB")
        if p.get("strip_index") is not None:
            half = im.width // 2
            im = im.crop((half * p["strip_index"], 0, half * (p["strip_index"] + 1), im.height))
        im = im.resize((width, int(im.height * width / im.width)))
        tiles.append(im)
    if not tiles:
        return
    tile_h = max(t.height for t in tiles)
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width + (columns + 1) * 10, rows * tile_h + (rows + 1) * 10), "white")
    for i, t in enumerate(tiles):
        r, c = divmod(i, columns)
        sheet.paste(t, (10 + c * (width + 10), 10 + r * (tile_h + 10)))
    sheet.save(out, quality=85)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir")
    ap.add_argument("--contact-sheet")
    args = ap.parse_args()
    d = Path(args.job_dir)
    print(report(d))
    if args.contact_sheet:
        contact_sheet(d, Path(args.contact_sheet))
        print("contact sheet:", args.contact_sheet)

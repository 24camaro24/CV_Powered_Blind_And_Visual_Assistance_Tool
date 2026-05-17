import argparse
import json
import os
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = REPO_ROOT / "evaluation_runs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an HTML report for an evaluation run."
    )
    parser.add_argument(
        "run_path",
        nargs="?",
        help="Path to an evaluation run directory or its summary.json file.",
    )
    parser.add_argument(
        "--runs-dir",
        default=str(DEFAULT_RUNS_DIR),
        help="Directory containing evaluation runs. Used when run_path is omitted.",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also export the generated report.html to report.pdf using a headless browser.",
    )
    parser.add_argument(
        "--browser",
        default="auto",
        choices=["auto", "edge", "chrome"],
        help="Browser to use for PDF export.",
    )
    return parser.parse_args()


def find_browser(browser_choice):
    browser_paths = {
        "edge": [
            Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        ],
        "chrome": [
            Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
            Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        ],
    }

    if browser_choice != "auto":
        for path in browser_paths[browser_choice]:
            if path.exists():
                return path
        raise FileNotFoundError(f"Requested browser '{browser_choice}' was not found.")

    for name in ("edge", "chrome"):
        for path in browser_paths[name]:
            if path.exists():
                return path
    raise FileNotFoundError("No supported browser found for PDF export.")


def find_latest_run(runs_dir):
    run_dirs = sorted(
        [path for path in runs_dir.iterdir() if path.is_dir() and (path / "summary.json").exists()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No evaluation runs found in: {runs_dir}")
    return run_dirs[0]


def resolve_run_dir(run_path, runs_dir):
    if run_path:
        path = Path(run_path).resolve()
        if path.is_file():
            if path.name != "summary.json":
                raise FileNotFoundError("Expected a run directory or summary.json file.")
            return path.parent
        return path
    return find_latest_run(runs_dir)


def load_summary(run_dir):
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found in run directory: {run_dir}")
    with summary_path.open("r", encoding="utf-8") as summary_file:
        return json.load(summary_file)


def draw_comparison_image(image_path, predicted_bbox, ground_truth_bbox, output_path, target, match_status):
    if not image_path or not Path(image_path).exists():
        return ""

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    if ground_truth_bbox and len(ground_truth_bbox) == 4:
        draw.rectangle(ground_truth_bbox, outline=(34, 139, 34), width=6)
        draw.text((ground_truth_bbox[0] + 8, max(4, ground_truth_bbox[1] - 22)), "GT", fill=(34, 139, 34))

    if predicted_bbox and len(predicted_bbox) == 4:
        draw.rectangle(predicted_bbox, outline=(220, 53, 69), width=6)
        draw.text((predicted_bbox[0] + 8, predicted_bbox[1] + 8), "Pred", fill=(220, 53, 69))

    banner_color = (31, 122, 92) if match_status == "correct_match" else (173, 92, 43)
    draw.rectangle((0, 0, image.width, 36), fill=banner_color)
    draw.text((10, 9), f"{target} | {match_status}", fill=(255, 255, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return str(output_path)


def load_thumbnail(image_path, max_size):
    if not image_path or not Path(image_path).exists():
        return None
    image = Image.open(image_path).convert("RGB")
    image.thumbnail(max_size)
    return image


def create_text_page(lines, page_size, title=None):
    page = Image.new("RGB", page_size, "white")
    draw = ImageDraw.Draw(page)
    title_font = ImageFont.load_default()
    body_font = ImageFont.load_default()
    x = 60
    y = 50

    if title:
      draw.text((x, y), title, fill="black", font=title_font)
      y += 40

    for line in lines:
        draw.text((x, y), line, fill="black", font=body_font)
        y += 24
        if y > page_size[1] - 80:
            break
    return page


def build_pdf_report(run_dir, summary, records, output_path):
    page_size = (1240, 1754)
    pages = []

    summary_lines = [
        f"Created: {summary.get('created_at', '-')}",
        f"Run directory: {summary.get('output_dir', '-')}",
        f"Input directory: {summary.get('input_dir', '-')}",
        f"Device: {summary.get('device', '-')}",
        f"Pipeline init: {summary.get('pipeline_init_time_sec', 0):.3f}s",
        "",
        f"Total inferences: {summary['summary'].get('total_inferences', 0)}",
        f"Successful: {summary['summary'].get('successful_inferences', 0)}",
        f"Failed: {summary['summary'].get('failed_inferences', 0)}",
        f"Success rate: {summary['summary'].get('success_rate', 0) * 100:.1f}%",
        f"Ground-truth match rate: {summary['summary'].get('ground_truth_match_rate', 0) * 100:.1f}%",
        f"Average IoU: {summary['summary'].get('average_iou_on_ground_truth', 0):.4f}",
        f"Average process time: {summary['summary'].get('average_process_time_sec', 0):.3f}s",
        f"Average instruction time: {summary['summary'].get('average_instruction_time_sec', 0):.3f}s",
        f"Average total time: {summary['summary'].get('average_total_time_sec', 0):.3f}s",
        "",
        "Per-target summary:",
    ]
    for target, stats in (summary["summary"].get("per_target") or {}).items():
        summary_lines.append(
            f"- {target}: success {stats.get('success', 0)}/{stats.get('total', 0)}, "
            f"match rate {stats.get('match_rate', 0) * 100:.1f}%, "
            f"avg total {stats.get('average_total_time_sec', 0):.3f}s"
        )
    pages.append(create_text_page(summary_lines, page_size, title="Evaluation Report"))

    for record in records:
        page = Image.new("RGB", page_size, "white")
        draw = ImageDraw.Draw(page)
        font = ImageFont.load_default()
        title = f"{record.get('image_name', '-') } | target={record.get('target', '-')}"
        draw.text((50, 40), title, fill="black", font=font)
        draw.text((50, 72), f"status={record.get('match_status', '-')} | matched={record.get('matched', False)}", fill="black", font=font)

        original = load_thumbnail(run_dir / record["image_rel"], (520, 520)) if record.get("image_rel") else None
        comparison = load_thumbnail(run_dir / record["comparison_rel"], (520, 520)) if record.get("comparison_rel") else None
        visualization = load_thumbnail(run_dir / record["visualization_rel"], (520, 360)) if record.get("visualization_rel") else None

        if original:
            page.paste(original, (50, 120))
            draw.text((50, 650), "Original", fill="black", font=font)
        if comparison:
            page.paste(comparison, (670, 120))
            draw.text((670, 650), "Predicted vs Ground Truth", fill="black", font=font)
        if visualization:
            page.paste(visualization, (50, 720))
            draw.text((50, 1090), "Model Visualization", fill="black", font=font)

        metrics_x = 670
        metrics_y = 720
        metric_lines = [
            f"Confidence: {record.get('confidence', '-')}",
            f"Distance (m): {record.get('distance_meters', '-')}",
            f"Steps: {record.get('steps', '-')}",
            f"Angle: {record.get('angle', '-')}",
            f"Direction: {record.get('direction', '-')}",
            f"Process time: {record.get('process_time_sec', '-')}s",
            f"Instruction time: {record.get('instruction_time_sec', '-')}s",
            f"Total time: {record.get('total_time_sec', '-')}s",
            f"Best IoU: {record.get('best_iou', '-')}",
            f"Predicted box: {record.get('bbox_parsed', record.get('bbox', '-'))}",
            f"Ground-truth box: {record.get('best_gt_bbox_parsed', record.get('best_gt_bbox', '-'))}",
            f"Expected present: {record.get('expected_present', '-')}",
            f"Error: {record.get('error', '-') or '-'}",
        ]
        for line in metric_lines:
            draw.text((metrics_x, metrics_y), line, fill="black", font=font)
            metrics_y += 28

        pages.append(page)

    first_page, *rest_pages = pages
    first_page.save(output_path, save_all=True, append_images=rest_pages, resolution=150.0)
    return output_path


def to_relative_path(report_dir, absolute_path):
    if not absolute_path:
        return ""
    path = Path(absolute_path).resolve()
    return Path(os.path.relpath(path, report_dir.resolve())).as_posix()


def normalize_record(run_dir, record):
    normalized = dict(record)
    normalized["image_rel"] = to_relative_path(run_dir, record.get("image_path", ""))
    normalized["visualization_rel"] = to_relative_path(run_dir, record.get("visualization_path", ""))
    normalized["audio_rel"] = to_relative_path(run_dir, record.get("audio_path", ""))
    normalized["json_rel"] = to_relative_path(run_dir, record.get("json_path", ""))

    bbox = record.get("bbox")
    surfaces = record.get("surfaces")
    try:
        normalized["bbox_parsed"] = json.loads(bbox) if isinstance(bbox, str) and bbox else bbox
    except json.JSONDecodeError:
        normalized["bbox_parsed"] = bbox
    try:
        normalized["surfaces_parsed"] = json.loads(surfaces) if isinstance(surfaces, str) and surfaces else surfaces
    except json.JSONDecodeError:
        normalized["surfaces_parsed"] = surfaces
    best_gt_bbox = record.get("best_gt_bbox")
    try:
        normalized["best_gt_bbox_parsed"] = (
            json.loads(best_gt_bbox) if isinstance(best_gt_bbox, str) and best_gt_bbox else best_gt_bbox
        )
    except json.JSONDecodeError:
        normalized["best_gt_bbox_parsed"] = best_gt_bbox
    return normalized


def flatten_records(run_dir, summary):
    comparison_dir = run_dir / "report_assets"
    records = []
    for image_entry in summary.get("images", []):
        for record in image_entry.get("records", []):
            normalized = normalize_record(run_dir, record)
            comparison_path = comparison_dir / Path(record.get("image_name", "unknown")).stem / f"{record.get('target', 'target')}_comparison.png"
            generated_path = draw_comparison_image(
                image_path=record.get("image_path", ""),
                predicted_bbox=normalized.get("bbox_parsed"),
                ground_truth_bbox=normalized.get("best_gt_bbox_parsed"),
                output_path=comparison_path,
                target=record.get("target", "target"),
                match_status=record.get("match_status", "not_evaluated"),
            )
            normalized["comparison_rel"] = to_relative_path(run_dir, generated_path) if generated_path else ""
            normalized["image_group"] = image_entry.get("image_name", record.get("image_name", ""))
            records.append(normalized)
    return records


def build_html(summary, records):
    summary_json = json.dumps(summary)
    records_json = json.dumps(records)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Evaluation Report</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: #fffaf3;
      --panel-strong: #fff;
      --ink: #1f1c18;
      --muted: #6c6259;
      --accent: #1f7a5c;
      --accent-soft: #dff3eb;
      --warn: #ad5c2b;
      --error: #b33b32;
      --border: #e2d8cb;
      --shadow: 0 18px 45px rgba(71, 52, 32, 0.08);
      --radius: 18px;
      --radius-sm: 12px;
      --mono: "Consolas", "SFMono-Regular", monospace;
      --sans: "Segoe UI", "Inter", sans-serif;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      font-family: var(--sans);
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(31, 122, 92, 0.10), transparent 26%),
        radial-gradient(circle at top left, rgba(173, 92, 43, 0.10), transparent 24%),
        var(--bg);
    }}

    .page {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 28px;
    }}

    .hero {{
      background: linear-gradient(135deg, rgba(31, 122, 92, 0.96), rgba(14, 55, 42, 0.96));
      color: #f7fff8;
      border-radius: 28px;
      padding: 28px;
      box-shadow: var(--shadow);
      margin-bottom: 24px;
    }}

    .hero h1 {{
      margin: 0 0 10px;
      font-size: 2.2rem;
      letter-spacing: -0.03em;
    }}

    .hero p {{
      margin: 4px 0;
      color: rgba(247, 255, 248, 0.86);
    }}

    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin: 24px 0;
    }}

    .stat-card {{
      background: var(--panel-strong);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 18px;
      box-shadow: var(--shadow);
    }}

    .stat-card .label {{
      color: var(--muted);
      font-size: 0.92rem;
      margin-bottom: 8px;
    }}

    .stat-card .value {{
      font-size: 1.8rem;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}

    .layout {{
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 22px;
      align-items: start;
    }}

    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }}

    .panel-header {{
      padding: 18px 20px 12px;
      border-bottom: 1px solid var(--border);
    }}

    .panel-header h2 {{
      margin: 0;
      font-size: 1.08rem;
    }}

    .panel-body {{
      padding: 18px 20px 20px;
    }}

    .filter-group {{
      margin-bottom: 14px;
    }}

    .filter-group label {{
      display: block;
      font-size: 0.88rem;
      color: var(--muted);
      margin-bottom: 6px;
    }}

    .filter-group input,
    .filter-group select {{
      width: 100%;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: #fff;
      color: var(--ink);
      font: inherit;
    }}

    .mini-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 12px;
    }}

    .mini-stat {{
      padding: 12px;
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 12px;
    }}

    .mini-stat strong {{
      display: block;
      font-size: 1.15rem;
      margin-bottom: 4px;
    }}

    .target-list {{
      display: grid;
      gap: 10px;
      margin-top: 14px;
    }}

    .target-item {{
      padding: 12px;
      border-radius: 12px;
      background: #fff;
      border: 1px solid var(--border);
    }}

    .target-item .name {{
      font-weight: 700;
      margin-bottom: 4px;
    }}

    .records-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 14px;
      flex-wrap: wrap;
    }}

    .records-header h2 {{
      margin: 0;
    }}

    .record-count {{
      color: var(--muted);
      font-size: 0.95rem;
    }}

    .cards {{
      display: grid;
      gap: 18px;
    }}

    .record-card {{
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 22px;
      overflow: hidden;
      box-shadow: var(--shadow);
    }}

    .record-top {{
      padding: 20px;
      display: flex;
      justify-content: space-between;
      align-items: start;
      gap: 16px;
      background: linear-gradient(135deg, rgba(31, 122, 92, 0.08), rgba(173, 92, 43, 0.06));
    }}

    .record-title {{
      margin: 0;
      font-size: 1.22rem;
    }}

    .record-subtitle {{
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 0.95rem;
      word-break: break-word;
    }}

    .badge-row {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 12px;
    }}

    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 0.83rem;
      font-weight: 600;
      background: #f4f0ea;
      color: var(--ink);
      border: 1px solid var(--border);
    }}

    .badge.success {{
      background: var(--accent-soft);
      color: var(--accent);
      border-color: rgba(31, 122, 92, 0.18);
    }}

    .badge.fail {{
      background: rgba(179, 59, 50, 0.10);
      color: var(--error);
      border-color: rgba(179, 59, 50, 0.18);
    }}

    .record-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      padding: 20px;
    }}

    .media-block {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px;
    }}

    .media-block h3,
    .metrics-block h3 {{
      margin: 0 0 10px;
      font-size: 0.98rem;
    }}

    .media-block img {{
      width: 100%;
      height: auto;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #f9f5ef;
    }}

    .metrics-block {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px;
    }}

    .metric-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }}

    .metric-table td {{
      padding: 8px 0;
      border-bottom: 1px solid #eee5d9;
      vertical-align: top;
    }}

    .metric-table td:first-child {{
      color: var(--muted);
      width: 44%;
    }}

    .audio {{
      margin-top: 12px;
      width: 100%;
    }}

    .artifact-links {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 12px;
    }}

    .artifact-links a {{
      text-decoration: none;
      color: var(--accent);
      font-weight: 600;
    }}

    .surfaces {{
      margin-top: 10px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}

    .surface-chip {{
      padding: 6px 10px;
      border-radius: 999px;
      background: #fff;
      border: 1px solid var(--border);
      font-size: 0.83rem;
    }}

    .error {{
      margin-top: 12px;
      color: var(--error);
      font-weight: 600;
    }}

    .empty {{
      padding: 34px;
      text-align: center;
      color: var(--muted);
      background: #fff;
      border: 1px dashed var(--border);
      border-radius: 18px;
    }}

    .mono {{
      font-family: var(--mono);
      word-break: break-word;
    }}

    @media (max-width: 1180px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .record-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Evaluation Report</h1>
      <p><strong>Created:</strong> <span id="createdAt"></span></p>
      <p><strong>Run directory:</strong> <span id="runDir" class="mono"></span></p>
      <p><strong>Input directory:</strong> <span id="inputDir" class="mono"></span></p>
      <p><strong>Device:</strong> <span id="device"></span> | <strong>Pipeline init:</strong> <span id="initTime"></span></p>
    </section>

    <section class="stats" id="summaryStats"></section>

    <div class="layout">
      <aside class="panel">
        <div class="panel-header">
          <h2>Filters</h2>
        </div>
        <div class="panel-body">
          <div class="filter-group">
            <label for="searchInput">Search image or target</label>
            <input id="searchInput" type="text" placeholder="phone, spectacle, bottle...">
          </div>
          <div class="filter-group">
            <label for="targetFilter">Target</label>
            <select id="targetFilter"></select>
          </div>
          <div class="filter-group">
            <label for="statusFilter">Status</label>
            <select id="statusFilter">
              <option value="all">All</option>
              <option value="success">Success only</option>
              <option value="fail">Failures only</option>
            </select>
          </div>
          <div class="filter-group">
            <label for="matchFilter">Match</label>
            <select id="matchFilter">
              <option value="all">All match states</option>
              <option value="correct_match">Correct matches only</option>
              <option value="wrong_match">Wrong matches only</option>
              <option value="no_ground_truth_for_target">No ground truth only</option>
              <option value="not_evaluated">Not evaluated only</option>
            </select>
          </div>
          <div class="filter-group">
            <label for="sortFilter">Sort by</label>
            <select id="sortFilter">
              <option value="image">Image name</option>
              <option value="time_desc">Slowest first</option>
              <option value="time_asc">Fastest first</option>
              <option value="confidence_desc">Highest confidence</option>
              <option value="confidence_asc">Lowest confidence</option>
            </select>
          </div>

          <div class="panel-header" style="padding-left:0; padding-right:0; border-bottom:none; margin-top:10px;">
            <h2>Snapshot</h2>
          </div>
          <div class="mini-grid" id="snapshotGrid"></div>

          <div class="panel-header" style="padding-left:0; padding-right:0; border-bottom:none; margin-top:10px;">
            <h2>Per Target</h2>
          </div>
          <div class="target-list" id="targetSummary"></div>
        </div>
      </aside>

      <main class="panel">
        <div class="panel-body">
          <div class="records-header">
            <h2>Inference Records</h2>
            <div class="record-count" id="recordCount"></div>
          </div>
          <div id="recordList" class="cards"></div>
        </div>
      </main>
    </div>
  </div>

  <script>
    const summary = {summary_json};
    const allRecords = {records_json};

    const createdAt = document.getElementById("createdAt");
    const runDir = document.getElementById("runDir");
    const inputDir = document.getElementById("inputDir");
    const device = document.getElementById("device");
    const initTime = document.getElementById("initTime");
    const summaryStats = document.getElementById("summaryStats");
    const snapshotGrid = document.getElementById("snapshotGrid");
    const targetSummary = document.getElementById("targetSummary");
    const targetFilter = document.getElementById("targetFilter");
    const statusFilter = document.getElementById("statusFilter");
    const matchFilter = document.getElementById("matchFilter");
    const sortFilter = document.getElementById("sortFilter");
    const searchInput = document.getElementById("searchInput");
    const recordCount = document.getElementById("recordCount");
    const recordList = document.getElementById("recordList");

    createdAt.textContent = summary.created_at || "-";
    runDir.textContent = summary.output_dir || "-";
    inputDir.textContent = summary.input_dir || "-";
    device.textContent = summary.device || "-";
    initTime.textContent = `${{(summary.pipeline_init_time_sec || 0).toFixed(3)}}s`;

    const topStats = [
      ["Inferences", summary.summary.total_inferences],
      ["Successes", summary.summary.successful_inferences],
      ["Failures", summary.summary.failed_inferences],
      ["Success Rate", `${{(summary.summary.success_rate * 100).toFixed(1)}}%`],
      ["Match Rate", `${{(summary.summary.ground_truth_match_rate * 100).toFixed(1)}}%`],
      ["Avg IoU", `${{summary.summary.average_iou_on_ground_truth.toFixed(4)}}`],
      ["Avg Process", `${{summary.summary.average_process_time_sec.toFixed(3)}}s`],
      ["Avg Instruction", `${{summary.summary.average_instruction_time_sec.toFixed(3)}}s`],
      ["Avg Total", `${{summary.summary.average_total_time_sec.toFixed(3)}}s`],
      ["Avg Confidence", summary.summary.average_confidence_success_only.toFixed(4)]
    ];

    summaryStats.innerHTML = topStats.map(([label, value]) => `
      <div class="stat-card">
        <div class="label">${{label}}</div>
        <div class="value">${{value}}</div>
      </div>
    `).join("");

    const snapshotStats = [
      ["Images", summary.images.length],
      ["Targets", Object.keys(summary.summary.per_target || {{}}).length],
      ["Device", summary.device],
      ["Init", `${{(summary.pipeline_init_time_sec || 0).toFixed(2)}}s`]
    ];

    snapshotGrid.innerHTML = snapshotStats.map(([label, value]) => `
      <div class="mini-stat">
        <strong>${{value}}</strong>
        <span>${{label}}</span>
      </div>
    `).join("");

    targetSummary.innerHTML = Object.entries(summary.summary.per_target || {{}}).map(([target, stats]) => `
      <div class="target-item">
        <div class="name">${{target}}</div>
        <div>${{stats.success}} / ${{stats.total}} success</div>
        <div>${{stats.matched || 0}} / ${{stats.grounded_total || 0}} matched</div>
        <div>Rate: ${{(stats.success_rate * 100).toFixed(1)}}%</div>
        <div>Match rate: ${{((stats.match_rate || 0) * 100).toFixed(1)}}%</div>
        <div>Avg total: ${{stats.average_total_time_sec.toFixed(3)}}s</div>
      </div>
    `).join("");

    const targets = [...new Set(allRecords.map(record => record.target))].sort();
    targetFilter.innerHTML = ['<option value="all">All targets</option>']
      .concat(targets.map(target => `<option value="${{target}}">${{target}}</option>`))
      .join("");

    function fmt(value, digits = 4) {{
      if (value === null || value === undefined || value === "") return "-";
      if (typeof value === "number") return value.toFixed(digits);
      return value;
    }}

    function renderRecords() {{
      const search = searchInput.value.trim().toLowerCase();
      const target = targetFilter.value;
      const status = statusFilter.value;
      const match = matchFilter.value;
      const sort = sortFilter.value;

      let records = allRecords.filter(record => {{
        const matchesSearch = !search ||
          record.image_name.toLowerCase().includes(search) ||
          record.target.toLowerCase().includes(search);
        const matchesTarget = target === "all" || record.target === target;
        const matchesStatus =
          status === "all" ||
          (status === "success" && record.success) ||
          (status === "fail" && !record.success);
        const matchesMatch = match === "all" || (record.match_status || "not_evaluated") === match;
        return matchesSearch && matchesTarget && matchesStatus && matchesMatch;
      }});

      if (sort === "time_desc") {{
        records.sort((a, b) => (b.total_time_sec || 0) - (a.total_time_sec || 0));
      }} else if (sort === "time_asc") {{
        records.sort((a, b) => (a.total_time_sec || 0) - (b.total_time_sec || 0));
      }} else if (sort === "confidence_desc") {{
        records.sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
      }} else if (sort === "confidence_asc") {{
        records.sort((a, b) => (a.confidence || 0) - (b.confidence || 0));
      }} else {{
        records.sort((a, b) => {{
          const imageCompare = a.image_name.localeCompare(b.image_name);
          if (imageCompare !== 0) return imageCompare;
          return a.target.localeCompare(b.target);
        }});
      }}

      recordCount.textContent = `${{records.length}} record(s) shown`;

      if (!records.length) {{
        recordList.innerHTML = '<div class="empty">No records match the current filters.</div>';
        return;
      }}

      recordList.innerHTML = records.map(record => {{
        const surfaces = Array.isArray(record.surfaces_parsed) ? record.surfaces_parsed : [];
        const bboxText = Array.isArray(record.bbox_parsed) ? record.bbox_parsed.join(", ") : fmt(record.bbox_parsed, 0);
        const gtBboxText = Array.isArray(record.best_gt_bbox_parsed)
          ? record.best_gt_bbox_parsed.join(", ")
          : fmt(record.best_gt_bbox_parsed, 0);
        const imageTag = record.image_rel
          ? `<img src="${{record.image_rel}}" alt="Original image for ${{record.target}}">`
          : `<div class="empty">Original image not available</div>`;
        const vizTag = record.visualization_rel
          ? `<img src="${{record.visualization_rel}}" alt="Visualization for ${{record.target}}">`
          : `<div class="empty">Visualization not available</div>`;
        const comparisonTag = record.comparison_rel
          ? `<img src="${{record.comparison_rel}}" alt="Comparison for ${{record.target}}">`
          : `<div class="empty">Comparison image not available</div>`;
        const audioTag = record.audio_rel
          ? `<audio class="audio" controls src="${{record.audio_rel}}"></audio>`
          : `<div class="empty" style="padding:14px;">No audio available</div>`;

        return `
          <article class="record-card">
            <div class="record-top">
              <div>
                <h3 class="record-title">${{record.target}}</h3>
                <p class="record-subtitle">${{record.image_name}}</p>
                <div class="badge-row">
                  <span class="badge ${{record.success ? "success" : "fail"}}">${{record.success ? "Success" : "Failed"}}</span>
                  <span class="badge">Expected present: ${{record.expected_present ? "yes" : "no"}}</span>
                  <span class="badge">Direction: ${{record.direction || "-"}}</span>
                  <span class="badge ${{record.matched ? "success" : "fail"}}">Match: ${{record.match_status || "-"}}</span>
                </div>
              </div>
              <div class="mono" style="font-size:0.85rem; color: var(--muted);">
                total: ${{fmt(record.total_time_sec)}}s
              </div>
            </div>

            <div class="record-grid">
              <section class="media-block">
                <h3>Original Image</h3>
                ${{imageTag}}
              </section>

              <section class="media-block">
                <h3>Model Visualization</h3>
                ${{vizTag}}
              </section>

              <section class="media-block">
                <h3>Predicted vs Ground Truth</h3>
                ${{comparisonTag}}
                <div class="artifact-links">
                  ${{record.comparison_rel ? `<a href="${{record.comparison_rel}}" target="_blank">Open comparison</a>` : ""}}
                </div>
              </section>

              <section class="metrics-block">
                <h3>Metrics</h3>
                <table class="metric-table">
                  <tr><td>Confidence</td><td>${{fmt(record.confidence)}}</td></tr>
                  <tr><td>Distance (m)</td><td>${{fmt(record.distance_meters)}}</td></tr>
                  <tr><td>Steps</td><td>${{fmt(record.steps)}}</td></tr>
                  <tr><td>Angle</td><td>${{fmt(record.angle)}}°</td></tr>
                  <tr><td>Process time</td><td>${{fmt(record.process_time_sec)}}s</td></tr>
                  <tr><td>Instruction time</td><td>${{fmt(record.instruction_time_sec)}}s</td></tr>
                  <tr><td>Total time</td><td>${{fmt(record.total_time_sec)}}s</td></tr>
                  <tr><td>Predicted box</td><td class="mono">${{bboxText}}</td></tr>
                  <tr><td>Ground-truth box</td><td class="mono">${{gtBboxText}}</td></tr>
                  <tr><td>Best IoU</td><td>${{fmt(record.best_iou)}}</td></tr>
                  <tr><td>Match status</td><td>${{record.match_status || "-"}}</td></tr>
                </table>

                <div class="surfaces">
                  ${{surfaces.length
                    ? surfaces.map(surface => `<span class="surface-chip">${{surface.surface}} (${{
                        Number(surface.confidence || 0).toFixed(3)
                      }})</span>`).join("")
                    : '<span class="surface-chip">No surfaces</span>'}}
                </div>

                <h3 style="margin-top:14px;">Audio</h3>
                ${{audioTag}}

                <div class="artifact-links">
                  ${{record.image_rel ? `<a href="${{record.image_rel}}" target="_blank">Open original</a>` : ""}}
                  ${{record.visualization_rel ? `<a href="${{record.visualization_rel}}" target="_blank">Open visualization</a>` : ""}}
                  ${{record.audio_rel ? `<a href="${{record.audio_rel}}" target="_blank">Open audio</a>` : ""}}
                  ${{record.json_rel ? `<a href="${{record.json_rel}}" target="_blank">Open JSON</a>` : ""}}
                </div>

                ${{record.error ? `<div class="error">Error: ${{record.error}}</div>` : ""}}
              </section>
            </div>
          </article>
        `;
      }}).join("");
    }}

    [searchInput, targetFilter, statusFilter, matchFilter, sortFilter].forEach(element => {{
      element.addEventListener("input", renderRecords);
      element.addEventListener("change", renderRecords);
    }});

    renderRecords();
  </script>
</body>
</html>
"""


def export_pdf(report_path, browser_choice):
    browser_path = find_browser(browser_choice)
    pdf_path = report_path.with_suffix(".pdf")
    report_uri = report_path.resolve().as_uri()

    if os.name == "nt":
        command = [
            "powershell",
            "-Command",
            (
                f"& '{browser_path}' "
                f"--headless=new --disable-gpu --use-angle=swiftshader "
                f"--allow-file-access-from-files "
                f"--print-to-pdf='{pdf_path}' "
                f"'{report_uri}'"
            ),
        ]
    else:
        command = [
            str(browser_path),
            "--headless=new",
            "--disable-gpu",
            "--use-angle=swiftshader",
            "--allow-file-access-from-files",
            f"--print-to-pdf={pdf_path}",
            report_uri,
        ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "PDF export failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    return pdf_path


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir).resolve()
    run_dir = resolve_run_dir(args.run_path, runs_dir).resolve()
    summary = load_summary(run_dir)
    records = flatten_records(run_dir, summary)

    report_path = run_dir / "report.html"
    html = build_html(summary, records)
    report_path.write_text(html, encoding="utf-8")

    print(f"Report generated: {report_path}")
    if args.pdf:
        pdf_path = report_path.with_suffix(".pdf")
        try:
            pdf_path = export_pdf(report_path, args.browser)
            if not pdf_path.exists():
                pdf_path = build_pdf_report(run_dir, summary, records, pdf_path)
        except Exception:
            pdf_path = build_pdf_report(run_dir, summary, records, pdf_path)
        print(f"PDF generated: {pdf_path}")


if __name__ == "__main__":
    main()

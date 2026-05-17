import argparse
import base64
import csv
import json
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = REPO_ROOT / "backend"
DEFAULT_INPUT_DIR = REPO_ROOT / "test_images"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluation_runs"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
YOLO_CLASS_NAMES = {
    0: "phone",
    1: "comb",
    2: "spectacle",
    3: "bottle",
    4: "watch",
}

sys.path.insert(0, str(BACKEND_DIR))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate test_images by running one target search at a time."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing test images named with targets separated by underscores.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where evaluation artifacts will be written.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Pipeline device to use: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Optional specific image filename(s) to evaluate.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Optional target(s) to evaluate. Only matching targets from filenames are used.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output to the final summary only.",
    )
    return parser.parse_args()


def slugify(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_")


def direction_from_angle(angle):
    if abs(angle) <= 5:
        return "straight"
    return "right" if angle > 0 else "left"


def derive_targets_from_filename(image_path):
    stem = image_path.stem.lower()
    parts = [part.strip() for part in stem.split("_") if part.strip()]
    seen = []
    for part in parts:
        if part not in seen:
            seen.append(part)
    return seen


def decode_base64_to_file(encoded, output_path):
    output_path.write_bytes(base64.b64decode(encoded))


def log(message, quiet=False):
    if not quiet:
        print(message)


def collect_images(input_dir, requested_images):
    if requested_images:
        images = []
        for image_name in requested_images:
            image_path = input_dir / image_name
            if not image_path.exists():
                raise FileNotFoundError(f"Requested image not found: {image_path}")
            images.append(image_path)
        return images

    return sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_yolo_annotations(image_path):
    label_path = image_path.with_suffix(".txt")
    if not label_path.exists():
        return []

    annotations = []
    lines = label_path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO annotation in {label_path} line {line_number}: {line}")
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])
        annotations.append(
            {
                "class_id": class_id,
                "class_name": YOLO_CLASS_NAMES.get(class_id, f"class_{class_id}"),
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
            }
        )
    return annotations


def yolo_to_xyxy(annotation, image_width, image_height):
    x_center = annotation["x_center"] * image_width
    y_center = annotation["y_center"] * image_height
    box_width = annotation["width"] * image_width
    box_height = annotation["height"] * image_height

    x1 = max(0.0, x_center - (box_width / 2.0))
    y1 = max(0.0, y_center - (box_height / 2.0))
    x2 = min(float(image_width), x_center + (box_width / 2.0))
    y2 = min(float(image_height), y_center + (box_height / 2.0))
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def compute_iou(box_a, box_b):
    if not box_a or not box_b:
        return 0.0

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def build_ground_truth(image_path):
    from PIL import Image

    annotations = read_yolo_annotations(image_path)
    with Image.open(image_path) as image:
        image_width, image_height = image.size

    ground_truth = []
    for annotation in annotations:
        ground_truth.append(
            {
                **annotation,
                "bbox_xyxy": yolo_to_xyxy(annotation, image_width, image_height),
                "image_width": image_width,
                "image_height": image_height,
            }
        )
    return ground_truth, image_width, image_height


def evaluate_prediction(target, predicted_bbox, ground_truth, iou_threshold=0.5):
    target_ground_truth = [item for item in ground_truth if item["class_name"] == target]
    if not target_ground_truth:
        return {
            "has_ground_truth": False,
            "target_present": False,
            "best_iou": 0.0,
            "best_gt_bbox": None,
            "best_gt_class": None,
            "matched": False,
            "match_status": "no_ground_truth_for_target",
        }

    best_match = None
    best_iou = -1.0
    for item in target_ground_truth:
        iou = compute_iou(predicted_bbox, item["bbox_xyxy"])
        if iou > best_iou:
            best_iou = iou
            best_match = item

    matched = best_iou >= iou_threshold
    return {
        "has_ground_truth": True,
        "target_present": True,
        "best_iou": round(best_iou, 4),
        "best_gt_bbox": best_match["bbox_xyxy"] if best_match else None,
        "best_gt_class": best_match["class_name"] if best_match else None,
        "matched": matched,
        "match_status": "correct_match" if matched else "wrong_match",
    }


def build_run_dir(output_root):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "inferences").mkdir()
    return run_dir


def make_record(image_path, target, expected_targets):
    return {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "target": target,
        "expected_targets": expected_targets,
        "expected_present": target in expected_targets,
    }


def save_json(output_path, payload):
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(csv_path, rows):
    fieldnames = [
        "image_name",
        "target",
        "expected_present",
        "has_ground_truth",
        "match_status",
        "matched",
        "best_iou",
        "success",
        "error",
        "process_time_sec",
        "instruction_time_sec",
        "total_time_sec",
        "confidence",
        "distance_meters",
        "steps",
        "angle",
        "direction",
        "bbox",
        "surfaces",
        "visualization_path",
        "audio_path",
        "json_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_records(records):
    per_target = defaultdict(lambda: {"total": 0, "success": 0, "total_time_sec": 0.0})
    success_records = [record for record in records if record["success"]]
    matched_records = [record for record in records if record.get("matched") is True]
    grounded_records = [record for record in records if record.get("has_ground_truth")]

    for record in records:
        target_stats = per_target[record["target"]]
        target_stats["total"] += 1
        target_stats["total_time_sec"] += record["total_time_sec"]
        if record["success"]:
            target_stats["success"] += 1
        target_stats.setdefault("matched", 0)
        target_stats.setdefault("grounded_total", 0)
        if record.get("has_ground_truth"):
            target_stats["grounded_total"] += 1
        if record.get("matched"):
            target_stats["matched"] += 1

    summary = {
        "total_inferences": len(records),
        "successful_inferences": len(success_records),
        "failed_inferences": len(records) - len(success_records),
        "success_rate": (len(success_records) / len(records)) if records else 0.0,
        "ground_truth_backed_inferences": len(grounded_records),
        "correct_matches": len(matched_records),
        "ground_truth_match_rate": (
            len(matched_records) / len(grounded_records) if grounded_records else 0.0
        ),
        "average_process_time_sec": (
            sum(record["process_time_sec"] for record in records) / len(records) if records else 0.0
        ),
        "average_instruction_time_sec": (
            sum(record["instruction_time_sec"] for record in records) / len(records) if records else 0.0
        ),
        "average_total_time_sec": (
            sum(record["total_time_sec"] for record in records) / len(records) if records else 0.0
        ),
        "average_confidence_success_only": (
            sum(record["confidence"] for record in success_records) / len(success_records)
            if success_records
            else 0.0
        ),
        "average_iou_on_ground_truth": (
            sum(record["best_iou"] for record in grounded_records) / len(grounded_records)
            if grounded_records
            else 0.0
        ),
        "per_target": {},
    }

    for target, stats in sorted(per_target.items()):
        summary["per_target"][target] = {
            "total": stats["total"],
            "success": stats["success"],
            "success_rate": (stats["success"] / stats["total"]) if stats["total"] else 0.0,
            "grounded_total": stats["grounded_total"],
            "matched": stats["matched"],
            "match_rate": (stats["matched"] / stats["grounded_total"]) if stats["grounded_total"] else 0.0,
            "average_total_time_sec": (
                stats["total_time_sec"] / stats["total"] if stats["total"] else 0.0
            ),
        }

    return summary


def main():
    args = parse_args()
    from pipeline import NavigationPipeline  # noqa: E402

    input_dir = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    requested_targets = {target.lower().strip() for target in args.target if target.strip()}

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_root.mkdir(parents=True, exist_ok=True)
    images = collect_images(input_dir, args.image)
    if not images:
        raise FileNotFoundError(f"No test images found in: {input_dir}")

    log("Evaluation configuration:", args.quiet)
    log(f"  input_dir:  {input_dir}", args.quiet)
    log(f"  output_dir: {output_root}", args.quiet)
    log(f"  device arg: {args.device}", args.quiet)
    log(f"  image count discovered: {len(images)}", args.quiet)
    if requested_targets:
        log(f"  target filter: {', '.join(sorted(requested_targets))}", args.quiet)
    else:
        log("  target filter: none", args.quiet)
    log("", args.quiet)

    log(f"Initializing pipeline on device={args.device}", args.quiet)
    pipeline_init_started = time.perf_counter()
    pipeline = NavigationPipeline(device=args.device)
    pipeline_init_elapsed = time.perf_counter() - pipeline_init_started
    log(f"Pipeline initialized in {pipeline_init_elapsed:.3f}s", args.quiet)
    log(f"Active device: {pipeline.get_device_str()}", args.quiet)
    if getattr(pipeline, "model_status", None):
        log("Model status:", args.quiet)
        for model_name, status in pipeline.model_status.items():
            log(f"  - {model_name}: {status}", args.quiet)
    log("", args.quiet)

    run_dir = build_run_dir(output_root)
    records = []

    run_manifest = {
        "created_at": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(run_dir),
        "device": pipeline.get_device_str(),
        "pipeline_init_time_sec": round(pipeline_init_elapsed, 4),
        "model_status": getattr(pipeline, "model_status", {}),
        "images": [],
    }

    log(f"Artifacts will be written to: {run_dir}", args.quiet)
    log("", args.quiet)

    for image_path in images:
        expected_targets = derive_targets_from_filename(image_path)
        if requested_targets:
            expected_targets = [target for target in expected_targets if target in requested_targets]
        if not expected_targets:
            log(f"Skipping {image_path.name}: no targets left after filtering", args.quiet)
            continue

        ground_truth, image_width, image_height = build_ground_truth(image_path)

        image_run_dir = run_dir / "inferences" / slugify(image_path.stem)
        image_run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, image_run_dir / image_path.name)

        image_manifest = {
            "image_name": image_path.name,
            "image_path": str(image_path),
            "targets": expected_targets,
            "ground_truth_count": len(ground_truth),
            "ground_truth": ground_truth,
            "records": [],
        }

        log("=" * 72, args.quiet)
        log(f"Image: {image_path.name}", args.quiet)
        log(f"Copied input image to: {image_run_dir / image_path.name}", args.quiet)
        log(f"Targets from filename: {', '.join(expected_targets)}", args.quiet)
        log(f"Ground-truth boxes loaded: {len(ground_truth)}", args.quiet)

        for target in expected_targets:
            record = make_record(image_path, target, expected_targets)
            record_dir = image_run_dir / slugify(target)
            record_dir.mkdir(parents=True, exist_ok=True)

            log("-" * 72, args.quiet)
            log(f"Running target search: '{target}'", args.quiet)
            log(f"Per-target artifact directory: {record_dir}", args.quiet)

            process_started = time.perf_counter()
            process_result = pipeline.process_image(str(image_path), target)
            process_elapsed = time.perf_counter() - process_started

            record["process_time_sec"] = round(process_elapsed, 4)
            record["success"] = bool(process_result.get("success"))
            record["error"] = ""
            record["instruction_time_sec"] = 0.0
            record["total_time_sec"] = round(process_elapsed, 4)
            record["confidence"] = 0.0
            record["distance_meters"] = None
            record["steps"] = None
            record["angle"] = None
            record["direction"] = None
            record["bbox"] = None
            record["surfaces"] = None
            record["visualization_path"] = ""
            record["audio_path"] = ""
            record["has_ground_truth"] = False
            record["match_status"] = "not_evaluated"
            record["matched"] = False
            record["best_iou"] = 0.0
            record["best_gt_bbox"] = None

            artifact_payload = {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "target": target,
                "expected_targets": expected_targets,
                "ground_truth": ground_truth,
                "timing": {
                    "process_time_sec": round(process_elapsed, 4),
                    "instruction_time_sec": 0.0,
                    "total_time_sec": round(process_elapsed, 4),
                },
                "process_result": process_result,
                "instruction_result": None,
                "artifacts": {},
            }

            log(f"process_image time: {process_elapsed:.4f}s", args.quiet)

            if process_result.get("success"):
                record["confidence"] = round(float(process_result.get("confidence", 0.0)), 4)
                record["distance_meters"] = round(float(process_result.get("distance_meters", 0.0)), 4)
                record["steps"] = round(float(process_result.get("steps", 0.0)), 4)
                record["angle"] = round(float(process_result.get("angle", 0.0)), 4)
                record["direction"] = direction_from_angle(record["angle"])
                record["bbox"] = json.dumps(process_result.get("bbox", []))
                record["surfaces"] = json.dumps(process_result.get("surfaces", []))

                match_evaluation = evaluate_prediction(
                    target=target,
                    predicted_bbox=process_result.get("bbox", []),
                    ground_truth=ground_truth,
                )
                record["has_ground_truth"] = match_evaluation["has_ground_truth"]
                record["match_status"] = match_evaluation["match_status"]
                record["matched"] = match_evaluation["matched"]
                record["best_iou"] = match_evaluation["best_iou"]
                record["best_gt_bbox"] = json.dumps(match_evaluation["best_gt_bbox"]) if match_evaluation["best_gt_bbox"] else None
                artifact_payload["match_evaluation"] = match_evaluation

                visualization_base64 = process_result.get("visualization")
                if visualization_base64:
                    visualization_path = record_dir / "visualization.png"
                    decode_base64_to_file(visualization_base64, visualization_path)
                    record["visualization_path"] = str(visualization_path)
                    artifact_payload["artifacts"]["visualization_path"] = str(visualization_path)
                    log(f"Saved visualization: {visualization_path}", args.quiet)

                instruction_started = time.perf_counter()
                instruction_result = pipeline.generate_instruction(
                    process_result["target"],
                    process_result["steps"],
                    process_result["angle"],
                    process_result.get("distance_meters"),
                    process_result.get("confidence"),
                    process_result.get("depth"),
                    surfaces=process_result.get("surfaces", []),
                )
                audio_path = pipeline.text_to_speech(instruction_result["conversational"])
                instruction_elapsed = time.perf_counter() - instruction_started

                saved_audio_path = record_dir / "instruction.wav"
                shutil.copy2(audio_path, saved_audio_path)

                record["instruction_time_sec"] = round(instruction_elapsed, 4)
                record["total_time_sec"] = round(process_elapsed + instruction_elapsed, 4)
                record["audio_path"] = str(saved_audio_path)

                artifact_payload["instruction_result"] = instruction_result
                artifact_payload["timing"]["instruction_time_sec"] = round(instruction_elapsed, 4)
                artifact_payload["timing"]["total_time_sec"] = round(process_elapsed + instruction_elapsed, 4)
                artifact_payload["artifacts"]["audio_path"] = str(saved_audio_path)

                log(
                    "Result: "
                    f"success | confidence={record['confidence']:.4f} | "
                    f"distance={record['distance_meters']:.4f}m | "
                    f"steps={record['steps']:.4f} | "
                    f"angle={record['angle']:.4f} | "
                    f"direction={record['direction']}",
                    args.quiet,
                )
                log(
                    f"Ground-truth check: status={record['match_status']} | "
                    f"matched={record['matched']} | best_iou={record['best_iou']:.4f}",
                    args.quiet,
                )
                log(f"Surfaces: {record['surfaces']}", args.quiet)
                log(f"Instruction generation + audio time: {instruction_elapsed:.4f}s", args.quiet)
                log(f"Saved audio: {saved_audio_path}", args.quiet)
                log(f"Total target time: {record['total_time_sec']:.4f}s", args.quiet)
            else:
                record["error"] = process_result.get("error", "Unknown error")
                log(f"Result: failed | error={record['error']}", args.quiet)

            record_json_path = record_dir / "result.json"
            artifact_payload["record"] = record
            save_json(record_json_path, artifact_payload)
            record["json_path"] = str(record_json_path)
            log(f"Saved JSON result: {record_json_path}", args.quiet)

            records.append(record)
            image_manifest["records"].append(record)

        run_manifest["images"].append(image_manifest)

    summary = summarize_records(records)
    run_manifest["summary"] = summary

    save_json(run_dir / "summary.json", run_manifest)
    write_csv(run_dir / "summary.csv", records)

    print("")
    print("=" * 72)
    print("Evaluation run complete")
    print(f"Run directory: {run_dir}")
    print(f"Pipeline init time: {pipeline_init_elapsed:.3f}s")
    print(f"Total inferences: {summary['total_inferences']}")
    print(f"Successful: {summary['successful_inferences']}")
    print(f"Failed: {summary['failed_inferences']}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(f"Ground-truth backed inferences: {summary['ground_truth_backed_inferences']}")
    print(f"Correct matches: {summary['correct_matches']}")
    print(f"Ground-truth match rate: {summary['ground_truth_match_rate']:.2%}")
    print(f"Average IoU on ground truth: {summary['average_iou_on_ground_truth']:.4f}")
    print(f"Average process time: {summary['average_process_time_sec']:.3f}s")
    print(f"Average instruction time: {summary['average_instruction_time_sec']:.3f}s")
    print(f"Average total time: {summary['average_total_time_sec']:.3f}s")
    print(f"Average confidence on successes: {summary['average_confidence_success_only']:.4f}")
    print(f"Summary JSON: {run_dir / 'summary.json'}")
    print(f"Summary CSV:  {run_dir / 'summary.csv'}")

    if summary["per_target"]:
        print("")
        print("Per-target summary:")
        for target, stats in summary["per_target"].items():
            print(
                f"  {target}: total={stats['total']}, success={stats['success']}, "
                f"success_rate={stats['success_rate']:.2%}, "
                f"match_rate={stats['match_rate']:.2%}, "
                f"avg_total_time={stats['average_total_time_sec']:.3f}s"
            )


if __name__ == "__main__":
    main()

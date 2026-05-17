# CV Powered Blind And Visual Assistance Tool

This project is a computer-vision-assisted navigation tool for blind and visually impaired users. It combines image understanding, target detection, depth estimation, and spoken guidance into a simple web interface with two modes:

- `Blind Mode`: capture or upload an image, speak the object you want to find, and receive voice guidance.
- `Visually Impaired Mode`: capture or upload an image, specify a target, and receive visual plus spoken navigation output.

## Main Features

- Object detection with `GroundingDINO`
- Depth estimation with `Depth Anything 3`
- Speech-to-text with `faster-whisper`
- Spoken instruction generation with local TTS
- Optional few-shot matching support in `few_shot/`
- Evaluation runner for per-image, per-target testing
- HTML report generation for reviewing test results

## Project Structure

```text
backend/
  app.py                  Flask API endpoints
  pipeline.py             Core CV and speech pipeline
  requirements.txt        Python dependencies
  uploads/                Uploaded images and temporary assets

frontend/
  index.html              UI shell
  app.js                  Frontend app logic

test_images/              Evaluation images and YOLO label files
evaluation_runs/          Generated evaluation outputs and reports

app.py                    Root app entrypoint for frontend + backend
main.py                   Alternative app runner
evaluate_test_images.py   Batch evaluation script
generate_evaluation_report.py  HTML report generator
```

## Requirements

- Python 3.10+
- Windows or Linux
- CUDA-capable GPU recommended, but CPU mode is supported

## Install

1. Create and activate a virtual environment.
2. Install backend dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r backend/requirements.txt
```

## Run The App

Start the combined app from the project root:

```powershell
python app.py
```

The backend serves the API and the frontend UI. By default, the Flask app runs on port `5001`.

Health check:

```text
GET /health
```

Main API routes:

- `POST /api/upload`
- `POST /api/process`
- `POST /api/transcribe`
- `POST /api/generate-instruction`

## Evaluation Workflow

The repository includes an evaluation runner that tests one target at a time for each image.

### Test Data Naming

Images in `test_images/` are named using the objects present in the scene, separated by underscores.

Example:

```text
phone_comb_watch.jpeg
```

This means the evaluator will run three separate searches:

- `phone`
- `comb`
- `watch`

### Ground Truth Labels

Ground-truth labels can live next to the images in YOLO format:

```text
test_images/
  phone_comb_watch.jpeg
  phone_comb_watch.txt
```

Recommended class mapping:

```text
0 phone
1 comb
2 spectacle
3 bottle
4 watch
```

### Run Evaluation

Run all images:

```powershell
python evaluate_test_images.py
```

Run a single image:

```powershell
python evaluate_test_images.py --image phone.jpeg
```

Run a single image and target:

```powershell
python evaluate_test_images.py --image phone_comb_watch.jpeg --target phone
```

CPU only:

```powershell
python evaluate_test_images.py --device cpu
```

The script:

- loads models once per run
- evaluates each image-target pair separately
- records timings, confidence, distance, steps, angle, bbox, surfaces, and audio
- compares predicted boxes against YOLO ground truth
- computes best IoU and flags `correct_match` vs `wrong_match`

Outputs are written to a timestamped directory under `evaluation_runs/`.

## Generate HTML Report

Generate a report for the latest run:

```powershell
python generate_evaluation_report.py
```

Generate a report for a specific run:

```powershell
python generate_evaluation_report.py evaluation_runs/run_20260517_101813
```

The generated `report.html` shows:

- run summary
- per-target summary
- original image
- model visualization
- audio output
- timings and metrics
- predicted vs ground-truth box information
- IoU and match status

## Annotation Recommendation

For easy manual labeling, use `make-sense`:

- App: https://skalskip.github.io/make-sense/
- Repository: https://github.com/SkalskiP/make-sense

Recommended annotation type:

- bounding boxes
- YOLO export format

Overlapping boxes are fine and expected for nearby objects.

## Notes

- Model loading can take time on first run.
- Large checkpoints and generated outputs are intentionally excluded in `.gitignore`.
- `backend/pipeline.py` is the main place to inspect or improve detection, depth, and instruction logic.

## Next Good Improvements

- Draw predicted and ground-truth boxes together in evaluation artifacts
- Add precision/recall metrics per class
- Add failure-only filtered report views
- Add automated regression tests for API routes

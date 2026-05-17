# CV Powered Blind And Visual Assistance Tool

This project is a computer-vision-assisted navigation tool for blind and visually impaired users. It combines target detection, depth estimation, speech input, and spoken guidance in a browser-based interface.

The app currently supports three main user flows:

- `Blind Mode`: capture or upload an image, speak what you want to find, and receive spoken guidance.
- `Visually Impaired Mode`: capture or upload an image, type or speak a target, and receive visual plus spoken guidance.
- `Add Reference`: save a named reference image such as `comb`, `phone`, or `spectacle` so the system can use reference-assisted matching during search.

## Main Features

- Open-vocabulary object detection with `GroundingDINO`
- Depth estimation with `Depth Anything 3`
- Speech transcription with `faster-whisper`
- Instruction generation plus text-to-speech audio output
- Optional reference-image-assisted matching with a Siamese few-shot matcher
- Batch evaluation with YOLO ground-truth support
- Static HTML and PDF evaluation reports

## Project Structure

```text
backend/
  app.py                    Backend API routes
  pipeline.py               Core CV, speech, depth, and reference-matching pipeline
  references/               Saved reference images + manifest
  requirements.txt          Python dependencies
  uploads/                  Uploaded images and temporary assets

frontend/
  index.html                UI shell
  app.js                    Frontend app logic

reference_images/           Optional evaluation-time reference images
test_images/                Evaluation images and YOLO label files
evaluation_runs/            Generated evaluation outputs and reports

app.py                      Root app entrypoint
main.py                     Alternative app runner
evaluate_test_images.py     Batch evaluation script
generate_evaluation_report.py
README.md
```

## Requirements

- Python 3.10+
- Windows or Linux
- CUDA-capable GPU recommended for best speed
- CPU mode is supported, but slower

## Install

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r backend/requirements.txt
```

## Run The App

Start the app from the project root:

```powershell
python app.py
```

The app serves the frontend and backend together. By default it runs on port `5001`.

Useful routes:

- `GET /health`
- `POST /api/upload`
- `POST /api/process`
- `POST /api/transcribe`
- `POST /api/generate-instruction`
- `GET /api/references`
- `POST /api/references`
- `DELETE /api/references/<reference_name>`

## Reference Images

The `Add Reference` tile opens a dedicated reference-management screen in the frontend.

From there you can:

- save a reference image with a name
- list saved references
- delete saved references

Saved references are stored under:

```text
backend/references/
```

and tracked in:

```text
backend/references/references.json
```

### How Reference Matching Works

Reference matching does not replace GroundingDINO. The current flow is:

1. GroundingDINO proposes candidate boxes
2. If a matching saved reference exists, the pipeline may rerank those candidates using the reference image
3. The best candidate is selected
4. Depth, navigation, and instruction generation continue as usual

To avoid hurting strong detections, the pipeline only uses reference reranking when the top GroundingDINO candidates are close. If the top DINO candidate is clearly ahead, the system keeps the original DINO choice.

This logic lives in:

- [backend/pipeline.py](backend/pipeline.py)

## Evaluation Workflow

The repository includes a batch evaluator that runs one target search at a time for each image.

### Test Image Naming

Images in `test_images/` are named using the objects present in the scene, separated by underscores.

Example:

```text
phone_comb_watch.jpeg
```

This produces three separate searches:

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

Current class mapping:

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

Run on CPU:

```powershell
python evaluate_test_images.py --device cpu
```

### Evaluation Reference Images

The evaluator can also register reference images before running tests.

Example currently supported in code:

```text
reference_images/comb.jpeg
```

That reference is used only for the `comb` target during evaluation.

The evaluator:

- loads models once per run
- can register reference images once at startup
- evaluates each image-target pair separately
- records timings, confidence, distance, steps, angle, bbox, surfaces, and audio
- compares predicted boxes against YOLO ground truth
- computes best IoU
- marks `correct_match` vs `wrong_match`
- records whether a reference was used for that inference

Outputs are written to a timestamped directory under `evaluation_runs/`.

## Generate Evaluation Reports

Generate a report for the latest run:

```powershell
python generate_evaluation_report.py
```

Generate a report for a specific run:

```powershell
python generate_evaluation_report.py evaluation_runs/run_20260517_141120
```

Generate HTML plus PDF:

```powershell
python generate_evaluation_report.py evaluation_runs/run_20260517_141120 --pdf
```

The generated report includes:

- run summary
- per-target summary
- filters including match status
- original image
- model visualization
- comparison overlays
- audio output
- timings and metrics
- predicted vs ground-truth box information
- IoU and match status

## Annotation Recommendation

For easy manual labeling, `make-sense` is a good lightweight option:

- App: https://skalskip.github.io/make-sense/
- Repository: https://github.com/SkalskiP/make-sense

Recommended labeling setup:

- annotation type: bounding boxes
- export format: YOLO

Overlapping boxes are normal and acceptable.

## Current Notes

- Model loading can take time, especially on first startup.
- The evaluator and report generator are useful for comparing pipeline changes across runs.
- Reference-assisted matching can help on ambiguous scenes, but should be evaluated carefully because a poor reference or overly aggressive reranking can hurt performance.
- `backend/pipeline.py` is the main file for detection, reranking, depth, and navigation logic.

## Good Next Improvements

- Support multiple reference images per object
- Log whether reference reranking was actually applied vs only available
- Add confusion summaries for common target mix-ups
- Add stricter uncertainty handling for weak detections
- Extend evaluation metrics with precision/recall style summaries

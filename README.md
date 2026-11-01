# Basketball Play Analysis

> Detect and track the ball and five numbered offensive players in animated basketball play-diagram videos, then export every position to a CSV and plot each object's movement path.

This is a student computer-vision project (Katz School of Science and Health, Yeshiva University, Fall 2026). It uses a YOLO model fine-tuned on Roboflow (`weights.pt`) and the Ultralytics BoT-SORT tracker to turn a play-diagram video into:

- an annotated video with a bounding box and label on each tracked object (`tracked_basketball.mp4`)
- a per-frame position log (`player_positions.csv`)
- a movement map that draws each object's path over an empty court image (displayed inline in the notebook)

## Why this exists

Coaches break down offensive plays by hand. The long-term goal in the project proposal is to recover structured play data (who had the ball, where each player moved) automatically from video. The current code tackles the first, simplest step: tracking the ball and players in clean, top-down, animated play diagrams (the "Coach Calls Timeout" style videos linked in `Notes.txt`) and saving their positions for later analysis.

> **Scope note.** The proposal (`../Basketball_Play_Analysis_Proposal.docx`) describes a much larger system for real single-camera game footage. Most of that is not implemented. See [Planned vs. implemented](docs/ARCHITECTURE.md#planned-vs-implemented) before you rely on any feature mentioned in the proposal or slides.

## Quick start

```bash
cd Final_Project_Computer_Vision
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install ultralytics opencv-python matplotlib pillow pandas tqdm jupyterlab
jupyter lab main.ipynb
```

In Jupyter, run cells 1–3 of `main.ipynb` in order (cell 0 is the `%pip install`). Cell 2 writes `tracked_basketball.mp4` and `player_positions.csv`; cell 3 displays the movement map.

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3 | The notebooks were last run on Python **3.14.6** (from notebook metadata). Use any Python version that your `torch` / `ultralytics` build supports. |
| Jupyter | Both pipelines are `.ipynb` notebooks. Cell 0 of `main.ipynb` uses the `%pip` magic, so it only works inside IPython/Jupyter. |
| `weights.pt` | Fine-tuned YOLO weights (about 44 MB) in the project folder. They are loaded by relative path; you cannot run detection without them. |
| A GPU (optional) | Ultralytics falls back to CPU automatically. CPU works but processes the 1,120-frame test video more slowly. |

### Python packages

These are every third-party package the notebooks import. Versions come from the `%pip install` output saved in `main.ipynb` (the last environment that ran it).

| Package | Imported as | Used in | Last-run version |
|---|---|---|---|
| `ultralytics` | `from ultralytics import YOLO` | `main.ipynb` cells 1, 2 | 8.4.177 |
| `opencv-python` | `cv2` | both notebooks | 5.0.0.93 |
| `matplotlib` | `matplotlib.pyplot`, `matplotlib.lines.Line2D` | `main.ipynb` cells 3, 5, 6 | 3.11.2 |
| `numpy` | `numpy` | `main.ipynb` cells 3, 5, 6, 7 | 2.4.1 (installed by ultralytics) |
| `pandas` | `pandas` | `main.ipynb` cells 3, 5, 6, 7 | not recorded |
| `tqdm` | `from tqdm.auto import tqdm` | `main.ipynb` cell 2 | not recorded |
| `pillow` | not imported directly | installed by cell 0 only | 12.3.0 |
| `torch` / `torchvision` | not imported directly | pulled in by `ultralytics` | 2.10.0 / 0.25.0 |

> **Important:** cell 0 installs only `ultralytics opencv-python matplotlib pillow`. It does **not** install `pandas` or `tqdm`, which cells 2–7 import. Install them yourself (the Quick start command includes them). There is no `requirements.txt`.

## Installation

1. Open a terminal in `Final_Project_Computer_Vision/`. Every path in the notebooks is relative to this folder, so you must launch Jupyter from here (or set the kernel's working directory here).
2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   ```

3. Install the dependencies:

   ```bash
   pip install ultralytics opencv-python matplotlib pillow pandas tqdm jupyterlab
   ```

4. Confirm `weights.pt`, `test_2.mp4`, and `background.jpeg` exist in the folder.

## Usage

Run the notebooks in this order. Step 1 is only needed if you want to regenerate training frames; the tracking pipeline does not depend on it.

### Step 1 (optional): extract fine-tuning frames — `frame_video.ipynb`

This notebook has a single cell. It opens every `.mp4` in `input_folder`, skips to `start_time_sec` (1 s), and saves the next `frames_per_video` (10) consecutive frames as JPEGs.

| Setting | Value in code | Meaning |
|---|---|---|
| `input_folder` | `"more_frames"` | Folder scanned for `*.mp4` |
| `output_folder` | `"frames"` | Created if missing; JPEGs are written here |
| `start_time_sec` | `1` | Seek position before saving starts |
| `frames_per_video` | `10` | Consecutive frames saved per video |

> **Path mismatch:** the code reads from `more_frames/`, but the videos in this repository live in `videos_for_frames/`. Before you run the cell, either change `input_folder = "videos_for_frames"` or rename the folder. Otherwise the cell prints `No .mp4 files found in 'more_frames'` and writes nothing.

Output: `frames/<video_name>_frame_NN.jpg` (for example `frames/fine_tune_3_frame_07.jpg`). The repository already contains 100 such frames (10 videos × 10 frames).

These frames were annotated and used to fine-tune the model on Roboflow (link in `Notes.txt`). **Annotation and training happened outside this codebase**; no labels, dataset YAML, or training script are included.

### Step 2: track the video — `main.ipynb`

| Cell | What it does |
|---|---|
| 0 | `%pip install ultralytics opencv-python matplotlib pillow` |
| 1 | Loads `weights.pt` and prints the class map. Expected output: `Classes: {0: 'ball', 1: 'player_1', 2: 'player_2', 3: 'player_3', 4: 'player_4', 5: 'player_5'}` |
| 2 | **Main pipeline.** Runs detection + BoT-SORT tracking on `test_2.mp4`, applies a jump filter, draws boxes, writes `tracked_basketball.mp4` and `player_positions.csv` |
| 3 | Reads the CSV and shows a movement map over `background.jpeg` |
| 4 | Markdown: "IGNORE THIS - I WAS CHECKING SOMETINHG WITH THE THRESHOLD" |
| 5–7 | Diagnostics used to choose the jump threshold: biggest `player_2` jumps (cell 5), `player_1` position/speed plots (cell 6), movement percentiles for `player_1` and `player_2` (cell 7) |
| 8 | Empty |

To process a different video, edit these variables at the top of cell 2:

```python
video_path = "test_2.mp4"
output_path = "tracked_basketball.mp4"
csv_path = "player_positions.csv"

CONFIDENCE = 0.25
JUMP_THRESHOLD = 10.0  # Maximum allowed movement in pixels per frame
```

Cell 3 reads `"player_positions.csv"` and `"test_2.mp4"` by literal name, not from the cell 2 variables. If you change `video_path` or `csv_path`, update cell 3 too.

### Inputs and outputs

| File | Direction | Produced / used by |
|---|---|---|
| `videos_for_frames/fine_tune_*.mp4` | input | `frame_video.ipynb` (after you fix `input_folder`) |
| `frames/*.jpg` | output | `frame_video.ipynb` |
| `weights.pt` | input | `main.ipynb` cells 1, 2 |
| `test_2.mp4` | input | `main.ipynb` cells 2, 3 |
| `background.jpeg` | input | `main.ipynb` cell 3 |
| `tracked_basketball.mp4` | output | `main.ipynb` cell 2 |
| `player_positions.csv` | output (input to cells 3, 5–7) | `main.ipynb` cell 2 |

See [docs/DATA.md](docs/DATA.md) for formats and the CSV schema.

## Project layout

```text
Computer Vision/
├── Basketball_Play_Analysis_Proposal.docx      # Project proposal (goals, planned method)
├── Basketball_Play_Analysis_Presentation.pptx  # Proposal slides
└── Final_Project_Computer_Vision/
    ├── README.md                 # This file
    ├── docs/
    │   ├── ARCHITECTURE.md       # Pipeline internals, parameters, planned vs. implemented
    │   └── DATA.md               # File formats and CSV schema
    ├── main.ipynb                # Detection, tracking, CSV export, movement map, diagnostics
    ├── frame_video.ipynb         # Extracts JPEG frames from videos for annotation
    ├── Notes.txt                 # Data sources, Roboflow link, instructor feedback
    ├── weights.pt                # Fine-tuned YOLO weights (6 classes)
    ├── test_2.mp4                # Video that main.ipynb processes
    ├── test_3.mp4                # Unused by code (byte-identical to fine_tune_10.mp4)
    ├── background.jpeg           # Empty court image for the movement map
    ├── tracked_basketball.mp4    # Output: annotated video
    ├── player_positions.csv      # Output: per-frame positions
    ├── videos_for_frames/        # 10 source videos: fine_tune_1.mp4 … fine_tune_10.mp4
    └── frames/                   # 100 extracted JPEGs: fine_tune_N_frame_00..09.jpg
```

`__MACOSX/`, `.ipynb_checkpoints/`, and `.DS_Store` files are macOS/Jupyter artifacts. You can ignore or delete them.

## Troubleshooting

**`No .mp4 files found in 'more_frames'`** (`frame_video.ipynb`)
The code reads from `more_frames/`, which does not exist in this repository. Set `input_folder = "videos_for_frames"`.

**`FileNotFoundError` / Ultralytics tries to download `weights.pt`** (`main.ipynb` cell 1 or 2)
`YOLO("weights.pt")` resolves relative to the kernel's working directory. Start Jupyter from `Final_Project_Computer_Vision/`, or check with `import os; os.getcwd()`. If the file is genuinely missing, you must re-export it from the Roboflow project listed in `Notes.txt`; the repository has no training script to rebuild it.

**`RuntimeError: Cannot open video: test_2.mp4`** (cell 2)
Same cause: wrong working directory, or a typo in `video_path`. OpenCV also fails silently on some codecs; confirm the file opens in a video player.

**`ValueError: Could not read video dimensions.`** (cell 3)
Cell 3 opens `"test_2.mp4"` by literal name. If you renamed or moved the video, edit cell 3.

**`FileNotFoundError: Could not find background.jpeg`** (cell 3)
`background.jpeg` must sit in the working directory.

**`ModuleNotFoundError: No module named 'pandas'` or `'tqdm'`**
Cell 0 does not install them. Run `pip install pandas tqdm`.

**Progress bar shows as plain text or a widget error** (cell 2)
`tqdm.auto` uses Jupyter widgets when available. Install `ipywidgets` for a graphical bar; the pipeline works either way.

**`RuntimeError: Cannot create output video: tracked_basketball.mp4`**
OpenCV could not open the `mp4v` writer. Check that the folder is writable and that the file is not open in another program.

**`tracked_basketball.mp4` will not play in a browser or some players**
The writer uses the `mp4v` (MPEG-4 Part 2) codec, which many browsers do not support. Open it in VLC, or re-encode it to H.264 with a tool such as ffmpeg.

**Mac paths in saved outputs**
Printed outputs in `main.ipynb` show `/Users/aradmishkal/...` paths and a pipx JupyterLab environment. These come from the author's machine and are **not** used by the code: every path in the code is relative. You can ignore them.

**A player's box disappears for a while, then returns**
The jump filter rejects any detection that moves more than `JUMP_THRESHOLD × frames_since_last_accept` pixels from the last accepted position. A detection that is rejected is neither drawn nor logged. See [Jump filter](docs/ARCHITECTURE.md#jump-filter) for tuning.

**Diagnostic cells 5–7 show numbers that do not match the current CSV**
The saved outputs of cells 5–7 came from an older `player_positions.csv`. For example, they report `player_2` jumps of up to 488.9 px between consecutive frames, while the CSV in the repository has a maximum per-frame `player_2` movement of about 3.2 px, consistent with the 10 px jump filter. The old outputs most likely predate the filter. Re-run the cells to get current values.

## Further reading

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — pipeline walkthrough, parameters, magic numbers, and planned vs. implemented features
- [docs/DATA.md](docs/DATA.md) — every input and output file, with the CSV schema and example rows
- `Notes.txt` — video sources, Roboflow project link, and instructor feedback that shaped the current scope

## Authors

Kenneth Charumuka, Ezekiel Ponde, Arad Mishkal, Faith Mashava (from the project proposal).

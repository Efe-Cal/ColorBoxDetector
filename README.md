# ColorBoxDetector

Color-based box (and corner/parallelogram) detection utilities with multiple algorithm versions and Raspberry Pi deployment support.

## Features
- Multiple detection pipelines (V1.2 HSV + morphology, V3 channel subtraction).
- Configurable HSV color ranges via interactive ROI tools.
- Support for detecting 3 colored reference boxes + a yellow locator box.
- Parallelogram / corner arrangement utilities.
- Single-file deploy script for Raspberry Pi: [raspi_functions.py](raspi_functions.py).
- Modular refactored detector: [color_detector_refactored.py](color_detector_refactored.py).

## Project Layout (Key Files)
- Core (versioned) scripts:  
  - [cbd_v1.2.py](cbd_v1.2.py) (HSV + distance transform)  
  - [cbd_v2.py](cbd_v2.py)  
  - [cbd_v3.py](cbd_v3.py) (newer experiments)  
  - [color_detector_refactored.py](color_detector_refactored.py)  
  - [raspi_functions.py](raspi_functions.py) (all-in-one)
- Tools (interactive config generation):  
  - [tools/create_config.py](tools/create_config.py) (main object colors)  
  - [tools/create_boxes_config.py](tools/create_boxes_config.py) (4 small boxes set)  
  - [tools/new_obj.py](tools/new_obj.py) (extend existing config)
- Parallelogram helpers:  
  - [parallelogram_detector_dual_method.py](parallelogram_detector_dual_method.py)  
  - [parallelogram_detector_v1_2.py](parallelogram_detector_v1_2.py)

## Configuration
JSON configs are ignored by Git (`.gitignore` contains *.json) so generate them locally.

Main: `config.json`  
Secondary (4-box mode): `config_boxes.json`

Both contain:
```
{
  "color_ranges": {
    "red": [ [ [low_H,low_S,low_V], [high_H,high_S,high_V] ], ... ],
    "green": [...],
    "blue": [...],
    "yellow": [...]
  },
  "big_box_crop": [x, y, w, h]          (optional)
  "boxes_crop":   [x, y, w, h]          (in config_boxes.json)
}
```

Loaded via:
- [`load_config`](cbd_v1.2.py)
- [`load_config`](raspi_functions.py) (two variants: one for `config.json` near line ~227 and one for `config_boxes.json` near line ~377)
- [`load_config`](color_detector_refactored.py)

If missing, some scripts fall back to hard‑coded defaults.

## Generating Configs
Interactive (requires GUI display):
```
python tools/create_config.py         # builds config.json
python tools/create_boxes_config.py   # builds config_boxes.json
python tools/new_obj.py               # append/merge new object-specific ranges
```
Follow on-screen ROI selections. Color ranges are auto-generated per ROI.

## Detection Pipelines

### V1.2 (HSV + Distance Transform)
Implemented in:
- [`preprocess_v1_2_method`](raspi_functions.py)
- Supporting utilities: [`detect_contours_and_centroids`](raspi_functions.py)

Flow:
1. HSV threshold per color range.
2. Morphological cleanup.
3. Distance transform to tighten mask.
4. Contour filtering (area + aspect ratio).
5. Representative centroid per channel.

### V3 (Channel Isolation & Subtraction)
Implemented in:
- [`preprocess_v3`](color_detector_refactored.py)
- Inline helper: `isolate_and_subtract_channel` inside that function and standalone in [raspi_functions.py](raspi_functions.py) (top section).

Flow:
1. Split BGR.
2. For target channel c: subtract other channels → enhances dominance.
3. Otsu threshold.
4. Morph open + close.
5. Contours extracted from cleaned mask.

### Corner / Arrangement Logic
Corner arrangement utilities (e.g., identifying relative positions) live around mid/lower sections of [raspi_functions.py](raspi_functions.py) (see functions near detection of yellow + ordering, e.g., `identify_corner_arrangement`).

## Raspberry Pi Deployment
Use [raspi_functions.py](raspi_functions.py) for minimal dependency deployment (aggregation of all logic). Copy only needed image + config assets beside the script.
## Usage
1. Create a configuration file (`config.json`) in the project root. Use the interactive tool:
  ```
  python tools/create_config.py
  ```
  (Place the generated `config.json` next to the scripts.)

2. HSV-only detection (simple pipeline):
  ```
  python cbd_v3.py
  ```
  This uses HSV ranges from `config.json`.

3. Channel isolation + HSV (combined method):
  ```
  python color_detector_refactored.py
  ```
  Provide an image path; the script loads `config.json` automatically.

Notes:
- Regenerate `config.json` if lighting changes.
- Keep `config.json` in the same directory as the scripts.

## Color Range Tuning Tips
- Start with evenly lit images.
- Avoid over-tight ranges; include slight illumination variance.
- Re-generate after lighting/environment changes.
- Yellow often needs separate (smaller) morphology kernel (`MORPH_KERNEL_YELLOW` in [raspi_functions.py](raspi_functions.py)).

## Extending
1. Add new color: update generation tool to prompt for it.
2. Add algorithm variant: create `preprocess_vX` function and mirror usage pattern.
3. Unify duplicate `load_config` functions if maintenance becomes difficult.

## Dependencies
Install (typical):
```
pip install opencv-python numpy
```
(If running on Raspberry Pi, use `opencv-contrib-python`)

## Performance Notes
- Reduce resolution before processing for speed.
- Tune morphology kernel sizes (`MORPH_KERNEL_COLOR`, `MORPH_KERNEL_YELLOW`) per camera.
- Cache configs; avoid reloading each frame.

## Troubleshooting
- Blank masks: verify HSV range order and lighting.
- Wrong color dominance in V3: ensure channel subtraction not saturating (inspect intermediate single-channel result).
- No contours: relax min area thresholds in functions like [`detect_contours_and_centroids`](raspi_functions.py).

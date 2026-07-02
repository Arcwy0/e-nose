"""Vision endpoints — Florence-2 grounding, decoupled from training.

Two routes:

* ``POST /predict/object`` — single label, ``<OPEN_VOCABULARY_DETECTION>``.
  This restores the endpoint the existing client at
  ``enose/client/api.py:detect_object`` has always POSTed to. Until this module
  was added the only Florence-2 entry point was ``/training_pipeline``, which
  couples grounding with online-learning — fine for the chemist UI, wrong for
  a robotics policy that needs to inspect the grounding score *before*
  deciding whether to record/commit.

* ``POST /predict/scene`` — multi-label grounding in a single Florence-2 call
  via ``<CAPTION_TO_PHRASE_GROUNDING>``. Used by the robot's SEARCH state to
  score a list of candidate labels at the current waypoint without paying N
  separate inference round-trips.
"""

from __future__ import annotations

import json
import os
import tempfile
import traceback
from typing import Any, Dict, List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from enose.vision.florence import process_image_with_vlm

from .. import state

router = APIRouter()


def _require_vlm() -> None:
    if state.vlm_model is None or state.vlm_processor is None:
        raise HTTPException(status_code=503, detail="VLM not loaded")


async def _save_upload(image: UploadFile) -> str:
    suffix = os.path.splitext(image.filename or "image.png")[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(await image.read())
        return f.name


def _normalize_label(s: str) -> str:
    # Florence-2 sometimes echoes back "A rose" instead of "rose"; drop the article.
    s = s.strip()
    low = s.lower()
    for prefix in ("a ", "an ", "the "):
        if low.startswith(prefix):
            return s[len(prefix):].strip()
    return s


@router.post("/predict/object")
async def predict_object(
    image: UploadFile = File(...),
    text: str = Form(...),
) -> Dict[str, Any]:
    """Florence-2 open-vocabulary detection for a single label.

    Body (multipart/form-data):
        image: PNG/JPEG file
        text:  the full prompt, e.g. ``"Find rose"`` (matches what the
               existing client sends).

    Response:
        ``{"prompt": ..., "task": "<OPEN_VOCABULARY_DETECTION>", "result": <florence post_process output>}``
    """
    _require_vlm()
    temp: str | None = None
    try:
        temp = await _save_upload(image)
        result = process_image_with_vlm(
            state.vlm_model,
            state.vlm_processor,
            temp,
            "<OPEN_VOCABULARY_DETECTION>",
            text,
        )
        return {
            "prompt": text,
            "task": "<OPEN_VOCABULARY_DETECTION>",
            "result": result,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[predict/object] error: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp and os.path.exists(temp):
            os.unlink(temp)


@router.post("/predict/scene")
async def predict_scene(
    image: UploadFile = File(...),
    labels: str = Form(...),  # JSON-encoded list of strings
) -> Dict[str, Any]:
    """Ground a list of candidate labels against one image in a single Florence-2 call.

    Body (multipart/form-data):
        image:  PNG/JPEG file
        labels: JSON list of label strings,
                e.g. ``'["rose", "coffee cup", "book"]'``.

    Response::

        {
          "labels": [...],
          "detections": [
              {"label": "rose", "bbox": [x1,y1,x2,y2], "score": float, "area_fraction": float},
              ...
          ],
          "image_size": [W, H],
          "task": "<CAPTION_TO_PHRASE_GROUNDING>",
          "caption": "A rose. A coffee cup. A book."
        }

    Notes:
        Florence-2's phrase-grounding head does not emit a native confidence
        score. ``score`` here is a bbox-area-fraction proxy (clamped to [0,1])
        which is monotone in "how close / how prominent" the object is in the
        frame — good enough for the robot's ``τ_grounding``/``τ_commit``
        thresholding. A label without any returned bbox is omitted from
        ``detections``; the policy treats that as score = 0.
    """
    # Validate inputs first so clients get a 400 even in NO_VLM smoke mode.
    try:
        label_list = json.loads(labels)
        if not isinstance(label_list, list) or not all(isinstance(l, str) for l in label_list):
            raise ValueError("labels must be a JSON list of strings")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid labels: {e}")
    if not label_list:
        raise HTTPException(status_code=400, detail="labels list cannot be empty")
    _require_vlm()

    temp: str | None = None
    try:
        temp = await _save_upload(image)

        # CAPTION_TO_PHRASE_GROUNDING takes a caption containing the phrases
        # to ground. Using "A <label>. A <label>. ..." gives Florence-2 the
        # phrase-like shape it grounds best on.
        caption = " ".join(f"A {l}." for l in label_list)

        result = process_image_with_vlm(
            state.vlm_model,
            state.vlm_processor,
            temp,
            "<CAPTION_TO_PHRASE_GROUNDING>",
            caption,
        )

        inner = result.get("<CAPTION_TO_PHRASE_GROUNDING>", {}) or {}
        bboxes: List[List[float]] = inner.get("bboxes", []) or []
        out_labels: List[str] = inner.get("labels", []) or []

        with Image.open(temp) as img:
            W, H = img.size
        img_area = float(W * H) if W and H else 1.0

        detections: List[Dict[str, Any]] = []
        for bbox, lbl in zip(bboxes, out_labels):
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            x1, y1, x2, y2 = (float(c) for c in bbox)
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_frac = (area / img_area) if img_area > 0 else 0.0
            detections.append({
                "label": _normalize_label(lbl),
                "bbox": [x1, y1, x2, y2],
                "score": float(min(1.0, area_frac)),
                "area_fraction": float(area_frac),
            })

        return {
            "labels": label_list,
            "detections": detections,
            "image_size": [int(W), int(H)],
            "task": "<CAPTION_TO_PHRASE_GROUNDING>",
            "caption": caption,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[predict/scene] error: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp and os.path.exists(temp):
            os.unlink(temp)

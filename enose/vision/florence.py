"""Florence-2 loader + single-image inference.

Keeps the loader separate from the model cache so routes can call `process_image_with_vlm`
with an already-loaded (model, processor) pair held in server state.
"""

from __future__ import annotations

import os
import traceback
from typing import Any, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from enose.config import VLM_MODEL_PATH


def load_florence_model(device: str, model_path: str = VLM_MODEL_PATH) -> Tuple[Any, Any]:
    """Load Florence-2 from local snapshot. Returns (model, processor). Raises on failure."""
    print(f"Loading Florence-2 from {model_path} …")
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"  dtype={dtype}")

    try:
        # `torch_dtype` is the correct kwarg for transformers < 5.0 (and the
        # version we pin for Florence-2 compatibility is 4.44.2). transformers 5
        # renamed it to `dtype`, which is why this call previously failed with
        # `Florence2ForConditionalGeneration.__init__() got an unexpected
        # keyword argument 'dtype'` — the model constructor received a stray
        # kwarg because from_pretrained didn't consume it.
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
                attn_implementation="eager",
            )
            .eval()
            .to(device)
        )
        # `use_fast=False` forces the Python (slow) tokenizer, built from
        # vocab.json + merges.txt. The fast path reads tokenizer.json through
        # the Rust `tokenizers` library, which in our pinned version
        # (tokenizers 0.19, required by transformers 4.44.2) can't parse
        # tokenizer.json files serialised by newer versions — the failure
        # mode is the cryptic "data did not match any variant of untagged
        # enum ModelWrapper" error. The slow tokenizer is slightly slower at
        # startup but functionally identical for Florence-2 inference.
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        actual_dtype = next(model.parameters()).dtype
        print(f"✓ Florence-2 loaded ({actual_dtype})")
        return model, processor
    except Exception as e:
        print(f"✗ Florence-2 load failed: {type(e).__name__}: {e}")
        if os.path.isdir(model_path):
            files = sorted(os.listdir(model_path))[:10]
            print(f"  Directory has {len(os.listdir(model_path))} entries, first 10: {files}")
        else:
            print(f"  Path not a directory: {model_path}")
        raise


def process_image_with_vlm(
    vlm_model,
    vlm_processor,
    image_path: str,
    prompt: str,
    additional_text: Optional[str] = None,
) -> dict:
    """Run Florence-2 on one image. Returns parsed detection result."""
    if vlm_model is None or vlm_processor is None:
        raise RuntimeError("Florence-2 not loaded")

    full_prompt = f"{prompt} {additional_text}" if additional_text else prompt

    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    device = next(vlm_model.parameters()).device
    model_dtype = next(vlm_model.parameters()).dtype

    inputs = vlm_processor(text=full_prompt, images=image, return_tensors="pt")
    processed: dict = {}
    for k, v in inputs.items():
        # pixel_values are float tensors → follow model dtype; token ids stay int
        processed[k] = v.to(device=device, dtype=model_dtype) if k == "pixel_values" else v.to(device=device)

    try:
        with torch.no_grad():
            generated_ids = vlm_model.generate(
                input_ids=processed["input_ids"],
                pixel_values=processed["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )
        generated_text = vlm_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        return vlm_processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
    except Exception as e:
        print(f"VLM processing error: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise

"""Critical contracts: preserve generated coordinates and display API detections."""

import json
import re
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from transformers import BartTokenizer

from enose.client.webcam import _parse_detection
from enose.vision.florence import process_image_with_vlm


class FlorenceDetectionTests(unittest.TestCase):
    def test_slow_bart_coordinates_reach_postprocessor_without_inserted_spaces(self):
        task = "<OPEN_VOCABULARY_DETECTION>"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vocabulary = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "bottle": 4}
            (root / "vocab.json").write_text(json.dumps(vocabulary), encoding="utf-8")
            (root / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
            tokenizer = BartTokenizer(str(root / "vocab.json"), str(root / "merges.txt"))
            coordinates = [f"<loc_{value}>" for value in (100, 200, 300, 400)]
            tokenizer.add_special_tokens({"additional_special_tokens": coordinates})
            ids = torch.tensor([[0, 4, *tokenizer.convert_tokens_to_ids(coordinates), 2]])
            # Establish that this fixture exercises the actual slow-tokenizer bug.
            self.assertIn("<loc_100> <loc_200>", tokenizer.batch_decode(ids)[0])

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

                def generate(self, **kwargs):
                    return ids

            class Processor:
                batch_decode = tokenizer.batch_decode

                def __call__(self, **kwargs):
                    return {"input_ids": torch.tensor([[0]]), "pixel_values": torch.zeros(1, 3, 2, 2)}

                def post_process_generation(self, text, task, image_size):
                    # Florence's real parser uses this contiguous-coordinate contract.
                    boxes = re.findall(r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>", text)
                    return {task: {"bboxes": [[int(value) for value in box] for box in boxes]}}

            image = root / "image.png"
            Image.new("RGB", (2, 2)).save(image)
            result = process_image_with_vlm(Model(), Processor(), str(image), task, "bottle")
            self.assertEqual(result[task]["bboxes"], [[100, 200, 300, 400]])

    def test_api_envelope_and_legacy_payloads_expose_the_same_detections(self):
        task = "<OPEN_VOCABULARY_DETECTION>"
        payload = {task: {"bboxes": [[10, 20, 30, 40]], "bboxes_labels": ["bottle"]}}
        envelope = {"prompt": "Find bottle", "task": task, "result": payload}
        for value in (payload, envelope, json.dumps(payload), json.dumps(envelope), repr(payload), repr(envelope)):
            with self.subTest(value=value):
                self.assertEqual(_parse_detection(value), payload)


if __name__ == "__main__":
    unittest.main()

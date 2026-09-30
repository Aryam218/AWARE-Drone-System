import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from .wrapper_base import WrapperBase


class GroundingDinoHuggingfaceWrapper(WrapperBase):
    def __init__(
        self,
        cfg_in=None,
        use_sam_hq=True,
        box_threshold=0.15,
        text_threshold=0.15,
        args="",
    ):
        model_id = "IDEA-Research/grounding-dino-tiny"

        # Use GPU if available, otherwise CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Load Grounding DINO processor and model
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(self.device)

    def run_inference(
        self,
        image_pil: Image,
        prompt: str,
        print_results=True,
        mark_results=True,
    ):
        # Clean the text prompt
        prompt = prompt.strip()

        # Grounding DINO prompts should end with a period
        if not prompt.endswith("."):
            prompt += "."

        # Let the processor handle the PIL image directly
        inputs = self.processor(
            images=image_pil,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        # Run Grounding DINO
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert raw model output into bounding boxes and scores
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image_pil.size[::-1]],
        )[0]

        masks = None

        scores = results["scores"].detach().cpu().numpy()
        boxes_filt = results["boxes"].detach().cpu().numpy()

        # Debug output so we can see whether Grounding DINO
        # is actually detecting anything
        if print_results:
            print(
                f"[GroundingDINO] prompt={prompt!r} | "
                f"detections={len(boxes_filt)} | "
                f"scores={scores}"
            )

        return image_pil, masks, boxes_filt, scores
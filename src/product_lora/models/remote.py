from __future__ import annotations

import base64
import os
import time

import requests

from product_lora.config import RemoteSettings
from product_lora.models.base import GenerationRequest, GenerationResult


class BflGenerator:
    """Minimal client for BFL's submit-and-poll image API."""

    def __init__(self, settings: RemoteSettings) -> None:
        self.settings = settings

    def generate(self, request: GenerationRequest) -> GenerationResult:
        key = os.environ.get(self.settings.api_key_env)
        if not key:
            raise RuntimeError(f"Missing API key environment variable: {self.settings.api_key_env}")
        payload: dict[str, object] = {
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "seed": request.seed,
        }
        if request.reference_images:
            payload["input_image"] = base64.b64encode(
                request.reference_images[0].read_bytes()
            ).decode("ascii")
        started = time.perf_counter()
        response = requests.post(
            self.settings.endpoint,
            headers={"x-key": key, "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        submitted = response.json()
        polling_url = submitted.get("polling_url")
        if not polling_url:
            raise RuntimeError(f"BFL response omitted polling_url: {submitted}")
        deadline = time.monotonic() + self.settings.timeout_seconds
        while time.monotonic() < deadline:
            polled = requests.get(polling_url, headers={"x-key": key}, timeout=60)
            polled.raise_for_status()
            result = polled.json()
            if result.get("status") == "Ready":
                image_url = result["result"]["sample"]
                image = requests.get(image_url, timeout=120)
                image.raise_for_status()
                request.output.parent.mkdir(parents=True, exist_ok=True)
                request.output.write_bytes(image.content)
                output_mp = request.width * request.height / 1_000_000
                input_mp = sum(_image_megapixels(p) for p in request.reference_images)
                cost = (
                    max(1.0, output_mp) * self.settings.cost_per_output_megapixel
                    + input_mp * self.settings.cost_per_input_megapixel
                )
                return GenerationResult(request.output, time.perf_counter() - started, cost)
            if result.get("status") in {"Error", "Failed"}:
                raise RuntimeError(f"BFL generation failed: {result}")
            time.sleep(self.settings.poll_interval_seconds)
        raise TimeoutError("Timed out waiting for BFL generation")


def _image_megapixels(path) -> float:
    from PIL import Image

    with Image.open(path) as image:
        return max(1.0, image.width * image.height / 1_000_000)

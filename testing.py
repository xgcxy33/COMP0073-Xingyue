# All code logic was fully understood, and AI was only used in an assistive role. All methods, analysis, and logical reasoning were developed independently by the author. This usage falls under Category 2 of UCL’s Generative AI policy: assistive use only.
# tests/test_pipeline_min.py
import os
import time
import re
import glob
import requests
import pytest
from PIL import Image

from app_gradio import multimodal_understanding, get_final_output

TGI_URL = os.getenv("TGI_URL", "http://127.0.0.1:8090")
# Prefer TEST_IMAGE env var; otherwise automatically pick one from repo
DEFAULT_GLOB = os.getenv("TEST_IMAGE_GLOB", "images/**/*.png")

def _tgi_alive(url: str = TGI_URL, timeout: float = 1.0) -> bool:
    for u in [url, url.rstrip("/") + "/health", url.rstrip("/") + "/docs", url.rstrip("/") + "/"]:
        try:
            r = requests.get(u, timeout=timeout)
            if r.status_code < 500:
                return True
        except Exception:
            pass
    return False

@pytest.fixture(scope="session")
def test_image():
    # 1) Explicitly provided
    img_path = os.getenv("TEST_IMAGE")
    if img_path and os.path.exists(img_path):
        return Image.open(img_path)

    # 2) Auto-pick first image from repo
    candidates = sorted(glob.glob(DEFAULT_GLOB, recursive=True))
    for p in candidates:
        if p.lower().endswith((".png", ".jpg", ".jpeg")) and os.path.getsize(p) > 0:
            return Image.open(p)

    pytest.skip("No test image found. Set $TEST_IMAGE or place one in images/ directory.")

@pytest.mark.end2end
@pytest.mark.skipif(not _tgi_alive(), reason="TGI not reachable, skipping end-to-end test.")
def test_e2e_minimal(test_image):
    """Minimal but essential end-to-end test: image → description → probe advice with latency/content checks."""
    q = "Describe this segmented ultrasound image of the brachial plexus."
    seed, top_p, temperature = 42, 0.9, 0.1

    t0 = time.time()
    desc = multimodal_understanding(test_image, q, seed, top_p, temperature)
    assert isinstance(desc, str) and desc.strip(), "Should return a non-empty description."
    assert any(k in desc.lower() for k in ["nerve", "artery", "muscle", "plexus", "scalene"]), \
        "Description should mention common anatomy terms."

    prompt_template = (
        "You are a clinical expert in ultrasound-guided regional anesthesia.\n"
        "Provide numbered probe adjustment steps based on:\n<input>\n"
    )
    final = ""
    for tok in get_final_output(desc, prompt_template, temperature=0.1, top_p=0.9, max_new_tokens=256):
        final += tok

    # Shape & operational content checks
    assert final.strip(), "Should return non-empty advice text."
    assert re.search(r"(?:^|\n)\s*1[.)]\s+", final), "Advice should be presented as numbered steps (e.g. '1.' or '1)')."
    assert any(w in final.lower() for w in ["probe", "angle", "depth", "gain", "in-plane", "out-of-plane"]), \
        "Advice should include probe/optimization terms."

    total = time.time() - t0
    assert total < 30, f"End-to-end latency too high: {total:.1f}s (threshold 30s)."



"""
Smoke test: iterate all device-map combinations and run text / image / audio inference.

Usage:
    uv run python test_smoke.py
"""

import os
import sys
import json
import time
import itertools
import traceback

sys.path.insert(0, os.path.dirname(__file__))

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "gemma-4-E2B-it-ONNX")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "engine_config.json")
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "uploads", "ws_smoke_image.png")
AUDIO_PATH = os.path.join(os.path.dirname(__file__), "uploads", "ws_smoke_audio.wav")

MAX_TOKENS = 5  # tiny generation just to confirm the path works


def write_config(text_dev: str, image_dev: str, audio_dev: str) -> dict:
    cfg = {
        "text": text_dev,
        "image": image_dev,
        "audio": audio_dev,
        "prefill_chunk_size": 256,
        "gpu_mem_gb": 4,
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    return cfg


def make_text_message():
    return [{"role": "user", "content": [{"type": "text", "text": "Say hello in one word."}]}]


def make_image_message():
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image briefly."},
                {"type": "image", "image": IMAGE_PATH},
            ],
        }
    ]


def make_audio_message():
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this audio."},
                {"type": "audio", "audio": AUDIO_PATH},
            ],
        }
    ]


def run_inference(engine, messages, label: str) -> tuple[bool, str]:
    """Run a tiny generation and return (success, info)."""
    try:
        tokens = []
        for tok in engine.generate_stream(messages, max_new_tokens=MAX_TOKENS, max_context_tokens=4096):
            tokens.append(tok)
        output = "".join(tokens).strip()
        return True, f"{label}: OK ({len(tokens)} tokens) → {output[:60]!r}"
    except Exception as e:
        return False, f"{label}: FAIL → {e}"


def main():
    from scripts.gemma_engine import GemmaEngine

    devices = ["cpu", "gpu"]
    combos = list(itertools.product(devices, repeat=3))  # text, image, audio

    results = []
    print(f"\n{'='*70}")
    print(f" SMOKE TEST: {len(combos)} device-map combinations × 3 modalities")
    print(f" Image: {IMAGE_PATH}")
    print(f" Audio: {AUDIO_PATH}")
    print(f"{'='*70}\n")

    for i, (text_dev, image_dev, audio_dev) in enumerate(combos, 1):
        cfg = write_config(text_dev, image_dev, audio_dev)
        tag = f"[{i}/{len(combos)}] text={text_dev}, image={image_dev}, audio={audio_dev}"
        print(f"\n{'─'*70}")
        print(f" {tag}")
        print(f"{'─'*70}")

        try:
            engine = GemmaEngine(MODEL_DIR, device="cuda", enable_vision=True, enable_audio=True)
        except Exception as e:
            msg = f"  ENGINE INIT FAILED: {e}"
            print(msg)
            results.append((cfg, "INIT_FAIL", str(e)))
            continue

        combo_results = []

        # Text-only
        ok, info = run_inference(engine, make_text_message(), "TEXT")
        print(f"  {info}")
        combo_results.append(("text", ok, info))

        # Image
        if os.path.isfile(IMAGE_PATH):
            ok, info = run_inference(engine, make_image_message(), "IMAGE")
            print(f"  {info}")
            combo_results.append(("image", ok, info))
        else:
            print(f"  IMAGE: SKIPPED (no file at {IMAGE_PATH})")
            combo_results.append(("image", None, "skipped"))

        # Audio
        if os.path.isfile(AUDIO_PATH):
            ok, info = run_inference(engine, make_audio_message(), "AUDIO")
            print(f"  {info}")
            combo_results.append(("audio", ok, info))
        else:
            print(f"  AUDIO: SKIPPED (no file at {AUDIO_PATH})")
            combo_results.append(("audio", None, "skipped"))

        results.append((cfg, combo_results))

        # Free sessions
        del engine

    # Summary
    print(f"\n\n{'='*70}")
    print(" SUMMARY")
    print(f"{'='*70}")
    total_pass = 0
    total_fail = 0
    for entry in results:
        cfg = entry[0]
        if entry[1] == "INIT_FAIL":
            print(f"  {cfg} → ENGINE INIT FAILED: {entry[2]}")
            total_fail += 1
            continue
        combo_results = entry[1]
        for modality, ok, info in combo_results:
            if ok is True:
                total_pass += 1
            elif ok is False:
                total_fail += 1
        status = "✓ ALL PASS" if all(r[1] for r in combo_results if r[1] is not None) else "✗ HAS FAILURES"
        print(f"  {cfg} → {status}")

    print(f"\n  Total: {total_pass} passed, {total_fail} failed")
    print(f"{'='*70}\n")

    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()

"""Isolated audio/tracking jobs. No generated output is production game state."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import time
import wave

from .core import atomic_json, read_json, finite
from .local_models import checkpoint, nvidia_uuid


def wave_data(path):
    import numpy as np

    with wave.open(str(path), "rb") as source:
        if (
            source.getnchannels() != 1
            or source.getsampwidth() != 2
            or source.getframerate() != 16000
        ):
            raise ValueError("Audio jobs require mono 16 kHz PCM16")
        return (
            np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(
                "float32"
            )
            / 32768
        )


def bind_audio_alignment(chunk, timestamps):
    """Keep model timings as evidence; expose only words inside their audio crop."""
    candidates, words, rejected = [], [], []
    for item in timestamps:
        start = chunk["source_start"] + float(item["start_time"])
        end = chunk["source_start"] + float(item["end_time"])
        candidate = {
            "text": item["text"],
            "start": start if finite(start) else None,
            "end": end if finite(end) else None,
        }
        candidates.append(candidate)
        if not (
            finite(start)
            and finite(end)
            and chunk["source_start"] <= start <= end <= chunk["source_end"]
        ):
            rejected.append(
                {
                    **candidate,
                    "reason": "alignment outside the source audio span or invalid",
                }
            )
        else:
            words.append(candidate)
    chunk["alignment_candidates"] = candidates
    chunk["words"] = words
    chunk["alignment"] = {
        "status": "partial" if rejected else "aligned",
        "rejected_words": rejected,
        "accepted_words": len(words),
        "candidate_words": len(candidates),
    }


def audio_job(job):
    import numpy as np
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps
    from transformers import (
        AutoProcessor,
        AutoModelForMultimodalLM,
        AutoModelForTokenClassification,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Audio experiment requires one allocated CUDA GPU")
    model_root, asr_pin = checkpoint(job["cache"], "asr")
    aligner_root, aligner_pin = checkpoint(job["cache"], "aligner")
    vad = load_silero_vad(onnx=False)
    chunks = []
    for segment in job["segments"]:
        audio = wave_data(segment["path"])
        speech = get_speech_timestamps(
            torch.from_numpy(audio.copy()),
            vad,
            sampling_rate=16000,
            return_seconds=False,
            max_speech_duration_s=30,
        )
        for index, interval in enumerate(speech):
            lower, upper = max(0, interval["start"] - 1600), min(
                len(audio), interval["end"] + 1600
            )
            path = Path(job["output"]).parent / (
                segment["id"] + f"-speech-{index:03d}.wav"
            )
            with wave.open(str(path), "wb") as target:
                target.setnchannels(1)
                target.setsampwidth(2)
                target.setframerate(16000)
                target.writeframes(
                    (np.clip(audio[lower:upper], -1, 1) * 32767).astype("<i2").tobytes()
                )
            chunks.append(
                {
                    "path": str(path),
                    "source_start": segment["start"] + lower / 16000,
                    "source_end": segment["start"] + upper / 16000,
                    "segment_id": segment["id"],
                }
            )
    if not chunks:
        return {
            "experimental": True,
            "evidence_modality": "audio",
            "chunks": [],
            "source_segments": job["segments"],
            "asr_model": asr_pin,
            "aligner_model": aligner_pin,
            "vad_version": importlib.metadata.version("silero-vad"),
            "silence_is_not_transcribed": True,
        }
    processor = AutoProcessor.from_pretrained(model_root, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_root, dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    model.eval()
    for chunk in chunks:
        inputs = processor.apply_transcription_request(
            audio=chunk["path"], language="English"
        ).to("cuda", torch.bfloat16)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        generated = output[:, inputs["input_ids"].shape[1] :]
        if generated.shape[1] >= 512:
            chunk.update(status="unresolved", text=None, reason="ASR truncation")
        else:
            chunk.update(
                status="transcribed",
                text=processor.decode(generated, return_format="transcription_only")[0],
            )
        atomic_json(
            Path(job["output"]).with_name("transcription-progress.json"),
            {"chunks": chunks, "experimental": True},
        )
    del model, processor, inputs, output
    gc.collect()
    torch.cuda.empty_cache()
    processor = AutoProcessor.from_pretrained(aligner_root, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(
        aligner_root, dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    model.eval()
    for chunk in chunks:
        if not chunk.get("text"):
            chunk["words"] = []
            continue
        inputs, words = processor.prepare_forced_aligner_inputs(
            audio=chunk["path"], transcript=chunk["text"], language="English"
        )
        inputs = inputs.to("cuda", torch.bfloat16)
        with torch.inference_mode():
            output = model(**inputs)
        timestamps = processor.decode_forced_alignment(
            logits=output.logits,
            input_ids=inputs["input_ids"],
            word_lists=words,
            timestamp_token_id=model.config.timestamp_token_id,
        )[0]
        bind_audio_alignment(chunk, timestamps)
        atomic_json(
            Path(job["output"]).with_name("alignment-progress.json"),
            {"chunks": chunks, "experimental": True},
        )
    return {
        "experimental": True,
        "evidence_modality": "audio",
        "chunks": chunks,
        "source_segments": job["segments"],
        "asr_model": asr_pin,
        "aligner_model": aligner_pin,
        "vad_version": importlib.metadata.version("silero-vad"),
        "silence_is_not_transcribed": True,
        "alignment_rejected_words": sum(
            len(c.get("alignment", {}).get("rejected_words", [])) for c in chunks
        ),
    }


def tracking_job(job):
    import numpy as np
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    root, pin = checkpoint(job["cache"], "tracking")
    if len(job["clips"]) != 5 or any(
        not 1 <= len(c["prompts"]) <= 3 for c in job["clips"]
    ):
        raise ValueError(
            "Tracking requires five labeled clips and one to three prompted objects per clip"
        )
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=str(root / "sam3.1_multiplex.pt"),
        max_num_objects=3,
        use_fa3=False,
        compile=False,
        warm_up=False,
    )
    results = []
    for clip in job["clips"]:
        response = predictor.handle_request(
            request={"type": "start_session", "resource_path": clip["frames_dir"]}
        )
        session = response["session_id"]
        try:
            for prompt in clip["prompts"]:
                # Explicit points in native-frame coordinates; no text-based auto-detection.
                point = torch.tensor(
                    [
                        [
                            prompt["point"][0] / clip["width"],
                            prompt["point"][1] / clip["height"],
                        ]
                    ],
                    dtype=torch.float32,
                )
                predictor.handle_request(
                    request={
                        "type": "add_prompt",
                        "session_id": session,
                        "frame_index": prompt.get("frame_index", 0),
                        "points": point,
                        "point_labels": torch.tensor([1], dtype=torch.int32),
                        "obj_id": prompt["object_id"],
                    }
                )
            frames = []
            for response in predictor.handle_stream_request(
                request={"type": "propagate_in_video", "session_id": session}
            ):
                output = response["outputs"]
                masks, ids = output["out_binary_masks"], output["out_obj_ids"]
                paths = []
                for mask, object_id in zip(masks, ids):
                    if hasattr(mask, "detach"):
                        mask = mask.detach().cpu().numpy()
                    path = (
                        Path(job["output"]).parent
                        / f'{clip["id"]}-{response["frame_index"]:04d}-{int(object_id)}.png'
                    )
                    Image.fromarray(
                        np.asarray(mask).squeeze().astype("uint8") * 255
                    ).save(path)
                    paths.append({"object_id": int(object_id), "mask_path": str(path)})
                frames.append({"frame_index": response["frame_index"], "masks": paths})
            results.append({"clip_id": clip["id"], "frames": frames})
        finally:
            predictor.handle_request(
                request={"type": "close_session", "session_id": session}
            )
    return {
        "experimental": True,
        "model": pin,
        "coordinate_system": "screen_pixels",
        "semantic_events": False,
        "clips": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = read_json(args.job)
    started = time.monotonic()
    import subprocess
    import threading
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU for an experiment")
    properties = torch.cuda.get_device_properties(0)
    gpu_id = nvidia_uuid(getattr(properties, "uuid", ""))
    if not gpu_id:
        import os

        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    stop = threading.Event()
    resource = {"peak_memory_bytes": 0, "errors": []}

    def monitor():
        while not stop.is_set():
            try:
                probe = subprocess.run(
                    [
                        "nvidia-smi",
                        "-i",
                        gpu_id,
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                resource["peak_memory_bytes"] = max(
                    resource["peak_memory_bytes"], int(probe.stdout.strip()) * 1024**2
                )
                if resource["peak_memory_bytes"] >= properties.total_memory * 0.9:
                    resource["errors"].append("Exceeded 90% GPU capacity")
            except Exception as exc:
                resource["errors"].append(str(exc))
            stop.wait(0.25)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    if job["kind"] not in {"audio", "tracking"}:
        raise ValueError("Unknown isolated experiment kind")
    try:
        result = audio_job(job) if job["kind"] == "audio" else tracking_job(job)
    finally:
        stop.set()
        thread.join(timeout=12)
    result["resource_measurement"] = resource
    result["total_memory_bytes"] = properties.total_memory
    result["resource_passed"] = (
        not resource["errors"] and resource["peak_memory_bytes"] > 0
    )
    result["wall_seconds"] = time.monotonic() - started
    import torch

    result["torch_peak_memory_bytes"] = torch.cuda.max_memory_allocated()
    result["gpu"] = torch.cuda.get_device_name(0)
    result["runtime"] = {
        name: importlib.metadata.version(name) for name in ("torch", "transformers")
    }
    atomic_json(Path(job["output"]), result)


if __name__ == "__main__":
    main()

# Copyright (c) Microsoft. All rights reserved.
"""FastAPI backend for VibeVoice TTS inference.

This module provides a REST API for text-to-speech using VibeVoice model.
Run with: python tts_server.py
"""

import os
import argparse
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import io
import numpy as np

# VibeVoice imports - streaming version
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference

# VibeVoice imports - non-streaming version (for full-text synthesis)
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    global voices_dir, model_mode

    print(f"Loading models in mode: {model_mode}")

    # Load models based on mode
    if model_mode in ("streaming", "both"):
        try:
            load_model()
            print("Streaming model loaded")
        except Exception as e:
            print(f"Warning: Streaming model not loaded: {e}")
            print("Use --model-path to specify a local model path")

    if model_mode in ("non-streaming", "both"):
        try:
            load_model_non_streaming()
            print("Non-streaming model loaded")
        except Exception as e:
            print(f"Warning: Non-streaming model not loaded: {e}")

    yield
    # Shutdown logic if needed


app = FastAPI(
    title="VibeVoice TTS API",
    description="Text-to-speech API using VibeVoice model",
    version="1.0.0",
    lifespan=lifespan
)

# CORS - allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global model and processor instances (streaming)
model = None
processor = None
device = None

# Global model and processor instances (non-streaming)
model_ns = None
processor_ns = None

# Global model loading mode: streaming, non-streaming, or both
model_mode = "both"


def get_device():
    """Determine the best device to use."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_model(model_path: str = "microsoft/VibeVoice-1.5b"):
    """Load the VibeVoice model and processor."""
    global model, processor, device

    if model is not None:
        return model, processor

    device = get_device()
    print(f"Using device: {device}")

    dtype = torch.float32
    if device == "cuda":
        try:
            import flash_attn
            dtype = torch.bfloat16
        except ImportError:
            dtype = torch.float16

    print(f"Loading model from {model_path}...")
    processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()

    print("Model loaded successfully!")
    return model, processor


def load_model_non_streaming(model_path: str = "microsoft/VibeVoice-1.5b"):
    """Load the VibeVoice model and processor for non-streaming inference."""
    global model_ns, processor_ns

    if model_ns is not None:
        return model_ns, processor_ns

    device = get_device()
    print(f"Non-streaming: Using device: {device}")

    dtype = torch.float32
    if device == "cuda":
        try:
            import flash_attn
            dtype = torch.bfloat16
        except ImportError:
            dtype = torch.float16

    print(f"Loading non-streaming model from {model_path}...")
    processor_ns = VibeVoiceProcessor.from_pretrained(model_path)
    model_ns = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
    )
    model_ns.eval()

    print("Non-streaming model loaded successfully!")
    return model_ns, processor_ns


# Global config for voices directory
voices_dir = "voices"


def find_voice_file(speaker_name: str) -> Optional[str]:
    """Find a voice file for the given speaker name."""
    if not os.path.exists(voices_dir):
        return None

    # Exact match
    for f in os.listdir(voices_dir):
        if f.endswith(".wav") and speaker_name in f:
            return os.path.join(voices_dir, f)

    # Fuzzy match - first word match
    for f in os.listdir(voices_dir):
        if f.endswith(".wav"):
            name_parts = f.split("_")[0].replace(".wav", "")
            if speaker_name[:2] == name_parts[:2]:  # Partial match
                return os.path.join(voices_dir, f)

    # Return first available voice
    for f in os.listdir(voices_dir):
        if f.endswith(".wav"):
            return os.path.join(voices_dir, f)

    return None


# Request/Response models
class TTSRequest(BaseModel):
    text: str
    voice_key: str = "zh-WHTest_man"
    cfg_scale: float = 1.5
    speed: float = 1.0


class VoicesResponse(BaseModel):
    voices: List[str]
    default_voice: str


app = FastAPI(
    title="VibeVoice TTS API",
    description="Text-to-speech API using VibeVoice model",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {"message": "VibeVoice TTS API", "status": "ready"}


@app.get("/config")
async def get_config() -> VoicesResponse:
    """Get available voices."""
    voices_dir = "voices"
    voices = []

    if os.path.exists(voices_dir):
        for f in os.listdir(voices_dir):
            if f.endswith(".wav"):
                # Extract speaker name from filename
                name = f.replace(".wav", "").replace("_", " ")
                voices.append(name)

    # Default preset voices if no local files
    if not voices:
        voices = [
            "zh-WHFemale",
            "zh-WHMale",
            "zh-ZHTest_man",
            "zh-ZHTest_woman",
        ]

    return VoicesResponse(voices=voices, default_voice=voices[0] if voices else "")


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech."""
    global model, processor

    if model is None or processor is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not loaded: {str(e)}")

    try:
        # Find voice file
        voice_file = find_voice_file(request.voice_key)
        if voice_file is None:
            # Use default voice samples from model
            voice_samples = None
        else:
            voice_samples = [voice_file]
            print(f"Using voice file: {voice_file}")

        # Prepare inputs
        inputs = processor(
            text=[request.text],
            voice_samples=voice_samples,
            padding=True,
            return_tensors="pt",
        )

        if device == "cuda":
            inputs = inputs.to(device="cuda")
        elif device == "mps":
            inputs = inputs.to(device="mps")

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                cfg_scale=request.cfg_scale,
                tokenizer=processor.tokenizer,
                is_prefill=not True,  # Enable prefill
            )

        # Get audio data
        speech = outputs.speech_outputs[0]

        # Convert to WAV format
        audio_data = speech.cpu().float().numpy()

        # Normalize if needed
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        # Convert to 16-bit PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)

        # Write to bytes buffer
        buffer = io.BytesIO()
        import wave
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes = 16 bits
            wav_file.setframerate(24000)  # VibeVoice sample rate
            wav_file.writeframes(audio_int16.tobytes())

        buffer.seek(0)

        return StreamingResponse(
            iter([buffer.read()]),
            media_type="audio/wav"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def clean_text_for_tts(text: str) -> str:
    """Clean and preprocess text for TTS.

    VibeVoice processor may have issues with certain characters.
    """
    import re
    # Replace problematic punctuation with periods
    text = text.replace('、', '。')
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Ensure text ends with proper punctuation for better prosody
    if text and text[-1] not in '。！？.!?':
        text = text + '。'
    return text.strip()


@app.post("/tts/full")
async def text_to_speech_full(request: TTSRequest):
    """Convert full text to speech (non-streaming mode).

    This endpoint processes the entire text at once, suitable for
    shorter texts that benefit from full-context understanding.
    """
    global model_ns, processor_ns

    # Log request details
    print(f"[TTS/FULL] Request received:")
    print(f"  - voice_key: {request.voice_key}")
    print(f"  - cfg_scale: {request.cfg_scale}")
    print(f"  - original_text: {request.text[:100]}..." if len(request.text) > 100 else f"  - original_text: {request.text}")

    if model_ns is None or processor_ns is None:
        try:
            load_model_non_streaming()
            print(f"[TTS/FULL] Model loaded successfully")
        except Exception as e:
            print(f"[TTS/FULL] Model load failed: {e}")
            raise HTTPException(status_code=503, detail=f"Model not loaded: {str(e)}")

    try:
        # Clean text before processing
        clean_text = clean_text_for_tts(request.text)
        print(f"[TTS/FULL] Cleaned text: {clean_text[:100]}..." if len(clean_text) > 100 else f"[TTS/FULL] Cleaned text: {clean_text}")

        # Find voice file
        voice_file = find_voice_file(request.voice_key)
        print(f"[TTS/FULL] voice_key={request.voice_key}, voice_file={voice_file}")

        if voice_file is None:
            # For non-streaming processor, voice_samples cannot be None
            # Use model's built-in default voice if available
            # Try to use the voice_key as a built-in voice preset
            built_in_voices = ["zh-WHFemale", "zh-WHMale", "zh-ZHTest_man", "zh-ZHTest_woman"]
            if request.voice_key in built_in_voices:
                voice_samples = None  # None means use model's default voice
                print(f"[TTS/FULL] Using model's default voice (built-in: {request.voice_key})")
            else:
                # Map player names to built-in voices
                voice_mapping = {
                    "小红": "zh-WHFemale",
                    "小明": "zh-WHMale",
                    "张三": "zh-WHMale",
                    "李四": "zh-WHMale",
                    "你": "zh-WHFemale",
                }
                mapped_voice = voice_mapping.get(request.voice_key, built_in_voices[0])
                voice_samples = None
                print(f"[TTS/FULL] Using mapped built-in voice: {mapped_voice} (for player: {request.voice_key})")
        else:
            voice_samples = [voice_file]
            print(f"[TTS/FULL] Using custom voice file: {voice_file}")

        # Prepare inputs - non-streaming processor takes full text at once
        print(f"[TTS/FULL] Calling processor_ns()...")
        inputs = processor_ns(
            text=[clean_text],
            voice_samples=voice_samples,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        print(f"[TTS/FULL] Processor call succeeded, input_ids shape: {inputs.get('input_ids', 'N/A').shape if hasattr(inputs, 'get') else 'N/A'}")

        device = get_device()
        if device == "cuda":
            inputs = inputs.to(device="cuda")
        elif device == "mps":
            inputs = inputs.to(device="mps")

        # Generate - full text at once
        print(f"[TTS/FULL] Calling model.generate()...")
        with torch.no_grad():
            outputs = model_ns.generate(
                **inputs,
                cfg_scale=request.cfg_scale,
                tokenizer=processor_ns.tokenizer,
                generation_config={'do_sample': False},
                verbose=True,
                is_prefill=not True,  # Enable prefill for better quality
            )
        print(f"[TTS/FULL] Generation succeeded")

        # Get audio data
        speech = outputs.speech_outputs[0]
        print(f"[TTS/FULL] Speech shape: {speech.shape}")

        # Convert to WAV format
        audio_data = speech.cpu().float().numpy()

        # Normalize if needed
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        # Convert to 16-bit PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)

        # Write to bytes buffer
        buffer = io.BytesIO()
        import wave
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes = 16 bits
            wav_file.setframerate(24000)  # VibeVoice sample rate
            wav_file.writeframes(audio_int16.tobytes())

        buffer.seek(0)

        return StreamingResponse(
            iter([buffer.read()]),
            media_type="audio/wav"
        )

    except Exception as e:
        import traceback
        print(f"[TTS/FULL] ERROR: {type(e).__name__}: {e}")
        print(f"[TTS/FULL] Traceback:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


@app.post("/tts/batch")
async def text_to_speech_batch(texts: List[str], voice_key: str = "zh-WHTest_man", cfg_scale: float = 1.5):
    """Convert multiple texts to speech."""
    global model, processor

    if model is None or processor is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not loaded: {str(e)}")

    results = []

    for text in texts:
        try:
            # Find voice file
            voice_file = find_voice_file(voice_key)
            voice_samples = [voice_file] if voice_file else None

            # Prepare inputs
            inputs = processor(
                text=[text],
                voice_samples=voice_samples,
                padding=True,
                return_tensors="pt",
            )

            if device == "cuda":
                inputs = inputs.to(device="cuda")
            elif device == "mps":
                inputs = inputs.to(device="mps")

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    cfg_scale=cfg_scale,
                    tokenizer=processor.tokenizer,
                    is_prefill=not True,
                )

            # Get audio data
            speech = outputs.speech_outputs[0]
            audio_data = speech.cpu().float().numpy()

            # Normalize
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))

            audio_int16 = (audio_data * 32767).astype(np.int16)

            # Save to temp file and read
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                import wave
                with wave.open(f.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(24000)
                    wav_file.writeframes(audio_int16.tobytes())

                with open(f.name, 'rb') as f:
                    audio_bytes = f.read()

                os.unlink(f.name)

            results.append({"text": text[:20] + "..." if len(text) > 20 else text, "audio_size": len(audio_bytes)})

        except Exception as e:
            results.append({"text": text[:20] + "..." if len(text) > 20 else text, "error": str(e)})

    return {"results": results}


def main():
    global voices_dir, model_mode

    parser = argparse.ArgumentParser(description="VibeVoice TTS Server")
    parser.add_argument("--model-path", type=str, default="microsoft/VibeVoice-1.5b",
                        help="Model path or HF repo ID (可从 HuggingFace 下载或本地路径)")
    parser.add_argument("--model-mode", type=str, default="both",
                        choices=["streaming", "non-streaming", "both"],
                        help="Which model to load: streaming, non-streaming, or both (default: both)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind (云服务器必须用 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8001,
                        help="Port to bind (确保云安全组已开放此端口)")
    parser.add_argument("--voices-dir", type=str, default="voices",
                        help="Directory containing voice sample WAV files (可选)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes (GPU服务器建议设为1)")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token for gated models (可选)")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload for development")

    args = parser.parse_args()

    # Set global config
    voices_dir = args.voices_dir
    model_mode = args.model_mode

    # Set HF token if provided
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    print(f"=" * 50)
    print(f"🚀 VibeVoice TTS Server")
    print(f"=" * 50)
    print(f"📡 Server: http://{args.host}:{args.port}")
    print(f"📁 Model: {args.model_path}")
    print(f"🔧 Mode: {model_mode} (streaming: /tts, non-streaming: /tts/full)")
    print(f"🎵 Voices: {voices_dir if os.path.exists(voices_dir) else 'using model defaults'}")
    print(f"=" * 50)

    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1
    )


if __name__ == "__main__":
    main()

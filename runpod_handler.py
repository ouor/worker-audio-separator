"""
RunPod Serverless Handler for Audio Separation

This module provides a serverless handler for audio separation using
various ML models. It handles audio ingestion, separation, and output
encoding to Opus format.
"""

import os
import base64
import requests
import logging
import time
import subprocess
import json
import uuid6
import traceback
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

import torch
import boto3
from botocore.config import Config
import runpod
from runpod.serverless.utils import rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from audio_separator.separator import Separator
from runpod_schemas import INPUT_SCHEMA
from runpod_exceptions import (
    AudioSeparatorError,
    AudioValidationError,
    StorageError,
    InferenceError,
    NetworkError,
)

# =============================================================================
# Constants
# =============================================================================

S3_BASE_PREFIX = "worker-audio-separator"
SAMPLE_RATE = 48000
OPUS_BITRATE = "128k"
PRESIGNED_URL_EXPIRY_SECONDS = 3600 * 24  # 24 hours
DOWNLOAD_CHUNK_SIZE = 8192
DOWNLOAD_TIMEOUT_SECONDS = 300
FFPROBE_TIMEOUT_SECONDS = 30
MODEL_DIR = "/models"
WORK_DIR_BASE = "/tmp"

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("AudioWorker")


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class JobContext:
    """Context object for tracking job state throughout the pipeline."""

    request_id: str
    job_id: str
    tag: Optional[str]
    work_dir: str
    input_dir: str
    output_dir: str
    s3_prefix: str
    start_time: float
    timing_ms: Dict[str, int] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    input_meta: Optional[Dict[str, Any]] = None
    gpu_peak_mb: int = 0


@dataclass
class AudioMetadata:
    """Metadata extracted from an audio file."""

    size_bytes: int
    duration_ms: int
    sample_rate: int
    channels: int
    codec: str
    format: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "size_bytes": self.size_bytes,
            "duration_ms": self.duration_ms,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "codec": self.codec,
            "format": self.format,
        }


# =============================================================================
# Main Worker Class
# =============================================================================


class AudioWorker:
    """
    Main worker class for audio separation jobs.

    Handles the complete pipeline of:
    1. Ingesting audio from base64 or presigned URL
    2. Encoding to Opus format
    3. Running ML model separation
    4. Encoding outputs and uploading to R2 storage
    """

    def __init__(self) -> None:
        """Initialize the AudioWorker with R2 storage and GPU info."""
        self.r2_bucket = os.environ.get("R2_BUCKET_NAME")
        self.r2_configured = self._check_r2_configuration()
        self.s3_client = self._init_s3_client()
        self.current_model_name: Optional[str] = None
        self.separator: Optional[Separator] = None
        self.gpu_info = self._get_gpu_info()

    def _check_r2_configuration(self) -> bool:
        """Check if all required R2 environment variables are set."""
        required_vars = [
            "R2_ENDPOINT_URL",
            "R2_ACCESS_KEY_ID",
            "R2_SECRET_ACCESS_KEY",
            "R2_BUCKET_NAME",
        ]
        return all(os.environ.get(var) for var in required_vars)

    def _init_s3_client(self) -> Optional[boto3.client]:
        """
        Initialize the S3/R2 client for file uploads.

        Returns:
            boto3 S3 client if configured, None otherwise.
        """
        if not self.r2_configured:
            logger.warning(
                "R2 storage is not fully configured. Presigned URLs will be unavailable."
            )
            return None

        return boto3.client(
            service_name="s3",
            endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
            aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"),
            region_name="auto",
            config=Config(signature_version="s3v4"),
        )

    def _get_gpu_info(self) -> Dict[str, Any]:
        """
        Gather GPU information for diagnostics.

        Returns:
            Dictionary containing GPU availability, name, memory, and versions.
        """
        info = {
            "enabled": torch.cuda.is_available(),
            "name": "None",
            "total_memory_mb": 0,
            "driver": os.environ.get("DRIVER_VERSION", "unknown"),
            "cuda": os.environ.get("CUDA_VERSION", "unknown"),
        }

        if info["enabled"]:
            try:
                device = 0
                info["name"] = torch.cuda.get_device_name(device)
                _, total_mem = torch.cuda.mem_get_info(device)
                info["total_memory_mb"] = total_mem // (1024 * 1024)
            except Exception as e:
                logger.warning(f"Could not retrieve full GPU info: {e}")

        return info

    # =========================================================================
    # Audio Processing Methods
    # =========================================================================

    def get_audio_metadata(self, path: str) -> AudioMetadata:
        """
        Extract metadata from an audio file using ffprobe.

        Args:
            path: Path to the audio file.

        Returns:
            AudioMetadata object with file information.

        Raises:
            AudioValidationError: If ffprobe fails or no audio stream found.
        """
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_format",
            "-show_streams",
            "-of", "json",
            path,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=FFPROBE_TIMEOUT_SECONDS,
            )
            if result.returncode != 0:
                raise AudioValidationError(f"FFprobe failed: {result.stderr}")
            data = json.loads(result.stdout)
        except subprocess.TimeoutExpired:
            raise AudioValidationError("FFprobe timed out.")

        format_info = data.get("format", {})
        streams = data.get("streams", [])
        audio_stream = next(
            (s for s in streams if s.get("codec_type") == "audio"),
            None,
        )

        if not audio_stream:
            raise AudioValidationError("No valid audio stream found.")

        return AudioMetadata(
            size_bytes=int(format_info.get("size", 0)),
            duration_ms=int(float(format_info.get("duration", 0.0)) * 1000),
            sample_rate=int(audio_stream.get("sample_rate", 0)),
            channels=int(audio_stream.get("channels", 0)),
            codec=audio_stream.get("codec_name", "unknown"),
            format=format_info.get("format_name", "unknown").split(",")[0],
        )

    def convert_to_opus(self, input_path: str, output_path: str) -> None:
        """
        Convert an audio file to Opus format.

        Args:
            input_path: Path to source audio file.
            output_path: Path for output Opus file.

        Raises:
            AudioValidationError: If input audio is invalid.
            AudioSeparatorError: If FFmpeg conversion fails.
        """
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "warning",
            "-stats",
            "-i", input_path,
            "-c:a", "libopus",
            "-b:a", OPUS_BITRATE,
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            stderr_lower = result.stderr.lower()
            invalid_indicators = ["invalid data", "invalid argument", "error opening input"]

            if any(indicator in stderr_lower for indicator in invalid_indicators):
                raise AudioValidationError(f"Invalid audio data: {result.stderr}")
            raise AudioSeparatorError(f"FFmpeg failed: {result.stderr}", "FFMPEG_ERROR")

    # =========================================================================
    # Storage Methods
    # =========================================================================

    def upload_file(self, local_path: str, s3_key: str) -> Optional[str]:
        """
        Upload a file to R2 storage and generate a presigned URL.

        Args:
            local_path: Path to the local file.
            s3_key: Key for the file in R2 storage.

        Returns:
            Presigned URL for the uploaded file, or None if R2 not configured.

        Raises:
            StorageError: If upload fails.
        """
        if not self.s3_client:
            return None

        try:
            self.s3_client.upload_file(local_path, self.r2_bucket, s3_key)
            return self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.r2_bucket, "Key": s3_key},
                ExpiresIn=PRESIGNED_URL_EXPIRY_SECONDS,
            )
        except Exception as e:
            raise StorageError(f"Upload failed: {str(e)}")

    # =========================================================================
    # Job Processing Pipeline
    # =========================================================================

    def process_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for processing a separation job.

        Args:
            job: RunPod job dictionary containing input and id.

        Returns:
            Response dictionary with results or error information.
        """
        start_time_total = time.perf_counter()
        request_id = str(uuid6.uuid7())
        job_id = job["id"]

        # Validate input
        validated = validate(job["input"], INPUT_SCHEMA)
        if "errors" in validated:
            return self._build_error_response(
                request_id,
                f"Validation failed: {validated['errors']}",
                "VALIDATION_ERROR",
            )

        input_data = validated["validated_input"]
        tag = input_data.get("tag")

        # Setup working directories
        work_dir = os.path.join(WORK_DIR_BASE, job_id)
        input_dir = os.path.join(work_dir, "input")
        output_dir = os.path.join(work_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Initialize job context
        ctx = JobContext(
            request_id=request_id,
            job_id=job_id,
            tag=tag,
            work_dir=work_dir,
            input_dir=input_dir,
            output_dir=output_dir,
            s3_prefix=f"{S3_BASE_PREFIX}/requests-{request_id}",
            start_time=start_time_total,
        )

        try:
            # Pipeline: Ingest -> Separate -> Egress
            input_opus_path = self._ingest(input_data, ctx)
            raw_outputs = self._separate(input_data, input_opus_path, ctx)
            response = self._egress(input_data, raw_outputs, ctx)
            return response

        except AudioSeparatorError as e:
            logger.error(f"Business logic error: {e.error_code} - {e.message}")
            return self._build_error_response(
                request_id, e.message, e.error_code, tag, ctx.timing_ms
            )
        except Exception as e:
            logger.error(f"Unexpected crash: {str(e)}\n{traceback.format_exc()}")
            return self._build_error_response(
                request_id,
                f"Unexpected error: {str(e)}",
                "UNEXPECTED_ERROR",
                tag,
                ctx.timing_ms,
                refresh=True,
            )
        finally:
            rp_cleanup.clean([work_dir])

    def _ingest(self, input_data: Dict[str, Any], ctx: JobContext) -> str:
        """
        Ingest audio from source and encode to Opus.

        Downloads or decodes input audio, converts to Opus format,
        and uploads to R2 storage.

        Args:
            input_data: Validated input configuration.
            ctx: Job context for tracking state.

        Returns:
            Path to the encoded Opus input file.

        Raises:
            AudioValidationError: If base64 decoding fails.
            NetworkError: If URL download fails.
        """
        t_start = time.perf_counter()
        input_config = input_data.get("input", {})
        input_orig = os.path.join(ctx.input_dir, "original")
        input_opus = os.path.join(ctx.input_dir, "input.opus.ogg")

        # Fetch input audio
        if input_config.get("mode") == "base64":
            self._decode_base64_input(input_config["base64"], input_orig)
        else:
            self._download_url_input(input_config["presigned_url"], input_orig)

        # Convert to Opus and get metadata
        self.convert_to_opus(input_orig, input_opus)
        meta = self.get_audio_metadata(input_opus)

        # Upload to R2
        s3_key = f"{ctx.s3_prefix}/input.opus.ogg"
        self.upload_file(input_opus, s3_key)

        # Update context
        ctx.input_meta = {"s3_key": s3_key, **meta.to_dict()}
        ctx.timing_ms["encode_in"] = int((time.perf_counter() - t_start) * 1000)

        return input_opus

    def _decode_base64_input(self, base64_data: str, output_path: str) -> None:
        """
        Decode base64 audio data and save to file.

        Args:
            base64_data: Base64 encoded audio data.
            output_path: Path to save decoded file.

        Raises:
            AudioValidationError: If decoding fails.
        """
        try:
            decoded = base64.b64decode(base64_data)
            with open(output_path, "wb") as f:
                f.write(decoded)
        except Exception as e:
            raise AudioValidationError(f"Invalid base64 data: {str(e)}")

    def _download_url_input(self, url: str, output_path: str) -> None:
        """
        Download audio from URL using streaming to handle large files.

        Args:
            url: Presigned URL to download from.
            output_path: Path to save downloaded file.

        Raises:
            NetworkError: If download fails.
        """
        try:
            with requests.get(
                url,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        f.write(chunk)
        except Exception as e:
            raise NetworkError(f"Source fetch failed: {str(e)}")

    def _separate(
        self,
        input_data: Dict[str, Any],
        input_path: str,
        ctx: JobContext,
    ) -> List[str]:
        """
        Run ML model separation on the input audio.

        Uses warm start optimization to reuse loaded models.

        Args:
            input_data: Validated input configuration.
            input_path: Path to the input Opus file.
            ctx: Job context for tracking state.

        Returns:
            List of output filenames from separation.

        Raises:
            InferenceError: If model loading or inference fails.
        """
        t_start = time.perf_counter()
        model_name = input_data.get("model")

        # Warm start: reuse separator if same model
        if self.current_model_name != model_name or self.separator is None:
            self._load_model(model_name, ctx.output_dir)
        else:
            self.separator.output_dir = ctx.output_dir

        # Run separation with GPU memory tracking
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            raw_outputs = self.separator.separate(input_path)

            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated()
                ctx.gpu_peak_mb = peak_mem // (1024 * 1024)
                # Clean up GPU memory after separation
                torch.cuda.empty_cache()

            ctx.timing_ms["separate"] = int((time.perf_counter() - t_start) * 1000)
            return raw_outputs

        except Exception as e:
            # Clean up GPU memory on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise InferenceError(f"Inference failed: {str(e)}")

    def _load_model(self, model_name: str, output_dir: str) -> None:
        """
        Load a separation model.

        Args:
            model_name: Name of the model file to load.
            output_dir: Output directory for separated files.

        Raises:
            InferenceError: If model loading fails.
        """
        logger.info(f"Loading/Switching model to: {model_name}")

        self.separator = Separator(
            output_dir=output_dir,
            output_format="WAV",
            sample_rate=SAMPLE_RATE,
            model_file_dir=MODEL_DIR,
        )

        try:
            self.separator.load_model(model_name)
            self.current_model_name = model_name
        except Exception as e:
            raise InferenceError(f"Model load failed: {str(e)}")

    def _egress(
        self,
        input_data: Dict[str, Any],
        raw_outputs: List[str],
        ctx: JobContext,
    ) -> Dict[str, Any]:
        """
        Process separation outputs and upload to storage.

        Filters stems based on request, encodes to Opus, uploads to R2,
        and builds the response.

        Args:
            input_data: Validated input configuration.
            raw_outputs: List of output filenames from separation.
            ctx: Job context for tracking state.

        Returns:
            Complete response dictionary.
        """
        t_start = time.perf_counter()

        # Parse requested stems filter
        requested = input_data.get("stems", [])
        requested_lower = [s.lower() for s in requested] if requested else []

        items = {}
        out_meta_items = {}

        for path in raw_outputs:
            # Extract stem name from pattern "*(Name).wav"
            match = re.search(r"\(([^)]+)\)", os.path.basename(path))
            stem_name_raw = match.group(1) if match else "Other"
            stem_name = stem_name_raw.lower()

            # Skip if not in requested stems
            if requested_lower and stem_name not in requested_lower:
                continue

            # Process this stem
            stem_result = self._process_stem(path, stem_name, ctx)
            items[stem_name] = stem_result["item"]
            out_meta_items[stem_name] = stem_result["meta"]

        ctx.timing_ms["encode_out"] = int((time.perf_counter() - t_start) * 1000)
        total_time = int((time.perf_counter() - ctx.start_time) * 1000)
        now = datetime.now(timezone.utc).isoformat()

        # Save and upload metadata
        self._save_metadata(input_data, ctx, out_meta_items, now)

        return {
            "request_id": ctx.request_id,
            "tag": ctx.tag,
            "result": "OK",
            "message": "Success",
            "returned_at": now,
            "outputs": {
                "codec": "opus",
                "format": "ogg",
                "items": items,
            },
            "timing_ms": {**ctx.timing_ms, "total": total_time},
            "error": None,
        }

    def _process_stem(
        self,
        source_path: str,
        stem_name: str,
        ctx: JobContext,
    ) -> Dict[str, Any]:
        """
        Process a single stem: encode to Opus and upload.

        Args:
            source_path: Path to the source WAV file.
            stem_name: Name of the stem (lowercase).
            ctx: Job context.

        Returns:
            Dictionary with 'item' (for response) and 'meta' (for storage).
        """
        # Build paths
        full_source_path = os.path.join(ctx.output_dir, os.path.basename(source_path))
        target_name = f"{stem_name}.opus.ogg"
        target_path = os.path.join(ctx.output_dir, target_name)

        # Convert and get metadata
        self.convert_to_opus(full_source_path, target_path)
        meta = self.get_audio_metadata(target_path)

        # Upload to R2
        s3_key = f"{ctx.s3_prefix}/{target_name}"
        presigned_url = self.upload_file(target_path, s3_key)

        return {
            "item": {"presigned_url": presigned_url, **meta.to_dict()},
            "meta": {"s3_key": s3_key, **meta.to_dict()},
        }

    def _save_metadata(
        self,
        input_data: Dict[str, Any],
        ctx: JobContext,
        out_meta_items: Dict[str, Any],
        timestamp: str,
    ) -> None:
        """
        Save and upload job metadata to R2.

        Args:
            input_data: Original input configuration.
            ctx: Job context.
            out_meta_items: Metadata for output items.
            timestamp: ISO format timestamp.
        """
        full_meta = {
            "request_id": ctx.request_id,
            "tag": ctx.tag,
            "created_at": timestamp,
            "updated_at": timestamp,
            "input": ctx.input_meta,
            "inference": {
                "model_name": input_data.get("model"),
                "gpu": {
                    **self.gpu_info,
                    "peak_memory_mb": ctx.gpu_peak_mb,
                },
                "timing_ms": ctx.timing_ms,
            },
            "output": {
                "codec": "opus",
                "format": "ogg",
                "items": out_meta_items,
            },
        }

        meta_path = os.path.join(ctx.work_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(full_meta, f, indent=2)

        self.upload_file(meta_path, f"{ctx.s3_prefix}/metadata.json")

    # =========================================================================
    # Response Builders
    # =========================================================================

    def _build_error_response(
        self,
        req_id: str,
        msg: str,
        code: str,
        tag: Optional[str] = None,
        timing: Optional[Dict[str, int]] = None,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Build a standardized error response.

        Args:
            req_id: Request ID for tracking.
            msg: Error message.
            code: Error code.
            tag: Optional tag from request.
            timing: Optional timing information.
            refresh: Whether to signal worker refresh.

        Returns:
            Error response dictionary.
        """
        return {
            "request_id": req_id,
            "tag": tag,
            "result": "ERROR",
            "message": msg,
            "returned_at": datetime.now(timezone.utc).isoformat(),
            "outputs": None,
            "timing_ms": {"total": timing.get("total", 0) if timing else 0},
            "error": code,
            "refresh_worker": refresh,
        }


# =============================================================================
# Module Entry Point
# =============================================================================

# Global worker instance to enable warm starts (model reuse)
worker = AudioWorker()


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod handler function."""
    return worker.process_job(job)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

"""
RunPod Serverless Handler for Audio Separation

Pipeline:
    download_file -> validate_input_file -> load_model -> separate_audio -> encode_stems -> upload_results

Response structure:
    { status, code, message, files[] }

Metadata structure (saved to R2):
    { request, response, debug: { trace, resource, performance, logs } }
"""

import os
import re
import base64
import json
import logging
import subprocess
import time
import traceback

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, unquote

import boto3
import requests as http_requests
import torch
import uuid6
from botocore.config import Config

import runpod
from runpod.serverless.utils import rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from audio_separator.separator import Separator
from runpod_schemas import INPUT_SCHEMA
from runpod_exceptions import (
    AudioSeparatorError,
    AudioValidationError,
    InferenceError,
    NetworkError,
    StorageError,
)


# =============================================================================
# Constants
# =============================================================================

R2_BASE_PREFIX = os.environ.get("R2_BASE_PREFIX", "worker-audio-separator")
PRESIGNED_URL_EXPIRY_SECONDS = 3600 * 24  # 24 hours
DOWNLOAD_CHUNK_SIZE = 8192
DOWNLOAD_TIMEOUT_SECONDS = 300
FFMPEG_TIMEOUT_SECONDS = 600
FFPROBE_TIMEOUT_SECONDS = 30
MODEL_DIR = os.environ.get("AUDIO_SEPARATOR_MODEL_DIR", "/models")
WORK_DIR_BASE = "/tmp"

# Format -> (container extension, ffmpeg codec, uses bitrate?)
FORMAT_PROFILES = {
    "OPUS": ("opus.ogg", "libopus", True),
    "MP3":  ("mp3",      "libmp3lame", True),
    "AAC":  ("aac",      "aac",       True),
    "M4A":  ("m4a",      "aac",       True),
    "OGG":  ("ogg",      "libvorbis", False),
    "FLAC": ("flac",     "flac",      False),
    "WAV":  ("wav",      "pcm_s16le", False),
}


# =============================================================================
# Log Capture Handler
# =============================================================================


class ListLogHandler(logging.Handler):
    """Captures log records into a list for embedding in metadata."""

    def __init__(self):
        super().__init__()
        self.records: List[str] = []

    def emit(self, record: logging.LogRecord):
        self.records.append(self.format(record))

    def flush_records(self) -> List[str]:
        out = list(self.records)
        self.records.clear()
        return out


# =============================================================================
# Global Logger
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("AudioWorker")

# Attach list-capture handler so we can store logs in metadata
_log_capture = ListLogHandler()
_log_capture.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_log_capture)


# =============================================================================
# Helper: file extension from format
# =============================================================================


def _ext_for_format(fmt: str) -> str:
    profile = FORMAT_PROFILES.get(fmt.upper())
    return profile[0] if profile else "opus.ogg"


# =============================================================================
# Main Worker Class
# =============================================================================


class AudioWorker:
    """
    Serverless worker for audio separation.

    Lifecycle per request:
        1. download_file    - fetch from URL / decode Data URL
        2. validate_input   - ffprobe + convert to WAV for inference
        3. load_model       - warm-start aware model loading
        4. separate_audio   - ML inference (always on WAV)
        5. encode_stems     - encode WAV stems to requested format
        6. upload_results   - upload to R2, generate presigned URLs
    """

    def __init__(self) -> None:
        self.r2_bucket = os.environ.get("R2_BUCKET_NAME")
        self.r2_configured = self._check_r2_config()
        self.s3_client = self._init_s3_client()
        self.current_model_name: Optional[str] = None
        self.separator: Optional[Separator] = None
        self.gpu_info = self._get_gpu_info()

    # ----- R2 / S3 initialisation -----

    def _check_r2_config(self) -> bool:
        required = ["R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID",
                     "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME"]
        return all(os.environ.get(v) for v in required)

    def _init_s3_client(self) -> Optional[boto3.client]:
        if not self.r2_configured:
            logger.warning("R2 not configured - presigned URLs unavailable.")
            return None
        return boto3.client(
            service_name="s3",
            endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
            aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"),
            region_name="auto",
            config=Config(signature_version="s3v4"),
        )

    # ----- GPU helpers -----

    def _get_gpu_info(self) -> Dict[str, Any]:
        info = {
            "available": torch.cuda.is_available(),
            "name": "N/A",
            "total_vram_mb": 0,
        }
        if info["available"]:
            try:
                info["name"] = torch.cuda.get_device_name(0)
                _, total = torch.cuda.mem_get_info(0)
                info["total_vram_mb"] = total // (1024 * 1024)
            except Exception as exc:
                logger.warning(f"GPU info unavailable: {exc}")
        return info

    @staticmethod
    def _get_gpu_power_draw() -> float:
        """Return current GPU power draw in watts via nvidia-smi, or 0."""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0:
                return float(out.stdout.strip().split("\n")[0])
        except Exception:
            pass
        return 0.0

    # ----- Low-level audio helpers -----

    @staticmethod
    def _ffprobe(path: str) -> Dict[str, Any]:
        """Run ffprobe and return parsed JSON."""
        cmd = ["ffprobe", "-v", "error", "-show_format",
               "-show_streams", "-of", "json", path]
        logger.info(f"Running ffprobe: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=FFPROBE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            raise AudioValidationError(
                f"ffprobe timed out after {FFPROBE_TIMEOUT_SECONDS}s - file may be corrupt")
        
        if result.returncode != 0:
            logger.error(f"ffprobe failed: {result.stderr}")
            raise AudioValidationError(f"ffprobe failed: {result.stderr}")
        
        if result.stderr:
            logger.debug(f"ffprobe output: {result.stderr}")
        return json.loads(result.stdout)

    def get_audio_metadata(self, path: str) -> Dict[str, Any]:
        """Return a flat dict of audio metadata for a file."""
        data = self._ffprobe(path)
        fmt = data.get("format", {})
        audio = next(
            (s for s in data.get("streams", [])
             if s.get("codec_type") == "audio"), None
        )
        if not audio:
            raise AudioValidationError("No audio stream found.")
        return {
            "size": int(fmt.get("size", 0)),
            "duration_sec": round(float(fmt.get("duration", 0.0)), 2),
            "sample_rate": int(audio.get("sample_rate", 0)),
            "channel": int(audio.get("channels", 0)),
            "codec": audio.get("codec_name", "unknown"),
            "format": fmt.get("format_name", "unknown").split(",")[0],
        }

    def _ffmpeg_convert(
        self, src: str, dst: str,
        out_format: str, bitrate: str, sample_rate: int,
    ) -> None:
        """Convert *src* to *dst* using the given format profile."""
        profile = FORMAT_PROFILES.get(out_format.upper())
        if profile is None:
            raise AudioValidationError(f"Unsupported format: {out_format}")

        _, codec, uses_bitrate = profile
        cmd = ["ffmpeg", "-y", "-loglevel", "warning",
               "-i", src, "-ar", str(sample_rate), "-c:a", codec]
        if uses_bitrate:
            cmd.extend(["-b:a", bitrate])
        cmd.append(dst)

        logger.info(f"Running FFmpeg: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=FFMPEG_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg timed out: {' '.join(cmd)}")
            raise AudioValidationError(
                f"FFmpeg timed out after {FFMPEG_TIMEOUT_SECONDS}s")

        if result.stderr:
            logger.info(f"FFmpeg output: {result.stderr.strip()}")

        if result.returncode != 0:
            stderr = result.stderr.lower()
            if any(k in stderr for k in ("invalid data", "invalid argument",
                                         "error opening input")):
                raise AudioValidationError(f"Invalid audio: {result.stderr}")
            raise AudioSeparatorError(f"FFmpeg failed: {result.stderr}",
                                      "FFMPEG_ERROR")

    def _convert_to_wav(self, src: str, dst: str) -> None:
        """Convert any source to 44100 Hz WAV for model inference."""
        cmd = ["ffmpeg", "-y", "-loglevel", "warning",
               "-i", src, "-ar", "44100", "-c:a", "pcm_s16le", dst]
        logger.info(f"Running FFmpeg (WAV conversion): {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=FFMPEG_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg (WAV) timed out: {' '.join(cmd)}")
            raise AudioValidationError(
                f"WAV conversion timed out after {FFMPEG_TIMEOUT_SECONDS}s")

        if result.stderr:
            logger.info(f"FFmpeg (WAV) output: {result.stderr.strip()}")

        if result.returncode != 0:
            raise AudioValidationError(
                f"WAV conversion failed: {result.stderr}")

    # ----- R2 upload -----

    def _upload_file(self, local: str, key: str) -> Optional[str]:
        if not self.s3_client:
            logger.warning(f"R2 skip (not configured): {key}")
            return None
        try:
            self.s3_client.upload_file(local, self.r2_bucket, key)
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.r2_bucket, "Key": key},
                ExpiresIn=PRESIGNED_URL_EXPIRY_SECONDS,
            )
            logger.info(f"Uploaded to R2: {key}")
            return url
        except Exception as exc:
            raise StorageError(f"Upload failed ({key}): {exc}") from exc

    # =====================================================================
    # Pipeline stages
    # =====================================================================

    # Stage 1: download_file
    def _download_file(self, url: str, dest: str) -> None:
        """Download from HTTP(S) or decode Data URL to *dest*."""
        if url.lower().startswith("data:audio/"):
            if ";base64," not in url:
                raise AudioValidationError("Data URL must use base64 encoding")
            _, b64 = url.split(";base64,", 1)
            try:
                data = base64.b64decode(b64)
            except Exception as exc:
                raise AudioValidationError(
                    f"Invalid base64 in Data URL: {exc}") from exc
            with open(dest, "wb") as f:
                f.write(data)
            logger.info(f"Decoded Data URL ({len(data)} bytes)")
        else:
            try:
                with http_requests.get(url, timeout=DOWNLOAD_TIMEOUT_SECONDS,
                                       stream=True) as resp:
                    resp.raise_for_status()
                    total = 0
                    with open(dest, "wb") as f:
                        for chunk in resp.iter_content(DOWNLOAD_CHUNK_SIZE):
                            f.write(chunk)
                            total += len(chunk)
                logger.info(f"Downloaded {url[:80]}... ({total} bytes)")
            except http_requests.exceptions.Timeout:
                raise NetworkError(
                    f"Download timed out after {DOWNLOAD_TIMEOUT_SECONDS}s")
            except http_requests.exceptions.HTTPError as exc:
                raise NetworkError(
                    f"HTTP {exc.response.status_code}: {exc}") from exc
            except Exception as exc:
                raise NetworkError(f"Download failed: {exc}") from exc

    # Stage 2: validate & convert to WAV
    def _validate_input_file(self, raw_path: str, wav_path: str) -> None:
        """Validate the raw download and create a WAV for inference."""
        # Quick probe to ensure it's a real audio file
        self._ffprobe(raw_path)
        self._convert_to_wav(raw_path, wav_path)
        logger.info("Input validated and converted to WAV")

    # Stage 3: load model (warm-start aware)
    def _load_model(self, model_name: str, output_dir: str) -> float:
        """
        Load model, returning elapsed seconds.

        Skips loading if the same model is already active (warm start).
        Returns 0.0 on warm hit.
        """
        if self.current_model_name == model_name and self.separator is not None:
            self.separator.output_dir = output_dir
            logger.info(f"Warm start - reusing model {model_name}")
            return 0.0

        logger.info(f"Loading model: {model_name}")
        t = time.perf_counter()
        self.separator = Separator(
            output_dir=output_dir,
            output_format="WAV",
            sample_rate=44100,
            model_file_dir=MODEL_DIR,
        )
        try:
            self.separator.load_model(model_name)
            self.current_model_name = model_name
        except Exception as exc:
            logger.error(f"Model load failed: {exc}", exc_info=True)
            raise InferenceError(f"Model load failed: {exc}") from exc
        elapsed = time.perf_counter() - t
        logger.info(f"Model loaded in {elapsed:.2f}s")
        return elapsed

    # Stage 4: separate
    def _separate_audio(self, wav_path: str, custom_output_names: Optional[Dict[str, str]] = None) -> List[str]:
        """
        Run separation inference.

        Returns list of output file paths (WAV stems).
        """
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            logger.info(f"Separation starting: {wav_path}")
            outputs = self.separator.separate(wav_path, custom_output_names=custom_output_names)
            logger.info(f"Separation complete: {len(outputs)} stems produced")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return outputs
        except Exception as exc:
            logger.error(f"Separation failed: {exc}", exc_info=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise InferenceError(f"Inference failed: {exc}") from exc

    # Stage 5: encode stems
    def _encode_stem(
        self, wav_path: str, stem_name: str,
        out_dir: str, fmt: str, bitrate: str, sr: int,
    ) -> Dict[str, Any]:
        """Encode a single WAV stem to the target format and return metadata."""
        ext = _ext_for_format(fmt)
        out_name = f"{stem_name}.{ext}"
        out_path = os.path.join(out_dir, out_name)
        self._ffmpeg_convert(wav_path, out_path, fmt, bitrate, sr)
        meta = self.get_audio_metadata(out_path)
        return {
            "stem": stem_name.capitalize(),
            "wav_path": wav_path,
            "wav_name": f"{stem_name}.wav",
            "local_path": out_path,
            "file_name": out_name,
            "bitrate": bitrate,
            **meta,
        }

    # Stage 6: upload WAV original + encoded stem, build file entry
    def _upload_stem(
        self, file_info: Dict[str, Any], s3_prefix: str,
    ) -> Dict[str, Any]:
        """Upload WAV original and encoded stem to R2. Return response dict + keys."""
        # Upload WAV original
        wav_key = f"{s3_prefix}/{file_info['wav_name']}"
        self._upload_file(file_info["wav_path"], wav_key)

        # Upload encoded stem
        convert_key = f"{s3_prefix}/{file_info['file_name']}"
        url = self._upload_file(file_info["local_path"], convert_key) or ""

        return {
            "stem": file_info["stem"],
            "size": file_info["size"],
            "format": file_info["format"],
            "codec": file_info["codec"],
            "channel": file_info["channel"],
            "bitrate": file_info["bitrate"],
            "sample_rate": file_info["sample_rate"],
            "duration_sec": file_info["duration_sec"],
            "url": url,
            "_wav_key": wav_key,
            "_convert_key": convert_key,
        }

    # =====================================================================
    # Orchestrator
    # =====================================================================

    def process_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point.

        Validates input, runs the 6-stage pipeline, builds response
        and metadata, and cleans up.
        """
        started_at = datetime.now(timezone.utc)
        request_id = str(uuid6.uuid7())
        job_id = job["id"]
        raw_request = job.get("input", {})

        # Clear captured logs for this job
        _log_capture.flush_records()
        logger.info(f"Job {job_id} started (request_id={request_id})")

        # --- Validate input ---
        validated = validate(raw_request, INPUT_SCHEMA)
        if "errors" in validated:
            msg = f"Validation failed: {validated['errors']}"
            logger.warning(f"Input validation failed: {validated['errors']}")
            return self._fail("VALIDATION_ERROR", msg, raw_request, request_id,
                              started_at)

        inp = validated["validated_input"]
        input_urls: List[str] = inp.get("input_urls", [])
        model_name: str = inp.get("model_name",
                                  "model_bs_roformer_ep_317_sdr_12.9755.ckpt")
        requested_stems: List[str] = inp.get("stems") or []
        requested_lower = [s.lower() for s in requested_stems]
        out_format: str = inp.get("format", "OPUS")
        out_bitrate: str = inp.get("bitrate", "128k")
        out_sr: int = inp.get("sample_rate", 48000)

        # Internal mapping for predictable temporary filenames
        custom_names = {
            "Vocals": "vocals",
            "Instrumental": "instrumental",
            "Instruments": "instrumental",
            "Drums": "drums",
            "Bass": "bass",
            "Other": "other",
            "Guitar": "guitar",
            "Piano": "piano",
            "Synthesizer": "synth",
            "Strings": "strings",
            "Woodwinds": "woodwinds",
            "Brass": "brass",
        }

        # --- Prepare working dirs ---
        work_dir = os.path.join(WORK_DIR_BASE, job_id)
        input_dir = os.path.join(work_dir, "input")
        output_dir = os.path.join(work_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        s3_prefix = f"{R2_BASE_PREFIX}/requests-{request_id}"

        # --- Performance timers (seconds) ---
        perf = {
            "download_file": 0.0,
            "validate_input_file": 0.0,
            "load_model": 0.0,
            "separate_audio": 0.0,
            "encode_stems": 0.0,
            "upload_results": 0.0,
        }
        peak_vram: float = 0.0
        peak_power: float = 0.0

        try:
            # ---- Stage 1: download ----
            t0 = time.perf_counter()
            if not input_urls:
                return self._fail("EMPTY_INPUT_URL", "At least one URL required",
                                  raw_request, request_id, started_at,
                                  work_dir=work_dir)
            source_url = input_urls[0]
            
            # Use temporary download path to detect extension
            download_path = os.path.join(input_dir, "download.tmp")
            self._download_file(source_url, download_path)
            perf["download_file"] = round(time.perf_counter() - t0, 3)

            # Determine extension
            ext = ".bin"
            try:
                # 1. Try detection via ffprobe (most reliable)
                meta = self.get_audio_metadata(download_path)
                fmt_name = meta.get("format", "").split(',')[0]
                if fmt_name:
                    # Map some ffprobe format names to common extensions
                    ext_map = {
                        "matroska": "mkv",
                        "mov,mp4,m4a,3gp,3g2,mj2": "m4a",
                        "asf": "wma",
                        "ogg": "ogg",
                    }
                    ext = ext_map.get(fmt_name, fmt_name)
                    if not ext.startswith("."):
                        ext = f".{ext}"
            except Exception:
                # 2. Fallback to URL path extension
                try:
                    p = unquote(urlparse(source_url).path)
                    url_ext = os.path.splitext(p)[1].lower()
                    if url_ext:
                        ext = url_ext
                except:
                    pass

            raw_path = os.path.join(input_dir, f"original{ext}")
            os.rename(download_path, raw_path)

            # Upload original input file to R2
            input_key = f"{s3_prefix}/original{ext}"
            self._upload_file(raw_path, input_key)
            logger.info(f"Uploaded original input: {input_key}")

            # ---- Stage 2: validate & prep WAV ----
            t0 = time.perf_counter()
            wav_path = os.path.join(input_dir, "input.wav")
            self._validate_input_file(raw_path, wav_path)
            perf["validate_input_file"] = round(time.perf_counter() - t0, 3)

            # ---- Stage 3: load model ----
            t0 = time.perf_counter()
            self._load_model(model_name, output_dir)
            perf["load_model"] = round(time.perf_counter() - t0, 3)

            # ---- Stage 4: separate ----
            t0 = time.perf_counter()
            power_before = self._get_gpu_power_draw()
            raw_outputs = self._separate_audio(wav_path, custom_output_names=custom_names)
            power_after = self._get_gpu_power_draw()
            peak_power = max(power_before, power_after)
            if torch.cuda.is_available():
                peak_vram = round(
                    torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
            perf["separate_audio"] = round(time.perf_counter() - t0, 3)

            # ---- Stage 5: encode ----
            t0 = time.perf_counter()
            encoded: List[Dict[str, Any]] = []
            for wav_stem_path in raw_outputs:
                fname = os.path.basename(wav_stem_path)
                full_path = os.path.join(output_dir, fname)
                
                # Extract stem name
                # 1. Try to find name in parentheses (default behavior)
                # 2. Else use filename root (custom name behavior)
                m = re.search(r"\(([^)]+)\)", fname)
                if m:
                    stem = m.group(1)
                else:
                    stem = os.path.splitext(fname)[0]
                
                # Filter
                if requested_lower and stem.lower() not in requested_lower:
                    logger.info(f"Skipping stem '{stem}' (not in requested: {requested_stems})")
                    continue
                logger.info(f"Encoding stem '{stem}' -> {out_format} ({out_bitrate}, {out_sr}Hz)")
                info = self._encode_stem(
                    full_path, stem, output_dir,
                    out_format, out_bitrate, out_sr,
                )
                encoded.append(info)
            logger.info(f"Encoding complete: {len(encoded)} stems encoded")
            perf["encode_stems"] = round(time.perf_counter() - t0, 3)

            # ---- Stage 6: upload ----
            t0 = time.perf_counter()
            files: List[Dict[str, Any]] = []
            stem_keys: List[Dict[str, str]] = []
            for info in encoded:
                entry = self._upload_stem(info, s3_prefix)
                # Collect R2 keys for metadata, then strip from response
                stem_keys.append({
                    "stem": entry["stem"],
                    "stem_original_key": entry.pop("_wav_key"),
                    "stem_convert_key": entry.pop("_convert_key"),
                })
                files.append(entry)
            logger.info(f"Upload complete: {len(files)} files uploaded")
            perf["upload_results"] = round(time.perf_counter() - t0, 3)

            # ---- Build files_keys for metadata ----
            files_keys = [{
                "input_key": input_key,
                "stems": stem_keys,
            }]

            # ---- Build response ----
            response = {
                "status": "success",
                "code": "OK",
                "message": "Audio separation completed successfully",
                "files": files,
            }

            # ---- Save metadata ----
            ended_at = datetime.now(timezone.utc)
            metadata = self._build_metadata(
                raw_request, response, request_id,
                started_at, ended_at, peak_vram, peak_power, perf,
                files_keys=files_keys,
            )
            meta_path = os.path.join(work_dir, "metadata.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            self._upload_file(meta_path, f"{s3_prefix}/metadata.json")

            logger.info(f"Job {job_id} completed successfully")
            return response

        except AudioSeparatorError as exc:
            logger.error(f"{exc.error_code}: {exc.message}")
            return self._fail(exc.error_code, exc.message,
                              raw_request, request_id, started_at,
                              work_dir=work_dir, perf=perf,
                              peak_vram=peak_vram, peak_power=peak_power)
        except Exception as exc:
            logger.error(f"Unexpected: {exc}\n{traceback.format_exc()}")
            return self._fail("UNEXPECTED_ERROR", f"Unexpected error: {exc}",
                              raw_request, request_id, started_at,
                              work_dir=work_dir, perf=perf,
                              peak_vram=peak_vram, peak_power=peak_power)
        finally:
            rp_cleanup.clean([work_dir])

    # =====================================================================
    # Response / Metadata builders
    # =====================================================================

    def _fail(
        self, code: str, message: str,
        raw_request: Dict[str, Any],
        request_id: str,
        started_at: datetime,
        *,
        work_dir: Optional[str] = None,
        perf: Optional[Dict[str, float]] = None,
        peak_vram: float = 0.0,
        peak_power: float = 0.0,
    ) -> Dict[str, Any]:
        """Build a fail response and persist metadata if possible."""
        response = {
            "status": "fail",
            "code": code,
            "message": message,
            "files": [],
        }
        # Best-effort metadata save
        try:
            ended_at = datetime.now(timezone.utc)
            metadata = self._build_metadata(
                raw_request, response, request_id,
                started_at, ended_at, peak_vram, peak_power,
                perf or {},
            )
            if work_dir:
                os.makedirs(work_dir, exist_ok=True)
                meta_path = os.path.join(work_dir, "metadata.json")
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)
                s3_prefix = f"{R2_BASE_PREFIX}/requests-{request_id}"
                self._upload_file(meta_path, f"{s3_prefix}/metadata.json")
        except Exception:
            logger.warning("Could not persist error metadata", exc_info=True)

        return response

    def _build_metadata(
        self,
        raw_request: Dict[str, Any],
        response: Dict[str, Any],
        request_id: str,
        started_at: datetime,
        ended_at: datetime,
        peak_vram: float,
        peak_power: float,
        perf: Dict[str, float],
        *,
        files_keys: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Assemble the full metadata document."""
        return {
            "request": raw_request,
            "response": response,
            "files": files_keys or [],
            "debug": {
                "trace": {
                    "request_id": request_id,
                    "started_at": started_at.isoformat(),
                    "ended_at": ended_at.isoformat(),
                },
                "resource": {
                    "gpu_model": self.gpu_info.get("name", "N/A"),
                    "peak_vram": peak_vram,
                    "peak_power": peak_power,
                },
                "performance": {
                    "download_file": perf.get("download_file", 0.0),
                    "validate_input_file": perf.get("validate_input_file", 0.0),
                    "load_model": perf.get("load_model", 0.0),
                    "separate_audio": perf.get("separate_audio", 0.0),
                    "encode_stems": perf.get("encode_stems", 0.0),
                    "upload_results": perf.get("upload_results", 0.0),
                },
                "logs": _log_capture.flush_records(),
            },
        }


# =============================================================================
# Entry Point
# =============================================================================

worker = AudioWorker()


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod handler function."""
    return worker.process_job(job)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

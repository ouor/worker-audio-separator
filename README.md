# Audio Separator – RunPod Deployment

This project provides a serverless audio stem separation API, deployable on RunPod using GPU acceleration. It leverages state-of-the-art models (MDX-Net, Demucs, Roformer, etc.) to separate audio files into stems (vocals, drums, etc.).

## Features
- Separate audio into multiple stems (vocals, drums, bass, etc.)
- Supports common audio formats: WAV, MP3, FLAC, M4A, OGG, OPUS, AAC
- Fast GPU inference (CUDA 12.3.1 base image)
- Cloud storage integration (Cloudflare R2/S3)
- REST API via RunPod serverless handler
- Model auto-download and caching

## Quickstart (RunPod)

### 1. Build Docker Image

```sh
docker build -f Dockerfile.runpod -t audio-separator-runpod .
```

### 2. Environment Variables

Copy `.env.example` to `.env` and fill in your Cloudflare R2 credentials:

```
R2_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your-access-key-id
R2_SECRET_ACCESS_KEY=your-secret-access-key
R2_BUCKET_NAME=your-bucket-name
R2_BASE_PREFIX="worker-audio-separator"
```

### 3. Deploy to RunPod

Push your image to a registry and deploy using RunPod's serverless API template. Set environment variables as above.

### 4. API Usage

Send a POST request to your RunPod endpoint with a JSON body:

```json
{
  "input_urls": ["https://example.com/audio.wav"],
  "model_name": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
  "stems": ["vocals", "instrumental"],
  "format": "MP3",
  "bitrate": "192k",
  "sample_rate": 44100
}
```

**Response:**
```json
{
  "status": "success",
  "code": "OK",
  "message": "Audio separation completed successfully",
  "files": [
    {
      "stem": "Vocals",
      "format": "mp3",
      "codec": "libmp3lame",
      "channel": 2,
      "bitrate": "192k",
      "sample_rate": 44100,
      "duration_sec": 180.0,
      "url": "https://.../vocals.mp3"
    },
    ...
  ]
}
```

## Handler Pipeline
- **download_file**: Download or decode input audio
- **validate_input_file**: Check and convert to WAV
- **load_model**: Load ML model (warm start supported)
- **separate_audio**: Run inference
- **encode_stems**: Encode stems to requested format
- **upload_results**: Upload to R2, return presigned URLs

## Error Handling
Custom exceptions for audio validation, storage, inference, and network errors are defined in `runpod_exceptions.py`.

## Input Schema
See `runpod_schemas.py` for input validation logic. Required fields:
- `input_urls`: List of HTTP(S) or data:audio/ URLs
- `model_name`: Model file name (default provided)
- `stems`: List of stems to extract (optional)
- `format`: Output format (WAV, MP3, FLAC, etc.)
- `bitrate`: Output bitrate (e.g., 128k)
- `sample_rate`: Output sample rate (e.g., 44100)

## Development
- See [README.md.old](README.md.old) for full CLI and Python API usage
- Main handler: `runpod_handler.py`
- Exception classes: `runpod_exceptions.py`
- Input schema: `runpod_schemas.py`
- Dockerfile for RunPod: `Dockerfile.runpod`

## License
MIT License. See [LICENSE](LICENSE).

"""
Input validation schema for RunPod audio separation handler.
"""


def _validate_input_urls(input_urls: list) -> bool:
    """
    Validate that input_urls is a non-empty list of strings.
    
    Args:
        input_urls: List of URLs or Data URLs.
        
    Returns:
        True if validation passes.
        
    Raises:
        ValueError: If validation fails.
    """
    if not input_urls:
        raise ValueError("'input_urls' must contain at least one URL")
    
    if not isinstance(input_urls, list):
        raise ValueError("'input_urls' must be a list")
    
    for idx, url in enumerate(input_urls):
        if not isinstance(url, str):
            raise ValueError(f"'input_urls[{idx}]' must be a string")
        if not url.strip():
            raise ValueError(f"'input_urls[{idx}]' cannot be empty")
        
        # Validate URL format (basic check)
        url_lower = url.lower()
        is_http = url_lower.startswith(("http://", "https://"))
        is_data_url = url_lower.startswith("data:audio/")
        
        if not (is_http or is_data_url):
            raise ValueError(
                f"'input_urls[{idx}]' must be a valid HTTP(S) URL or Data URL starting with 'data:audio/'"
            )
    
    return True


def _validate_format(format_str: str) -> bool:
    """
    Validate output format.
    
    Args:
        format_str: Output format string.
        
    Returns:
        True if validation passes.
        
    Raises:
        ValueError: If format is not supported.
    """
    supported_formats = ["WAV", "MP3", "FLAC", "OGG", "M4A", "OPUS", "AAC"]
    format_upper = format_str.upper()
    
    if format_upper not in supported_formats:
        raise ValueError(
            f"Unsupported format '{format_str}'. Supported formats: {', '.join(supported_formats)}"
        )
    
    return True


def _validate_bitrate(bitrate: str) -> bool:
    """
    Validate bitrate format.
    
    Args:
        bitrate: Bitrate string (e.g., "128k", "320k").
        
    Returns:
        True if validation passes.
        
    Raises:
        ValueError: If bitrate format is invalid.
    """
    import re
    
    # Match patterns like "128k", "320k", "256k"
    pattern = r"^\d+k$"
    if not re.match(pattern, bitrate.lower()):
        raise ValueError(
            f"Invalid bitrate '{bitrate}'. Must be in format like '128k', '192k', '320k'"
        )
    
    # Extract numeric value
    bitrate_value = int(bitrate.lower().rstrip("k"))
    
    # Reasonable range check
    if bitrate_value < 32 or bitrate_value > 512:
        raise ValueError(
            f"Bitrate {bitrate_value}k is out of reasonable range (32k - 512k)"
        )
    
    return True


def _validate_sample_rate(sample_rate: int) -> bool:
    """
    Validate sample rate.
    
    Args:
        sample_rate: Sample rate in Hz.
        
    Returns:
        True if validation passes.
        
    Raises:
        ValueError: If sample rate is invalid.
    """
    common_rates = [8000, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000, 192000]
    
    if sample_rate not in common_rates:
        raise ValueError(
            f"Uncommon sample rate {sample_rate}Hz. Common rates: {', '.join(map(str, common_rates))}"
        )
    
    return True


INPUT_SCHEMA = {
    "tag": {
        "type": str,
        "required": False,
        "default": None,
    },
    "input_urls": {
        "type": list,
        "required": True,
        "constraints": _validate_input_urls,
    },
    "model_name": {
        "type": str,
        "required": False,
        "default": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    },
    "stems": {
        "type": list,
        "required": False,
        "default": None,
    },
    "format": {
        "type": str,
        "required": False,
        "default": "OPUS",
        "constraints": _validate_format,
    },
    "bitrate": {
        "type": str,
        "required": False,
        "default": "128k",
        "constraints": _validate_bitrate,
    },
    "sample_rate": {
        "type": int,
        "required": False,
        "default": 48000,
        "constraints": _validate_sample_rate,
    },
}

"""
Input validation schema for RunPod audio separation handler.
"""


def _validate_input_mode(input_dict: dict) -> bool:
    """
    Cross-validate that the required field exists for the specified mode.
    
    Args:
        input_dict: The 'input' dictionary from the request.
        
    Returns:
        True if validation passes.
        
    Raises:
        ValueError: If mode-specific field is missing or empty.
    """
    mode = input_dict.get("mode")
    
    if mode == "base64":
        if not input_dict.get("base64"):
            raise ValueError("'base64' field is required when mode is 'base64'")
    elif mode == "presigned_url":
        if not input_dict.get("presigned_url"):
            raise ValueError("'presigned_url' field is required when mode is 'presigned_url'")
    
    return True


INPUT_SCHEMA = {
    "tag": {
        "type": str,
        "required": False,
        "default": None,
    },
    "input": {
        "type": dict,
        "required": True,
        "constraints": _validate_input_mode,
        "mode": {
            "type": str,
            "required": True,
            "constraints": lambda mode: mode in ["base64", "presigned_url"],
        },
        "base64": {
            "type": str,
            "required": False,
            "default": None,
        },
        "presigned_url": {
            "type": str,
            "required": False,
            "default": None,
        },
    },
    "model": {
        "type": str,
        "required": False,
        "default": "UVR-MDX-NET-Voc_FT.onnx",
    },
    "stems": {
        "type": list,
        "required": False,
        "default": None,
    },
}

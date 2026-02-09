class AudioSeparatorError(Exception):
    """Base class for exceptions in this project."""
    def __init__(self, message, error_code="INTERNAL_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class AudioValidationError(AudioSeparatorError):
    """Raised when the input audio file is invalid or cannot be processed."""
    def __init__(self, message):
        super().__init__(message, error_code="AUDIO_VALIDATION_ERROR")

class StorageError(AudioSeparatorError):
    """Raised when there is an error during storage operations (R2/S3)."""
    def __init__(self, message):
        super().__init__(message, error_code="STORAGE_ERROR")

class InferenceError(AudioSeparatorError):
    """Raised when the AI model fails during separation."""
    def __init__(self, message):
        super().__init__(message, error_code="INFERENCE_ERROR")

class NetworkError(AudioSeparatorError):
    """Raised when external resource downloads fail."""
    def __init__(self, message):
        super().__init__(message, error_code="NETWORK_ERROR")
    """Raised when external resource downloads fail."""
    def __init__(self, message):
        super().__init__(message, error_code="NETWORK_ERROR")

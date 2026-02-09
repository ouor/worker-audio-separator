import os
import logging
from audio_separator.separator import Separator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pre_download_models():
    model_dir = '/models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize separator just to use its download capability
    separator = Separator(model_file_dir=model_dir)
    
    # List of models to pre-download
    models_to_download = [
        'model_bs_roformer_ep_317_sdr_12.9755.ckpt',
        'UVR-MDX-NET-Voc_FT.onnx',
        '5_HP-Karaoke-UVR.pth',
    ]
    
    for model in models_to_download:
        try:
            logger.info(f"Pre-downloading model: {model}")
            separator.download_model_and_data(model)
        except Exception as e:
            logger.error(f"Failed to download {model}: {e}")

if __name__ == "__main__":
    pre_download_models()

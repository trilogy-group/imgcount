import os
import uuid
from datetime import datetime

def get_output_path(prefix: str, extension: str = "png", output_dir: str = "output") -> str:
    """Generates a unique file path with timestamp and UUID."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{prefix}_{timestamp}_{unique_id}.{extension}"
    return os.path.join(output_dir, filename)


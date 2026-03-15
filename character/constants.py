import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = os.getenv("OCT_DATA_PATH", str(ROOT / "data"))
MODEL_PATH = os.getenv("OCT_MODEL_PATH", str(Path.home() / "models"))
LORA_PATH = os.getenv("OCT_LORA_PATH", str(Path.home() / "loras"))
CONSTITUTION_PATH = os.getenv("OCT_CONSTITUTION_PATH", str(ROOT / "constitutions"))

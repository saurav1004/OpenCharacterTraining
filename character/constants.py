import os

_WORKSPACE = os.environ.get("OCT_WORKSPACE", "/workspace")

DATA_PATH = os.path.join(_WORKSPACE, "OpenCharacterTraining", "data")
MODEL_PATH = os.path.join(_WORKSPACE, "models")
LORA_PATH = os.path.join(_WORKSPACE, "loras")
CONSTITUTION_PATH = os.path.join(_WORKSPACE, "OpenCharacterTraining", "constitutions")

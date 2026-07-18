"""Central settings for the whole project. Change things here, not inside scripts."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "inputs"      # base image and test images go here
RUNS_DIR = DATA_DIR / "runs"         # every experiment writes its outputs here
COSTS_CSV = DATA_DIR / "costs.csv"   # running receipt of every API call

# Model slug on OpenRouter. Verify the exact name at openrouter.ai/models
# (filter by image output). Nano Banana 2 is the family Weon names in the JD.
PRIMARY_MODEL = "google/gemini-3.1-flash-image"

# Every image is resized so its longest side is this many pixels before a chain
# starts. Editing models output around 1K anyway; starting at a matched size
# keeps every step comparable to the last.
WORKING_LONG_SIDE = 1024

# Hard spending cap. The client refuses to call the API once the receipts pass
# this. Leaves a safety buffer inside the 20 dollars Weon provided.
MAX_BUDGET_USD = 16.0

# Settings for the crop-edit-composite treatment (used later by compositing.py):
# how much padding to add around the edit rectangle before cropping, and how
# soft the paste-back edge should be, in pixels.
CROP_PAD_FRAC = 0.35
FEATHER_PX = 16

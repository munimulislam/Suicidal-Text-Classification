from pathlib import Path
import os

IS_KAGGLE = os.path.exists("/kaggle/working")

SEED = 42
KAGGLE = "kaggle"
SWMH = "swmh"

INPUT_DIR = (
    Path("/kaggle/input/datasets/mdmunimulislam/mh-crisis")
    if IS_KAGGLE
    else Path(__file__).parent.resolve()
)
DATA_DIR = INPUT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

OUT_DIR = Path("/kaggle/working/") if IS_KAGGLE else Path(__file__).parent.resolve()
RESULTS_DIR = OUT_DIR / "results"
CHECKPOINT_IN_DIR = (
    Path("/kaggle/input/models/mdmunimulislam/mh-crisis/pytorch/default/2/checkpoints")
    if IS_KAGGLE
    else OUT_DIR / "checkpoints"
)
CHECKPOINT_OUT_DIR = OUT_DIR / "checkpoints"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_OUT_DIR.mkdir(parents=True, exist_ok=True)

KAGGLE_DATA_DIR = DATA_DIR / KAGGLE
SWMH_DATA_DIR = DATA_DIR / SWMH

KAGGLE_CSV = KAGGLE_DATA_DIR / "Suicide_Detection.csv"
SWMH_TRAIN_CSV = SWMH_DATA_DIR / "train.csv"
SWMH_VAL_CSV = SWMH_DATA_DIR / "val.csv"
SWMH_TEST_CSV = SWMH_DATA_DIR / "test.csv"

KAGGLE_PROCESSED = PROCESSED_DIR / KAGGLE
SWMH_PROCESSED = PROCESSED_DIR / SWMH

KAGGLE_LABELS = ["suicide", "non-suicide"]
SWMH_LABELS = ["Anxiety", "bipolar", "depression", "SuicideWatch", "offmychest"]
KAGGLE_LABEL_MAP = {
    "suicide": 1,
    "non-suicide": 0,
}
SWMH_LABEL_MAP = {
    "self.SuicideWatch": 0,
    "self.depression": 1,
    "self.Anxiety": 2,
    "self.bipolar": 3,
    "self.offmychest": 4,
}

GLOVE_DIR = DATA_DIR / "glove"

MAX_LEN = 256

XAI_DIR = RESULTS_DIR / "xai"

XAI_DIR.mkdir(parents=True, exist_ok=True)

from pathlib import Path

SEED = 42
KAGGLE = 'kaggle'
SWMH = 'swmh'

ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / 'data'
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / 'results'
CHECKPOINT_DIR = ROOT_DIR / 'checkpoints'

KAGGLE_DATA_DIR = DATA_DIR / KAGGLE
SWMH_DATA_DIR = DATA_DIR / SWMH

KAGGLE_CSV = KAGGLE_DATA_DIR / "Suicide_Detection.csv"
SWMH_TRAIN_CSV = SWMH_DATA_DIR / "train.csv"
SWMH_VAL_CSV = SWMH_DATA_DIR / "val.csv"
SWMH_TEST_CSV  = SWMH_DATA_DIR / "test.csv"

KAGGLE_PROCESSED = PROCESSED_DIR / KAGGLE
SWMH_PROCESSED = PROCESSED_DIR / SWMH

KAGGLE_LABELS = ['suicide', 'non-suicide']
SWMH_LABELS = ['Anxiety', 'bipolar', 'depression', 'SuicideWatch', 'offmychest']
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

GLOVE_DIR = DATA_DIR / 'glove'
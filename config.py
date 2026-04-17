from pathlib import Path

SEED = 42
KAGGLE = 'kaggle'
SWMH = 'swmh'

ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / 'data'
PROCESSED_DIR = DATA_DIR / "processed"

KAGGLE_DATA_DIR = DATA_DIR / KAGGLE
SWMH_DATA_DIR = DATA_DIR / SWMH

KAGGLE_CSV = KAGGLE_DATA_DIR / "Suicide_Detection.csv"
SWMH_TRAIN_CSV = SWMH_DATA_DIR / "train.csv"
SWMH_VAL_CSV   = SWMH_DATA_DIR / "val.csv"
SWMH_TEST_CSV  = SWMH_DATA_DIR / "test.csv"

KAGGLE_PROCESSED = PROCESSED_DIR / KAGGLE
SWMH_PROCESSED   = PROCESSED_DIR / SWMH
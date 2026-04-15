import os
from pathlib import Path

SEED = 42

ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / 'data'
KAGGLE_DATA_DIR = DATA_DIR / 'kaggle'
SWMH_DATA_DIR = DATA_DIR / 'swmh'

KAGGLE_CSV = KAGGLE_DATA_DIR / "Suicide_Detection.csv"

SWMH_TRAIN_CSV = SWMH_DATA_DIR / "train.csv"
SWMH_VAL_CSV   = SWMH_DATA_DIR / "val.csv"
SWMH_TEST_CSV  = SWMH_DATA_DIR / "test.csv"
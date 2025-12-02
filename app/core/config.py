from pathlib import Path

class Settings:
    # Paths
    ARTIFACTS_DIR = Path("../artifacts")
    MODELS_DIR = Path("../models")
    DATA_DIR = Path("../data")

    # Encoders
    TARGET_ENCODER_PATH = ARTIFACTS_DIR / "target_encoder.pkl"
    OHE_ENCODER_PATH = ARTIFACTS_DIR / "ohe.pkl"

    # Model
    BEST_MODEL_PATH = MODELS_DIR / "model_Catboost.pkl"

    # Expected feature order (from training)
    FEATURE_ORDER = [
        'ItemWeight', 'MRP', 'OutletAge', 'Visibility', 'IsVisibile', 'IsGroceryStore', 'PricePerWeight',
        'ItemType',
        'FatContent_Regular',
        'OutletSize_Medium', 'OutletSize_Small',
        'LocationType_Tier 2', 'LocationType_Tier 3',
        'OutletType_Supermarket Type1', 'OutletType_Supermarket Type2', 'OutletType_Supermarket Type3'
    ]

settings = Settings()
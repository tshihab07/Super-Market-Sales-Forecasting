from pathlib import Path

# Auto-detect project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

class Settings:
    ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
    MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
    DATA_DIR = PROJECT_ROOT / "data"

    # Now paths are absolute and robust
    TARGET_ENCODER_PATH = ARTIFACTS_DIR / "feature-selection" / "target_encoder.pkl"
    OHE_ENCODER_PATH = ARTIFACTS_DIR / "feature-selection" / "ohe.pkl"
    BEST_MODEL_PATH = MODELS_DIR / "model_Catboost.pkl"

    FEATURE_ORDER = [
        'ItemWeight', 'MRP', 'OutletAge', 'Visibility', 'IsVisibile', 'IsGroceryStore', 'PricePerWeight',
        'ItemType',
        'FatContent_Regular',
        'OutletSize_Medium', 'OutletSize_Small',
        'LocationType_Tier 2', 'LocationType_Tier 3',
        'OutletType_Supermarket Type1', 'OutletType_Supermarket Type2', 'OutletType_Supermarket Type3'
    ]

    # Debug helper
    def __init__(self):
        print(f"Project root: {PROJECT_ROOT}")
        print(f"Target encoder path: {self.TARGET_ENCODER_PATH}")
        print(f"Exists? {self.TARGET_ENCODER_PATH.exists()}")
        if not self.TARGET_ENCODER_PATH.exists():
            print("WARNING: Encoder file not found!")


settings = Settings()
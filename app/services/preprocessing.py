import numpy as np
import pandas as pd
import joblib
from app.core.config import settings
from app.schemas.prediction import SaleInput

# Load encoders ONCE (module-level)
try:
    target_encoder = joblib.load(settings.TARGET_ENCODER_PATH)
    ohe_encoder = joblib.load(settings.OHE_ENCODER_PATH)
    print("✅ Encoders loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load encoders: {e}")


def preprocess_input(input_obj: SaleInput) -> pd.DataFrame:
    """
    Converts validated SaleInput → preprocessed DataFrame matching training schema.
    """
    # Convert to dict for processing
    input_data = input_obj.dict()

    # Step 1: Create DataFrame
    df = pd.DataFrame([input_data])

    # Step 2: Compute derived features
    df['PricePerWeight'] = np.where(
        df['ItemWeight'] > 0,
        df['MRP'] / df['ItemWeight'],
        df['MRP']
    )

    df['IsGroceryStore'] = (df['OutletType'] == 'Grocery Store').astype(int)

    # Step 3: Handle Visibility
    VISIBILITY_IF_VISIBLE = 0.056
    df['Visibility_raw'] = df['IsVisible'].map({True: VISIBILITY_IF_VISIBLE, False: 0.0})
    df['Visibility'] = np.log(df['Visibility_raw'] + 1)
    df['IsVisibile'] = df['IsVisible'].astype(int)

    df = df.drop(columns=['Visibility_raw', 'IsVisible'])

    # Step 4: MAP USER-FRIENDLY VALUES TO INTERNAL NAMES
    # ← THIS IS THE KEY FIX ←
    df['FatContent'] = df['FatContent'].replace({'Low Fat': 'Low Fat', 'Regular': 'Regular'})
    df['OutletSize'] = df['OutletSize'].replace({
        'Large': 'High',
        'Medium': 'Medium',
        'Small': 'Small'
    })
    df['LocationType'] = df['LocationType'].replace({
        'City': 'Tier 1',
        'Semi-Urban': 'Tier 2',
        'Rural': 'Tier 3'
    })
    df['OutletType'] = df['OutletType'].replace({
        'Small-Format Supermarket': 'Supermarket Type1',
        'Medium-Format Supermarket': 'Supermarket Type2',
        'Large-Format Supermarket': 'Supermarket Type3'
    })

    # Step 5: Define columns for encoding
    cat_cols = ['FatContent', 'OutletSize', 'LocationType', 'OutletType']
    target_col = ['ItemType']

    # Step 6: One-Hot Encoding
    try:
        ohe_array = ohe_encoder.transform(df[cat_cols])
        ohe_df = pd.DataFrame(
            ohe_array,
            columns=ohe_encoder.get_feature_names_out(cat_cols),
            index=df.index
        )
        # ✅ CONVERT TO INT — CatBoost requires int/str for categorical features
        ohe_df = ohe_df.astype(int)
    except Exception as e:
        print(f"OHE fallback: {e}")
        expected_cols = ohe_encoder.get_feature_names_out(cat_cols)
        ohe_df = pd.DataFrame(0, index=df.index, columns=expected_cols).astype(int)

    # Step 7: Target Encoding
    try:
        item_type_encoded = target_encoder.transform(df[target_col])
        item_type_df = item_type_encoded[['ItemType']].copy()
    except Exception as e:
        print(f"Target encoding fallback: {e}")
        try:
            global_mean = target_encoder.mapping['ItemType']['mean']
        except:
            global_mean = 7.5
        item_type_df = pd.DataFrame([global_mean], columns=['ItemType'], index=df.index)

    # Step 8: Combine numerical + encoded features
    numerical_cols = ['ItemWeight', 'MRP', 'OutletAge', 'Visibility', 'IsVisibile', 'IsGroceryStore', 'PricePerWeight']
    final_df = pd.concat([
        df[numerical_cols].reset_index(drop=True),
        item_type_df.reset_index(drop=True),
        ohe_df.reset_index(drop=True)
    ], axis=1)

    # Step 9: Reorder to match training
    try:
        final_df = final_df[settings.FEATURE_ORDER]
    except KeyError as e:
        missing = set(settings.FEATURE_ORDER) - set(final_df.columns)
        raise ValueError(f"Missing columns after encoding: {missing}")

    return final_df
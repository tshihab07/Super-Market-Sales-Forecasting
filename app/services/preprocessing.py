import numpy as np
import pandas as pd
import joblib
from app.core.config import settings

# Load encoders ONCE (module-level)
try:
    target_encoder = joblib.load(settings.TARGET_ENCODER_PATH)
    ohe_encoder = joblib.load(settings.OHE_ENCODER_PATH)
    print("✅ Encoders loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load encoders: {e}")

def preprocess_input(input_data: dict) -> pd.DataFrame:
    """
    Converts raw user input (dict) → preprocessed DataFrame matching training schema.
    """
    # Step 1: Create DataFrame
    df = pd.DataFrame([input_data])

    # Step 2: Compute derived features
    # PricePerWeight = MRP / ItemWeight (with safe division)
    df['PricePerWeight'] = np.where(
        df['ItemWeight'] > 0,
        df['MRP'] / df['ItemWeight'],
        df['MRP']  # fallback if weight=0
    )

    # IsGroceryStore flag (1 if OutletType == 'Grocery Store', else 0)
    # But note: user never selects 'Grocery Store' — only Supermarket types
    # So this will always be 0 — safe.
    df['IsGroceryStore'] = (df['OutletType'] == 'Grocery Store').astype(int)

    # Step 3: Handle Visibility
    # User gives IsVisible → we infer raw Visibility
    # From EDA: median non-zero Visibility ≈ 0.056
    VISIBILITY_IF_VISIBLE = 0.056
    df['Visibility_raw'] = df['IsVisible'].map({True: VISIBILITY_IF_VISIBLE, False: 0.0})
    df['Visibility'] = np.log(df['Visibility_raw'] + 1)  # matches training log(visibility + 1)
    df['IsVisibile'] = df['IsVisible'].astype(int)  # 1 if visible, 0 otherwise

    # Step 4: Drop temporary/helper columns
    df = df.drop(columns=['Visibility_raw', 'IsVisible'])

    # Step 5: Define columns for encoding
    cat_cols = ['FatContent', 'OutletSize', 'LocationType', 'OutletType']
    target_col = ['ItemType']

    # Step 6: One-Hot Encoding (handle unknown gracefully)
    try:
        ohe_array = ohe_encoder.transform(df[cat_cols])
        ohe_df = pd.DataFrame(
            ohe_array,
            columns=ohe_encoder.get_feature_names_out(cat_cols),
            index=df.index
        )
    except Exception as e:
        print(f"⚠️ OHE fallback: {e}")
        # Create zero-filled DataFrame with expected OHE columns
        expected_cols = ohe_encoder.get_feature_names_out(cat_cols)
        ohe_df = pd.DataFrame(0, index=df.index, columns=expected_cols)

    # Step 7: Target Encoding
    try:
        item_type_encoded = target_encoder.transform(df[target_col])
        item_type_df = item_type_encoded[['ItemType']].copy()
    except Exception as e:
        print(f"⚠️ Target encoding fallback: {e}")
        # Use global mean from encoder (if available)
        try:
            global_mean = target_encoder.mapping['ItemType']['mean']
        except:
            # Fallback: use 7.5 (approx mean of OutletSales_log)
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
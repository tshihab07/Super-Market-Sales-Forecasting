from pydantic import BaseModel, Field, validator
from typing import Literal

class SaleInput(BaseModel):
    ItemType: Literal[
        "Fruits and Vegetables", "Snack Foods", "Household", "Frozen Foods", "Dairy",
        "Canned", "Baking Goods", "Health and Hygiene", "Soft Drinks", "Meat", "Others"
    ]
    FatContent: Literal["Low Fat", "Regular"]
    ItemWeight: float = Field(gt=0, description="Item weight in kg")
    MRP: float = Field(gt=0, description="Maximum Retail Price")
    IsVisible: bool
    OutletSize: Literal["Large", "Medium", "Small"]
    LocationType: Literal["City", "Semi-Urban", "Rural"]
    OutletType: Literal["Small-Format Supermarket", "Medium-Format Supermarket", "Large-Format Supermarket"]
    OutletAge: int = Field(ge=0, le=100, description="Outlet age in years")

    @validator("OutletSize", pre=True)
    def map_outlet_size(cls, v):
        mapping = {"Large": "High", "Medium": "Medium", "Small": "Small"}
        if v in mapping:
            return mapping[v]
        return v

    @validator("LocationType", pre=True)
    def map_location_type(cls, v):
        mapping = {"City": "Tier 1", "Semi-Urban": "Tier 2", "Rural": "Tier 3"}
        if v in mapping:
            return mapping[v]
        return v

    @validator("OutletType", pre=True)
    def map_outlet_type(cls, v):
        mapping = {
            "Small-Format Supermarket": "Supermarket Type1",
            "Medium-Format Supermarket": "Supermarket Type2",
            "Large-Format Supermarket": "Supermarket Type3"
        }
        if v in mapping:
            return mapping[v]
        return v

    @validator("IsVisible", pre=True)
    def parse_is_visible(cls, v):
        if isinstance(v, str):
            return v.lower() in ["yes", "true", "1"]
        return bool(v)

class PredictionResult(BaseModel):
    log_sales: float
    predicted_sales: float
    input_data: SaleInput
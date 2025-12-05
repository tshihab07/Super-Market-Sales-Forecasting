from pydantic import BaseModel, Field
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
    OutletType: Literal[
        "Small-Format Supermarket",
        "Medium-Format Supermarket",
        "Large-Format Supermarket"
    ]
    OutletAge: int = Field(ge=0, le=100, description="Outlet age in years")


class PredictionResult(BaseModel):
    log_sales: float
    predicted_sales: float
    input_data: SaleInput
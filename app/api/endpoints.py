from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from app.schemas.prediction import SaleInput
from app.services.preprocessing import preprocess_input
from app.services.prediction import predict_sales

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/sale_information", response_class=HTMLResponse)
async def sale_information(request: Request):
    return templates.TemplateResponse("sale_information.html", {"request": request})


@router.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    ItemType: str = Form(...),
    FatContent: str = Form(...),
    ItemWeight: float = Form(...),
    MRP: float = Form(...),
    IsVisible: str = Form(...),  # "yes" or "no"
    OutletSize: str = Form(...),
    LocationType: str = Form(...),
    OutletType: str = Form(...),
    OutletAge: int = Form(...)
):
    try:
        # Parse form data
        input_dict = {
            "ItemType": ItemType,
            "FatContent": FatContent,
            "ItemWeight": ItemWeight,
            "MRP": MRP,
            "IsVisible": IsVisible.lower() == "yes",
            "OutletSize": OutletSize,
            "LocationType": LocationType,
            "OutletType": OutletType,
            "OutletAge": OutletAge
        }

        # Validate with Pydantic
        validated_input = SaleInput(**input_dict)

        # Preprocess
        preprocessed = preprocess_input(validated_input)

        # Predict
        log_pred, orig_pred = predict_sales(preprocessed)

        # Format for display
        predicted_sales = round(orig_pred, 2)

        return templates.TemplateResponse("result.html", {
            "request": request,
            "predicted_sales": predicted_sales,
            "log_prediction": round(log_pred, 4)
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return templates.TemplateResponse("result.html", {
            "request": request,
            "error": f"Prediction failed: {str(e)}",
            "predicted_sales": None
        }, status_code=400)
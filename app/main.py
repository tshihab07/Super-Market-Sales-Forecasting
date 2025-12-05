from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from app.api.endpoints import router as api_router

app = FastAPI(
    title="Supermarket Sales Forecasting API",
    description="AI-powered sales prediction for supermarket outlets using CatBoost",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# mount static files (CSS)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# include API routes
app.include_router(api_router)

# health check
@app.get("/health")
async def health_check():
    return {"status": "OK", "model": "CatBoost (BayesianSearch)"}
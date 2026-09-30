import os
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router

app = FastAPI(
    title="EVOX — Adaptive AI Intelligence Platform API",
    description="Production ML Platform with Scikit-Learn, Pandas, IsolationForest, Optuna, and SHAP",
    version="2.1.0",
)

# -------------------------------------------------------------
# FIX 6 — PRODUCTION CORS CONFIGURATION
# -------------------------------------------------------------
environment = os.getenv("ENVIRONMENT", "development").lower()
is_production = environment == "production" or os.getenv("NODE_ENV", "").lower() == "production"
cors_origins_env = os.getenv("CORS_ORIGINS", "").strip()

if is_production:
    # Production: Strictly parse explicit allowed domains, reject arbitrary '*'
    raw_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
    allowed_origins = [o for o in raw_origins if o != "*"]
    if not allowed_origins:
        # Fallback to internal container / deployment origin
        allowed_origins = ["http://localhost:3000"]
else:
    # Development: Allow common local Vite / development ports
    if cors_origins_env:
        allowed_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
    else:
        allowed_origins = [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "*",
        ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# FIX 15 — CLEAN JSON ERROR HANDLING (No raw stack traces)
# -------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(_req: Request, exc: Exception):
    # Sanitize message to avoid exposing raw system paths or python tracebacks
    err_str = str(exc)
    clean_msg = err_str.split("\n")[-1] if err_str else "An unexpected error occurred during ML execution."
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Machine learning operation encountered an issue.",
            "details": clean_msg,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )

# Include the full API router
app.include_router(api_router)

@app.get("/")
def root():
    return {
        "platform": "EVOX — Adaptive AI Intelligence Platform",
        "status": "ONLINE",
        "environment": "production" if is_production else "development",
        "documentation": "/docs",
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=not is_production)

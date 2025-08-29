from fastapi import FastAPI
from app.core.logging import logger
from app.api.v1.endpoints.passport import router as passport_v1_router
from app.api.v2.endpoints.passport import router as passport_v2_router 

app = FastAPI(title="Passport Validation API")

app.include_router(passport_v1_router, prefix="/api/v1", tags=["v1"])
app.include_router(passport_v2_router, prefix="/api/v2", tags=["v2"])

@app.get("/healthz")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

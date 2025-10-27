from fastapi import FastAPI
from app.core.logging import logger
from app.api.v1.endpoints.passport import router as passport_v1_router
from app.api.v2.endpoints.passport import router as passport_v2_router
from prometheus_fastapi_instrumentator import Instrumentator


def create_app() -> FastAPI:
    app = FastAPI(title="Passport Validation API")

    # Include all API routes
    app.include_router(passport_v1_router, prefix="/api/v1", tags=["v1"])
    app.include_router(passport_v2_router, prefix="/api/v2", tags=["v2"])

    # Health check route (Prometheus can also use this)
    @app.get("/healthz")
    def health_check():
        return {"status": "ok"}

    # Initialize Prometheus metrics
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

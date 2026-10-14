import logging
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routes.students import router as students_router
from routes.attendance import router as attendance_router
from routes.auth import router as auth_router
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("backend_app")

app = FastAPI(
    title="SNN-Based Smart Attendance System API",
    description="RFID + Face Recognition + SNN confidence check attendance backend.",
    version="1.0.0",
)

origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred."},
    )


@app.get("/", tags=["Health"])
async def root_health_check():
    return {"status": "healthy", "system": "SNN-Based Smart Attendance System API", "version": "1.0.0"}


app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(students_router, prefix="/api/students", tags=["Students"])
app.include_router(attendance_router, prefix="/api/attendance", tags=["Attendance"])


if __name__ == "__main__":
    import uvicorn
    logger.info("Launching FastAPI Development Server...")
    uvicorn.run("app:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)

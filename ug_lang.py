import os
import asyncio
import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from routers import auth, assessment, profile, chat, pdf, learn
from services.pdf.pdf_service import PDFProcessingService

# Initialize Redis client
redis_client = redis.from_url(settings.REDIS_URL)

# Initialize FastAPI app
app = FastAPI(title="Sahasra AI Education API")
allow_origin_regex = (
    r"^(https?:\/\/("
    r"localhost:3000|"
    r"localhost:3001|"
    r"fastapi\.tiangolo\.com|"
    r"sahasraai\.vercel\.app|"
    r"www\.sahasra\.ai|"
    r"questionbank-one\.vercel\.app|"
    r"v0-sahas\.vercel\.app|"
    r".+\.lite\.vusercontent\.net"
    r"))$"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
    allow_headers=["X-Auth-Session", "Content-Type","Content-Disposition","Cache-Control","Transfer-Encoding"],
    expose_headers=["X-Auth-Session"]
)

# Mount static directory for serving files
static_dir = settings.static_dir_path
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)
    
# Create pdfs directory if it doesn't exist
pdfs_dir = os.path.join(static_dir, "pdfs")
if not os.path.exists(pdfs_dir):
    os.makedirs(pdfs_dir, exist_ok=True)
    
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include routers
app.include_router(auth.router)
app.include_router(assessment.router)
app.include_router(profile.router)
app.include_router(chat.router)
app.include_router(pdf.router)
app.include_router(learn.router)

# PDF processing worker with Redis
pdf_processing_service = PDFProcessingService(
    redis_client=redis_client,
    max_workers=settings.REDIS_MAX_WORKERS
)

@app.on_event("startup")
async def startup_events():
    """Startup events for the application."""
    # Start the PDF processing worker
    asyncio.create_task(pdf_processing_service.process_queued_pdfs())
    
    # Log startup
    print(f"Started PDF processing worker with {settings.REDIS_MAX_WORKERS} worker threads")

@app.on_event("shutdown")
async def shutdown_events():
    """Shutdown events for the application."""
    # Close Redis connection
    await redis_client.close()
    print("Closed Redis connection")

@app.get("/")
def root():
    """Root endpoint that provides API information."""
    return {
        "message": "Welcome to Sahasra AI Education API",
        "version": "1.0.0",
        "documentation": "/docs"
    }

# Run the app with uvicorn if this file is executed directly
if __name__ == "__main__":
    '''import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)'''
    app.run(host="0.0.0.0", port=8000)







import os
import asyncio
import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from routers import auth, assessment, profile, chat, pdf, learn, health
from services.pdf.pdf_service import PDFProcessingService

# Initialize Redis client with connection parameters for replica set
redis_client = redis.from_url(
    settings.REDIS_URL,
    socket_connect_timeout=settings.REDIS_SOCKET_CONNECT_TIMEOUT,
    socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
    socket_keepalive=True,
    retry_on_timeout=True,
    health_check_interval=settings.REDIS_HEALTH_CHECK_INTERVAL
)

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
app.include_router(health.router)  # Health check endpoints first
app.include_router(auth.router)
app.include_router(assessment.router)
app.include_router(profile.router)
app.include_router(chat.router)
app.include_router(pdf.router)
app.include_router(learn.router)

@app.on_event("startup")
async def startup_events():
    """Startup events for the application with database health checks."""
    print("=" * 70)
    print("🚀 Starting Sahasra AI Education API")
    print("=" * 70)
    
    # Verify database connections
    all_healthy = True
    
    # Test MongoDB connection
    try:
        from repositories.mongo_repository import MongoRepository
        mongo_repo = MongoRepository()
        server_info = mongo_repo.client.admin.command('ismaster')
        
        if server_info.get('setName'):
            print(f"✓ MongoDB Connected - Replica Set: {server_info.get('setName')}")
            print(f"  Primary: {server_info.get('primary', 'N/A')}")
            print(f"  Hosts: {len(server_info.get('hosts', []))} nodes")
        else:
            print(f"✓ MongoDB Connected (single instance, no replica set)")
            
    except Exception as e:
        print(f"✗ MongoDB connection failed: {e}")
        print(f"  WARNING: Application starting with MongoDB connection issues")
        all_healthy = False
    
    # Test Redis connection
    try:
        await redis_client.ping()
        info = await redis_client.info()
        redis_role = info.get('role', 'unknown')
        print(f"✓ Redis Connected - Role: {redis_role}")
        
        if redis_role == 'master':
            connected_slaves = info.get('connected_slaves', 0)
            print(f"  Connected replicas: {connected_slaves}")
        
    except Exception as e:
        print(f"✗ Redis connection failed: {e}")
        print(f"  WARNING: Application starting with Redis connection issues")
        all_healthy = False
    
    # Test PostgreSQL connection
    try:
        from repositories.postgres_text_repository import PostgresTextRepository
        pg_repo = PostgresTextRepository()
        print(f"✓ PostgreSQL Connected")
        print(f"  Host: {pg_repo.host}:{pg_repo.port}")
        print(f"  Base DB: {pg_repo.base_db}")
        
    except Exception as e:
        print(f"✗ PostgreSQL connection failed: {e}")
        print(f"  WARNING: Application starting with PostgreSQL connection issues")
        all_healthy = False
    
    print("-" * 70)
    
    if all_healthy:
        print("✓ All database connections healthy")
    else:
        print("⚠ Some database connections failed - check logs above")
    
    print("-" * 70)
    
    # Import websocket manager from pdf router to ensure WebSocket notifications work
    from routers.pdf import websocket_manager
    
    # PDF processing worker with Redis and WebSocket support
    pdf_processing_service = PDFProcessingService(
        redis_client=redis_client,
        websocket_manager=websocket_manager,
        max_workers=settings.REDIS_MAX_WORKERS
    )
    
    # Start the PDF processing worker
    asyncio.create_task(pdf_processing_service.process_queued_pdfs())
    
    print(f"✓ PDF processing worker started ({settings.REDIS_MAX_WORKERS} workers)")
    print(f"✓ WebSocket support enabled")
    print("=" * 70)
    print("🎉 Application startup complete!")
    print("=" * 70)

@app.on_event("shutdown")
async def shutdown_events():
    """Shutdown events for the application."""
    print("=" * 70)
    print("🛑 Shutting down Sahasra AI Education API")
    print("=" * 70)
    
    # Close Redis connection
    try:
        await redis_client.close()
        print("✓ Closed Redis connection")
    except Exception as e:
        print(f"✗ Error closing Redis: {e}")
    
    # Close PostgreSQL connection pools (if needed)
    try:
        from repositories.postgres_text_repository import PostgresTextRepository
        # Connection pools will be closed when objects are garbage collected
        print("✓ PostgreSQL connection pools marked for cleanup")
    except Exception as e:
        print(f"⚠ PostgreSQL cleanup warning: {e}")
    
    print("=" * 70)
    print("✓ Shutdown complete")
    print("=" * 70)

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







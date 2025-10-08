from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os

from .core.config import settings
from .core.logging import setup_logging, app_logger
from .repositories.db import init_mongodb
from .api.v1.routers import health, evaluate, feedback, benchmark, celery_evaluate

# Setup logging
setup_logging()

# Initialize FastAPI app
app = FastAPI(
    title="AI Assignment Checker",
    description="AI-powered assignment evaluation system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MongoDB on startup
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    app_logger.info("=" * 60)
    app_logger.info("AI Assignment Checker Starting")
    app_logger.info("=" * 60)
    
    # Initialize MongoDB
    if init_mongodb():
        app_logger.info("MongoDB initialized successfully")
    else:
        app_logger.warning("MongoDB initialization failed")

# Mount static files (only if directory exists)
static_dir = "app/static"
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    # Create empty static directory if it doesn't exist
    os.makedirs(static_dir, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize templates (only if directory exists)
templates_dir = "app/templates"
if os.path.exists(templates_dir):
    templates = Jinja2Templates(directory=templates_dir)
else:
    # Create empty templates directory if it doesn't exist
    os.makedirs(templates_dir, exist_ok=True)
    templates = Jinja2Templates(directory=templates_dir)

# Include routers
app.include_router(health.router)
app.include_router(evaluate.router)
app.include_router(evaluate.router_tokens)
app.include_router(feedback.router)
app.include_router(benchmark.router)
app.include_router(celery_evaluate.router)

# Legacy landing page (temporary)
@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    """Legacy landing page - will be replaced by React frontend."""
    return templates.TemplateResponse("index.html", {"request": request})

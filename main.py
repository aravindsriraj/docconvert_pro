from fastapi import FastAPI, UploadFile, Form, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
import uvicorn
from pathlib import Path
import aiofiles
import tempfile
import os
import logging
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, HttpUrl
from typing import Optional
import time

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Sentry
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN", ""),
    integrations=[FastApiIntegration()],
    traces_sample_rate=1.0,
    environment=os.getenv("ENVIRONMENT", "production"),
)

# Initialize metrics
REQUESTS = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
CONVERSION_TIME = Counter('conversion_time_seconds', 'Time spent converting documents')

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

app = FastAPI(
    title="Document Converter Pro",
    description="A professional document conversion service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create templates and static directories
templates_dir = Path("templates")
static_dir = Path("static")
templates_dir.mkdir(exist_ok=True)
static_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# API key validation
async def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
    if not api_key:
        return None
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

class ConversionRequest(BaseModel):
    url: HttpUrl
    
class ConversionResponse(BaseModel):
    markdown: str
    error: Optional[str] = None

def convert_document(file_path: str) -> str:
    start_time = time.time()
    try:
        pipeline_options = PdfPipelineOptions(do_table_structure=True)
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        pipeline_options.do_ocr = True

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        result = converter.convert(file_path)
        return result.document.export_to_markdown()
    finally:
        conversion_time = time.time() - start_time
        CONVERSION_TIME.inc(conversion_time)

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/", response_class=HTMLResponse)
@limiter.limit("60/minute")
async def home(request: Request):
    REQUESTS.labels(method='GET', endpoint='/', status=200).inc()
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/convert-url")
@limiter.limit("10/minute")
async def convert_url(request: Request, url: str = Form(...)):
    try:
        markdown_content = convert_document(url)
        REQUESTS.labels(method='POST', endpoint='/convert-url', status=200).inc()
        return {"markdown": markdown_content}
    except Exception as e:
        logger.error(f"Error converting URL: {e}", exc_info=True)
        REQUESTS.labels(method='POST', endpoint='/convert-url', status=500).inc()
        return {"error": str(e)}

@app.post("/convert-file")
@limiter.limit("10/minute")
async def convert_file(request: Request, file: UploadFile):
    if file.content_type != 'application/pdf':
        REQUESTS.labels(method='POST', endpoint='/convert-file', status=400).inc()
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            if len(content) > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(status_code=400, detail="File too large")
            temp_file.write(content)
            temp_file_path = temp_file.name

        markdown_content = convert_document(temp_file_path)
        os.unlink(temp_file_path)
        
        REQUESTS.labels(method='POST', endpoint='/convert-file', status=200).inc()
        return {"markdown": markdown_content}
    except Exception as e:
        logger.error(f"Error converting file: {e}", exc_info=True)
        REQUESTS.labels(method='POST', endpoint='/convert-file', status=500).inc()
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)
        return {"error": str(e)}

@app.get("/download-markdown")
@limiter.limit("60/minute")
async def download_markdown(request: Request, content: str):
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        response = FileResponse(
            temp_file_path,
            media_type='text/markdown',
            filename='converted_document.md'
        )
        
        REQUESTS.labels(method='GET', endpoint='/download-markdown', status=200).inc()
        return response
    except Exception as e:
        logger.error(f"Error downloading markdown: {e}", exc_info=True)
        REQUESTS.labels(method='GET', endpoint='/download-markdown', status=500).inc()
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)
        raise HTTPException(status_code=500, detail="Error creating download file")

@app.post("/api/convert", response_model=ConversionResponse)
@limiter.limit("10/minute")
async def api_convert_url(
    request: Request,
    conversion_request: ConversionRequest,
    api_key: str = Depends(verify_api_key)
):
    try:
        markdown_content = convert_document(str(conversion_request.url))
        REQUESTS.labels(method='POST', endpoint='/api/convert', status=200).inc()
        return ConversionResponse(markdown=markdown_content)
    except Exception as e:
        logger.error(f"API conversion error: {e}", exc_info=True)
        REQUESTS.labels(method='POST', endpoint='/api/convert', status=500).inc()
        return ConversionResponse(markdown="", error=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global error handler caught: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error. Please try again later."}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Header, Request, Body
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any, Union
import json
import redis.asyncio as redis
import httpx
from datetime import datetime

from config import settings
from models.pdf_models import PDFDocument, PDFUploadRequest, PDFLearnRequest
from services.pdf.pdf_service import PDFUploadService, PDFProcessingService, PDFQuestionGenerationService
from services.langchain.langchain_service import LangchainService
from routers.auth import auth_middleware

# Create router
router = APIRouter(prefix="/api/pdf", tags=["PDF"])

# Redis client
print(f"Redis URL: {settings.dict()}")
redis_client = redis.from_url(settings.REDIS_URL)

# Service instances
pdf_upload_service = PDFUploadService(redis_client=redis_client)
pdf_processing_service = PDFProcessingService(
    redis_client=redis_client,
    max_workers=settings.REDIS_MAX_WORKERS
)
pdf_question_service = PDFQuestionGenerationService()
langchain_service = LangchainService()


class UGJSONResponse(JSONResponse):
    """Custom JSON response for consistent formatting."""
    def __init__(self, data: Any, message: str = "Success", status_code: int = 200):
        content = {"status": status_code, "message": message, "data": data}
        super().__init__(content=content, status_code=status_code)


@router.post("/upload")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form(...),
    description: Optional[str] = Form(None),
    subject: Optional[str] = Form(None),
    grade: Optional[str] = Form(None),
    x_auth_session: Optional[str] = Header(None),
    user_id: str = Depends(auth_middleware)
):
    """Upload a PDF file for processing.
    
    Args:
        request: The request object
        file: PDF file to upload
        title: Title of the PDF
        description: Description of the PDF (optional)
        subject: Subject of the PDF (optional)
        grade: Grade level of the PDF (optional)
        x_auth_session: JWT token for authentication
        user_id: User ID extracted from JWT token
        
    Returns:
        UGJSONResponse with PDF document data
    """
    try:
        # Check file size
        file.file.seek(0, 2)  # Go to end of file
        file_size = file.file.tell()  # Get current position (file size)
        file.file.seek(0)  # Reset file position to beginning
        
        if file_size > settings.PDF_MAX_FILE_SIZE:
            return UGJSONResponse(
                data={},
                message=f"File is too large. Maximum size is {settings.PDF_MAX_FILE_SIZE / 1024 / 1024}MB",
                status_code=400
            )
        
        # Validate file extension
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in settings.PDF_UPLOAD_EXTENSIONS:
            return UGJSONResponse(
                data={},
                message=f"Invalid file format. Only {', '.join(settings.PDF_UPLOAD_EXTENSIONS)} files are allowed",
                status_code=400
            )
        
        # Create upload request model
        upload_request = PDFUploadRequest(
            title=title,
            description=description,
            subject=subject,
            grade=grade
        )
        
        # Upload the PDF
        pdf_document = await pdf_upload_service.upload_pdf(
            file=file,
            user_id=user_id,
            metadata=upload_request
        )
        
        return UGJSONResponse(
            data=pdf_document.model_dump_json(),
            message="PDF uploaded successfully and queued for processing"
        )
    except ValueError as e:
        return UGJSONResponse(
            data={},
            message=str(e),
            status_code=400
        )
    except Exception as e:
        return UGJSONResponse(
            data={},
            message=f"Error uploading PDF: {str(e)}",
            status_code=500
        )


@router.get("/list")
async def list_pdfs(
    request: Request,
    subject: Optional[str] = None,
    x_auth_session: Optional[str] = Header(None),
    user_id: str = Depends(auth_middleware)
):
    """Get all PDFs for the current user.
    
    Args:
        request: The request object
        subject: Optional subject to filter PDFs by
        x_auth_session: JWT token for authentication
        user_id: User ID extracted from JWT token
        
    Returns:
        UGJSONResponse with list of PDF documents
    """
    try:
        pdfs = pdf_upload_service.get_user_pdfs(user_id, subject)
        return UGJSONResponse(
            data=[pdf.model_dump_json() for pdf in pdfs],
            message="PDF documents retrieved successfully"
        )
    except Exception as e:
        return UGJSONResponse(
            data=[],
            message=f"Error retrieving PDFs: {str(e)}",
            status_code=500
        )


@router.get("/{pdf_id}")
async def get_pdf(
    pdf_id: str,
    request: Request,
    x_auth_session: Optional[str] = Header(None),
    user_id: str = Depends(auth_middleware)
):
    """Get a specific PDF document by ID.
    
    Args:
        pdf_id: ID of the PDF document
        request: The request object
        x_auth_session: JWT token for authentication
        user_id: User ID extracted from JWT token
        
    Returns:
        UGJSONResponse with PDF document data
    """
    try:
        pdf = pdf_upload_service.get_pdf(pdf_id)
        
        if not pdf:
            return UGJSONResponse(
                data={},
                message="PDF not found",
                status_code=404
            )
            
        # Check authorization
        if pdf.user_id != user_id:
            return UGJSONResponse(
                data={},
                message="You don't have permission to access this PDF",
                status_code=403
            )
            
        # Get processing status from Redis
        processing_status = await redis_client.hgetall(f"pdf_processing_status:{pdf_id}")
        
        # Combine with PDF data
        pdf_data = pdf.model_dump_json()
       
        print(f"PDF data: {pdf_data}")
        return UGJSONResponse(
            data=pdf_data,
            message="PDF document retrieved successfully"
        )
    except Exception as e:
        return UGJSONResponse(
            data={},
            message=f"Error retrieving PDF: {str(e)}",
            status_code=500
        )


@router.delete("/{pdf_id}")
async def delete_pdf(
    pdf_id: str,
    request: Request,
    x_auth_session: Optional[str] = Header(None),
    user_id: str = Depends(auth_middleware)
):
    """Delete a PDF document.
    
    Args:
        pdf_id: ID of the PDF document
        request: The request object
        x_auth_session: JWT token for authentication
        user_id: User ID extracted from JWT token
        
    Returns:
        UGJSONResponse with deletion status
    """
    try:
        success = pdf_upload_service.delete_pdf(pdf_id, user_id)
        
        if not success:
            return UGJSONResponse(
                data={},
                message="PDF not found",
                status_code=404
            )
        
        # Also clean up Redis keys related to this PDF
        await redis_client.delete(f"pdf_processing_status:{pdf_id}")
        await redis_client.delete(f"pdf_processing_errors:{pdf_id}")
            
        return UGJSONResponse(
            data={"deleted": True},
            message="PDF document deleted successfully"
        )
    except PermissionError as e:
        return UGJSONResponse(
            data={},
            message=str(e),
            status_code=403
        )
    except Exception as e:
        return UGJSONResponse(
            data={},
            message=f"Error deleting PDF: {str(e)}",
            status_code=500
        )


@router.post("/{pdf_id}/process")
async def process_pdf(
    pdf_id: str,
    request: Request,
    x_auth_session: Optional[str] = Header(None),
    user_id: str = Depends(auth_middleware)
):
    """Manually trigger processing for a PDF document.
    
    Args:
        pdf_id: ID of the PDF document
        request: The request object
        x_auth_session: JWT token for authentication
        user_id: User ID extracted from JWT token
        
    Returns:
        UGJSONResponse with processing status
    """
    try:
        # Get the PDF document
        pdf = pdf_upload_service.get_pdf(pdf_id)
        
        if not pdf:
            return UGJSONResponse(
                data={},
                message="PDF not found",
                status_code=404
            )
            
        # Check authorization
        if pdf.user_id != user_id:
            return UGJSONResponse(
                data={},
                message="You don't have permission to process this PDF",
                status_code=403
            )
            
        # Check if already processed
        if pdf.processing_status == "completed":
            return UGJSONResponse(
                data={"status": "completed"},
                message="PDF already processed"
            )
            
        # Process the PDF
        await pdf_processing_service.process_specific_pdf(pdf_id, user_id)
        
        return UGJSONResponse(
            data={"status": "processing"},
            message="PDF processing started"
        )
    except Exception as e:
        return UGJSONResponse(
            data={},
            message=f"Error processing PDF: {str(e)}",
            status_code=500
        )


@router.get("/{pdf_id}/status")
async def check_processing_status(
    pdf_id: str,
    request: Request,
    x_auth_session: Optional[str] = Header(None),
    user_id: str = Depends(auth_middleware)
):
    """Check the processing status of a PDF.
    
    Args:
        pdf_id: ID of the PDF document
        request: The request object
        x_auth_session: JWT token for authentication
        user_id: User ID extracted from JWT token
        
    Returns:
        UGJSONResponse with processing status
    """
    try:
        # Get the PDF document
        pdf = pdf_upload_service.get_pdf(pdf_id)
        
        if not pdf:
            return UGJSONResponse(
                data={},
                message="PDF not found",
                status_code=404
            )
            
        # Check authorization
        if pdf.user_id != user_id:
            return UGJSONResponse(
                data={},
                message="You don't have permission to access this PDF",
                status_code=403
            )
        
        # Get status from Redis
        status = await redis_client.hgetall(f"pdf_processing_status:{pdf_id}")
        errors = await redis_client.hgetall(f"pdf_processing_errors:{pdf_id}")
        print(f"Status: {status}")
        print(f"Errors: {errors}")
        
        # Prepare the response
        status_data = {
            "mongodb_status": pdf.processing_status,
            "started_at": str(pdf.process_start_time),
            "completed_at": str(pdf.process_end_time),
            "error": pdf.processing_error
        }
        
        # Add Redis status if available
        if status:
            status_data["redis_status"] = {
                k.decode('utf-8'): v.decode('utf-8') 
                for k, v in status.items()
            }
            
        if errors:
            status_data["redis_errors"] = {
                k.decode('utf-8'): v.decode('utf-8') 
                for k, v in errors.items()
            }
        print(f"Status data: {status_data}")
        return UGJSONResponse(
            data=status_data,
            message="Processing status retrieved successfully"
        )
    except Exception as e:
        return UGJSONResponse(
            data={},
            message=f"Error retrieving processing status: {str(e)}",
            status_code=500
        )


@router.post("/{pdf_id}/learn")
async def learn_from_pdf(
    pdf_id: str,
    request_body: PDFLearnRequest,
    request: Request,
    x_auth_session: Optional[str] = Header(None),
    user_id: str = Depends(auth_middleware)
):
    """Learn from a PDF by asking questions about its content.
    
    Args:
        pdf_id: ID of the PDF document
        request_body: Request body containing the question and optional parameters
        request: The request object
        x_auth_session: JWT token for authentication
        user_id: User ID extracted from JWT token
        
    Returns:
        UGJSONResponse with answer to the question
    """
    try:
        # Get the PDF document
        pdf = pdf_upload_service.get_pdf(pdf_id)
        
        if not pdf:
            return UGJSONResponse(
                data={},
                message="PDF not found",
                status_code=404
            )
            
        # Check authorization
        if pdf.user_id != user_id:
            return UGJSONResponse(
                data={},
                message="You don't have permission to learn from this PDF",
                status_code=403
            )
            
        # Check if PDF has been processed
        if pdf.processing_status != "completed":
            return UGJSONResponse(
                data={},
                message="PDF has not been fully processed yet. Please wait until processing is complete before learning from it.",
                status_code=400
            )
            
        # Get question and parameters from the request body model
        question = request_body.question
        similarity_threshold = request_body.similarity_threshold
        
        # Use LangchainService to answer the question
        response, status_code = await langchain_service.learn_from_pdf(
            student_id=user_id,
            pdf_id=pdf_id,
            question=question,
            session_id=x_auth_session,
            similarity_threshold=similarity_threshold or 0.75  # Use default if not provided
        )
        
        if status_code != 200:
            # Handle error case - response is a string message
            return UGJSONResponse(
                data={},
                message=response if isinstance(response, str) else "Error processing request",
                status_code=status_code
            )
        
        # Return the structured response directly
        return UGJSONResponse(
            data=response,
            message="Learning answer generated successfully"
        )
        
    except Exception as e:
        return UGJSONResponse(
            data={},
            message=f"Error learning from PDF: {str(e)}",
            status_code=500
        )


@router.post("/generate-questions")
async def generate_questions_from_pdf(
    file: UploadFile = File(...),
    subject: str = Form(...),
    topic: str = Form(...),
    subtopic: str = Form(...),
    request: Request = None,
    x_auth_session: Optional[str] = Header(None),
    user_id: str = Depends(auth_middleware)
):
    """Generate questions from a PDF using Gemini OCR and insert them into question bank.
    
    Args:
        file: PDF file to process
        subject: Subject for the questions
        topic: Topic for the questions
        subtopic: Subtopic for the questions
        request: The request object
        x_auth_session: JWT token for authentication
        user_id: User ID extracted from JWT token
        
    Returns:
        UGJSONResponse with generated questions data
    """
    try:
        # Validate file extension
        if not file.filename.lower().endswith('.pdf'):
            return UGJSONResponse(
                data={},
                message="Only PDF files are allowed",
                status_code=400
            )
        
        # Check file size
        file.file.seek(0, 2)  # Go to end of file
        file_size = file.file.tell()  # Get current position (file size)
        file.file.seek(0)  # Reset file position to beginning
        
        if file_size > settings.PDF_MAX_FILE_SIZE:
            return UGJSONResponse(
                data={},
                message=f"File is too large. Maximum size is {settings.PDF_MAX_FILE_SIZE / 1024 / 1024}MB",
                status_code=400
            )
        
        # Read the PDF file content
        pdf_content = await file.read()
        
        # Use the PDF question generation service
        result = await pdf_question_service.generate_questions_from_pdf(
            pdf_content=pdf_content,
            subject=subject,
            topic=topic,
            subtopic=subtopic
        )
        
        return UGJSONResponse(
            data=result,
            message=f"Successfully generated and inserted {result['total_questions_generated']} questions into question bank"
        )
        
    except Exception as e:
        return UGJSONResponse(
            data={},
            message=f"Error generating questions from PDF: {str(e)}",
            status_code=500
        ) 
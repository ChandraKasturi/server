from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Header, Request, Body, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any, Union
import json
import redis.asyncio as redis
import httpx
import asyncio
from datetime import datetime

from config import settings
from models.pdf_models import PDFDocument, PDFUploadRequest, PDFLearnRequest
from models.u_models import (
    PDFUploadResponse, PDFListResponse, PDFGetResponse, PDFDeleteResponse,
    PDFProcessingStatusResponse, PDFProcessingStartResponse, PDFLearnResponse,
    PDFQuestionGenerationResponse
)
from services.pdf.pdf_service import PDFUploadService, PDFProcessingService, PDFQuestionGenerationService
from services.langchain.langchain_service import LangchainService
from routers.auth import auth_middleware

# Create router
router = APIRouter(prefix="/api/pdf", tags=["PDF"])

# Redis client with connection parameters for replica set
print(f"Redis URL: {settings.dict()}")
redis_client = redis.from_url(
    settings.REDIS_URL,
    socket_connect_timeout=settings.REDIS_SOCKET_CONNECT_TIMEOUT,
    socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
    socket_keepalive=True,
    retry_on_timeout=True,
    health_check_interval=settings.REDIS_HEALTH_CHECK_INTERVAL
)


class WebSocketManager:
    """Manages WebSocket connections for PDF processing status updates."""
    
    def __init__(self):
        # Dictionary to store active connections: {pdf_id: [websocket_connections]}
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, pdf_id: str):
        """Connect a WebSocket for a specific PDF."""
        await websocket.accept()
        if pdf_id not in self.active_connections:
            self.active_connections[pdf_id] = []
        self.active_connections[pdf_id].append(websocket)
        print(f"WebSocket connected for PDF {pdf_id}. Total connections: {len(self.active_connections[pdf_id])}")
    
    def disconnect(self, websocket: WebSocket, pdf_id: str):
        """Disconnect a WebSocket for a specific PDF."""
        if pdf_id in self.active_connections:
            try:
                self.active_connections[pdf_id].remove(websocket)
                if not self.active_connections[pdf_id]:
                    # Remove the PDF entry if no connections left
                    del self.active_connections[pdf_id]
                print(f"WebSocket disconnected for PDF {pdf_id}")
            except ValueError:
                pass  # WebSocket was not in the list
    
    async def send_to_pdf_connections(self, pdf_id: str, message: dict):
        """Send a message to all WebSocket connections for a specific PDF."""
        if pdf_id not in self.active_connections:
            return
        
        # Create a copy of the list to avoid modification during iteration
        connections = self.active_connections[pdf_id].copy()
        disconnected = []
        
        for websocket in connections:
            try:
                await websocket.send_json(message)
            except Exception as e:
                print(f"Error sending WebSocket message: {str(e)}")
                disconnected.append(websocket)
        
        # Remove disconnected websockets
        for websocket in disconnected:
            self.disconnect(websocket, pdf_id)
    
    async def broadcast_to_all(self, message: dict):
        """Broadcast a message to all active WebSocket connections."""
        for pdf_id in list(self.active_connections.keys()):
            await self.send_to_pdf_connections(pdf_id, message)


# WebSocket manager instance
websocket_manager = WebSocketManager()

# Service instances
pdf_upload_service = PDFUploadService(redis_client=redis_client)
pdf_processing_service = PDFProcessingService(
    redis_client=redis_client,
    websocket_manager=websocket_manager,
    max_workers=settings.REDIS_MAX_WORKERS
)
pdf_question_service = PDFQuestionGenerationService()
langchain_service = LangchainService()


class UGJSONResponse(JSONResponse):
    """Custom JSON response for consistent formatting."""
    def __init__(self, data: Any, message: str = "Success", status_code: int = 200):
        content = {"status": status_code, "message": message, "data": data}
        super().__init__(content=content, status_code=status_code)


@router.post("/upload", response_model=PDFUploadResponse)
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


@router.get("/list", response_model=PDFListResponse)
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


@router.get("/{pdf_id}", response_model=PDFGetResponse)
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


@router.delete("/{pdf_id}", response_model=PDFDeleteResponse)
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


@router.post("/{pdf_id}/process", response_model=PDFProcessingStartResponse)
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


@router.get("/{pdf_id}/status", response_model=PDFProcessingStatusResponse)
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


@router.websocket("/{pdf_id}/status-ws")
async def websocket_processing_status(
    websocket: WebSocket,
    pdf_id: str,
    token: Optional[str] = None
):
    """WebSocket endpoint for real-time PDF processing status updates.
    
    Args:
        websocket: WebSocket connection
        pdf_id: ID of the PDF document
        token: JWT token for authentication (passed as query parameter)
    """
    try:
        # Authenticate using the token query parameter
        if not token:
            await websocket.close(code=4001, reason="Authentication token required")
            return
        
        # Verify the token and get user_id
        try:
            from services.auth.auth_service import AuthService
            auth_service = AuthService()
            user_id = auth_service.verify_token(token)
            if not user_id:
                await websocket.close(code=4001, reason="Invalid authentication token")
                return
        except Exception as e:
            await websocket.close(code=4001, reason="Authentication failed")
            return
        
        # Verify PDF ownership
        pdf = pdf_upload_service.get_pdf(pdf_id)
        if not pdf:
            await websocket.close(code=4004, reason="PDF not found")
            return
        
        if pdf.user_id != user_id:
            await websocket.close(code=4003, reason="You don't have permission to access this PDF")
            return
        
        # Connect the WebSocket
        await websocket_manager.connect(websocket, pdf_id)
        
        # Send initial status
        try:
            current_status = await redis_client.hgetall(f"pdf_processing_status:{pdf_id}")
            if current_status:
                status_data = {
                    k.decode('utf-8'): v.decode('utf-8') 
                    for k, v in current_status.items()
                }
                # Add MongoDB status for completeness
                status_data['mongodb_status'] = pdf.processing_status
                status_data['started_at'] = str(pdf.process_start_time)
                status_data['completed_at'] = str(pdf.process_end_time)
                status_data['error'] = pdf.processing_error
                
                await websocket.send_json({
                    "type": "status",
                    "data": status_data
                })
            else:
                # Send current MongoDB status if no Redis status exists
                initial_status = {
                    'mongodb_status': pdf.processing_status,
                    'started_at': str(pdf.process_start_time),
                    'completed_at': str(pdf.process_end_time),
                    'error': pdf.processing_error
                }
                await websocket.send_json({
                    "type": "status",
                    "data": initial_status
                })
        except Exception as e:
            print(f"Error sending initial status: {str(e)}")
        
        # Keep the connection alive and handle client messages
        try:
            while True:
                # Wait for client messages (can be used for heartbeat or commands)
                try:
                    data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                    
                    # Handle client messages
                    if data.get("type") == "ping":
                        await websocket.send_json({
                            "type": "pong",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    elif data.get("type") == "get_status":
                        # Client requesting current status
                        current_status = await redis_client.hgetall(f"pdf_processing_status:{pdf_id}")
                        if current_status:
                            status_data = {
                                k.decode('utf-8'): v.decode('utf-8') 
                                for k, v in current_status.items()
                            }
                            await websocket.send_json({
                                "type": "status",
                                "data": status_data
                            })
                
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    await websocket.send_json({
                        "type": "heartbeat",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    continue
                
        except WebSocketDisconnect:
            print(f"WebSocket disconnected for PDF {pdf_id}")
        except Exception as e:
            print(f"Error in WebSocket connection: {str(e)}")
        finally:
            # Clean up the connection
            websocket_manager.disconnect(websocket, pdf_id)
    
    except Exception as e:
        print(f"Error in WebSocket endpoint: {str(e)}")
        try:
            await websocket.close(code=4000, reason="Internal server error")
        except:
            pass


@router.websocket("/{pdf_id}/learn")
async def learn_from_pdf_websocket(
    websocket: WebSocket,
    pdf_id: str
):
    """WebSocket endpoint for streaming learning responses from PDF.
    
    Requires authentication via query parameter: ?x-auth-session=<jwt_token>
    
    Args:
        websocket: WebSocket connection
        pdf_id: ID of the PDF document to learn from
    """
    # Extract token from query parameters manually
    query_params = dict(websocket.query_params)
    token = query_params.get("x-auth-session", None)
    
    # Authenticate before accepting connection
    if not token:
        await websocket.close(code=1008, reason="No authentication token provided")
        return
    
    # Verify token and get student_id
    from services.auth.auth_service import AuthService
    auth_service = AuthService()
    student_id = auth_service.verify_token(token)
    
    if not student_id:
        await websocket.close(code=1008, reason="Invalid or expired token")
        return
    
    # Verify PDF exists and check authorization BEFORE accepting connection
    pdf = pdf_upload_service.get_pdf(pdf_id)
    
    if not pdf:
        await websocket.close(code=1008, reason="PDF not found")
        return
    
    if pdf.user_id != student_id:
        await websocket.close(code=1008, reason="You don't have permission to learn from this PDF")
        return
    
    if pdf.processing_status != "completed":
        await websocket.close(code=1008, reason="PDF has not been fully processed yet")
        return
    
    # Accept connection after successful authentication and validation
    await websocket.accept()
    
    try:
        # Receive initial message with question
        data = await websocket.receive_text()
        request_data = json.loads(data)
        
        # Extract parameters
        question = request_data.get("question")
        similarity_threshold = request_data.get("similarity_threshold", 0.75)
        
        # session_id can be provided in message or default to token
        session_id = request_data.get("session_id", token)
        
        # Validate required fields
        if not question:
            await websocket.send_json({
                "type": "error",
                "content": "Missing required field: question"
            })
            await websocket.close()
            return
        
        # Stream response using authenticated student_id
        async for chunk in langchain_service.learn_from_pdf_stream(
            student_id=student_id,
            pdf_id=pdf_id,
            question=question,
            session_id=session_id,
            similarity_threshold=similarity_threshold
        ):
            await websocket.send_json(chunk)
        
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for PDF: {pdf_id}, student: {student_id}")
    except json.JSONDecodeError as e:
        await websocket.send_json({
            "type": "error",
            "content": f"Invalid JSON format: {str(e)}"
        })
    except Exception as e:
        print(f"Error in PDF WebSocket for student {student_id}: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "content": f"Error in WebSocket connection: {str(e)}"
        })
    finally:
        try:
            await websocket.close()
        except:
            pass


@router.post("/generate-questions", response_model=PDFQuestionGenerationResponse)
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
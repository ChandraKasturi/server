from fastapi import APIRouter, Depends, Request, Header, Query, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional
from datetime import datetime, timedelta

from models.pdf_models import (SubjectLearnRequest, TTSRequest, LearningPDFUploadRequest, 
                              LearningPDFUploadResponse, LearningPDFProcessingStatus,
                              LearningImageUploadRequest, LearningImageUploadResponse)
from models.u_models import (LearnAnswerResponse, TTSVoicesResponse, LearningInfoResponse, 
                           FetchQuestionsRequest, FetchQuestionsResponse, 
                           UpdateQuestionRequest, UpdateQuestionResponse)
from services.learning.learning_service import LearningService
from services.learning.tts_service import TTSService
from routers.auth import auth_middleware
from utils.json_response import UGJSONResponse

# Create router
router = APIRouter(prefix="/api/learn", tags=["Learning"])

# Service instances
learning_service = LearningService()
tts_service = TTSService()

@router.post("/science", response_model=LearnAnswerResponse)
async def learn_science(
    request_body: SubjectLearnRequest,
    request: Request,
    x_auth_session: Optional[str] = Header(None),
    user_id: str = Depends(auth_middleware)
):
    """Learn about science by asking questions.
    
    Args:
        request_body: Request containing the question and options
        request: FastAPI request object
        x_auth_session: JWT token for authentication
        user_id: User ID extracted from JWT token
        
    Returns:
        UGJSONResponse with answer to the question
    """
    try:
        # Get question and options from request
        question = request_body.question
        include_pdfs = request_body.include_pdfs
        
        # Use service to get answer (now returns structured response with images)
        response_data, status_code = await learning_service.learn_science(
            question=question,
            student_id=user_id,
            session_id=x_auth_session,
            include_pdfs=include_pdfs
        )
        
        return UGJSONResponse(
            content=response_data,
            status_code=status_code
        )
        
    except Exception as e:
        return UGJSONResponse(
            content={"answer": f"Error learning science: {str(e)}"},
            status_code=500
        )

@router.post("/social-science", response_model=LearnAnswerResponse)
async def learn_social_science(
    request_body: SubjectLearnRequest,
    request: Request,
    x_auth_session: Optional[str] = Header(None),
    user_id: str = Depends(auth_middleware)
):
    """Learn about social science by asking questions.
    
    Args:
        request_body: Request containing the question and options
        request: FastAPI request object
        x_auth_session: JWT token for authentication
        user_id: User ID extracted from JWT token
        
    Returns:
        UGJSONResponse with answer to the question
    """
    try:
        # Get question and options from request
        question = request_body.question
        include_pdfs = request_body.include_pdfs
        
        # Use service to get answer (now returns structured response with images)
        response_data, status_code = await learning_service.learn_social_science(
            question=question,
            student_id=user_id,
            session_id=x_auth_session,
            include_pdfs=include_pdfs
        )
        
        return UGJSONResponse(
            content=response_data,
            status_code=status_code
        )
        
    except Exception as e:
        return UGJSONResponse(
            content={"answer": f"Error learning social science: {str(e)}"},
            status_code=500
        )

@router.post("/mathematics", response_model=LearnAnswerResponse)
async def learn_mathematics(
    request_body: SubjectLearnRequest,
    request: Request,
    x_auth_session: Optional[str] = Header(None),
    user_id: str = Depends(auth_middleware)
):
    """Learn about mathematics by asking questions.
    
    Args:
        request_body: Request containing the question and options
        request: FastAPI request object
        x_auth_session: JWT token for authentication
        user_id: User ID extracted from JWT token
        
    Returns:
        UGJSONResponse with answer to the question
    """
    try:
        # Get question and options from request
        question = request_body.question
        include_pdfs = request_body.include_pdfs
        
        # Use service to get answer (now returns structured response with images)
        response_data, status_code = await learning_service.learn_mathematics(
            question=question,
            student_id=user_id,
            session_id=x_auth_session,
            include_pdfs=include_pdfs
        )
        
        return UGJSONResponse(
            content=response_data,
            status_code=status_code
        )
        
    except Exception as e:
        return UGJSONResponse(
            content={"answer": f"Error learning mathematics: {str(e)}"},
            status_code=500
        )

@router.post("/english", response_model=LearnAnswerResponse)
async def learn_english(
    request_body: SubjectLearnRequest,
    request: Request,
    x_auth_session: Optional[str] = Header(None),
    user_id: str = Depends(auth_middleware)
):
    """Learn about English by asking questions.
    
    Args:
        request_body: Request containing the question and options
        request: FastAPI request object
        x_auth_session: JWT token for authentication
        user_id: User ID extracted from JWT token
        
    Returns:
        UGJSONResponse with answer to the question
    """
    try:
        # Get question and options from request
        question = request_body.question
        include_pdfs = request_body.include_pdfs
        
        # Use service to get answer (now returns structured response with images)
        response_data, status_code = await learning_service.learn_english(
            question=question,
            student_id=user_id,
            session_id=x_auth_session,
            include_pdfs=include_pdfs
        )
        
        return UGJSONResponse(
            content=response_data,
            status_code=status_code
        )
        
    except Exception as e:
        return UGJSONResponse(
            content={"answer": f"Error learning English: {str(e)}"},
            status_code=500
        )

@router.post("/hindi", response_model=LearnAnswerResponse)
async def learn_hindi(
    request_body: SubjectLearnRequest,
    request: Request,
    x_auth_session: Optional[str] = Header(None),
    user_id: str = Depends(auth_middleware)
):
    """Learn about Hindi by asking questions.
    
    Args:
        request_body: Request containing the question and options
        request: FastAPI request object
        x_auth_session: JWT token for authentication
        user_id: User ID extracted from JWT token
        
    Returns:
        UGJSONResponse with answer to the question
    """
    try:
        # Get question and options from request
        question = request_body.question
        include_pdfs = request_body.include_pdfs
        
        # Use service to get answer (now returns structured response with images)
        response_data, status_code = await learning_service.learn_hindi(
            question=question,
            student_id=user_id,
            session_id=x_auth_session,
            include_pdfs=include_pdfs
        )
        
        return UGJSONResponse(
            content=response_data,
            status_code=status_code
        )
        
    except Exception as e:
        return UGJSONResponse(
            content={"answer": f"Error learning Hindi: {str(e)}"},
            status_code=500
        )

@router.post("/tts/stream")
async def stream_tts_audio(
    request_body: TTSRequest,
    request: Request,
    x_auth_session: Optional[str] = Header(None),
    user_id: str = Depends(auth_middleware)
):
    """Stream text-to-speech audio chunks as WAV format.
    
    Args:
        request_body: Request containing text, voice, and speed options
        request: FastAPI request object
        x_auth_session: JWT token for authentication
        user_id: User ID extracted from JWT token
        
    Returns:
        StreamingResponse with WAV audio chunks
    """
    try:
        # Get parameters from request
        text = request_body.text
        voice = request_body.voice
        speed = request_body.speed
        
        # Validate text
        if not text or not text.strip():
            return UGJSONResponse(
                content={"answer": "Text is required for TTS conversion"},
                status_code=400
            )
        
        # Generate audio chunks using the TTS service
        async def generate_audio():
            try:
                async for chunk in tts_service.generate_audio_chunks(text, voice, speed):
                    yield chunk
            except Exception as e:
                print(f"Error generating TTS audio: {str(e)}")
                # Could yield an error chunk here if needed
        
        # Return streaming response with appropriate headers
        return StreamingResponse(
            generate_audio(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=tts_audio_24khz_mono.wav",
                "Access-Control-Expose-Headers": "Content-Disposition",
                "Cache-Control": "no-cache",
                "Transfer-Encoding": "chunked",
                "X-Audio-Format": "WAV 24kHz Mono PCM_S16LE"
            }
        )
        
    except Exception as e:
        return UGJSONResponse(
            content={"answer": f"Error streaming TTS audio: {str(e)}"},
            status_code=500
        )


@router.get("/tts/voices", response_model=TTSVoicesResponse)
async def get_available_voices(
    request: Request,
    x_auth_session: Optional[str] = Header(None),
    user_id: str = Depends(auth_middleware)
):
    """Get list of available TTS voices.
    
    Args:
        request: FastAPI request object
        x_auth_session: JWT token for authentication
        user_id: User ID extracted from JWT token
        
    Returns:
        UGJSONResponse with list of available voices
    """
    try:
        voices = tts_service.get_available_voices()
        
        return UGJSONResponse(
            content={"voices": voices},
            status_code=200
        )
        
    except Exception as e:
        return UGJSONResponse(
            content={"answer": f"Error retrieving voices: {str(e)}"},
            status_code=500
        ) 

@router.get("/learning-info", response_model=LearningInfoResponse)
async def get_learning_info(
    request: Request,
    subject: Optional[str] = Query(None, description="Optional subject to filter by (science, social_science, mathematics, english, hindi, or 'all' for all subjects)"),
    days_back: Optional[int] = Query(None, description="Number of days to look back for questions answered count (default: None for all time)"),
    x_auth_session: Optional[str] = Header(None),
    user_id: str = Depends(auth_middleware)
):
    """Get comprehensive learning information for a student.
    
    Returns:
    - Learning streak information (consecutive days of learning activity)
    - Questions answered count (AI responses) with breakdown by subject
    - Random educational quote to inspire learning
    
    Args:
        request: FastAPI request object
        subject: Optional subject filter (science, social_science, mathematics, english, hindi, or 'all')
        days_back: Number of days to look back for questions answered count
        x_auth_session: JWT token for authentication
        user_id: User ID extracted from JWT token
        
    Returns:
        UGJSONResponse with comprehensive learning information
    """
    try:
        # Calculate from_date for questions answered count
        from_date = None
        if days_back and days_back > 0:
            from_date = datetime.utcnow() - timedelta(days=days_back)
        
        # Normalize subject parameter
        if subject and subject.lower() == "all":
            subject = None
        elif subject:
            subject = subject.replace("-", "_").lower()
        
        # Get comprehensive learning info
        learning_info, status_code = await learning_service.get_learning_info(
            student_id=user_id,
            subject=subject,
            from_date=from_date
        )
        
        if status_code != 200:
            return UGJSONResponse(
                content=learning_info,
                status_code=status_code
            )
        
        return UGJSONResponse(
            content=learning_info,
            status_code=200
        )
        
    except Exception as e:
        return UGJSONResponse(
            content={
                "message": f"Error getting learning info: {str(e)}",
                "learning_streak": {
                    "current_streak": 0,
                    "last_activity_date": None,
                    "longest_streak": 0,
                    "total_active_days": 0
                },
                "questions_answered": {
                    "total_questions_answered": 0,
                    "by_subject": {}
                },
                "educational_quote": {
                    "quote": "Education is the passport to the future, for tomorrow belongs to those who prepare for it today.",
                    "author": "Malcolm X"
                },
                "generated_at": datetime.utcnow().isoformat(),
                "student_id": user_id
            },
            status_code=500
        )

@router.post("/questions/fetch", response_model=FetchQuestionsResponse)
async def fetch_questions_by_topic_subject(
    request_body: FetchQuestionsRequest,
    request: Request,
    x_auth_session: Optional[str] = Header(None),
    user_id: str = Depends(auth_middleware)
):
    """Fetch questions from question_bank collection by topic and subject.
    
    Args:
        request_body: Request containing subject, topic, and amount
        request: FastAPI request object
        x_auth_session: JWT token for authentication
        user_id: User ID extracted from JWT token
        
    Returns:
        UGJSONResponse with list of questions
    """
    try:
        # Get parameters from request
        subject = request_body.subject
        topic = request_body.topic

        # Use service to fetch questions
        result, status_code = await learning_service.fetch_questions_by_topic_subject(
            subject=subject,
            topic=topic
        )

        return UGJSONResponse(
            content=result,
            status_code=status_code
        )
        
    except Exception as e:
        return UGJSONResponse(
            content={"message": f"Error fetching questions: {str(e)}"},
            status_code=500
        )

@router.put("/questions/update", response_model=UpdateQuestionResponse)
async def update_question_document(
    request_body: UpdateQuestionRequest,
    request: Request,
    x_auth_session: Optional[str] = Header(None),
    user_id: str = Depends(auth_middleware)
):
    """Update an entire document in the question_bank collection.
    
    Args:
        request_body: Request containing the complete question document with _id
        request: FastAPI request object
        x_auth_session: JWT token for authentication
        user_id: User ID extracted from JWT token
        
    Returns:
        UGJSONResponse with update result
    """
    try:
        # Get question data from request
        question_data = request_body.question_data

        # Use service to update question document
        result, status_code = await learning_service.update_question_document(question_data)

        return UGJSONResponse(
            content=result,
            status_code=status_code
        )
        
    except Exception as e:
        return UGJSONResponse(
            content={
                "success": False,
                "message": f"Error updating question document: {str(e)}"
            },
            status_code=500
        )


@router.post("/test-upload")
async def test_upload(
    file: UploadFile = File(...),
    title: str = Form(...),
    subject: str = Form(...)
):
    """Test endpoint for multipart uploads"""
    try:
        # Simple test response
        return UGJSONResponse(
            content={
                "success": True,
                "message": "Test upload successful",
                "filename": file.filename,
                "content_type": file.content_type,
                "title": title,
                "subject": subject,
                "file_size": file.size if file.size else "unknown"
            },
            status_code=200
        )
    except Exception as e:
        return UGJSONResponse(
            content={
                "success": False,
                "message": f"Test upload failed: {str(e)}"
            },
            status_code=500
        )


@router.post("/upload-pdf-noauth")
async def upload_learning_pdf_noauth(
    file: UploadFile = File(...),
    title: str = Form(...), 
    subject: str = Form(...),
    description: Optional[str] = Form(None),
    topic: Optional[str] = Form(None),
    grade: Optional[str] = Form(None)
):
    """Upload PDF without authentication for testing"""
    try:
        # Mock user_id for testing
        user_id = "test_user_123"
        
        # Use the learning service to upload and process the PDF
        result, status_code = await learning_service.upload_learning_pdf(
            file=file,
            user_id=user_id,
            title=title,
            subject=subject,
            description=description,
            topic=topic,
            grade=grade
        )
        
        return UGJSONResponse(
            content=result,
            status_code=status_code
        )
        
    except Exception as e:
        return UGJSONResponse(
            content={
                "success": False,
                "message": f"Error uploading learning PDF: {str(e)}"
            },
            status_code=500
        )


@router.post("/upload-pdf")
async def upload_learning_pdf(
    file: UploadFile = File(...),
    title: str = Form(...), 
    subject: str = Form(...),
    description: Optional[str] = Form(None),
    topic: Optional[str] = Form(None),
    grade: Optional[str] = Form(None),
    x_auth_session: Optional[str] = Header(None),
    user_id: str = Depends(auth_middleware)
):
    """Upload and process a PDF for subject-specific learning with image extraction.
    
    This endpoint allows students to upload PDF documents that will be processed for learning purposes.
    The PDF will be:
    1. Text extracted using Gemini OCR
    2. Split into chunks and stored in subject-specific vector database
    3. Images extracted and captions generated using Gemini
    4. Image captions stored in vector database for retrieval
    
    Args:
        file: PDF file to upload (must be .pdf format)
        title: Title for the PDF document
        subject: Subject category (science, social_science, mathematics, english, hindi)
        description: Optional description of the PDF content
        topic: Optional topic within the subject
        grade: Optional grade level for the content
        x_auth_session: JWT token for authentication
        user_id: User ID extracted from JWT token
        
    Returns:
        UGJSONResponse with upload and processing results including:
        - pdf_id: Unique identifier for the uploaded PDF
        - processing_status: Status of the processing (completed/failed)
        - chunks_created: Number of text chunks created
        - images_extracted: Number of images extracted
        - file_size: Size of the uploaded file
        - message: Success or error message
    """
    try:
        # Early validation to prevent binary data processing issues
        if not file:
            return UGJSONResponse(
                content={
                    "success": False,
                    "message": "No file provided"
                },
                status_code=400
            )
        
        # Validate file is actually a PDF by checking content type and filename
        if not (file.filename and file.filename.lower().endswith('.pdf')):
            return UGJSONResponse(
                content={
                    "success": False,
                    "message": "File must be a PDF (.pdf extension required)"
                },
                status_code=400
            )
        
        # Validate content type if provided
        if file.content_type and not file.content_type.startswith(('application/pdf', 'application/octet-stream')):
            return UGJSONResponse(
                content={
                    "success": False,
                    "message": f"Invalid content type: {file.content_type}. Expected application/pdf"
                },
                status_code=400
            )
        
        # Validate required fields
        if not title or not title.strip():
            return UGJSONResponse(
                content={
                    "success": False,
                    "message": "Title is required and cannot be empty"
                },
                status_code=400
            )
        
        if not subject or not subject.strip():
            return UGJSONResponse(
                content={
                    "success": False,
                    "message": "Subject is required and cannot be empty"
                },
                status_code=400
            )
        
        # Validate file size (optional, adjust as needed)
        if file.size and file.size > 50 * 1024 * 1024:  # 50MB limit
            return UGJSONResponse(
                content={
                    "success": False,
                    "message": "File size too large. Maximum size is 50MB."
                },
                status_code=400
            )
        
        # Use the learning service to upload and process the PDF
        result, status_code = await learning_service.upload_learning_pdf(
            file=file,
            user_id=user_id,
            title=title,
            subject=subject,
            description=description,
            topic=topic,
            grade=grade
        )
        
        return UGJSONResponse(
            content=result,
            status_code=status_code
        )
        
    except Exception as e:
        return UGJSONResponse(
            content={
                "success": False,
                "message": f"Error uploading learning PDF: {str(e)}"
            },
            status_code=500
        )


@router.post("/upload-image", response_model=LearningImageUploadResponse)
async def upload_learning_image(
    file: UploadFile = File(...),
    caption: str = Form(...),
    subject: str = Form(...),
    topic: Optional[str] = Form(None),
    grade: Optional[str] = Form(None),
    page_number: Optional[int] = Form(None),
    x_auth_session: Optional[str] = Header(None),
    user_id: str = Depends(auth_middleware)
):
    """Upload an image with caption for learning purposes.
    
    This endpoint allows students to upload images with captions that will be stored
    in the subject-specific learning vector database for retrieval during learning sessions.
    
    Args:
        file: Image file to upload (jpg, jpeg, png, gif, webp, bmp)
        caption: Caption or description for the image
        subject: Subject category (science, social_science, mathematics, english, hindi)
        topic: Optional topic within the subject
        grade: Optional grade level for the content
        page_number: Optional page number reference if applicable
        x_auth_session: JWT token for authentication
        user_id: User ID extracted from JWT token
        
    Returns:
        UGJSONResponse with upload results including:
        - success: Whether the upload was successful
        - message: Success or error message
        - image_id: Unique identifier for the stored image
        - image_url: URL path to the stored image
        - subject: Subject category
        - caption: Image caption
        - upload_date: Upload timestamp
    """
    try:
        # Early validation to prevent binary data processing issues
        if not file:
            return UGJSONResponse(
                content={
                    "success": False,
                    "message": "No file provided"
                },
                status_code=400
            )
        
        # Validate required fields
        if not caption or not caption.strip():
            return UGJSONResponse(
                content={
                    "success": False,
                    "message": "Caption is required and cannot be empty"
                },
                status_code=400
            )
        
        if not subject or not subject.strip():
            return UGJSONResponse(
                content={
                    "success": False,
                    "message": "Subject is required and cannot be empty"
                },
                status_code=400
            )
        
        # Use the learning service to upload and process the image
        result, status_code = await learning_service.upload_learning_image(
            file=file,
            user_id=user_id,
            caption=caption,
            subject=subject,
            topic=topic,
            grade=grade,
            page_number=page_number
        )
        
        return UGJSONResponse(
            content=result,
            status_code=status_code
        )
        
    except Exception as e:
        return UGJSONResponse(
            content={
                "success": False,
                "message": f"Error uploading learning image: {str(e)}"
            },
            status_code=500
        )


# Generic endpoint for any subject - MUST BE LAST to avoid catching specific routes
@router.post("/{subject}", response_model=LearnAnswerResponse)
async def learn_subject(
    subject: str,
    request_body: SubjectLearnRequest,
    request: Request,
    x_auth_session: Optional[str] = Header(None),
    user_id: str = Depends(auth_middleware)
):
    """Learn about any subject by asking questions.
    
    Args:
        subject: Subject to learn about (science, social_science, mathematics, english, hindi)
        request_body: Request containing the question and options
        request: FastAPI request object
        x_auth_session: JWT token for authentication
        user_id: User ID extracted from JWT token
        
    Returns:
        UGJSONResponse with answer to the question
    """
    try:
        # Get question and options from request
        question = request_body.question
        include_pdfs = request_body.include_pdfs
        
        # Normalize the subject name
        subject = subject.replace("-", "_").lower()
        
        # Use service to get answer (now returns structured response with images)
        response_data, status_code = await learning_service.learn_subject(
            subject=subject,
            question=question,
            student_id=user_id,
            session_id=x_auth_session,
            include_pdfs=include_pdfs,
            include_images=True
        )
        
        return UGJSONResponse(
            content=response_data,
            status_code=status_code
        )
        
    except Exception as e:
        return UGJSONResponse(
            content={"answer": f"Error learning {subject}: {str(e)}"},
            status_code=500
        )
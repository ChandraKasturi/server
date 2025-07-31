from fastapi import APIRouter, Depends, Request, Header, Query
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional
from datetime import datetime, timedelta

from models.pdf_models import SubjectLearnRequest, TTSRequest
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
        
        # Use service to get answer
        answer, status_code = await learning_service.learn_science(
            question=question,
            student_id=user_id,
            session_id=x_auth_session,
            include_pdfs=include_pdfs
        )
        
        if status_code != 200:
            return UGJSONResponse(
                content={"answer": answer},
                status_code=status_code
            )
            
        return UGJSONResponse(
            content={"answer": answer},
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
        
        # Use service to get answer
        answer, status_code = await learning_service.learn_social_science(
            question=question,
            student_id=user_id,
            session_id=x_auth_session,
            include_pdfs=include_pdfs
        )
        
        if status_code != 200:
            return UGJSONResponse(
                content={"answer": answer},
                status_code=status_code
            )
            
        return UGJSONResponse(
            content={"answer": answer},
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
        
        # Use service to get answer
        answer, status_code = await learning_service.learn_mathematics(
            question=question,
            student_id=user_id,
            session_id=x_auth_session,
            include_pdfs=include_pdfs
        )
        
        if status_code != 200:
            return UGJSONResponse(
                content={"answer": answer},
                status_code=status_code
            )
            
        return UGJSONResponse(
            content={"answer": answer},
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
        
        # Use service to get answer
        answer, status_code = await learning_service.learn_english(
            question=question,
            student_id=user_id,
            session_id=x_auth_session,
            include_pdfs=include_pdfs
        )
        
        if status_code != 200:
            return UGJSONResponse(
                content={"answer": answer},
                status_code=status_code
            )
            
        return UGJSONResponse(
            content={"answer": answer},
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
        
        # Use service to get answer
        answer, status_code = await learning_service.learn_hindi(
            question=question,
            student_id=user_id,
            session_id=x_auth_session,
            include_pdfs=include_pdfs
        )
        
        if status_code != 200:
            return UGJSONResponse(
                content={"answer": answer},
                status_code=status_code
            )
            
        return UGJSONResponse(
            content={"answer": answer},
            status_code=status_code
        )
        
    except Exception as e:
        return UGJSONResponse(
            content={"answer": f"Error learning Hindi: {str(e)}"},
            status_code=500
        )

# Generic endpoint for any subject
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
        
        # Use service to get answer
        answer, status_code = await learning_service.learn_subject(
            subject=subject,
            question=question,
            student_id=user_id,
            session_id=x_auth_session,
            include_pdfs=include_pdfs
        )
        
        if status_code != 200:
            return UGJSONResponse(
                content={"answer": answer},
                status_code=status_code
            )
            
        return UGJSONResponse(
            content={"answer": answer},
            status_code=status_code
        )
        
    except Exception as e:
        return UGJSONResponse(
            content={"answer": f"Error learning {subject}: {str(e)}"},
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
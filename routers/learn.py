from fastapi import APIRouter, Depends, Request, Header
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional

from models.pdf_models import SubjectLearnRequest, TTSRequest
from services.learning.learning_service import LearningService
from services.learning.tts_service import TTSService
from routers.auth import auth_middleware
from utils.json_response import UGJSONResponse

# Create router
router = APIRouter(prefix="/api/learn", tags=["Learning"])

# Service instances
learning_service = LearningService()
tts_service = TTSService()

@router.post("/science")
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

@router.post("/social-science")
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
                data={},
                message=answer,
                status_code=status_code
            )
            
        return UGJSONResponse(
            data={"answer": answer},
            message="Social Science answer generated successfully"
        )
        
    except Exception as e:
        return UGJSONResponse(
            data={},
            message=f"Error learning social science: {str(e)}",
            status_code=500
        )

@router.post("/mathematics")
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
                data={},
                message=answer,
                status_code=status_code
            )
            
        return UGJSONResponse(
            data={"answer": answer},
            message="Mathematics answer generated successfully"
        )
        
    except Exception as e:
        return UGJSONResponse(
            data={},
            message=f"Error learning mathematics: {str(e)}",
            status_code=500
        )

@router.post("/english")
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
                content={},
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

@router.post("/hindi")
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
                data={},
                message=answer,
                status_code=status_code
            )
            
        return UGJSONResponse(
            data={"answer": answer},
            message="Hindi answer generated successfully"
        )
        
    except Exception as e:
        return UGJSONResponse(
            data={},
            message=f"Error learning Hindi: {str(e)}",
            status_code=500
        )

# Generic endpoint for any subject
@router.post("/{subject}")
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
                data={},
                message=answer,
                status_code=status_code
            )
            
        return UGJSONResponse(
            data={"answer": answer},
            message=f"{subject.capitalize()} answer generated successfully"
        )
        
    except Exception as e:
        return UGJSONResponse(
            data={},
            message=f"Error learning {subject}: {str(e)}",
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
                data={},
                message="Text is required for TTS conversion",
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
            data={},
            message=f"Error streaming TTS audio: {str(e)}",
            status_code=500
        )


@router.get("/tts/voices")
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
            data={"voices": voices},
            message="Available voices retrieved successfully"
        )
        
    except Exception as e:
        return UGJSONResponse(
            data={},
            message=f"Error retrieving voices: {str(e)}",
            status_code=500
        ) 
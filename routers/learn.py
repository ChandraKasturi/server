from fastapi import APIRouter, Depends, Request, Header
from fastapi.responses import JSONResponse
from typing import Optional

from models.pdf_models import SubjectLearnRequest
from services.learning.learning_service import LearningService
from routers.auth import auth_middleware
from utils.json_response import UGJSONResponse

# Create router
router = APIRouter(prefix="/api/learn", tags=["Learning"])

# Service instance
learning_service = LearningService()

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
        answer, status_code = learning_service.learn_science(
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
            message="Science answer generated successfully"
        )
        
    except Exception as e:
        return UGJSONResponse(
            data={},
            message=f"Error learning science: {str(e)}",
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
        answer, status_code = learning_service.learn_social_science(
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
        answer, status_code = learning_service.learn_mathematics(
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
        answer, status_code = learning_service.learn_english(
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
            message="English answer generated successfully"
        )
        
    except Exception as e:
        return UGJSONResponse(
            data={},
            message=f"Error learning English: {str(e)}",
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
        answer, status_code = learning_service.learn_hindi(
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
        answer, status_code = learning_service.learn_subject(
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
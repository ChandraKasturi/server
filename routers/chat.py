from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from models.u_models import (
    uTweet, uAnswer, uploadVectorUmodel, TranslateRequest,
    TweetResponse, AnswerResponse, TranslateResponse, UploadVectorResponse
)
from services.langchain.langchain_service import LangchainService
from routers.auth import auth_middleware

router = APIRouter(tags=["AI Chat"])

langchain_service = LangchainService()

@router.post("/tweet", response_model=TweetResponse)
def generate_tweet(body: uTweet, request: Request, student_id: str = Depends(auth_middleware)):
    """Generate a tweet about a topic.
    
    Args:
        body: Request containing tweet topic
        request: FastAPI request object to access the auth token
        student_id: ID of the student (from auth middleware)
        
    Returns:
        JSON response with generated tweet
    """
    # Get JWT token to use as session ID
    session_id = request.headers.get("X-Auth-Session")
    
    tweet, status_code = langchain_service.generate_tweet(body.tweet)
    
    return JSONResponse(content={"tweet": tweet}, status_code=status_code)

@router.post("/answer", response_model=AnswerResponse)
def get_answer(body: uAnswer, request: Request, student_id: str = Depends(auth_middleware)):
    """Get an answer to a question using AI.
    
    Args:
        body: Request containing the question
        request: FastAPI request object to access the auth token
        student_id: ID of the student (from auth middleware)
        
    Returns:
        JSON response with the answer
    """
    # Get JWT token to use as session ID
    session_id = request.headers.get("X-Auth-Session")
    
    answer, status_code = langchain_service.answer_question(
        body.question, 
        student_id,
        session_id
    )
    
    return JSONResponse(content={"answer": answer}, status_code=status_code)

@router.post("/translate", response_model=TranslateResponse)
def translate_text(body: TranslateRequest, request: Request, student_id: str = Depends(auth_middleware)):
    """Translate text to another language.
    
    Args:
        body: Request containing text and target language
        request: FastAPI request object to access the auth token
        student_id: ID of the student (from auth middleware)
        
    Returns:
        JSON response with translated text
    """
    text = body.text
    target_language = body.language
    
    # Get JWT token to use as session ID
    session_id = request.headers.get("X-Auth-Session")
    
    translated, status_code = langchain_service.translate_text(text, target_language)
    
    return JSONResponse(content={"translated": translated}, status_code=status_code)

@router.post("/uploadvector", response_model=UploadVectorResponse)
def upload_vector(body: uploadVectorUmodel, request: Request, student_id: str = Depends(auth_middleware)):
    """Upload text to vector storage.
    
    Args:
        body: Request containing text and subject
        request: FastAPI request object to access the auth token
        student_id: ID of the student (from auth middleware)
        
    Returns:
        JSON response with upload status
    """
    message, status_code = langchain_service.upload_vector(body.text, body.subject)
    
    return JSONResponse(content={"Message": message}, status_code=status_code) 
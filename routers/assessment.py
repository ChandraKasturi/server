from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from typing import Optional
from datetime import datetime

from models.u_models import AssessUmodel, FeedBackUmodel, uAnswer, PDFAssessmentRequest
from services.assessment.assessment_service import AssessmentService
from routers.auth import auth_middleware
from utils.json_response import UGJSONResponse
from repositories.mongo_repository import FeedbackRepository, HistoryRepository

router = APIRouter(tags=["Assessment"])

assessment_service = AssessmentService()
feedback_repo = FeedbackRepository()
history_repository = HistoryRepository()

@router.post("/assessment", response_class=UGJSONResponse)
def submit_assessment(body: AssessUmodel, request: Request, student_id: str = Depends(auth_middleware)):
    """Submit assessment answers for grading.
    
    Args:
        body: Assessment data including assessment ID and questions with answers
        request: FastAPI request object
        student_id: ID of the student (from auth middleware)
        
    Returns:
        JSON response with assessment results
    """
    result, status_code = assessment_service.submit_assessment(
        assessment_id=body.assessment_id,
        student_answers=body.questions,
        student_id=student_id
    )
    
    return UGJSONResponse(content=result, status_code=status_code)

@router.post("/api/chat", response_class=UGJSONResponse)
def check_assessment(body: uAnswer, request: Request, student_id: str = Depends(auth_middleware)):
    """Check if the input is an assessment request or a regular chat question.
    
    Args:
        body: Request containing the question
        request: FastAPI request object
        student_id: ID of the student (from auth middleware)
        
    Returns:
        JSON response with assessment results or answer
    """
    # Get session ID from JWT token
    session_id = request.headers.get("X-Auth-Session")
    
    # Extract question from the request body
    question = body.question
    
    result, status_code = assessment_service.check_assessment_content(
        question, 
        session_id,
        student_id
    )
    
    return UGJSONResponse(content=result, status_code=status_code)

@router.post("/generate-pdf-assessment", response_class=UGJSONResponse)
def generate_pdf_assessment(
    body: PDFAssessmentRequest, 
    request: Request, 
    student_id: str = Depends(auth_middleware)
):
    """Generate assessment questions from a PDF document.
    
    Args:
        body: Request containing PDF ID and assessment parameters
        request: FastAPI request object
        student_id: ID of the student (from auth middleware)
        
    Returns:
        JSON response with generated assessment
    """
    result, status_code = assessment_service.generate_assessment_from_pdf(
        body.pdf_id,
        student_id,
        body.question_type,
        body.num_questions
    )
    
    return UGJSONResponse(content=result, status_code=status_code)

@router.get("/assessments", response_class=UGJSONResponse)
def get_assessments(request: Request, student_id: str = Depends(auth_middleware), time: str = None):
    """Get all assessments for a student.
    
    Args:
        request: FastAPI request object
        student_id: ID of the student (from auth middleware)
        time: Optional time filter
        
    Returns:
        JSON response with list of assessments
    """
    assessments, status_code = assessment_service.get_assessments(student_id, time)
    
    return UGJSONResponse(content=assessments, status_code=status_code)

@router.get("/history", response_class=UGJSONResponse)
def get_history(request: Request, student_id: str = Depends(auth_middleware), time: str = None):
    """Get assessment history for a student.
    
    Args:
        request: FastAPI request object
        student_id: ID of the student (from auth middleware)
        time: Optional time filter
        
    Returns:
        JSON response with assessment history
    """
    # Get session ID from JWT token
    session_id = request.headers.get("X-Auth-Session")
    
    history, status_code = assessment_service.get_history(student_id, time)
    
    return UGJSONResponse(content=history, status_code=status_code)

@router.get("/assessment/{assessment_id}", response_class=UGJSONResponse)
def get_assessment_by_id(assessment_id: str, request: Request, student_id: str = Depends(auth_middleware)):
    """Get a specific assessment by ID.
    
    Args:
        assessment_id: ID of the assessment to retrieve
        request: FastAPI request object
        student_id: ID of the student (from auth middleware)
        
    Returns:
        JSON response with assessment details
    """
    assessment, status_code = assessment_service.get_assessment_by_id(student_id, assessment_id)
    
    return UGJSONResponse(content=assessment, status_code=status_code)

@router.get("/pdf-assessments/{pdf_id}", response_class=UGJSONResponse)
def get_pdf_assessments(
    pdf_id: str, 
    request: Request, 
    student_id: str = Depends(auth_middleware)
):
    """Get all assessments for a specific PDF.
    
    Args:
        pdf_id: ID of the PDF
        request: FastAPI request object
        student_id: ID of the student (from auth middleware)
        
    Returns:
        JSON response with list of assessments for the PDF
    """
    # Get assessments and filter by PDF ID
    all_assessments, status_code = assessment_service.get_assessments(student_id)
    
    if status_code != 200:
        return UGJSONResponse(content=all_assessments, status_code=status_code)
    
    # Filter assessments for the specified PDF
    pdf_assessments = [
        assessment for assessment in all_assessments 
        if assessment.get("pdf_id") == pdf_id
    ]
    
    return UGJSONResponse(
        content={
            "pdf_id": pdf_id,
            "assessments": pdf_assessments,
            "count": len(pdf_assessments)
        }, 
        status_code=200
    )

@router.post("/feedback", response_class=UGJSONResponse)
def add_feedback(body: FeedBackUmodel, request: Request, student_id: str = Depends(auth_middleware)):
    """Add feedback from a student.
    
    Args:
        body: Feedback data
        request: FastAPI request object
        student_id: ID of the student (from auth middleware)
        
    Returns:
        JSON response with feedback submission status
    """
    success = feedback_repo.add_feedback(student_id, body.feedback)
    
    if success:
        return UGJSONResponse(content={"Message": "Feedback Received Successfully"}, status_code=200)
    else:
        return UGJSONResponse(content={"Message": "Something Went Wrong"}, status_code=400)

@router.get("/progress")
def get_progress():
    """Get progress information (placeholder endpoint).
    
    Returns:
        JSON response with progress data
    """
    # This is a placeholder that returns static data
    # In a real implementation, this would fetch actual progress data
    return JSONResponse(
        content={
            "Physics": {"Lesson1": "Beginner", "Lesson2": "Beginner", "Lesson3": "Beginner", "Overall": "Beginner"},
            "Biology": {"Lesson1": "Beginner", "Lesson2": "Beginner", "Lesson3": "Beginner", "Overall": "Beginner"},
            "Chemistry": {"Lesson1": "Beginner", "Lesson2": "Beginner", "Lesson3": "Beginner", "Overall": "Beginner"}
        },
        status_code=200
    )

@router.get("/assessment-submissions", response_class=UGJSONResponse)
def get_assessment_submissions(request: Request, student_id: str = Depends(auth_middleware)):
    """Get all assessment submissions for a student.
    
    Args:
        request: FastAPI request object
        student_id: ID of the student (from auth middleware)
        
    Returns:
        JSON response with list of assessment submissions
    """
    # Get all assessments that have submissions
    assessments, status_code = assessment_service.get_assessments(student_id)
    
    if status_code != 200:
        return UGJSONResponse(content=assessments, status_code=status_code)
    
    # Filter assessments that have submissions
    submissions = []
    for assessment in assessments:
        if "last_submission" in assessment:
            submission = assessment["last_submission"]
            submission["assessment_title"] = assessment.get("title", "")
            submission["assessment_type"] = assessment.get("question_type", "")
            submission["pdf_id"] = assessment.get("pdf_id", "")
            submissions.append(submission)
    
    return UGJSONResponse(
        content={
            "submissions": submissions,
            "count": len(submissions)
        }, 
        status_code=200
    )

@router.get("/assessment-submission/{assessment_id}", response_class=UGJSONResponse)
def get_assessment_submission(
    assessment_id: str, 
    request: Request, 
    student_id: str = Depends(auth_middleware)
):
    """Get a specific assessment submission by ID.
    
    Args:
        assessment_id: ID of the assessment
        request: FastAPI request object
        student_id: ID of the student (from auth middleware)
        
    Returns:
        JSON response with assessment submission details
    """
    # Get the assessment
    assessment, status_code = assessment_service.get_assessment_by_id(student_id, assessment_id)
    
    if status_code != 200:
        return UGJSONResponse(content=assessment, status_code=status_code)
    
    # Check if assessment has a submission
    if "last_submission" not in assessment:
        return UGJSONResponse(
            content={"Message": "No submission found for this assessment"}, 
            status_code=404
        )
    
    # Get the submission
    submission = assessment["last_submission"]
    submission["assessment_title"] = assessment.get("title", "")
    submission["assessment_type"] = assessment.get("question_type", "")
    submission["pdf_id"] = assessment.get("pdf_id", "")
    
    return UGJSONResponse(
        content=submission, 
        status_code=200
    )

@router.get("/get-history/{subject}")
def get_subject_history(
    subject: str,
    request: Request, 
    student_id: str = Depends(auth_middleware),
    time: Optional[str] = None
):
    """Get learning history for a specific subject.
    
    Args:
        subject: The subject to get history for (science, social_science, mathematics, english, hindi)
        request: FastAPI request object
        student_id: ID of the student (from auth middleware)
        time: Optional time filter in ISO format (YYYY-MM-DDTHH:MM:SS)
        
    Returns:
        JSON response with subject history
    """
    try:
        # Convert time string to datetime if provided
        from_date = None
        if time:
            try:
                from_date = datetime.fromisoformat(time)
            except ValueError:
                return UGJSONResponse(
                    content={"Message": "Invalid time format. Use ISO format (YYYY-MM-DDTHH:MM:SS)"},
                    status_code=400
                )
        
        # Get all history items
        history_items = history_repository.get_history(student_id, from_date)
        
        # Filter by subject if specified
        if subject.lower() != "all":
            # Normalize subject name (replace hyphens with underscores)
            normalized_subject = subject.replace("-", "_").lower()
            
            # Filter history items by subject
            subject_history = [
                item for item in history_items
                if item.get("subject", "").lower() == normalized_subject
            ]
        else:
            # Return all subjects if "all" is specified
            subject_history = history_items
        
        # Format history items for response
        formatted_history = []
        for item in subject_history:
            formatted_item = {
                "subject": item.get("subject", "unknown"),
                "message": item.get("message", ""),
                "is_ai": item.get("is_ai", False),
                "time": item.get("time", datetime.utcnow()).isoformat(),
                "session_id": item.get("session_id", "")
            }
            formatted_history.append(formatted_item)
        
        # Sort by time (most recent first)
        formatted_history.sort(key=lambda x: x["time"], reverse=True)
        
        return UGJSONResponse(
            content={
                "subject": subject,
                "history": formatted_history,
                "count": len(formatted_history)
            },
            status_code=200
        )
    
    except Exception as e:
        return UGJSONResponse(
            content={"Message": f"Error getting history: {str(e)}"},
            status_code=500
        ) 
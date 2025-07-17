from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from typing import Optional, List
from datetime import datetime

from models.u_models import (
    AssessUmodel, FeedBackUmodel, uAnswer, PDFAssessmentRequest, AssessmentRequest,
    AssessmentGenerationResponse, AssessmentSubmissionResponse, AssessmentListResponse,
    AssessmentByIdResponse, FeedbackResponse, ProgressResponse, SubjectProgress,
    AssessmentSubmissionsListResponse, AssessmentPDFListResponse, SubjectHistoryResponse,
    AssessmentHistoryItem, AssessmentSubmissionItem, HistoryDatesResponse, HistoryDatesData
)
from services.assessment.assessment_service import AssessmentService
from routers.auth import auth_middleware
from utils.json_response import UGJSONResponse
from repositories.mongo_repository import FeedbackRepository, HistoryRepository

router = APIRouter(tags=["Assessment"])

assessment_service = AssessmentService()
feedback_repo = FeedbackRepository()
history_repository = HistoryRepository()

@router.post("/assessment", response_model=AssessmentSubmissionResponse)
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

@router.post("/api/assessment", response_model=AssessmentGenerationResponse)
def generate_assessment(body: AssessmentRequest, request: Request, student_id: str = Depends(auth_middleware)):
    """Generate assessment questions based on provided parameters.
    
    Args:
        body: Request containing subject, topics, subtopic, level, and number of questions
        request: FastAPI request object
        student_id: ID of the student (from auth middleware)
        
    Returns:
        JSON response with generated assessment
    """
    # Get session ID from JWT token
    session_id = request.headers.get("X-Auth-Session")
    
    result, status_code = assessment_service.check_assessment_content(
        subject=body.subject,
        topic=body.topic,  # Keep for backward compatibility
        topics=body.topics,  # New field for list of topics
        subtopic=body.subtopic,
        level=body.level,
        num_questions=body.number_of_questions,
        question_types=body.question_types,
        session_id=session_id,
        student_id=student_id
    )
    
    return UGJSONResponse(content=result, status_code=status_code)

@router.post("/generate-pdf-assessment", response_model=AssessmentGenerationResponse)
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
        body.question_types,
        body.num_questions
    )
    
    return UGJSONResponse(content=result, status_code=status_code)

@router.get("/assessments", response_model=List[AssessmentHistoryItem])
def get_assessments(request: Request, student_id: str = Depends(auth_middleware), time: str = None, subject: str = None, topic: str = None):
    """Get all assessments for a student.
    
    Args:
        request: FastAPI request object
        student_id: ID of the student (from auth middleware)
        time: Optional time filter
        subject: Optional subject filter
        topic: Optional topic filter to find assessments containing this topic
        
    Returns:
        JSON response with list of assessments
    """
    assessments, status_code = assessment_service.get_assessments(student_id, time, subject, topic)
    
    return UGJSONResponse(content=assessments, status_code=status_code)

@router.get("/history", response_model=List[AssessmentHistoryItem])
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

@router.get("/assessment/{assessment_id}", response_model=AssessmentHistoryItem)
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

@router.get("/pdf-assessments/{pdf_id}", response_model=AssessmentPDFListResponse)
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

@router.post("/feedback", response_model=FeedbackResponse)
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

@router.get("/progress", response_model=ProgressResponse)
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

@router.get("/assessment-submissions", response_model=AssessmentSubmissionsListResponse)
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

@router.get("/assessment-submission/{assessment_id}", response_model=AssessmentSubmissionItem)
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

@router.get("/get-history/{subject}", response_model=SubjectHistoryResponse)
def get_subject_history(
    subject: str,
    request: Request, 
    student_id: str = Depends(auth_middleware),
    time: Optional[str] = None,
    page: int = 1,
    oldest_first: bool = False
):
    """Get learning history for a specific subject with pagination.
    
    Args:
        subject: The subject to get history for (science, social_science, mathematics, english, hindi)
        request: FastAPI request object
        student_id: ID of the student (from auth middleware)
        time: Optional time filter in ISO format (YYYY-MM-DDTHH:MM:SS)
        page: Page number for pagination (default: 1, each page contains 10 items)
        oldest_first: If True, return oldest messages first; if False, return newest first (default: False)
        
    Returns:
        JSON response with subject history
    """
    try:
        # Validate page parameter
        if page < 1:
            return UGJSONResponse(
                content={"Message": "Page number must be 1 or greater"},
                status_code=400
            )
        
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
        
        # Get paginated history items with subject filtering at database level
        history_items = history_repository.get_history(
            student_id, 
            from_date, 
            page, 
            page_size=10, 
            oldest_first=oldest_first, 
            subject=subject
        )
        
        # Format history items for response (no need to filter by subject anymore)
        formatted_history = []
        for item in history_items:
            formatted_item = {
                "subject": item.get("subject", "unknown"),
                "message": item.get("message", ""),
                "is_ai": item.get("is_ai", False),
                "time": item.get("time", datetime.utcnow()).isoformat(),
                "session_id": item.get("session_id", "")
            }
            formatted_history.append(formatted_item)
        
        # Note: We don't sort here again since the repository already returns sorted data
        
        return UGJSONResponse(
            content={
                "subject": subject,
                "history": formatted_history,
                "count": len(formatted_history),
                "page": page,
                "page_size": 10,
                "oldest_first": oldest_first,
                "has_more": len(history_items) == 10  # If we got a full page, there might be more
            },
            status_code=200
        )
    
    except Exception as e:
        return UGJSONResponse(
            content={"Message": f"Error getting history: {str(e)}"},
            status_code=500
        )

@router.get("/history-dates", response_model=HistoryDatesData)
def get_history_dates(
    request: Request,
    student_id: str = Depends(auth_middleware),
    date: str = None
):
    """Get available dates where messages exist in history.
    
    Args:
        request: FastAPI request object
        student_id: ID of the student (from auth middleware)
        date: Date to search from in ISO format (YYYY-MM-DD). If not provided, uses current date.
        
    Returns:
        JSON response with list of available dates (up to 5 most recent)
    """
    try:
        # Parse the input date or use current date
        if date:
            try:
                # Parse date string (YYYY-MM-DD format)
                search_date = datetime.fromisoformat(date)
            except ValueError:
                return UGJSONResponse(
                    content={"Message": "Invalid date format. Use YYYY-MM-DD format"},
                    status_code=400
                )
        else:
            # Use current date if no date provided
            search_date = datetime.utcnow()
        
        # Get available dates from the repository
        available_dates = history_repository.get_available_dates(
            student_id=student_id,
            from_date=search_date,
            limit=5
        )
        
        return UGJSONResponse(
            content={
                "search_date": search_date.strftime("%Y-%m-%d"),
                "available_dates": available_dates,
                "count": len(available_dates)
            },
            status_code=200
        )
        
    except Exception as e:
        return UGJSONResponse(
            content={"Message": f"Error getting available dates: {str(e)}"},
            status_code=500
        ) 
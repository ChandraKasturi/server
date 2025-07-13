from pydantic import BaseModel,RootModel,Field
from typing import List,Dict,Optional,Union
from datetime import datetime


class uTweet(BaseModel):
	tweet:str


class uploadVectorUmodel(BaseModel):
	text:str
	subject:str

class uAnswer(BaseModel):
	question:str

class ucorrect(BaseModel):
	ucgrammar:dict

class loginUmodel(BaseModel):
	mobilenumberoremail:str
	password:str


class registerUmodel(BaseModel):
	phonenumber:str
	email:str

class confirmRegisterUmodel(BaseModel):
	email:str
	username:str
	password:str
	mobilenumber:str
	Class: str
	educationboard: str
	token:str


class UpdatePasswordUmodel(BaseModel):
	password:str
	token:str


class ForgotPasswordUmodel(BaseModel):
	mobilenumberoremail:str



class QuestionUmodel(BaseModel):
    Class: str
    subject: str
    topic: str
    subtopic: Optional[str] = None
    question: str
    option1: str
    option2: str
    option3: str
    option4: str
    correctanswer: Optional[str] = None
    level: str
    explaination: str
    explainationimage: Optional[str] = None 
    questionimage: Optional[str]  = None
    questionset: Optional[str]  = None
    schooldid: Optional[str]  = None
    qsetboard: Optional[str]  = None
    qsetdescription: Optional[str]  = None  
    marks: str  
    descriptiveanswer: Optional[str]   = None


class questionAssessUmodel(BaseModel):
	questionid: str
	studentanswer: Optional[str] = None
	question_type: Optional[str] = None  # MCQ, DESCRIPTIVE, FILL_BLANKS


class AssessUmodel(BaseModel):
	assessment_id: str
	questions: List[questionAssessUmodel]

class ProfileUmodel(BaseModel):
	email:str | None
	username:str | None
	mobilenumber:str | None
	Class: str | None
	educationboard: str | None
	bio: str | None

class FeedBackUmodel(BaseModel):
	feedback:str
	
class PDFAssessmentRequest(BaseModel):
    """Request model for generating assessments from PDF documents."""
    pdf_id: str
    question_types: Optional[List[str]] = None  # List of types: ["MCQ", "DESCRIPTIVE", "FILL_BLANKS", "TRUEFALSE"]
    num_questions: Optional[int] = 10
    
class AssessmentResultQuestion(BaseModel):
    """Individual question result in an assessment submission."""
    questionid: str
    is_correct: bool
    feedback: str
    
class AssessmentResult(BaseModel):
    """Response model for assessment submission results."""
    results: List[AssessmentResultQuestion]
    correct_count: int
    total_questions: int
    score_percentage: float
    
class AssessmentSubmissionResult(BaseModel):
    """Comprehensive response model for assessment submissions."""
    assessment_id: str
    student_id: str
    submission_time: datetime
    results: List[AssessmentResultQuestion]
    correct_count: int
    total_questions: int
    score_percentage: float

class AssessmentRequest(BaseModel):
    """Request model for generating assessments."""
    subject: str
    topic: Optional[str] = None  # Legacy field, kept for backward compatibility
    topics: Optional[List[str]] = None  # New field for multiple topics
    subtopic: Optional[str] = None
    question_types: Optional[List[str]] = None  # List of types: ["MCQ", "DESCRIPTIVE", "FILL_BLANKS", "TRUEFALSE"]
    number_of_questions: Optional[int] = 5
    level: Optional[int] = 1

# ========== EXACT RESPONSE SCHEMAS ==========

# MongoDB ObjectId structure
class MongoObjectId(BaseModel):
    """MongoDB ObjectId structure"""
    oid: str = Field(alias="$oid")

# PDF UGJSONResponse wrapper (matches UGJSONResponse format)
class UGJSONResponseModel(BaseModel):
    """Wrapper for UGJSONResponse format: {"status": int, "message": str, "data": any}"""
    status: int
    message: str
    data: Optional[BaseModel] = None  # Will be overridden in specific responses

# AUTH RESPONSES (matching JSONResponse content)
class AuthMessageResponse(BaseModel):
    """Standard auth response with Message field"""
    Message: str

# ASSESSMENT RESPONSES (matching service returns)
class AssessmentQuestionMCQ(BaseModel):
    """MCQ question from MongoDB sahasra_questions collection"""
    question: str
    option1: str
    option2: str
    option3: str
    option4: str
    correctanswer: str
    explaination: str
    question_type: str
    subject: str
    topic: str
    subtopic: str
    level: str
    questionset: str
    marks: str
    created_at: str
    id: str
    # Student response fields (when submitted)
    student_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    feedback: Optional[str] = None

class AssessmentQuestionShortAnswer(BaseModel):
    """Short answer question from MongoDB"""
    question: str
    model_answer: str
    grading_criteria: str
    question_type: str
    explaination: str
    expected_length: str
    subject: str
    topic: str
    subtopic: str
    level: str
    questionset: str
    marks: str
    created_at: str
    id: str
    # Student response fields (when submitted)
    student_answer: Optional[str] = None
    score: Optional[int] = None
    is_correct: Optional[bool] = None
    feedback: Optional[str] = None

class AssessmentQuestionVeryShortAnswer(BaseModel):
    """Very short answer question from MongoDB"""
    question: str
    model_answer: str
    grading_criteria: str
    question_type: str
    explaination: str
    expected_length: str
    subject: str
    topic: str
    subtopic: str
    level: str
    questionset: str
    marks: str
    created_at: str
    id: str
    # Student response fields (when submitted)
    student_answer: Optional[str] = None
    score: Optional[int] = None
    is_correct: Optional[bool] = None
    feedback: Optional[str] = None

# Union type for any question type
AssessmentQuestion = Union[AssessmentQuestionMCQ, AssessmentQuestionShortAnswer, AssessmentQuestionVeryShortAnswer]

class AssessmentGenerationResponse(BaseModel):
    """Response from check_assessment_content service"""
    message: str
    assessment_id: str
    questions: List[AssessmentQuestion]

class AssessmentSubmissionResultItem(BaseModel):
    """Individual result item from assessment submission"""
    questionid: str
    is_correct: bool
    feedback: str

class AssessmentSubmissionResponse(BaseModel):
    """Response from submit_assessment service"""
    results: List[AssessmentSubmissionResultItem]
    correct_count: int
    total_questions: int
    score_percentage: float

class AssessmentLastSubmission(BaseModel):
    """Last submission data from MongoDB assessments"""
    assessment_id: str
    student_id: str
    submission_time: datetime
    results: List[AssessmentSubmissionResultItem]
    correct_count: int
    total_questions: int
    score_percentage: float

class AssessmentHistoryItem(BaseModel):
    """Individual assessment from get_assessments"""
    id: MongoObjectId = Field(alias="_id")
    questions: List[AssessmentQuestion]
    student_id: str
    timestamp: datetime
    session_id: str
    question_types: List[str]
    subject: str
    topics: List[str]
    level: int
    created_at: datetime
    date: datetime
    last_submission: Optional[AssessmentLastSubmission] = None
    last_submission_time: Optional[datetime] = None
    submission_count: Optional[int] = None

class AssessmentListResponse(BaseModel):
    """Response from get_assessments service - returns List[AssessmentHistoryItem]"""
    assessments: List[AssessmentHistoryItem]

class AssessmentByIdResponse(BaseModel):
    """Response from get_assessment_by_id service - returns single AssessmentHistoryItem"""
    assessment: AssessmentHistoryItem

# LEARNING RESPONSES (matching service returns)
class LearnAnswerResponse(BaseModel):
    """Response from learning services - returns answer string"""
    answer: str

class TTSVoicesResponse(BaseModel):
    """Response from TTS voices endpoint"""
    voices: List[str]

# PROFILE RESPONSES (matching service returns)
class ProfileData(BaseModel):
    """User profile data from MongoDB sahasra_users collection"""
    email: str
    username: str
    password: str  # Filtered out in service
    mobilenumber: str
    Class: str
    educationboard: str
    token: str  # Filtered out in service
    student_id: str

class ProfileGetResponse(BaseModel):
    """Response from get_profile service - filtered user data without password/token"""
    email: str
    username: str
    mobilenumber: str
    Class: str
    educationboard: str
    student_id: str

class ProfileUpdateResponse(BaseModel):
    """Response from update_profile service"""
    Message: str

class ProfileImageUpdateResponse(BaseModel):
    """Response from update_profile_image service"""
    Message: str
    image_url: str

# CHAT RESPONSES (matching JSONResponse content)
class TweetResponse(BaseModel):
    """Response from generate_tweet"""
    tweet: str

class AnswerResponse(BaseModel):
    """Response from get_answer"""
    answer: str

class TranslateResponse(BaseModel):
    """Response from translate_text"""
    translated: str

class UploadVectorResponse(BaseModel):
    """Response from upload_vector"""
    Message: str

# Request model for translate endpoint
class TranslateRequest(BaseModel):
    text: str
    language: str

# PDF RESPONSES (matching UGJSONResponse data field)
class PDFUploadData(BaseModel):
    """Data field for PDF upload response"""
    pdf_document_json: str  # Serialized PDFDocument

class PDFUploadResponse(BaseModel):
    """Response from PDF upload"""
    status: int
    message: str
    data: PDFUploadData

class PDFListData(BaseModel):
    """Data field for PDF list response"""
    pdf_documents: List[str]  # List of serialized PDFDocument

class PDFListResponse(BaseModel):
    """Response from PDF list"""
    status: int
    message: str
    data: PDFListData

class PDFGetData(BaseModel):
    """Data field for PDF get response"""
    pdf_document_json: str  # Serialized PDFDocument

class PDFGetResponse(BaseModel):
    """Response from PDF get"""
    status: int
    message: str
    data: PDFGetData

class PDFDeleteData(BaseModel):
    """Data field for PDF delete response"""
    deleted: bool

class PDFDeleteResponse(BaseModel):
    """Response from PDF delete"""
    status: int
    message: str
    data: PDFDeleteData

class PDFProcessingRedisStatus(BaseModel):
    """Redis status fields from processing status"""
    status: Optional[str] = None
    progress: Optional[str] = None
    current_task: Optional[str] = None

class PDFProcessingRedisErrors(BaseModel):
    """Redis error fields from processing status"""
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_timestamp: Optional[str] = None

class PDFProcessingStatusData(BaseModel):
    """Processing status data from PDF status endpoint"""
    mongodb_status: str
    started_at: str
    completed_at: str
    error: Optional[str] = None
    redis_status: Optional[PDFProcessingRedisStatus] = None
    redis_errors: Optional[PDFProcessingRedisErrors] = None

class PDFProcessingStatusResponse(BaseModel):
    """Response from PDF processing status"""
    status: int
    message: str
    data: PDFProcessingStatusData

class PDFProcessingStartData(BaseModel):
    """Data field for PDF processing start"""
    status: str  # "processing" or "completed"

class PDFProcessingStartResponse(BaseModel):
    """Response from PDF process start"""
    status: int
    message: str
    data: PDFProcessingStartData

class PDFLearnAnswerData(BaseModel):
    """Answer data from PDF learn endpoint"""
    answer: str
    source_chunks: List[str]
    confidence_score: Optional[float] = None
    relevant_pages: Optional[List[int]] = None

class PDFLearnResponse(BaseModel):
    """Response from PDF learn endpoint"""
    status: int
    message: str
    data: PDFLearnAnswerData

class QuestionsByType(BaseModel):
    """Breakdown of questions generated by type"""
    MCQ: Optional[int] = None
    SHORT_ANSWER: Optional[int] = None
    VERY_SHORT_ANSWER: Optional[int] = None
    LONG_ANSWER: Optional[int] = None
    FILL_BLANKS: Optional[int] = None
    TRUEFALSE: Optional[int] = None

class PDFQuestionGenerationData(BaseModel):
    """Data from PDF question generation"""
    total_questions_generated: int
    questions_by_type: Optional[QuestionsByType] = None
    processing_time: Optional[float] = None

class PDFQuestionGenerationResponse(BaseModel):
    """Response from PDF question generation"""
    status: int
    message: str
    data: PDFQuestionGenerationData

# FEEDBACK RESPONSE
class FeedbackResponse(BaseModel):
    """Response from feedback submission"""
    Message: str

# PROGRESS RESPONSE (static data)
class SubjectProgress(BaseModel):
    """Progress data for a subject"""
    Lesson1: str
    Lesson2: str
    Lesson3: str
    Overall: str

class ProgressResponse(BaseModel):
    """Response from progress endpoint"""
    Physics: SubjectProgress
    Biology: SubjectProgress
    Chemistry: SubjectProgress

# ASSESSMENT SUBMISSIONS RESPONSE
class AssessmentSubmissionItem(BaseModel):
    """Individual submission item with assessment metadata"""
    assessment_id: str
    student_id: str
    submission_time: datetime
    results: List[AssessmentSubmissionResultItem]
    correct_count: int
    total_questions: int
    score_percentage: float
    assessment_title: str
    assessment_type: str
    pdf_id: str

class AssessmentSubmissionsListResponse(BaseModel):
    """Response from get_assessment_submissions"""
    submissions: List[AssessmentSubmissionItem]
    count: int

class AssessmentPDFListResponse(BaseModel):
    """Response from get_pdf_assessments"""
    pdf_id: str
    assessments: List[AssessmentHistoryItem]
    count: int

# SUBJECT HISTORY RESPONSE
class SubjectHistoryItem(BaseModel):
    """History item from MongoDB student_id.sahasra_history collection"""
    subject: str
    message: str
    is_ai: bool
    time: str  # ISO format
    session_id: str

class SubjectHistoryResponse(BaseModel):
    """Response from get_subject_history"""
    subject: str
    history: List[SubjectHistoryItem]
    count: int
    page: int
    page_size: int
    oldest_first: bool
    has_more: bool
from pydantic import BaseModel,RootModel,Field
from typing import List,Dict,Optional,Union,Any
from datetime import datetime
from models.pdf_models import PDFDocumentMetadata


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
	coupon_code: Optional[str] = None


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
	username:str | None
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
    language: Optional[str] = "English"  # Language for assessment title (English or Hindi)

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
    title: str

class AssessmentSubmissionResultItem(BaseModel):
    """Individual result item from assessment submission"""
    questionid: str
    is_correct: bool
    feedback: str

class AchievementProcessingResult(BaseModel):
    """Model for achievement processing results in assessment submission"""
    achievements_earned: List[str] = []
    badges_updated: List[str] = []
    streaks_updated: List[str] = []
    errors: List[str] = []
    achievement_details: List[Dict[str, Any]] = []
    badge_details: List[Dict[str, Any]] = []
    streak_details: List[Dict[str, Any]] = []

class AssessmentSubmissionResponse(BaseModel):
    """Response from submit_assessment service"""
    title: str
    results: List[AssessmentSubmissionResultItem]
    correct_count: int
    total_questions: int
    score_percentage: float
    achievements: Optional[AchievementProcessingResult] = None

class AssessmentLastSubmission(BaseModel):
    """Last submission data from MongoDB assessments"""
    assessment_id: str
    student_id: str
    submission_time: datetime
    results: List[AssessmentSubmissionResultItem]
    correct_count: int
    total_questions: int
    score_percentage: float

class AssessmentLastSubmissionFiltered(BaseModel):
    """Filtered last submission data for get_assessments response"""
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

class AssessmentHistoryItemFiltered(BaseModel):
    """Filtered individual assessment from get_assessments - returns only essential fields"""
    id: MongoObjectId = Field(alias="_id")
    subject: Optional[str] = None
    topics: List[str]
    level: Optional[int] = None
    created_at: Optional[datetime] = None
    # Additional fields needed for backward compatibility
    pdf_id: Optional[str] = None
    title: Optional[str] = None
    question_type: Optional[str] = None
    last_submission: Optional[AssessmentLastSubmissionFiltered] = None
    last_submission_time: Optional[datetime] = None

class AssessmentListResponse(BaseModel):
    """Response from get_assessments service - returns List[AssessmentHistoryItem]"""
    assessments: List[AssessmentHistoryItem]

class AssessmentByIdResponse(BaseModel):
    """Response from get_assessment_by_id service - returns single AssessmentHistoryItem"""
    assessment: AssessmentHistoryItem

# LEARNING RESPONSES (matching service returns)
class LearningImageData(BaseModel):
    """Individual image data for learning responses"""
    image_url: str
    caption: str
    page_number: Optional[int] = None
    score: Optional[float] = None
    pdf_id: Optional[str] = None
    subject: Optional[str] = None

class LearnAnswerResponse(BaseModel):
    """Enhanced response from learning services with multiple image support"""
    answer: str
    has_image: bool = False
    subject: str
    # Backward compatibility fields (use first image if available)
    image_url: Optional[str] = None
    image_caption: Optional[str] = None
    image_page: Optional[int] = None
    image_score: Optional[float] = None
    image_pdf_id: Optional[str] = None
    # New field for multiple images
    images: Optional[List[LearningImageData]] = None

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

# Mobile Verification Models
class MobileUpdateRequest(BaseModel):
    """Request model for mobile number update"""
    new_mobile: str

class MobileVerificationRequest(BaseModel):
    """Request model for mobile verification with OTP"""
    otp_token: str

class MobileUpdateResponse(BaseModel):
    """Response from mobile update request"""
    Message: str

class MobileVerificationResponse(BaseModel):
    """Response from mobile verification"""
    Message: str
    new_mobile: Optional[str] = None

class PendingMobileVerificationResponse(BaseModel):
    """Response for pending mobile verification status"""
    has_pending_verification: bool
    old_mobile: Optional[str] = None
    new_mobile: Optional[str] = None
    requested_at: Optional[datetime] = None

# Email Verification Models
class EmailUpdateRequest(BaseModel):
    """Request model for email address update"""
    new_email: str

class EmailVerificationRequest(BaseModel):
    """Request model for email verification with OTP"""
    otp_token: str

class EmailUpdateResponse(BaseModel):
    """Response from email update request"""
    Message: str

class EmailVerificationResponse(BaseModel):
    """Response from email verification"""
    Message: str
    new_email: Optional[str] = None

class PendingEmailVerificationResponse(BaseModel):
    """Response for pending email verification status"""
    has_pending_verification: bool
    old_email: Optional[str] = None
    new_email: Optional[str] = None
    requested_at: Optional[datetime] = None

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
class PDFDocumentData(BaseModel):
    """Actual PDF document data (not serialized JSON)"""
    id: str
    user_id: str
    file_name: str
    file_path: str
    file_size: int
    title: str
    description: Optional[str] = None
    pages: Optional[int] = None
    upload_date: str
    processing_status: str
    processing_error: Optional[str] = None
    process_start_time: Optional[str] = None
    process_end_time: Optional[str] = None
    metadata: PDFDocumentMetadata

class PDFUploadResponse(BaseModel):
    """Response from PDF upload"""
    status: int
    message: str
    data: PDFDocumentData

class PDFListResponse(BaseModel):
    """Response from PDF list"""
    status: int
    message: str
    data: List[PDFDocumentData]

class PDFGetResponse(BaseModel):
    """Response from PDF get"""
    status: int
    message: str
    data: PDFDocumentData

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
    start_time: Optional[str] = None
    step: Optional[str] = None
    end_time: Optional[str] = None
    images_extracted: Optional[str] = None
    timestamp: Optional[str] = None


class PDFProcessingRedisError(BaseModel):
    """Redis error fields from processing status"""
    error: Optional[str] = None
    timestamp: Optional[str] = None

class PDFProcessingStatusData(BaseModel):
    """Processing status data from PDF status endpoint"""
    mongodb_status: str
    started_at: str
    completed_at: str
    error: Optional[str] = None
    redis_status: Optional[PDFProcessingRedisStatus] = None
    redis_errors: Optional[PDFProcessingRedisError] = None

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
    has_image: bool
    image_url: Optional[str] = None
    image_caption: Optional[str] = None
    image_page: Optional[int] = None

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

# HISTORY DATES RESPONSE
class HistoryDatesData(BaseModel):
    """Data field for history dates response"""
    search_date: str
    available_dates: List[str]
    count: int

class HistoryDatesResponse(BaseModel):
    """Response from history dates endpoint"""
    status: int
    message: str
    data: HistoryDatesData

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

# LEARNING INFO MODELS
class LearningStreakInfo(BaseModel):
    """Learning streak information"""
    current_streak: int
    last_activity_date: Optional[str]
    longest_streak: int
    total_active_days: int

class QuestionsAnsweredInfo(BaseModel):
    """Questions answered count information"""
    total_questions_answered: int
    by_subject: Dict[str, int]
    weekly_change: int
    this_week_count: int
    last_week_count: int
    is_increase: bool
    is_decrease: bool

class LearningHoursInfo(BaseModel):
    """Learning hours information"""
    total_learning_hours: float
    learning_hours_today: float
    sessions_analyzed: int

class EducationalQuote(BaseModel):
    """Educational quote with author"""
    quote: str
    author: str

class LearningInfoResponse(BaseModel):
    """Response from learning-info endpoint"""
    learning_streak: LearningStreakInfo
    questions_answered: QuestionsAnsweredInfo
    learning_hours: LearningHoursInfo
    educational_quote: EducationalQuote
    generated_at: str
    student_id: str
    subject_filter: Optional[str] = None

class FetchQuestionsRequest(BaseModel):
    """Request model for fetching questions by topic and subject"""
    subject: str
    topic: str

class FetchQuestionsResponse(BaseModel):
    """Response model for fetched questions"""
    questions: List[Dict[str, Any]]
    total_count: int
    subject: str
    topic: str

class UpdateQuestionRequest(BaseModel):
    """Request model for updating a question document"""
    question_data: Dict[str, Any]

class UpdateQuestionResponse(BaseModel):
    """Response model for question update"""
    success: bool
    message: str
    document_id: Optional[str] = None
    updated_at: Optional[str] = None

# ASSESSMENT STATISTICS MODELS
class AssessmentStreakInfo(BaseModel):
    """Assessment streak information"""
    current_streak: int
    last_activity_date: Optional[str]
    longest_streak: int
    total_active_days: int

class SubjectAssessmentStats(BaseModel):
    """Assessment statistics for a specific subject"""
    total_assessments: int
    completed_assessments: int
    total_score: float
    average_score: float
    total_submissions: int

class AssessmentStatisticsResponse(BaseModel):
    """Response from assessment statistics endpoint"""
    total_assessments: int
    completed_assessments: int
    average_score: float
    total_submissions: int
    by_subject: Dict[str, SubjectAssessmentStats]
    assessment_streak: AssessmentStreakInfo
    days_filter: Optional[int] = None

# ACHIEVEMENT SYSTEM MODELS


class Achievement(BaseModel):
    """Model for a student achievement"""
    achievement_id: str
    achievement_type: str  # performance, consistency, coverage, difficulty
    name: str
    description: str
    icon: str
    count: int = 1
    first_earned: str
    last_earned: str
    metadata: Optional[Dict[str, Any]] = None

class Badge(BaseModel):
    """Model for a student badge"""
    badge_id: str
    badge_type: str  # topic_mastery, subject_mastery, consistency, etc.
    name: str
    tier: str  # bronze, silver, gold, platinum, diamond
    subject: Optional[str] = None
    topic: Optional[str] = None
    earned_date: str
    updated_at: str
    progress: Dict[str, Any]

class Streak(BaseModel):
    """Model for a student streak"""
    streak_type: str  # daily, subject, weekly
    subject: Optional[str] = None
    current_streak: int
    longest_streak: int
    last_activity_date: Optional[str] = None
    streak_start_date: Optional[str] = None
    updated_at: str
    grace_days_used: Optional[int] = 0
    weekly_progress: Optional[Dict[str, Any]] = None

class AchievementsResponse(BaseModel):
    """Response model for student achievements"""
    achievements: List[Achievement]
    total_count: int
    by_type: Dict[str, int]

class BadgesResponse(BaseModel):
    """Response model for student badges"""
    badges: List[Badge]
    total_count: int
    by_type: Dict[str, int]
    by_tier: Dict[str, int]

class StreaksResponse(BaseModel):
    """Response model for student streaks"""
    streaks: List[Streak]
    daily_streak: Optional[Streak] = None
    subject_streaks: List[Streak] = []
    weekly_streak: Optional[Streak] = None

class AchievementSummary(BaseModel):
    """Summary model for achievement overview"""
    total_achievements: int
    total_badges: int
    highest_badge_tier: str
    current_daily_streak: int
    longest_daily_streak: int
    recent_achievements: List[Achievement]
    featured_badges: List[Badge]

class AchievementSummaryResponse(BaseModel):
    """Response model for achievement summary"""
    summary: AchievementSummary
    quick_stats: Dict[str, Any]
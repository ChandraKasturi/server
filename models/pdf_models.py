from pydantic import BaseModel, Field
from typing import List, Optional, Union
from enum import Enum
from datetime import datetime
import uuid


class ProcessingStatus(str, Enum):
    """Enum for PDF processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class QuestionType(str, Enum):
    """Enum for question types."""
    MCQ = "mcq"
    FILL_BLANKS = "fill_in_blanks"
    VERY_SHORT_ANSWER = "very_short_answer"  # 1-2 words, definitions, terms
    SHORT_ANSWER = "short_answer"            # 1-3 sentences, brief explanations
    LONG_ANSWER = "long_answer"              # Detailed explanations, multiple paragraphs
    CASE_STUDY = "case_study"                # Scenario-based analysis questions
    TRUEFALSE = "truefalse"


class QuestionDifficulty(str, Enum):
    """Enum for question difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class PDFUploadRequest(BaseModel):
    """Request model for PDF upload."""
    title: str
    description: Optional[str] = None
    subject: Optional[str] = None
    grade: Optional[str] = None


class PDFDocumentMetadata(BaseModel):
    """Metadata for PDF documents - matches MongoDB structure"""
    subject: Optional[str] = None
    grade: Optional[str] = None
    author: Optional[str] = None
    tags: Optional[List[str]] = None
    language: Optional[str] = None
    category: Optional[str] = None


class PDFDocument(BaseModel):
    """Model for PDF document metadata."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    file_name: str
    file_path: str
    file_size: int
    title: str
    description: Optional[str] = None
    pages: Optional[int] = None
    upload_date: datetime = Field(default_factory=datetime.utcnow)
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    processing_error: Optional[str] = None
    process_start_time: Optional[datetime] = None
    process_end_time: Optional[datetime] = None
    metadata: PDFDocumentMetadata = Field(default_factory=PDFDocumentMetadata)
    
    class Config:
        use_enum_values = True


class PDFChunkMetadata(BaseModel):
    """Metadata for PDF chunks"""
    section_title: Optional[str] = None
    confidence_score: Optional[float] = None
    processing_method: Optional[str] = None
    text_quality: Optional[str] = None
    has_images: Optional[bool] = None
    image_count: Optional[int] = None


class PDFChunk(BaseModel):
    """Model for PDF text chunks."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pdf_id: str
    chunk_index: int
    page_number: Optional[int] = None
    content: str
    embedding: Optional[List[float]] = None
    metadata: PDFChunkMetadata = Field(default_factory=PDFChunkMetadata)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PDFTextMetadata(BaseModel):
    """Metadata for PDF text storage"""
    extraction_method: Optional[str] = None
    processing_time: Optional[float] = None
    language_detected: Optional[str] = None
    text_quality_score: Optional[float] = None
    has_tables: Optional[bool] = None
    has_images: Optional[bool] = None


class PDFText(BaseModel):
    """Model for storing full PDF text in PostgreSQL."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pdf_id: str
    title: str
    content: str
    page_count: int
    word_count: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: PDFTextMetadata = Field(default_factory=PDFTextMetadata)


class PDFTextChunk(BaseModel):
    """Model for storing PDF text chunks in PostgreSQL."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pdf_id: str
    chunk_index: int
    page_number: Optional[int] = None
    content: str
    word_count: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class GeneratedQuestionMetadata(BaseModel):
    """Metadata for generated questions"""
    generation_method: Optional[str] = None
    confidence_score: Optional[float] = None
    source_text_snippet: Optional[str] = None
    keywords: Optional[List[str]] = None
    bloom_taxonomy_level: Optional[str] = None
    estimated_time_minutes: Optional[int] = None


class GeneratedQuestion(BaseModel):
    """Model for generated questions."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pdf_id: str
    question_text: str
    question_type: QuestionType
    difficulty: QuestionDifficulty = QuestionDifficulty.MEDIUM
    source_chunk_id: Optional[str] = None
    page_reference: Optional[int] = None
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: GeneratedQuestionMetadata = Field(default_factory=GeneratedQuestionMetadata)
    
    class Config:
        use_enum_values = True


class QuestionOption(BaseModel):
    """Model for multiple-choice question options."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question_id: str
    option_text: str
    is_correct: bool = False
    option_order: int


class QuestionAnswer(BaseModel):
    """Model for question answers."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question_id: str
    answer_text: str
    explanation: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class LearningSession(BaseModel):
    """Model for learning sessions."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    pdf_id: str
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    total_questions: int = 0
    session_summary: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class LearningInteraction(BaseModel):
    """Model for learning interactions."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    query: str
    response: str
    source_chunks: List[str] = Field(default_factory=list)
    query_embedding: Optional[List[float]] = None
    interaction_time: datetime = Field(default_factory=datetime.utcnow)
    feedback_rating: Optional[int] = None
    feedback_comment: Optional[str] = None


class AssessmentStatus(str, Enum):
    """Enum for assessment status."""
    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"


class AssessmentMetadata(BaseModel):
    """Metadata for assessments"""
    difficulty_distribution: Optional[str] = None
    estimated_duration_minutes: Optional[int] = None
    question_categories: Optional[List[str]] = None
    auto_generated: Optional[bool] = None
    generation_timestamp: Optional[datetime] = None
    instructor_notes: Optional[str] = None


class Assessment(BaseModel):
    """Model for assessments."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    pdf_id: str
    title: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: AssessmentStatus = AssessmentStatus.DRAFT
    time_limit: Optional[int] = None  # in minutes
    passing_score: Optional[float] = None  # percentage
    is_randomized: bool = False
    metadata: AssessmentMetadata = Field(default_factory=AssessmentMetadata)
    
    class Config:
        use_enum_values = True


class AssessmentQuestion(BaseModel):
    """Model for assessment questions."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    assessment_id: str
    question_id: str
    question_order: int
    points: float = 1.0


class AssessmentAttemptStatus(str, Enum):
    """Enum for assessment attempt status."""
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"
    GRADED = "graded"


class AssessmentAttempt(BaseModel):
    """Model for assessment attempts."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    assessment_id: str
    user_id: str
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    status: AssessmentAttemptStatus = AssessmentAttemptStatus.IN_PROGRESS
    score: Optional[float] = None
    max_score: Optional[float] = None
    percentage: Optional[float] = None
    feedback: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


class StudentResponse(BaseModel):
    """Model for student responses to questions."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    attempt_id: str
    question_id: str
    response_text: Optional[str] = None
    selected_option_id: Optional[str] = None
    is_correct: Optional[bool] = None
    points_awarded: Optional[float] = None
    ai_evaluation: Optional[str] = None
    evaluation_confidence: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class PDFLearnRequest(BaseModel):
    """Model for PDF learning request."""
    question: str = Field(..., description="Question about the PDF content to learn from")
    similarity_threshold: Optional[float] = Field(default=None, description="Optional threshold for image similarity (0-1, lower is more similar)")


class SubjectLearnRequest(BaseModel):
    """Model for subject-specific learning request."""
    question: str = Field(..., description="Question about the subject to learn about")
    include_pdfs: bool = Field(default=True, description="Whether to include user's PDFs in the answer")


class TTSRequest(BaseModel):
    """Model for text-to-speech request."""
    text: str = Field(..., description="Text to convert to speech")
    voice: Optional[str] = Field(default='af_heart', description="Voice to use for TTS")
    speed: Optional[float] = Field(default=1.0, ge=0.25, le=4.0, description="Speed of speech (0.25 to 4.0)") 
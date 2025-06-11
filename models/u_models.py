from pydantic import BaseModel,RootModel
from typing import List,Dict,Optional
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
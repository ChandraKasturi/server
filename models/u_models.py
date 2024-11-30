from pydantic import BaseModel,RootModel
from typing import List,Dict,Optional


class uTweet(BaseModel):
	tweet:str


class uploadVectorUmodel(BaseModel):
	text:str

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
	questionid:str
	studentanswer:str | None


class AssessUmodel(BaseModel):
	questions:List[questionAssessUmodel]

class ProfileUmodel(BaseModel):
	email:str | None
	username:str | None
	mobilenumber:str | None
	Class: str | None
	educationboard: str | None
	bio: str | None

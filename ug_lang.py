import datetime
import uuid
from fastapi.middleware.cors import CORSMiddleware
import time
from langchain.schema import Document
from fastapi import FastAPI,Request,Response,Depends,HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from controllers.u_controllers import GetProfileImageU, UpdateProfileImageU, UploadUGVector, addFeedBackU, fetchAssessmentU, fetchHistoryU, getAssessmentsU,getUtweet,getUanswer,getUtranslate,UploadUGImageUUrl,getUimageuurl,AssessUContent,InsertQuestionU,loginU,registerU,forgotPasswordU,updatePasswordU,beforeRegisterU,checkAssessUContent,profileU,UpdateProfileU
from models.u_models import FeedBackUmodel, UpdatePasswordUmodel, uTweet,uAnswer,ucorrect,QuestionUmodel,loginUmodel,registerUmodel,ForgotPasswordUmodel,confirmRegisterUmodel,AssessUmodel,ProfileUmodel, uploadVectorUmodel
import jwt
import json
import typing
import pymongo
import uvicorn
from starlette import background
from bson.objectid import ObjectId
from fastapi.staticfiles import StaticFiles
from fastapi import File,UploadFile

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,  # Allows cookies and credentials
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


c = pymongo.MongoClient("mongodb://test:testug@localhost/")
class UGJSONResponse(JSONResponse):
	media_type = "application/json"

	def __init__(
        self,
        content: typing.Any,
        status_code: int = 200,
        headers: typing.Mapping[str, str] | None = None,
        media_type: str | None = None,
        background: background.BackgroundTask | None = None,
    ) -> None:
		super().__init__(content, status_code, headers, media_type, background)
	

	def render(self, content: typing.Any) -> bytes:
			print(f"UGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG CONTENT {content}")
			def changeUG(content):
				if type(content) == dict:
					for k,v in content.items():
						if type(v) == dict:
							changeUG(content[k])
						if type(v) == list:
							for ug in v:
								changeUG(ug)
						if type(v) == ObjectId:
							content[k] = str(v)
						if type(v) == datetime.datetime:
							content[k] = str(v)
				if type(content) == list:
					for ug in content:
						changeUG(ug)
				if type(content) == ObjectId:
					print("HERE UGUGUGGUUGGGGGGGGGGGGGGGGGGGGGGGGGMUUUUUUUUUUUUUUUUUUUUUUUU")
					content = str(content)
			changeUG(content)
			return json.dumps(
            	content,
            	ensure_ascii=False,
            	allow_nan=False,
            	indent=None,
            	separators=(",", ":"),
        		).encode("utf-8")

origins = [
    "http://localhost:3000",  
    "https://fastapi.tiangolo.com",
	"http://localhost:3001",
	"https://sahasraai.vercel.app",
	"https://www.sahasra.ai",
	"https://questionbank-one.vercel.app",
	"https://v0-sahas.vercel.app"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "PUT"],
    allow_headers=["X-Auth-Session"],
	expose_headers = ["X-Auth-Session"]
)


app.mount("/static",StaticFiles(directory="static"),name="static")



def auth_middleware(request:Request):
	token = request.headers.get("X-Auth-Session")
	print(token)
	if not token:
		raise HTTPException(status_code=401, detail="No Session Found")
	try:
		studentid  = jwt.decode(jwt=token,key="SECRET_UG",algorithms=["HS256"])["user"]
		print(f"STUDENT ID IS {studentid}")
		ugoldtoken = c.sahasra_tokens.auth_tokens.find_one({"student_id":studentid})["token"]
		if not ugoldtoken:
			raise HTTPException(status_code=401,detail="Session Expired")
		return studentid
	except Exception as e:
		print(e)
		raise HTTPException(status_code=401, detail="No Session Found")




@app.post("/login",response_class=JSONResponse)
def ulogin(ubody:loginUmodel,request:Request,response:Response):
	if type(ubody.mobilenumberoremail) != str or type(ubody.password) != str:
		return "Username and password Must be of type string"
	studentuid = loginU(ubody.mobilenumberoremail,ubody.password)
	if studentuid:
		try:
			iat = int(time.time())
			jti = uuid.uuid4().hex
			token = jwt.encode(payload={"user":studentuid,"iat":iat,"jti":jti},key="SECRET_UG",algorithm="HS256")
			c.sahasra_tokens.auth_tokens.create_index("ExpiresAt",expireAfterSeconds=2*60*60)
			c.sahasra_tokens.auth_tokens.insert_one({"student_id":studentuid,"token":token,"ExpiresAt":datetime.datetime.utcnow()})
			return JSONResponse(content={"Message":"Logged in SuccessFully"},status_code=200,headers={"X-Auth-Session":token})
		except Exception as e:
			print(e)
			return JSONResponse(content={"Message":"Something Went Wrong"},status_code=400)
	else:
		return JSONResponse(content={"Message":"Incorrect Username Or Password"},status_code=400)

			

			
@app.post("/forgotpassword",response_class=JSONResponse)
def uForgotPassword(ubody:ForgotPasswordUmodel):
	print(type(ubody.mobilenumberoremail))
	if type(ubody.mobilenumberoremail) != str:
		return JSONResponse(content={"Message":"email must be a string"},status_code=400)
	status ,status_code= forgotPasswordU(ubody)
	JSONResponse(content={"Message":status},status_code=status_code)

@app.get("/progress")
def uFetchProgress():
	return JSONResponse(content={"Physics":{"Lesson1":"Beginner","Lesson2":"Beginner","Lesson3":"Beginner","Overall":"Beginner"},"Biology":{"Lesson1":"Beginner","Lesson2":"Beginner","Lesson3":"Beginner","Overall":"Beginner"},"Chemistry":{"Lesson1":"Beginner","Lesson2":"Beginner","Lesson3":"Beginner","Overall":"Beginner"}},status_code=200)
@app.post("/updatepassword",response_class=JSONResponse)
def uResetPassword(ubody:UpdatePasswordUmodel):
	if ubody.password and ubody.token:
		resu = updatePasswordU(ubody.password,ubody.token)
		return JSONResponse(content={"Message":resu},status_code=200)
	if ubody.password and not ubody.token:
		return JSONResponse(content={"Message":"Provide a token"},status_code=400)
	if not ubody.password and ubody.token:
		return JSONResponse(content={"Message":"Provide A new Password"},status_code=400)
	return JSONResponse(content={"Message":"Provide a Token and a password"},status_code=400)



@app.post("/getotp",response_class=JSONResponse)
def uRegister(ubody:registerUmodel):
	resu,status_code = beforeRegisterU(ubody)
	return JSONResponse(content={"Message":resu},status_code=status_code)





@app.post("/register")
def uconfirmRegister(ubody:confirmRegisterUmodel):
	resu,studentid,status_code = registerU(ubody)
	if studentid:
		try:
			expdateug = time.time() + 2*3600
			token = jwt.encode(payload={"user":studentid,"exp":expdateug},key="SECRET_UG",algorithm="HS256")
			c.sahasra_tokens.auth_tokens.create_index("ExpiresAt",expireAfterSeconds=2*60*60)
			c.sahasra_tokens.auth_tokens.insert_one({"student_id":studentid,"token":token,"ExpiresAt":datetime.datetime.utcnow()})
			return JSONResponse(content={"Message":resu},status_code=status_code,headers={"X-Auth-Session":token})
		except Exception as e:
			print(e)
			return JSONResponse(content={"Message":"Something Went Wrong"},status_code=400)
	else:
		return JSONResponse(content={"Message":"Unable To Register User"},status_code=400)


@app.get("/logout")
def ulogOut(studentid: str  = Depends(auth_middleware)):
	try:
		c.sahasra_tokens.auth_tokens.delete_one({"studentid":studentid})
		return JSONResponse(content={"Message":"Logged Out SuccessFully"},status_code=200)
	except Exception as e:
		print(e)
		return JSONResponse(content={"Message":"Something Went Wrong"},status_code=400)




@app.post("/assessment",response_class=UGJSONResponse)
def uSubmitAssessment(ubody:AssessUmodel,request:Request,studentid:str = Depends(auth_middleware)):
	resu,status_code = checkAssessUContent(ubody,studentid)
	print(f"UG RESU{resu}")
	return UGJSONResponse(content=resu,status_code=status_code)


@app.get("/assessments",response_class=UGJSONResponse)
def uGetAssessment(studentid:str = Depends(auth_middleware),time:str|None = None):
	resu,status_code = getAssessmentsU(studentid,time)
	return UGJSONResponse(content=resu,status_code=status_code)
	return JSONResponse(content=resu,status_code=status_code)

@app.get("/history",response_class=UGJSONResponse)
def uGetHistory(studentid:str = Depends(auth_middleware),time:str | None = None):
	resu,status_code = fetchHistoryU(studentid,time)
	return UGJSONResponse(content=resu,status_code=status_code)


@app.get("/assessment/{assessment_id}",response_class=UGJSONResponse)
def uGetAssessmentWithId(assessment_id,studentid:str = Depends(auth_middleware)):
	resu,status_code = fetchAssessmentU(studentid,assessment_id)
	return UGJSONResponse(content=resu,status_code=status_code)

@app.post("/feedback",response_class=UGJSONResponse)
def uGetFeedBack(ubody:FeedBackUmodel,studentid:str=Depends(auth_middleware)):
	feedbackug = ubody.feedback
	resu,status_code = addFeedBackU(studentid,feedbackug)
	return UGJSONResponse(content=resu,status_code=status_code)

@app.get("/profile",response_class=JSONResponse)
def uProfile(request:Request,studentid:str = Depends(auth_middleware)):
	print(studentid)
	resu,status_code = profileU(studentid)
	print(resu)
	return JSONResponse(content=dict(resu),status_code=status_code)



@app.post("/updateprofileimage",response_class=UGJSONResponse)
async def uProfileImage(studentid:str=Depends(auth_middleware),file:UploadFile=File(...)):
	resu,status_code = await UpdateProfileImageU(studentid,file)
	return UGJSONResponse(content=resu,status_code=status_code)

@app.get("/getprofileimage",response_class=UGJSONResponse)
def uGetProfileImage(studentid:str=Depends(auth_middleware)):
	resu,status_code = GetProfileImageU(studentid)
	return UGJSONResponse(content=resu,status_code=status_code)
	


@app.post("/updateprofile",response_class=JSONResponse)
def updateUProfile(ubody:ProfileUmodel,request:Request,studentid:str = Depends(auth_middleware)):
	resu,status_code = UpdateProfileU(ubody,studentid)
	
	return JSONResponse(content=resu,status_code=status_code)




@app.get("/uFetchuUimageurl")
def ufetch_uimageuurl(ucontext: str = ""):
	uuimageurl = getUimageuurl(ucontext)
	return uuimageurl


@app.post("/api/question")
def insQuestion(body:QuestionUmodel):
	try:
		idU = InsertQuestionU(body)
		return {"STATUS":"SUCCESS","insertedID":idU}
	except Exception as e:
		print(e)
		return{"STATUS":"ERROR"}

@app.post("/api/uUploadVector")
def split_lang(ubody:uploadVectorUmodel):
		doc_ucontent = UploadUGVector(ubody.text,ubody.subject)
		return {"success_code":1}


@app.get("/api/uUploadImageuUrl")
def split_uimageurl():
	doc_ucontent = UploadUGImageUUrl("uimageuurl.txt")
	return doc_ucontent


@app.post("/api/utweet")
def get_utweet(body: uTweet):
	try:
		message = body.tweet
		u_tweetGenerated = getUtweet(message)
		return {"uGEN":u_tweetGenerated}
	except Exception as e:
		print(e)
		return {"uGEN":"ERROR"}

@app.post("/api/chat",response_class=UGJSONResponse)
def get_uanswer(body: uAnswer,request:Request,studentid:str = Depends(auth_middleware)):
	sessionug = request.headers.get("X-Auth-Session")
	question = body.question
	print(question)
	u_answerGenerated = AssessUContent(question,sessionug ,studentid)
	return UGJSONResponse(content=u_answerGenerated,status_code=200)

@app.post("/api/utranslate")
def get_utranslate(body: ucorrect):
	try:
		ugrammar = body.ucgrammar
		u_translateGenerated = getUtranslate(ugrammar)
		return {"UGEN":u_translateGenerated}
	except Exception as e:
		print(e)
		return {"uGEN":"ERROR"}







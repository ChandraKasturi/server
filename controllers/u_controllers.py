import os
import re
from supabase.client import create_client
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.vectorstores import SupabaseVectorStore
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from bson.objectid import ObjectId
from connections.pgvector import PostgresVectorStore
from langchain_postgres.vectorstores import PGVector
import secrets
import string
import datetime
import dateutil
import json
import secrets
import uuid

from services.questionbank.mongo.questionbankrepu import insertIntoQuestionsU,CheckifEmailPresent
from services.questionbank.mongo.emailserviceu import register_phoneNumberU, send_email,register_email, send_phoneNumberU

import pymongo

#U_KEYS
phone_ugregex = re.compile(r"[0-9]+$")
c = pymongo.MongoClient("mongodb://test:testug@localhost/")
u_openai_api_key = "-8v5huYC_6cjaoZ9Hxn_kJCWfhBIWdFIA"
u_supabase_url = "https://uuvgdpvtndnglygvblht.supabase.co"
u_supabase_api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InV1dmdkcHZ0bmRuZ2x5Z3ZibGh0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MDkxMDkzNTUsImV4cCI6MjAyNDY4NTM1NX0.MNSga3iZ_SnjdUVgxva71uqJJK9S5SFhD0MgJ-_boVs"

#create a splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50) # ["\n\n", "\n", " ", ""] are default values
openai_uembeddings = OpenAIEmbeddings(openai_api_key=u_openai_api_key)
supabase_uclient = create_client(u_supabase_url,u_supabase_api_key)
#creating llm
llm = ChatOpenAI(openai_api_key=u_openai_api_key)
ug_llm = ChatOpenAI(openai_api_key=u_openai_api_key,model="gpt-4o")
#creating retriever u_context
u_vecstore = SupabaseVectorStore(embedding=openai_uembeddings,client=supabase_uclient,query_name="match_subject",table_name="biology")
u_vecstore_uimageuurl = SupabaseVectorStore(embedding=openai_uembeddings,client=supabase_uclient,query_name="match_u_image_uurl",table_name="uimageurl")
a_u = u_vecstore.as_retriever()
a_uimageuurl = u_vecstore_uimageuurl.as_retriever()	
retrieveru_from_llm = MultiQueryRetriever.from_llm(retriever=a_u,llm=llm)
retrieveruimguurl_from_llm = MultiQueryRetriever.from_llm(retriever=a_uimageuurl,llm=llm)




def InsertQuestionU(collectionU):
	return insertIntoQuestionsU(c,collectionU)



def loginU(user,password):
	rowU = c.sahasra_users.users.find_one({"$and":[{"$or":[{"email":{"$eq":f"{user.lower()}"}},{"mobilenumber":{"$eq":f"{user}"}}]},{"password":{"$eq":f"{password}"}}]})
	if rowU:
		return rowU["student_id"]
	return False


def beforeRegisterU(nModel):
	print(dict(nModel))
	emailU = dict(nModel)["email"].lower()
	mobile = dict(nModel)["phonenumber"]
	fModelU = dict(nModel)
	token = ''.join([secrets.choice(string.digits) for i in range(6)])
	emailTakenU = CheckifEmailPresent(c,emailU,mobileno=mobile)
	if not emailTakenU:
		fModelU["token"] = token
		fModelU["ExpiresAt"] = datetime.datetime.utcnow()
		c.sahasra_tokens.register_tokens.create_index("ExpiresAt",expireAfterSeconds=10*60*60)
		c.sahasra_tokens.register_tokens.insert_one(fModelU)
		'''if phone_ugregex.match(emailU):
			register_phoneNumberU(emailU,token)
		else:
			register_email(emailU,token)'''
		register_phoneNumberU(mobile,token)
		register_email(emailU,token)
		return "Register Token Has been sent to your Email ID Or MobileNumber Please Use it to Confirm Your Registration",200
	else:
		return "Email Already Taken",400


def registerU(rModel):
	ugstudentid = secrets.token_hex(16)
	token = dict(rModel)["token"]
	print(token)
	token_dbu = c.sahasra_tokens.register_tokens.find_one({"token":token})
	print(token_dbu)
	finalregisterU = dict(rModel)
	finalregisterU["student_id"] = ugstudentid

	if token_dbu and token_dbu["token"] == token:
		try:
			_ = c.sahasra_users.users.insert_one(finalregisterU)
			return "User Registered SuccessFully You May Login Now",ugstudentid
		except Exception as e:
			print(e)
			return "Something Went Wrong",""
	elif token_dbu and token_dbu["token"] != token:
		return "Wrong Token",""
	else:
		return "Token Expired Try againU",""
	'''emailU = dict(rModel).email
	rowU = c.sahasra_users.users.find({"$or":[{"username":{"$eq":dict(rModel).username}},{"email":{"$eq":dict(rModel).email}}]})
	if rowU != 1:
		return False
	c.saharsa_users.users.insert(dict(rModel))
	return True'''
	

def forgotPasswordU(forgotuModel):
	emailU = dict(forgotuModel)["mobilenumberoremail"]
	token = ''.join([secrets.choice(string.digits) for i in range(6)])
	emailTakenU = CheckifEmailPresent(c,emailU)
	if emailTakenU:
		c.sahasra_tokens.password_tokens.create_index("ExpiresAt",expireAfterSeconds=10*60*60)
		c.sahasra_tokens.password_tokens.insert_one({"email":emailTakenU,"token":token,"ExpiresAt":datetime.datetime.utcnow()})
		try:
			if not phone_ugregex.match(emailTakenU):
				send_email(emailTakenU,token)
			else:
				send_phoneNumberU(emailTakenU,token)
			return "IF the Email or PhoneNumber You Provided Exists in our Database The reset Link Should be in Your Inbox Please Check your mail or Mobile",200
		except Exception as e:
			print(e)
			return "something Went Wrong",400
	return "IF the Email or PhoneNumber You Provided Exists in our Database The reset Link Should be in Your Inbox Please Check your mail or Mobile",200

def updatePasswordU(password,token):
	token_dbu = c.sahasra_tokens.password_tokens.find_one({"token":token})
	if token_dbu and token_dbu.token == token:
		try:
			_ = c.sahasra_users.users.update_one({"$or":[{"email":{"$eq":f"{token_dbu.email}"}},{"mobilenumber":{"$eq":f"{token_dbu.email}"}}]},{"$set":{"password":password}})
			return "Password Updated SuccessFully You May Login Now",200
		except Exception as e:
			print(e)
			return "Something Went Wrong",400
	elif token_dbu and token_dbu.token != token:
		return "Wrong Token",400
	else:
		return "Token Expired Try againU",400




def fetchHistoryU(studentid,time):
	try:
		if time:
			timeCheckUg = dateutil.parser.parse(time)-datetime.timedelta(weeks=1)
		else:
			timeCheckUg = datetime.datetime.utcnow()-datetime.timedelta(weeks=1)
	except Exception as e:
		timeCheckUg = datetime.datetime.utcnow()-datetime.timedelta(weeks=1)
	final_historyug = [ug for ug in c[studentid]["sahasra_history"].find({"time":{"$gte":timeCheckUg}})]
	return final_historyug,200







def getUimageuurl(ucontext):
	return retrieveruimguurl_from_llm.get_relevant_documents(query=ucontext)


def UploadUGImageUUrl(file):
	text_image_usplitter = CharacterTextSplitter(chunk_size=100,chunk_overlap=10,separator="\n") # need it store a valid image uurl
	file_ubytes_uimg = open(file,"rb").read()
	u_text_uimg = text_image_usplitter.create_documents([str(file_ubytes_uimg)])
	print(u_text_uimg)
	#file_ubytes_uimg = open(file,"rb").read().decode('utf-8').split("\n") # need to handle cases not utf-8
	#u_text_uimg = [Document(i) for i in file_ubytes_uimg]
	print(u_text_uimg)
	vector_ustore_uimg  = SupabaseVectorStore.from_documents(u_text_uimg,openai_uembeddings,client=supabase_uclient,table_name="uimageurl")



def UploadUGVector(file,subject):
	textug_splitter = RecursiveCharacterTextSplitter(chunk_size=len(file),chunk_overlap=0)
	u_text = textug_splitter.create_documents([str(file)])
	'''DB_PARAMS = {
    'dbname': 'x_cbse',
    'user': 'myuser',
    'password': 'mypassword',
    'host': 'localhost',  # or your host
    'port': '5432'        # default PostgreSQL port
	}
	vector_store = PostgresVectorStore(DB_PARAMS, str(subject).lower(),u_openai_api_key)'''
	vector_store = PGVector(
    	embeddings=openai_uembeddings,
    	collection_name=subject,
    	connection="postgresql+psycopg://myuser:mypassword@localhost:5432/x_cbse",
    	use_jsonb=True,
	)
	print(u_text[0].metadata)
	vector_store.add_documents(u_text,ids=[str(uuid.uuid4()) for doc in u_text])
	return "Done"


def AssessUContent(udata,sessionIdu,studentid):
	studentanswerug = {}
	lanswerug = {}
	finalquestionsug = {}
	assesug = {}
	u_response = {}
	u_response["response"] = []
	CheckUprompt = PromptTemplate.from_template("Answer Only AS YES OR NO IN CAPTIALS Answer whether the following statement is Strictly  Related to this Context: \n Context:{context}\n Statement:{statement}")
	checkuChain = CheckUprompt | llm | StrOutputParser()
	AssessUG = checkuChain.invoke({"context":"Question me,Ask me ,Assess me Evaluate me,Examine me, Test me, Analyze me","statement":udata})



	'''ug_page_content = [u.page_content for u in retrieveru_from_llm.get_relevant_documents(query=udata)]'''
	'''ug_topics = [supabase_uclient.table("science").select("topic,subtopic,imageu_uurl,videou_uurl,u_formula").eq('content',u).execute() for u in ug_page_content]
	print("HEREEEEEEEEEEEE")
	print(ug_topics)
	print(ug_topics[2].data[0]["topic"])
	print(ug_topics[2].data[0]["subtopic"])'''
	print(AssessUG)
	u_questions = []
	if AssessUG == "YES":
		assessug_name = ""
		assessug_time = datetime.datetime.utcnow()
		ugjson = """{"json":[{"subject":"Biology","topic":"Plants","subtopic":"Plants","NumberOfQuestions":3,"level":2},{"subject":"Physics","topic":"Plants","subtopic":"Plants","NumberOfQuestions":3,"level":2}]}"""
		jsonU = json.dumps([i for i in c.sahasra_subjectdata.topic_subtopic.find({},{"_id":0})])
		print(jsonU)
		GetFinalUjson = PromptTemplate.from_template("Given The Question Produce a correct JSON containing a key called questions which contains an array of json which in turn has subject,topic,subtopic,level,NumberOfQuestions for each subject from the Question Keep the Default value for NumberOfQuestions as 5 and Default Level as 1 if not found in the Question and the omit and remove fields from json if not found in the question\n Question: {question}")
		GetFinalUchain = GetFinalUjson | llm | StrOutputParser()
		json_withU = GetFinalUchain.invoke({"question":udata})
		print(f"FINAL JSONUGUGUGST{json_withU}")
		check_jsonU = json.loads(json_withU)
		print(f"FINAL JSONUG{json_withU}")
		for questionug in check_jsonU["questions"]:
			noqug = questionug["NumberOfQuestions"]
			_ = questionug.pop("NumberOfQuestions")
			for k,v in questionug.items():
				if type(v) == int:
					questionug[k] = str(v)
			print(f"FINALUG questionug {questionug}")

			collections_questionu = c.sahasra_questions.question_bank.find(questionug).limit(noqug)
			u_questions.extend([dict(ug) for ug in collections_questionu])
			print(u_questions)
		if len(check_jsonU["questions"]) == 1:
			assessug_name = check_jsonU["questions"][0]["subject"]+check_jsonU["questions"][0].get("topic","")+check_jsonU["questions"][0].get("subtopic","")
		else:
			assessug_name = "Multi-Subject-Assessment"
		

		
		'''ug_ugd = {}
		for k,v in check_jsonU.items():
			if v and type(v) == str:
				ug_ugd[k.lower()] = v

		print(check_jsonU)
		print(ug_ugd)
		collections_questionu = c.sahasra_questions.question_bank.find(ug_ugd).limit(noqug)
		u_questions.extend([dict(ug) for ug in collections_questionu])'''

		assesug["questions"] = []
		for uqug in u_questions:
			assesug["questions"].append(uqug["_id"])
		assesug["issubmit"] = False
		assesug["assessment_name"] = assessug_name
		assesug["assessment_date"] =  assessug_time
		c[studentid]["assessments"].insert_one(assesug)

		if not u_questions:
			return "Please Proivde The Right At least Subject for Assessment"
		finalquestionsug["questions"] = u_questions
		finalquestionsug["Assessment"] = True
		finalquestionsug["assessment_name"] = assessug_name
		finalquestionsug["assessment_date"] = assessug_time
		finalquestionsug["NumberOfQuestions"] = len(u_questions)

		return finalquestionsug
		if not check_jsonU["Subject"]:
			return "Please Provide A Valid Subject,Topic Or SubTopic" 

		if not check_jsonU["Topic"] and not check_jsonU["SubTopic"]:
			collection_questionu = c.sahasra_questions.questionbank.find({"subject":check_jsonU['Subject']})

		elif check_jsonU["Topic"] and not check_jsonU["SubTopic"]:
			collection_questionu = c.sahasra_questions.questionbank.find({"$and":[{"subject":{"$eq":f"{check_jsonU['Subject']}"}},{"topic":{"$eq":f"{check_jsonU['Topic']}"}}]})
		else:
			print("HERE UG")
			print(check_jsonU)
			collection_questionu = c.sahasra_questions.questionbank.find({"$and":[{"subject":{"$eq":f"{check_jsonU['Subject']}"}},{"topic":{"$eq":f"{check_jsonU['Topic']}"}},{"subtopic":{"$eq":f"{check_jsonU['SubTopic']}"}}]})

		u_questions.extend([dict(ug) for ug in collection_questionu ])
		"""for ug in ug_topics:
			if ug.data[0]["topic"] and ug.data[0]["subtopic"]:
				collection_questionu = c.sahasra_questions.questionbank.find({"topic":ug.data[0]["topic"]})
				u_questions.extend([dict(ug) for ug in collection_questionu ])"""
		print(u_questions)
		return u_questions
	ugresponse = getUanswer(udata,sessionIdu,studentid)
	u_response["response"].extend([{"text":ugresponse[0]},{"images":[ug for ug in ugresponse[1]["images"]]},{"videos":[ug for ug in ugresponse[1]["videos"]]},{"formula":[]}])
	timeug = datetime.datetime.utcnow()
	studentanswerug["role"] = "user"
	studentanswerug["content"] = udata
	studentanswerug["time"] = timeug
	studentanswerug["session"] = sessionIdu
	lanswerug["role"] = "assistant"
	lanswerug["content"] = u_response
	lanswerug["time"] = timeug
	lanswerug["session"] = sessionIdu
	u_response["Assessment"] = False
	c[studentid]["sahasra_history"].insert_one(studentanswerug)
	c[studentid]["sahasra_history"].insert_one(lanswerug)
	return u_response


def checkAssessUContent(umodel,studentid):
	student_answerU = []
	final_assessU = {}
	ug_assessU = {}
	progress_assessU = []
	ud = []
	assess_model = dict(umodel)["questions"]
	questionsug_assess = []
	for ug in assess_model:
		questionsug_assess.append(ObjectId(dict(ug)["questionid"]))
		student_answerU.append({"question":ObjectId(dict(ug)["questionid"]),"studentanswer":dict(ug)["studentanswer"]})
	for assessmentu in assess_model:
		questionu = c.sahasra_questions.question_bank.find_one({"_id":ObjectId(dict(assessmentu)["questionid"])})
		if not final_assessU.get(questionu['subject'],None):
			final_assessU[questionu["subject"]] = {"score":0,"questions":0,"topic":{}}
		studentuanswer = dict(assessmentu)["studentanswer"]
		if studentuanswer == questionu["correctanswer"]:
			print(final_assessU)
			final_assessU["total_score"] = final_assessU.get("total_score",0) + 1
			final_assessU[questionu["subject"]]["score"] = final_assessU[questionu["subject"]]["score"]+1
			final_assessU[questionu["subject"]]["questions"] = final_assessU[questionu['subject']]["questions"]+1
			final_assessU[questionu["subject"]]["topic"][questionu["topic"]] = final_assessU[questionu["subject"]]["topic"].get(questionu["topic"],0)+1
			progress_assessU.append(final_assessU[questionu['subject']])
			'''final_assessU["total_score"] = final_assessU.get("total_score",0)+ 1
			final_assessU[questionu["subject"]] = final_assessU.get(questionu["subject"],0) + 1
			print(f"total scoreug {final_assessU['total_score']}")
			print(f"subjectug {final_assessU[questionu['subject']]} ")
			print(f"Question UU {questionu['_id']}")'''
			ud.append(str(questionu["_id"]))
			print(f"final dictu {final_assessU}")
		else:
			final_assessU["total_score"] = final_assessU.get("total_score",0)+0
			final_assessU[questionu["subject"]]["questions"] = final_assessU[questionu['subject']]["questions"]+1
			final_assessU[questionu["subject"]]["topic"][questionu["topic"]] = final_assessU[questionu["subject"]]["topic"].get(questionu["topic"],0)+0
	ug_assessU["total_score"] = final_assessU["total_score"]
	ug_assessU["progress"] = final_assessU
	ug_assessU["uanswerd"] = ud
	ug_assessU["issubmit"] = True
	ug_assessU["questions"] = student_answerU
	print(f"questions TO UGGGGGGGGGGGGGGGGGGGGGGGGGGGG {questionsug_assess}")
	print(f'STUDENT IF UG {studentid}')
	
	c[studentid]["assessments"].update_one({"questions":questionsug_assess},{"$set":ug_assessU},upsert=True)

	assesug_response = []
	for ug in questionsug_assess:
		ug_f = {}
		questionug = dict(c.sahasra_questions.question_bank.find_one({"_id":ug}))
		for uug in student_answerU:
			if uug["question"] == ug:
				questionug["studentanswer"] = uug["studentanswer"]
		assesug_response.append(questionug)
	print(final_assessU)
	return assesug_response,200

def getAssessmentsU(studentid,time):
	try:
		if time:
			timeCheckUg = dateutil.parser.parse(time)-datetime.timedelta(weeks=1)
		else:
			timeCheckUg = datetime.datetime.utcnow()-datetime.timedelta(weeks=1)
	except Exception as e:
		timeCheckUg = datetime.datetime.utcnow()-datetime.timedelta(weeks=1)

	assessmentsUg = []
	for ug in c[studentid]["assessments"].find({"assessment_date":{"$gte":timeCheckUg}}):
		if type(ug["questions"]) == list:
			ug["questions"] = [str(u) for u in ug["questions"]]
		assessmentsUg.append(ug)
	return assessmentsUg,200

def fetchAssessmentU(studentid,assessmentid):
	assessquestionug = {}
	questionsu = c[studentid]["assessments"].find_one({"_id":ObjectId(assessmentid)},{"questions":1,"issubmit":1,"assessment_name":1,"assessment_date":1,"NumberOfQuestions":1,"total_score":1})
	assessquestionug["questions"] = []
	assessquestionug["assessment_name"] = questionsu["assessment_name"]
	assessquestionug["assessment_date"] = questionsu["assessment_date"]
	assessquestionug["NumberOfQuestions"] = len(questionsu["questions"])
	if questionsu["issubmit"]:
		print(questionsu)
		for ug in questionsu["questions"]:
			print(f"UGUGUGUUGUGUGGGGGGGGGG{ug}")
			questionassessug = dict(c.sahasra_questions.question_bank.find_one({"_id":ug["question"]}))
			questionassessug["studentanswer"] = ug["studentanswer"]
			assessquestionug["questions"].append(questionassessug)
		assessquestionug["total_score"] = questionsu["total_score"]
			
	else:
		for ug in questionsu["questions"]:
			print(f"UGUGUGGUUGUGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG{ug}")
			questionassessug = dict(c.sahasra_questions.question_bank.find_one({"_id":ug},{"correctanswer":0}))
			assessquestionug["questions"].append(questionassessug)

	print(f"ASSESSUMENTSUG {assessquestionug}")
	return assessquestionug,200




def profileU(studentid):
	uprofile = c.sahasra_users.users.find_one({"student_id":studentid},{"password":0,"token":0,"_id":0,"studentid":0})
	if uprofile:
		return uprofile,200
	return "No Student Id Found With the Student ID",400



def UpdateProfileU(updateUModel,studentid):
	c.sahasra_users.users.update_one({"student_id":studentid},{"$set":dict(updateUModel)})
	return "Updated Bio SuccessFully",200






def getUtweet(data):
	utweetPrompt = PromptTemplate.from_template("Generate a promotional tweet for a product, from this product description: {productDesc}")
	chain = utweetPrompt | llm | {"str": StrOutputParser()}
	return chain.invoke({"productDesc":data})["str"]


def getUanswer(data,sessionIdu,studentid):
	student_ug = {}
	l_ug = {}
	print("ugly")
	'''print(retrieveru_from_llm)
	questionualoneTemplate = "generate a question based on the following: {udata}"
	questionuPrompt = PromptTemplate.from_template(questionualoneTemplate)
	questionUchain = questionuPrompt | llm | StrOutputParser()
	print(questionUchain)
	generatedUquestion = questionUchain.invoke({"udata":data}) # a second invoke before main uinvoke bad way will change
	print(generatedUquestion)'''
	subjectugTemplate = "Given the Statment Reply with a valid JSON with the key as subject Which subject this statement Belongs to Given the subjects Array \n Subjects:{subjects},Statement:{statement}"	
	subjectugPrompt = PromptTemplate.from_template(subjectugTemplate)
	subjectugchain = subjectugPrompt | llm | StrOutputParser()
	jsonug = subjectugchain.invoke({"subjects":"['biology','english]","statement":data})
	print(f"THIS IS SUBJECTUG {jsonug}")
	ugsubject = json.loads(jsonug)["subject"].lower()
	'''ugsubject_vecstore = SupabaseVectorStore(embedding=openai_uembeddings,client=supabase_uclient,query_name=f"match_{ugsubject}",table_name=f"{ugsubject}")
	s_u = ugsubject_vecstore.as_retriever()
	retrieverugsubject_from_llm = MultiQueryRetriever.from_llm(retriever=s_u,llm=llm)'''
	standaloneTemplate = "Give the appropiate Formulas in Latex and Please Provide Related Image Urls In Between Your Answer appropriately as Markdown From the Context To the Question and Answer the Question with the references and Image urls only on the following context If The Question is Not Related to The Context Act As Usual\n Context: {context}\n\n"
	chatGeneratedUTemplate = ChatPromptTemplate.from_messages([("system",standaloneTemplate),MessagesPlaceholder(variable_name="history"),("human","Question: {question}")])
	#standalonePrompt = PromptTemplate.from_template(standaloneTemplate)
	'''DB_PARAMS = {
    'dbname': 'x_cbse',
    'user': 'myuser',
    'password': 'mypassword',
    'host': 'localhost',  # or your host
    'port': '5432'        # default PostgreSQL port
	}
	vector_store = PostgresVectorStore(DB_PARAMS, ugsubject,u_openai_api_key)
	query_vector = vector_store.embeddings.embed_documents([data])[0]
	print(f"QUERY VECTOR {query_vector}")
	ug_page_content = [u[1] for u in vector_store.search(query_vector,1)]'''
	vector_store = PGVector(
   	 embeddings=openai_uembeddings,
   	 collection_name=ugsubject,
    	connection="postgresql+psycopg://myuser:mypassword@localhost:5432/x_cbse",
    	use_jsonb=True,
	)
	retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})
	ug_page_content = [doc.page_content for doc in retriever.invoke(data)]
	'''ug_topics = [supabase_uclient.table("science").select("topic,subtopic").eq('content',u).execute() for u in ug_page_content]
	print(ug_topics)'''
	'''metaDatau = [u.metadata for u in retrieveru_from_llm.get_relevant_documents(query=data)]'''
	print(f"RETRIEV {ug_page_content}")
	retriever = "\n".join(ug_page_content)
	standAloneChain =  chatGeneratedUTemplate | ug_llm | StrOutputParser()
	standAloneUChainWithHistory = RunnableWithMessageHistory(standAloneChain,lambda sessionIdu: MongoDBChatMessageHistory(session_id=sessionIdu,connection_string="mongodb://test:testug@localhost/",database_name=studentid,collection_name="history"),input_messages_key="question",history_messages_key="history")
	config = {"configurable":{"session_id":sessionIdu}}
	print(f"THIS IS UG CONTEXT {retriever}")
	finalug = standAloneUChainWithHistory.invoke({"question":data,"context":retriever},config=config)
	'''ugTemplate = 'From this List Of Images And Videos Images=["https://www.sahasra.ai/class_X/Biology/Life_processes/Binary_Fission_Ameboa.PNG", "https://www.sahasra.ai/class_X/Biology/Life_processes/Multiple%20Fission%20In%20Plasmodium.PNG", "https://www.sahasra.ai/class_X/Biology/Life_processes/regeneration%20in%20planaria.png", "https://www.sahasra.ai/class_X/Biology/Life_processes/budding%20in%20hydra.png", "https://www.sahasra.ai/class_X/Biology/Life_processes/spore%20formation%20in%20Rhizopus.png", "https://www.sahasra.ai/class_X/Biology/Life_processes/sexual%20reproduction%20in%20flowering%20plants.png", "https://www.sahasra.ai/class_X/Biology/Life_processes/generation%20of%20pollen%20on%20stigma.png", "https://www.sahasra.ai/class_X/Biology/Life_processes/germination.png", "https://www.sahasra.ai/class_X/Biology/Life_processes/Human-male%20reproductive%20system.png", "https://www.sahasra.ai/class_X/Biology/Life_processes/Human-female%20reproductive%20system.png"] Videos ["https://www.youtube.com/watch?v=PFySHqo1e0w?title=Human+reproduction", "https://www.youtube.com/watch?v=68fVEQcftTw?title=Plant+reproduction", "https://www.youtube.com/watch?v=r8IoV0y4htU?title=Plant+Heridity"] Give me a JSON with keys images and vidoes which contain List of images and Videos Most relavent to this context\nContext:{context}'
	ugprompt = PromptTemplate.from_template(ugTemplate)
	ugChain = ugprompt | llm | StrOutputParser()
	ugImagesVidoes = ugChain.invoke({"context":finalug})
	print(F"THIS IS UGGUGUGUGUGUGUGUGUGUGGUGUGUGGUGGUGUGUGUGUGUG UGUGGU {ugImagesVidoes}")'''
	return finalug,{"images":[],"videos":[]}


def getUtranslate(data):
	punctuationuTemplate = "Given a sentence add punctuation when needed. sentence: {usentence} sentence with punctuation: "
	punctuationUprompt = PromptTemplate.from_template(punctuationuTemplate)
	punctuationUchain = punctuationUprompt | llm | StrOutputParser()


	grammaruTemplate = "Given a sentence correct the grammar. sentence:{upunctuated_sentence} sentence with correct grammar: "
	grammarUprompt = PromptTemplate.from_template(grammaruTemplate)
	grammarUchain = grammarUprompt | llm | StrOutputParser()

	translationuTemplate = "give a sentence translate that sentence into {ulanguage} sentence:{u_gramaticallyCorrect_Sentence} "
	translationUprompt = PromptTemplate.from_template(translationuTemplate)

	uchain = {"upunctuated_sentence":punctuationUchain} | {"u_gramaticallyCorrect_Sentence":grammarUchain,"ulanguage":RunnablePassthrough()} | translationUprompt | llm | StrOutputParser()
	print(uchain)
	return uchain.invoke({"usentence":data["usentence"],"ulanguage":data["ulanguage"],"upunctuated_sentence":punctuationUchain,"u_gramaticallyCorrect_Sentence":grammarUchain})	

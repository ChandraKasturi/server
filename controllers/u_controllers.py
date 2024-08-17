import os
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
import secrets
import string
import datetime
import json


from services.questionbank.mongo.questionbankrepu import insertIntoQuestionsU,CheckifEmailPresent
from services.questionbank.mongo.emailserviceu import send_email,register_email

import pymongo

#U_KEYS

c = pymongo.MongoClient("mongodb+srv://chandrakasturi:Bisleri1234@cluster0.ehbe5dz.mongodb.net/",server_api=pymongo.server_api.ServerApi('1'))
u_openai_api_key = ""
u_supabase_url = "https://uuvgdpvtndnglygvblht.supabase.co"
u_supabase_api_key = ""

#create a splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50) # ["\n\n", "\n", " ", ""] are default values
openai_uembeddings = OpenAIEmbeddings(openai_api_key=u_openai_api_key)
supabase_uclient = create_client(u_supabase_url,u_supabase_api_key)
#creating llm
llm = ChatOpenAI(openai_api_key=u_openai_api_key)
#creating retriever u_context
u_vecstore = SupabaseVectorStore(embedding=openai_uembeddings,client=supabase_uclient,query_name="match_science1",table_name="science")
u_vecstore_uimageuurl = SupabaseVectorStore(embedding=openai_uembeddings,client=supabase_uclient,query_name="match_u_image_uurl",table_name="uimageurl")
a_u = u_vecstore.as_retriever()
a_uimageuurl = u_vecstore_uimageuurl.as_retriever()	
retrieveru_from_llm = MultiQueryRetriever.from_llm(retriever=a_u,llm=llm)
retrieveruimguurl_from_llm = MultiQueryRetriever.from_llm(retriever=a_uimageuurl,llm=llm)




def InsertQuestionU(collectionU):
	return insertIntoQuestionsU(c,collectionU)



def loginU(user,password):
	rowU = c.sahasra_users.users.find({"$and":[{"username":{"$eq":f"{user}"}},{"password":{"$eq":f"{password}"}}]})
	if rowU:
		return True
	return False


def beforeRegisterU(nModel):
	print(dict(nModel))
	emailU = dict(nModel)["email"]
	fModelU = dict(nModel)
	token = ''.join([secrets.choice(string.ascii_uppercase+string.digits) for i in range(6)])
	emailTakenU = CheckifEmailPresent(c,emailU)
	if not emailTakenU:
		fModelU["token"] = token
		fModelU["ExpiresAt"] = datetime.datetime.utcnow()
		c.sahasra_tokens.register_tokens.create_index("ExpiresAt",expireAfterSeconds=10*60*60)
		c.sahasra_tokens.register_tokens.insert_one(fModelU)
		register_email(emailU,token)
		return "Register Token Has been sent to your Email ID Please Use it to Confirm Your Registration",200
	else:
		return "Email Already Taken",400


def registerU(rModel):
	token = dict(rModel)["token"]
	print(token)
	token_dbu = c.sahasra_tokens.register_tokens.find_one({"token":token})
	print(token_dbu)
	if token_dbu and token_dbu["token"] == token:
		try:
			_ = c.sahasra_users.users.insert_one(dict(rModel))
			return "User Registered SuccessFully You May Login Now"
		except Exception as e:
			print(e)
			return "Something Went Wrong"
	elif token_dbu and token_dbu["token"] != token:
		return "Wrong Token"
	else:
		return "Token Expired Try againU"
	'''emailU = dict(rModel).email
	rowU = c.sahasra_users.users.find({"$or":[{"username":{"$eq":dict(rModel).username}},{"email":{"$eq":dict(rModel).email}}]})
	if rowU != 1:
		return False
	c.saharsa_users.users.insert(dict(rModel))
	return True'''
	

def forgotPasswordU(forgotuModel):
	emailU = dict(forgotuModel)["email"]
	token = ''.join([secrets.choice(string.ascii_uppercase+string.digits) for i in range(6)])
	emailTakenU = CheckifEmailPresent(c,emailU)
	if emailTakenU:
		c.sahasra_tokens.password_tokens.create_index("ExpiresAt",expireAfterSeconds=10*60*60)
		c.sahasra_tokens.password_tokens.insert_one({"email":emailTakenU,"token":token,"ExpiresAt":datetime.datetime.utcnow()})
		try:
			send_email(emailTakenU,token)
			return "IF the Email You Provided Exists in our Database The reset Link Should be in Your Inbox Please Check your mail",200
		except Exception as e:
			print(e)
			return "something Went Wrong",400


def updatePasswordU(password,token):
	token_dbu = c.sahasra_tokens.password_tokens.find_one({"token":token})
	if token_dbu and token_dbu.token == token:
		try:
			_ = c.sahasra_users.users.update_one({"email":token_dbu.email},{"$set":{"password":password}})
			return "Password Updated SuccessFully You May Login Now",200
		except Exception as e:
			print(e)
			return "Something Went Wrong",400
	elif token_dbu and token_dbu.token != token:
		return "Wrong Token",400
	else:
		return "Token Expired Try againU",400










def getUimageuurl(ucontext):
	return retrieveruimguurl_from_llm.get_relevant_documents(query=ucontext)


def UploadUGImageUUrl(file):
	text_image_usplitter = CharacterTextSplitter(chunk_size=10,chunk_overlap=0,separator="\n") # need it store a valid image uurl
	file_ubytes_uimg = open(file,"rb").read()
	u_text_uimg = text_image_usplitter.create_documents([str(file_ubytes_uimg)])
	print(u_text_uimg)
	#file_ubytes_uimg = open(file,"rb").read().decode('utf-8').split("\n") # need to handle cases not utf-8
	#u_text_uimg = [Document(i) for i in file_ubytes_uimg]
	print(u_text_uimg)
	vector_ustore_uimg  = SupabaseVectorStore.from_documents(u_text_uimg,openai_uembeddings,client=supabase_uclient,table_name="uimageurl")



def UploadUGVector(file):
	file_ubytes = open(file,"rb").read()
	u_text = text_splitter.create_documents([str(file_ubytes)])
	vector_ustore = SupabaseVectorStore.from_documents(u_text,openai_uembeddings,client=supabase_uclient,table_name="science")
	return vector_ustore



def AssessUContent(udata,sessionIdu):
	u_response = {}
	u_response["response"] = []
	CheckUprompt = PromptTemplate.from_template("Answer Only AS YES OR NO IN CAPTIALS Answer whether the following statement is Related to this Context: \n Context:{context}\n Statement:{statement}")
	checkuChain = CheckUprompt | llm | StrOutputParser()
	AssessUG = checkuChain.invoke({"context":"Question me,Ask me,Assess me","statement":udata})



	ug_page_content = [u.page_content for u in retrieveru_from_llm.get_relevant_documents(query=udata)]
	ug_topics = [supabase_uclient.table("science").select("topic,subtopic,imageu_uurl,videou_uurl,u_formula").eq('content',u).execute() for u in ug_page_content]
	print("HEREEEEEEEEEEEE")
	print(ug_topics)
	print(ug_topics[2].data[0]["topic"])
	print(ug_topics[2].data[0]["subtopic"])
	print(AssessUG)
	u_questions = []
	if AssessUG == "YES":
		jsonU = json.dumps([i for i in c.sahasra_subjectdata.topic_subtopic.find({},{"_id":0})])
		print(jsonU)
		GetFinalUjson = PromptTemplate.from_template("Given The Question Produce a JSON containing Subject Topic Subtopic Level and NumberOfQuestions from the Question Keep the Default value for NumberOfQuestions as 5 if not found in the Question and the rest as empty if not found\n Question: {question}")
		GetFinalUchain = GetFinalUjson | llm | StrOutputParser()
		json_withU = GetFinalUchain.invoke({"question":udata})
		check_jsonU = json.loads(json_withU)
		noqug = check_jsonU["NumberOfQuestions"]
		ug_ugd = {}
		for k,v in check_jsonU.items():
			if v and type(v) == str:
				ug_ugd[k.lower()] = v

		print(check_jsonU)
		print(ug_ugd)
		collections_questionu = c.sahasra_questions.question_bank.find(ug_ugd).limit(noqug)
		u_questions.extend([dict(ug) for ug in collections_questionu])

		if not u_questions:
			return "Please Proivde The Right At least Subject for Assessment"

		return u_questions
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
	u_response["response"].extend([{"text":getUanswer(udata,sessionIdu)[0]},{"images":[ug.data[0]["imageu_uurl"] for ug in ug_topics]},{"videos":[ug.data[0]["videou_uurl"] for ug in ug_topics]},{"formula":[ug.data[0]["u_formula"] for ug in ug_topics]}])
	return u_response



def getUtweet(data):
	utweetPrompt = PromptTemplate.from_template("Generate a promotional tweet for a product, from this product description: {productDesc}")
	chain = utweetPrompt | llm | {"str": StrOutputParser()}
	return chain.invoke({"productDesc":data})["str"]


def getUanswer(data,sessionIdu):
	print("ugly")
	print(retrieveru_from_llm)
	questionualoneTemplate = "generate a question based on the following: {udata}"
	questionuPrompt = PromptTemplate.from_template(questionualoneTemplate)
	questionUchain = questionuPrompt | llm | StrOutputParser()
	print(questionUchain)
	generatedUquestion = questionUchain.invoke({"udata":data}) # a second invoke before main uinvoke bad way will change
	print(generatedUquestion)
	standaloneTemplate = "Keep the references or figures or links to any .png or keep the markdown syntax for the .png in between the sentence in the context Answer the Question with the references only on the following context If The Question is Not Related to The Context Act As Usual: {context}\n\n"
	chatGeneratedUTemplate = ChatPromptTemplate.from_messages([("system",standaloneTemplate),MessagesPlaceholder(variable_name="history"),("human","Question: {question}")])
	#standalonePrompt = PromptTemplate.from_template(standaloneTemplate)
	ug_page_content = [u.page_content for u in retrieveru_from_llm.get_relevant_documents(query=data)]
	ug_topics = [supabase_uclient.table("science").select("topic,subtopic").eq('content',u).execute() for u in ug_page_content]
	print(ug_topics)
	metaDatau = [u.metadata for u in retrieveru_from_llm.get_relevant_documents(query=data)]
	retriever = "".join(ug_page_content)
	standAloneChain =  chatGeneratedUTemplate | llm | StrOutputParser()
	standAloneUChainWithHistory = RunnableWithMessageHistory(standAloneChain,lambda sessionIdu: MongoDBChatMessageHistory(session_id=sessionIdu,connection_string="mongodb+srv://chandrakasturi:Bisleri1234@cluster0.ehbe5dz.mongodb.net/",database_name="sahasra_history",collection_name="history"),input_messages_key="question",history_messages_key="history")
	config = {"configurable":{"session_id":sessionIdu}}
	print(f"THIS IS UG CONTEXT {retriever}")
	return standAloneUChainWithHistory.invoke({"question":data,"context":retriever},config=config),metaDatau	


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

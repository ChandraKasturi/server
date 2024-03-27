import os
from supabase.client import create_client
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.vectorstores import SupabaseVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough

#U_KEYS

u_openai_api_key = os.environ.get("OPENAPI_API_KEY",None)
u_supabase_url = os.environ.get("SUPABASE_URL",None)
u_supabase_api_key = os.environ.get("SUPABASE_API_KEY",None)

#create a splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50) # ["\n\n", "\n", " ", ""] are default values
openai_uembeddings = OpenAIEmbeddings(openai_api_key=u_openai_api_key)
supabase_uclient = create_client(u_supabase_url,u_supabase_api_key)
#creating llm
llm = ChatOpenAI(openai_api_key=u_openai_api_key)
#creating retriever u_context
u_vecstore = SupabaseVectorStore(embedding=openai_uembeddings,client=supabase_uclient,query_name="match_flexnet",table_name="flexnet")
a_u = u_vecstore.as_retriever()
retrieveru_from_llm = MultiQueryRetriever.from_llm(retriever=a_u,llm=llm)


def UploadUGVector(file):
	file_ubytes = open(file,"rb").read()
	u_text = text_splitter.create_documents([str(file_ubytes)])
	vector_ustore = SupabaseVectorStore.from_documents(u_text,openai_uembeddings,client=supabase_uclient,table_name="science")
	return vector_ustore





def getUtweet(data):
	utweetPrompt = PromptTemplate.from_template("Generate a promotional tweet for a product, from this product description: {productDesc}")
	chain = utweetPrompt | llm | {"str": StrOutputParser()}
	return chain.invoke({"productDesc":data})["str"]


def getUanswer(data):
	questionualoneTemplate = "generate a question based on the following: {udata}"
	questionuPrompt = PromptTemplate.from_template(questionualoneTemplate)
	questionUchain = questionuPrompt | llm | StrOutputParser()
	generatedUquestion = questionUchain.invoke({"udata":data}) # a second invoke before main uinvoke bad way will change
	print(generatedUquestion)
	standaloneTemplate = "Answer the Question only on the following context: {context}\n\nQuestion: {question}"
	standalonePrompt = PromptTemplate.from_template(standaloneTemplate)
	retriever = "".join([u.page_content for u in retrieveru_from_llm.get_relevant_documents(query=data)])
	standAloneChain = {"context":lambda x: retriever,"question":RunnablePassthrough()} | standalonePrompt | llm | StrOutputParser()
	return standAloneChain.invoke(generatedUquestion)


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
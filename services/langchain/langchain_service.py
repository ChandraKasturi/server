from typing import List, Dict, Any, Optional, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI 
from langchain_classic.vectorstores import SupabaseVectorStore
from langchain_postgres import PGVectorStore
from langchain_postgres import PGEngine
from langchain_postgres.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.documents import Document
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from supabase.client import create_client
import uuid
import os
import json
from langchain_core.messages import HumanMessage, AIMessage
import asyncio
import concurrent.futures

from config import settings
from repositories.pgvector_repository import LangchainVectorRepository
from repositories.pdf_repository import PDFRepository

class LangchainService:
    """Service for LangChain-related operations."""
    
    def __init__(self, openai_api_key: Optional[str] = None, google_api_key: Optional[str] = None):
        """Initialize LangChain service.
        
        Args:
            openai_api_key: API key for OpenAI. If None, uses the one from settings.
            google_api_key: API key for Google. If None, uses the one from settings.
        """
        self.openai_api_key = openai_api_key or settings.OPENAI_API_KEY
        self.google_api_key = google_api_key or settings.GOOGLE_API_KEY
        
        # OpenAI models
        self.llm = ChatOpenAI(openai_api_key=self.openai_api_key)
        self.ug_llm = ChatOpenAI(openai_api_key=self.openai_api_key, model="gpt-4o")
        '''self.llm = ChatGoogleGenerativeAI(google_api_key=self.google_api_key, model="gemini-2.0-flash")
        self.ug_llm = ChatGoogleGenerativeAI(google_api_key=self.google_api_key, model="gemini-2.0-flash")'''
        

        # Google models (commented out but available)
        # self.llm = ChatGoogleGenerativeAI(google_api_key=self.google_api_key, model="gemini-1.5-pro")
        # self.ug_llm = ChatGoogleGenerativeAI(google_api_key=self.google_api_key, model="gemini-1.5-pro")
        
        # Embeddings
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        '''self.embeddings = GoogleGenerativeAIEmbeddings(google_api_key=self.google_api_key,model="models/gemini-embedding-exp-03-07")'''
        
        # Text splitters
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.image_text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10, separator="\n")
        
        # Supabase client
        self.supabase_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        
        # Vector stores
        # Biology vector store using PGVector
        print(f"PGVECTOR_CONNECTION_STRING: {settings.PGVECTOR_CONNECTION_STRING}")
        self.biology_vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name="class_x",
            connection=settings.PGVECTOR_CONNECTION_STRING,
            use_jsonb=True
        )
        
        self.image_url_vector_store = SupabaseVectorStore(
            embedding=self.embeddings,
            client=self.supabase_client,
            query_name=settings.SUPABASE_QUERY_MATCH_UIMAGEURL,
            table_name=settings.SUPABASE_TABLE_UIMAGEURL
        )
        
        # Retrievers
        self.biology_retriever = self.biology_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
        self.image_url_retriever = self.image_url_vector_store.as_retriever()
        
        # Multi-query retrievers
        self.multi_query_biology_retriever = MultiQueryRetriever.from_llm(
            retriever=self.biology_retriever,
            llm=self.llm
        )
        
        self.multi_query_image_url_retriever = MultiQueryRetriever.from_llm(
            retriever=self.image_url_retriever,
            llm=self.llm
        )
        
        # Thread pool for blocking operations
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=6)
    
    def setup_chat_history(self, student_id: str, session_id: str) -> MongoDBChatMessageHistory:
        """Set up a MongoDB-based chat history for a student.
        
        Args:
            student_id: ID of the student
            session_id: JWT token from X-Auth-Session to use as conversation ID
            
        Returns:
            MongoDBChatMessageHistory instance
        """
        return MongoDBChatMessageHistory(
            connection_string=settings.MONGO_URI,
            database_name=student_id,
            collection_name=settings.MONGO_DATABASE_HISTORY,
            session_id=session_id,
            # Use history_size to limit the number of messages stored
            history_size=10
        )
    
    def add_to_chat_history(self, student_id: str, session_id: str, 
                          message: str, is_ai: bool = True) -> None:
        """Add a message to the chat history.
        
        Args:
            student_id: ID of the student
            session_id: JWT token from X-Auth-Session to use as conversation ID
            message: Message content to add
            is_ai: Whether the message is from the AI (True) or user (False)
        """
        history = self.setup_chat_history(student_id, session_id)
        
        if is_ai:
            history.add_message(AIMessage(content=message))
        else:
            history.add_message(HumanMessage(content=message))
    
    def create_chat_chain_with_history(self, student_id: str, session_id: str) -> RunnableWithMessageHistory:
        """Create a chat chain with message history.
        
        Args:
            student_id: ID of the student
            session_id: JWT token from X-Auth-Session to use as conversation ID
            
        Returns:
            RunnableWithMessageHistory instance
        """
        # Set up retriever for relevant context
        retriever = self.multi_query_biology_retriever
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant for education focused on providing clear, accurate, and helpful information about academic subjects, particularly in the sciences. "
                       "Use the provided context to give accurate answers. "
                       "If you're unsure or the answer is not in the context, be honest about it."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        
        # Set up the retrieval chain
        chain = (
            {"context": retriever, "question": RunnablePassthrough()} 
            | prompt 
            | self.ug_llm 
            | StrOutputParser()
        )
        
        # Create a message history getter function that uses the exact JWT token as the session ID
        def get_chat_history(session_id_param):
            return self.setup_chat_history(student_id, session_id)
        
        # Wrap the chain with message history
        return RunnableWithMessageHistory(
            chain,
            get_chat_history,
            input_messages_key="question",
            history_messages_key="history"
        )
    
    def answer_question(self, question: str, student_id: str, session_id: str) -> Tuple[str, int]:
        """Answer a question using LangChain.
        
        Args:
            question: The user's question
            student_id: ID of the student
            session_id: JWT token from X-Auth-Session to use as conversation ID
            
        Returns:
            Tuple of (answer, status_code)
        """
        try:
            # Add the user question to the chat history
            self.add_to_chat_history(student_id, session_id, question, is_ai=False)
            
            # STEP 1: Get relevant documents from the general knowledge base
            retriever = self.multi_query_biology_retriever
            general_docs = retriever.get_relevant_documents(question)
            
            # Extract content from general knowledge documents
            general_context = [doc.page_content for doc in general_docs]
            
            # STEP 2: Try to get relevant documents from user's PDF collections
            pdf_docs = []
            
            try:
                # Get the repository to fetch user PDFs
                pdf_repository = PDFRepository()
                
                # Get all PDFs for this student that have been processed successfully
                user_pdfs = pdf_repository.get_user_pdf_documents(student_id)
                completed_pdfs = [pdf for pdf in user_pdfs if pdf.processing_status == "completed"]
                
                if completed_pdfs:
                    # Create student-specific connection string
                    base_connection = settings.POSTGRES_CONNECTION_STRING
                    
                    # Parse the base connection string to insert student_id
                    if "://" in base_connection and "@" in base_connection:
                        prefix = base_connection[:base_connection.rindex('@') + 1]
                        suffix = base_connection[base_connection.rindex('@') + 1:]
                        
                        # If suffix contains a slash, it has host:port/dbname
                        if '/' in suffix:
                            host_port = suffix[:suffix.index('/')]
                            # Replace the database name with student-specific one
                            student_db = f"student_{student_id}"
                            
                            # Reconstruct the connection string
                            connection_string = f"{prefix}{host_port}/{student_db}"
                        else:
                            # If no database name in original connection, just append it
                            connection_string = f"{base_connection}/student_{student_id}"
                    else:
                        # Fallback: just use base connection
                        connection_string = base_connection
                    
                    # For each processed PDF, try to retrieve relevant content
                    for pdf in completed_pdfs[:5]:  # Limit to 5 PDFs for performance
                        try:
                            # Create a collection name specific to this PDF
                            collection_name = f"pdf_{pdf.id}"
                            
                            # Initialize PGVector with the student-specific connection
                            pdf_vector_store = PGVector(
                                embeddings=self.embeddings,
                                collection_name=collection_name,
                                connection=connection_string,
                                use_jsonb=True
                            )
                            
                            # Create a retriever from the vector store
                            pdf_retriever = pdf_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2})
                            
                            # Get relevant documents from this PDF
                            pdf_results = pdf_retriever.get_relevant_documents(question)
                            
                            # Add source information to each document
                            for doc in pdf_results:
                                if not hasattr(doc, "metadata"):
                                    doc.metadata = {}
                                doc.metadata["source"] = f"From your document: {pdf.title}"
                            
                            # Add to the list of retrieved documents
                            pdf_docs.extend(pdf_results)
                        except Exception as e:
                            print(f"Error retrieving from PDF {pdf.id}: {str(e)}")
                            continue
            except Exception as e:
                print(f"Error accessing student PDFs: {str(e)}")
            
            # Extract content from PDF documents
            pdf_context = [f"{doc.metadata.get('source', '')}: {doc.page_content}" for doc in pdf_docs]
            
            # STEP 3: Combine both contexts
            all_context_parts = []
            
            # Add general knowledge context if available
            if general_context:
                all_context_parts.append("General Knowledge:")
                all_context_parts.extend(general_context)
            
            # Add personal PDF context if available
            if pdf_context:
                all_context_parts.append("\nFrom Your Documents:")
                all_context_parts.extend(pdf_context)
            
            # Join all context parts
            context = "\n".join(all_context_parts) if all_context_parts else ""
            
            # Create prompt template with combined context
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an AI assistant for education focused on providing clear, accurate, and helpful information about academic subjects. "
                          "Use the provided context to give accurate answers. "
                          "The context includes both general knowledge and content from the student's own documents if available. "
                          "When answering, clearly indicate if your information comes from the student's own documents. "
                          "If you're unsure or the answer is not in the context, be honest about it.\n\n"
                          "Context: {context}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}")
            ])
            
            # Create chain
            chain = prompt | self.ug_llm | StrOutputParser()
            
            # Get message history
            history = self.setup_chat_history(student_id, session_id)
            
            # Run chain with context
            answer = chain.invoke({
                "context": context,
                "question": question,
                "history": history.messages
            })
            
            # Add the AI's answer to the chat history
            self.add_to_chat_history(student_id, session_id, answer, is_ai=True)
            
            return answer, 200
        except Exception as e:
            return f"Error generating answer: {str(e)}", 500
    
    def upload_vector(self, text: str, subject: str) -> Tuple[str, int]:
        """Upload text to vector storage.
        
        Args:
            text: Text to upload
            subject: Subject category for the text
            
        Returns:
            Tuple of (message, status_code)
        """
        try:
            # Create documents from text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=len(text), chunk_overlap=0)
            documents = text_splitter.create_documents([str(text)])
            
            # Add document to vector store
            repo = LangchainVectorRepository(collection_name=subject)
            ids = repo.add_documents(documents)
            
            return "Vector uploaded successfully", 200
        except Exception as e:
            return f"Error uploading vector: {str(e)}", 500
    
    def upload_image_url(self, file_path: str) -> Tuple[str, int]:
        """Upload image URL data to vector storage.
        
        Args:
            file_path: Path to the file containing image URL data
            
        Returns:
            Tuple of (message, status_code)
        """
        try:
            # Read file
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            
            # Create documents
            documents = self.image_text_splitter.create_documents([str(file_bytes)])
            
            # Create vector store
            vector_store = SupabaseVectorStore.from_documents(
                documents,
                self.embeddings,
                client=self.supabase_client,
                table_name=settings.SUPABASE_TABLE_UIMAGEURL
            )
            
            return "Image URL data uploaded successfully", 200
        except Exception as e:
            return f"Error uploading image URL data: {str(e)}", 500
    
    def get_image_url(self, context: str) -> List[Document]:
        """Get relevant image URLs based on context.
        
        Args:
            context: Text context to find relevant images for
            
        Returns:
            List of document objects containing image URLs
        """
        return self.multi_query_image_url_retriever.get_relevant_documents(query=context)
    
    def generate_tweet(self, topic: str) -> Tuple[str, int]:
        """Generate a tweet about a topic.
        
        Args:
            topic: Topic to generate a tweet about
            
        Returns:
            Tuple of (tweet, status_code)
        """
        try:
            tweet_prompt = PromptTemplate.from_template(
                "Generate a concise, engaging tweet (under 280 characters) about {topic}. "
                "Include relevant hashtags and make it informative yet conversational."
            )
            
            tweet_chain = tweet_prompt | self.llm | StrOutputParser()
            
            tweet = tweet_chain.invoke({"topic": topic})
            
            return tweet, 200
        except Exception as e:
            return f"Error generating tweet: {str(e)}", 500
    
    def translate_text(self, text: str, target_language: str) -> Tuple[str, int]:
        """Translate text to another language.
        
        Args:
            text: Text to translate
            target_language: Target language
            
        Returns:
            Tuple of (translated_text, status_code)
        """
        try:
            translate_prompt = PromptTemplate.from_template(
                "Translate the following text to {language}:\n\n{text}"
            )
            
            translate_chain = translate_prompt | self.llm | StrOutputParser()
            
            translated = translate_chain.invoke({
                "language": target_language,
                "text": text
            })
            
            return translated, 200
        except Exception as e:
            return f"Error translating text: {str(e)}", 500
    
    async def _get_pdf_documents_async(self, student_id: str, question: str, connection_string: str, collection_name: str) -> List[Document]:
        """Async version of getting relevant documents from PDF.
        
        Args:
            student_id: ID of the student
            question: Question to search for
            connection_string: Database connection string
            collection_name: Collection name
            
        Returns:
            List of relevant documents
        """
        def _get_docs_sync():
            # Initialize PGVector with the student-specific connection
            ug = PGEngine.from_connection_string(url=connection_string)
            pdf_vector_store = PGVectorStore.create_sync(
                engine=ug,
                embedding_service=self.embeddings,
                table_name=collection_name,
            )
            
            # Create a retriever from the vector store
            pdf_retriever = pdf_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
            
            # Get relevant documents for the question
            return pdf_retriever.get_relevant_documents(question)
        
        # Run the synchronous operation in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, _get_docs_sync)

    async def _find_relevant_image_async(self, connection_string: str, collection_name: str, query: str, similarity_threshold: float = 0.4) -> Optional[Dict]:
        """Async version of finding a relevant image for the query.
        
        Args:
            connection_string: Connection string for the database
            collection_name: Name of the image captions collection
            query: The query to search for
            similarity_threshold: Minimum similarity score required
            
        Returns:
            Dictionary with image information if found, None otherwise
        """
        def _find_image_sync():
            return self._find_relevant_image(connection_string, collection_name, query, similarity_threshold)
        
        # Run the synchronous operation in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, _find_image_sync)

    async def _run_chain_async(self, chain, input_data: Dict) -> str:
        """Async version of running a LangChain chain.
        
        Args:
            chain: LangChain chain to run
            input_data: Input data for the chain
            
        Returns:
            Chain output
        """
        def _run_chain_sync():
            return chain.invoke(input_data)
        
        # Run the chain in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, _run_chain_sync)

    async def _add_to_chat_history_async(self, student_id: str, session_id: str, message: str, is_ai: bool = True) -> None:
        """Async version of adding to chat history.
        
        Args:
            student_id: ID of the student
            session_id: Session ID
            message: Message to add
            is_ai: Whether message is from AI
        """
        def _add_history_sync():
            return self.add_to_chat_history(student_id, session_id, message, is_ai)
        
        # Run the database operation in a thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.thread_pool, _add_history_sync)

    def _get_pdf_session_history(self, student_id: str, pdf_id: str):
        """Factory function to create chat history for PDF learning with RunnableWithMessageHistory.
        
        Args:
            student_id: ID of the student
            pdf_id: ID of the PDF document
            
        Returns:
            Function that creates MongoDBChatMessageHistory for a given session_id
        """
        def get_history(session_id: str) -> MongoDBChatMessageHistory:
            collection_name = f"{settings.MONGO_DATABASE_HISTORY}_{pdf_id}"
            return MongoDBChatMessageHistory(
                connection_string=settings.MONGO_URI,
                database_name=student_id,
                collection_name=collection_name,
                session_id=session_id,
                history_size=10
            )
        return get_history

    async def _run_pdf_chain_async(self, chain_with_history, chain_input: Dict, config: Dict) -> str:
        """Async version of running the PDF learning chain with history.
        
        Args:
            chain_with_history: Chain with message history
            chain_input: Input for the chain
            config: Configuration for the chain
            
        Returns:
            Generated answer
        """
        def _run_chain_sync():
            return chain_with_history.invoke(chain_input, config=config)
        
        # Run the chain in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, _run_chain_sync)

    async def learn_from_pdf(self, student_id: str, pdf_id: str, question: str, 
                  session_id: str = None, similarity_threshold: float = 0.75) -> Tuple[Dict, int]:
        """Learn from a specific PDF document using RAG (async version).
        
        Args:
            student_id: ID of the student
            pdf_id: ID of the PDF document to learn from
            question: The user's question about the PDF content
            session_id: Optional session ID for chat history
            similarity_threshold: Minimum similarity score for images (0-1, where lower is more similar for distance metrics)
            
        Returns:
            Tuple of (response_data, status_code)
        """
        try:
            # Create a collection name specific to this PDF
            collection_name = f"pdf_{pdf_id}"
            # Create collection name for images
            images_collection_name = f"pdf_{pdf_id}_images"
            
            # Create student-specific connection string
            # Format: postgresql+psycopg://myuser:mypassword@localhost:5432/student_{user_id}
            base_connection = settings.POSTGRES_CONNECTION_STRING
            
            # Parse the base connection string to insert student_id
            if "://" in base_connection and "@" in base_connection:
                prefix = base_connection[:base_connection.rindex('@') + 1]
                suffix = base_connection[base_connection.rindex('@') + 1:]
                
                # If suffix contains a slash, it has host:port/dbname
                if '/' in suffix:
                    host_port = suffix[:suffix.index('/')]
                    # Replace the database name with student-specific one
                    student_db = f"student_{student_id}"
                    
                    # Reconstruct the connection string
                    connection_string = f"{prefix}{host_port}/{student_db}"
                else:
                    # If no database name in original connection, just append it
                    connection_string = f"{base_connection}/student_{student_id}"
            else:
                # Fallback: just use base connection
                connection_string = base_connection
            
            # Get relevant documents for the question (async)
            retrieved_docs = await self._get_pdf_documents_async(student_id, question, connection_string, collection_name)
            
            # Extract content from retrieved documents
            context_content = [doc.page_content for doc in retrieved_docs]
            context = "\n".join(context_content) if context_content else ""
            
            # If no context was retrieved, return an error
            if not context:
                return "I couldn't find relevant information in this PDF to answer your question. Please try a different question or check if the PDF has been processed correctly.", 404
            
            # Find relevant images for the question (async)
            relevant_image = await self._find_relevant_image_async(
                connection_string, 
                images_collection_name, 
                question,
                similarity_threshold
            )
            
            # Create prompt template with context, history, and potential image reference
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an educational assistant helping a student learn from a specific PDF document. "
                          "Use only the provided context from the PDF to answer the question. "
                          "If the answer isn't in the context, acknowledge this and suggest what might be relevant. "
                          "If an image is provided, incorporate it into your explanation and mention that there's a visual reference available. "
                          "Keep your answers clear, educational, and directly related to the PDF content.\n\n"
                          "Context from the PDF:\n{context}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}")
            ])
            
            # Create chain
            chain = prompt | self.ug_llm | StrOutputParser()
            
            # Prepare input for the chain
            chain_input = {
                "context": context,
                "question": question
            }
            
            # Create structured response
            response = {
                "answer": "",
                "has_image": relevant_image is not None
            }
            
            # Add image information to the response if available
            if relevant_image:
                response["image_url"] = relevant_image["image_url"]
                response["image_caption"] = relevant_image["caption"]
                response["image_page"] = relevant_image.get("page_number")
            
            # Run chain with history if session_id is provided, otherwise without history
            if session_id:
                # Create chain with message history
                chain_with_history = RunnableWithMessageHistory(
                    chain,
                    self._get_pdf_session_history(student_id, pdf_id),
                    input_messages_key="question",
                    history_messages_key="history",
                )
                
                # Configure session
                config = {"configurable": {"session_id": session_id}}
                
                # Run chain with history (async)
                answer = await self._run_pdf_chain_async(chain_with_history, chain_input, config)
            else:
                # Fallback: run without history if no session_id (async)
                chain_input["history"] = []  # Empty history
                answer = await self._run_chain_async(chain, chain_input)
            
            # Set the answer in response
            response["answer"] = answer
            
            return response, 200
            
        except Exception as e:
            return f"Error learning from PDF: {str(e)}", 500
            
    def _find_relevant_image(self, connection_string: str, collection_name: str, query: str, similarity_threshold: float = 0.4) -> Optional[Dict]:
        """Find a relevant image for the query from the image captions collection.
        
        Args:
            connection_string: Connection string for the database
            collection_name: Name of the image captions collection
            query: The query to search for
            similarity_threshold: Minimum similarity score required (lower scores mean more similar for distance metrics)
            
        Returns:
            Dictionary with image information if found, None otherwise
        """
        try:
            # Try to connect to the image captions collection
            try:
                ug = PGEngine.from_connection_string(url=connection_string)
                image_vector_store = PGVectorStore.create_sync(
                    engine=ug,
                    embedding_service=self.embeddings,
                    table_name=collection_name,
                )
                
                # Perform similarity search to find relevant image
                results = image_vector_store.similarity_search_with_score(
                    query, 
                    k=1  # Get the most relevant image
                )
                
                if results and len(results) > 0:
                    # Extract image information from the document and metadata
                    doc, score = results[0]
                    
                    # Check if the similarity score meets the threshold
                    # Note: For distance-based metrics, lower scores mean more similar
                    # For PGVector, scores are typically distance metrics
                    if score <= similarity_threshold:
                        if doc.metadata and "image_url" in doc.metadata:
                            return {
                                "image_url": doc.metadata["image_url"],
                                "caption": doc.page_content,
                                "score": score,
                                "page_number": doc.metadata.get("page_number")
                            }
                    else:
                        print(f"Image found but similarity score {score} doesn't meet threshold {similarity_threshold}")
                        return None
            except Exception as e:
                print(f"Error searching for images: {str(e)}")
                return None
                
        except Exception as e:
            print(f"Error finding relevant image: {str(e)}")
            
        return None 
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_postgres.vectorstores import PGVector
from langchain.embeddings import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
import concurrent.futures

from config import settings
from repositories.pdf_repository import PDFRepository
from repositories.mongo_repository import HistoryRepository

class LearningService:
    """Service for subject-specific learning using RAG."""
    
    # Subject-specific collection names in PGVector
    SUBJECT_COLLECTIONS = {
        "science": "science",
        "social_science": "social_science",
        "mathematics": "mathematics", 
        "english": "english",
        "hindi": "hindi"
    }
    
    # Subject-specific system prompts
    SUBJECT_PROMPTS = {
        "science": "You are a science education assistant. Focus on providing clear, accurate explanations of scientific concepts, principles, and phenomena. Use precise scientific terminology and explain complex ideas in an accessible way. When possible, relate scientific concepts to real-world applications and everyday experiences.",
        
        "social_science": "You are a social science education assistant. Provide balanced, nuanced perspectives on historical events, geographical concepts, civics, and economic principles. Present multiple viewpoints when relevant and emphasize critical thinking about social issues. Avoid politically biased language and present information objectively.",
        
        "mathematics": "You are a mathematics education assistant. Present mathematical concepts with precision and clarity. Walk through problem-solving steps methodically, explaining each step's reasoning. Use mathematical notation appropriately, and when possible, provide visual representations or analogies to aid understanding of abstract concepts.",
        
        "english": "You are an English language and literature education assistant. Provide clear explanations of grammar rules, literary devices, and writing techniques. When analyzing texts, focus on themes, character development, narrative structure, and authorial intent. Offer constructive guidance for improving writing and language skills.",
        
        "hindi": "You are a Hindi language education assistant. Focus on proper grammar, vocabulary, and language usage. Explain linguistic concepts clearly and provide examples that aid understanding. When discussing Hindi literature, highlight cultural contexts, literary traditions, and notable works. Support students in developing their Hindi language skills with clear, practical guidance."
    }
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the learning service.
        
        Args:
            openai_api_key: API key for OpenAI. If None, uses the one from settings.
        """
        self.api_key = openai_api_key or settings.OPENAI_API_KEY
        self.embeddings = OpenAIEmbeddings(api_key=self.api_key)
        '''self.embeddings = GoogleGenerativeAIEmbeddings(google_api_key=settings.GOOGLE_API_KEY,model="models/gemini-embedding-exp-03-07")'''
        self.llm = ChatOpenAI(api_key=self.api_key)
        self.ug_llm = ChatOpenAI(api_key=self.api_key, model="gpt-4o")
        '''self.llm = ChatGoogleGenerativeAI(google_api_key=settings.GOOGLE_API_KEY, model="gemini-2.0-flash")
        self.ug_llm = ChatGoogleGenerativeAI(google_api_key=settings.GOOGLE_API_KEY, model="gemini-2.0-flash")'''
        self.pdf_repository = PDFRepository()
        self.history_repository = HistoryRepository()
        # Thread pool for blocking operations
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    def _setup_chat_history(self, student_id: str, session_id: str, subject: str) -> MongoDBChatMessageHistory:
        """Set up a MongoDB-based chat history for a student with subject-specific collection.
        
        Args:
            student_id: ID of the student
            session_id: JWT token from X-Auth-Session 
            subject: Subject for this chat history
            
        Returns:
            MongoDBChatMessageHistory instance
        """
        # Use subject-specific collection name
        collection_name = f"{settings.MONGO_DATABASE_HISTORY}_{subject}"
        
        return MongoDBChatMessageHistory(
            connection_string=settings.MONGO_URI,
            database_name=student_id,
            collection_name=collection_name,
            session_id=session_id,
            history_size=10
        )

    def _get_session_history(self, student_id: str, subject: str):
        """Factory function to create chat history for RunnableWithMessageHistory.
        
        Args:
            student_id: ID of the student
            subject: Subject for this chat history
            
        Returns:
            Function that creates MongoDBChatMessageHistory for a given session_id
        """
        def get_history(session_id: str) -> MongoDBChatMessageHistory:
            return self._setup_chat_history(student_id, session_id, subject)
        return get_history
    
    def _get_student_specific_connection_string(self, student_id: str) -> str:
        """Get a student-specific connection string for PGVector.
        
        Args:
            student_id: ID of the student
            
        Returns:
            Connection string for the student's database
        """
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
            
        return connection_string
    
    def _get_subject_pdf_content(self, student_id: str, subject: str, question: str) -> List[Dict]:
        """Retrieve content from user's PDFs related to the specified subject.
        
        Args:
            student_id: ID of the student
            subject: Subject to filter PDFs by
            question: Question to retrieve relevant content for
            
        Returns:
            List of documents with relevant content
        """
        pdf_docs = []
        
        try:
            # Get all PDFs for this student that have been processed successfully
            user_pdfs = self.pdf_repository.get_user_pdf_documents(student_id)
            completed_pdfs = [pdf for pdf in user_pdfs if pdf.processing_status == "completed"]
            
            # Filter PDFs by subject if specified
            subject_pdfs = []
            for pdf in completed_pdfs:
                pdf_subject = pdf.metadata.get("subject", "").lower() if pdf.metadata else ""
                
                # Match PDFs with this subject or with no subject specified
                if subject.lower() in pdf_subject or not pdf_subject:
                    subject_pdfs.append(pdf)
            
            if subject_pdfs:
                # Create student-specific connection string
                connection_string = self._get_student_specific_connection_string(student_id)
                
                # For each processed PDF, try to retrieve relevant content
                for pdf in subject_pdfs[:5]:  # Limit to 5 PDFs for performance
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
        
        return pdf_docs
    
    async def _get_subject_pdf_content_async(self, student_id: str, subject: str, question: str) -> List[Dict]:
        """Async version of getting content from user's PDFs related to the specified subject.
        
        Args:
            student_id: ID of the student
            subject: Subject to filter PDFs by
            question: Question to retrieve relevant content for
            
        Returns:
            List of documents with relevant content
        """
        def _get_pdf_content_sync():
            return self._get_subject_pdf_content(student_id, subject, question)
        
        # Run the synchronous operation in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, _get_pdf_content_sync)

    async def _get_vector_store_results_async(self, subject_collection: str, question: str) -> List:
        """Async version of getting results from vector store.
        
        Args:
            subject_collection: Subject collection name
            question: Question to search for
            
        Returns:
            List of relevant documents
        """
        def _get_results_sync():
            # Initialize PGVector with the subject collection
            subject_vector_store = PGVector(
                embeddings=self.embeddings,
                collection_name=subject_collection,
                connection=settings.PGVECTOR_CONNECTION_STRING,
                use_jsonb=True
            )
            
            # Create a retriever from the vector store
            subject_retriever = subject_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
            
            # Get relevant documents from subject collection
            return subject_retriever.get_relevant_documents(question)
        
        # Run the synchronous operation in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, _get_results_sync)

    async def _run_chain_async(self, chain_with_history, chain_input: Dict, config: Dict) -> str:
        """Async version of running the chain with history.
        
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

    async def _run_chain_without_history_async(self, chain, chain_input: Dict) -> str:
        """Async version of running the chain without history.
        
        Args:
            chain: Chain without history
            chain_input: Input for the chain
            
        Returns:
            Generated answer
        """
        def _run_chain_sync():
            return chain.invoke(chain_input)
        
        # Run the chain in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, _run_chain_sync)

    async def _store_history_async(self, student_id: str, history_data: Dict) -> None:
        """Async version of storing history data.
        
        Args:
            student_id: ID of the student
            history_data: History data to store
        """
        def _store_history_sync():
            return self.history_repository.add_history_item(student_id, history_data)
        
        # Run the database operation in a thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.thread_pool, _store_history_sync)

    async def learn_subject(self, 
                    subject: str, 
                    question: str, 
                    student_id: str, 
                    session_id: str = None,
                    include_pdfs: bool = True) -> Tuple[str, int]:
        """Learn about a specific subject (async version).
        
        Args:
            subject: Subject to learn about (science, social_science, mathematics, english, hindi)
            question: Question about the subject
            student_id: ID of the student
            session_id: JWT token from X-Auth-Session (optional)
            include_pdfs: Whether to include user's PDFs in the answer
            
        Returns:
            Tuple of (answer, status_code)
        """
        try:
            # Validate subject
            if subject not in self.SUBJECT_COLLECTIONS:
                return f"Invalid subject: {subject}. Valid subjects are: {', '.join(self.SUBJECT_COLLECTIONS.keys())}", 400
            
            # STEP 1: Get relevant documents from the subject knowledge base (async)
            subject_collection = self.SUBJECT_COLLECTIONS[subject]
            subject_docs = await self._get_vector_store_results_async(subject_collection, question)
            
            # Extract content from subject knowledge documents
            subject_context = [doc.page_content for doc in subject_docs]
            
            # STEP 2: Get relevant documents from user's PDFs if requested (async)
            pdf_docs = []
            if include_pdfs:
                pdf_docs = await self._get_subject_pdf_content_async(student_id, subject, question)
            
            # Extract content from PDF documents with source info
            pdf_context = [f"{doc.metadata.get('source', '')}: {doc.page_content}" for doc in pdf_docs]
            
            # STEP 3: Combine both contexts
            all_context_parts = []
            
            # Add subject knowledge context if available
            if subject_context:
                all_context_parts.append(f"{subject.capitalize()} Knowledge:")
                all_context_parts.extend(subject_context)
            
            # Add personal PDF context if available
            if pdf_context:
                all_context_parts.append("\nFrom Your Documents:")
                all_context_parts.extend(pdf_context)
            
            # Join all context parts
            context = "\n".join(all_context_parts) if all_context_parts else ""            
            
            # STEP 4: Create prompt with subject-specific system message and history
            system_prompt = self.SUBJECT_PROMPTS.get(subject, "You are an educational assistant.")
            
            prompt_messages = [
                ("system", f"{system_prompt}\n\n"
                          "Use the provided context to give accurate answers. "
                          "If your answer includes information from the student's own documents, clearly indicate this. "
                          "If you're unsure or the answer is not in the context, be honest about it.\n\n"
                          "Context: {context}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}")
            ]
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages(prompt_messages)
            
            # STEP 5: Create chain with history integration
            chain = prompt | self.ug_llm | StrOutputParser()
            
            # Prepare input for the chain
            chain_input = {
                "context": context,
                "question": question
            }
            
            # Create chain with message history if session_id is provided and run async
            if session_id:
                chain_with_history = RunnableWithMessageHistory(
                    chain,
                    self._get_session_history(student_id, subject),
                    input_messages_key="question",
                    history_messages_key="history",
                )
                
                # Configure session
                config = {"configurable": {"session_id": session_id}}
                
                # Run chain with history to get answer (async)
                answer = await self._run_chain_async(chain_with_history, chain_input, config)
            else:
                # Fallback: run without history if no session_id (async)
                chain_input["history"] = []  # Empty history
                answer = await self._run_chain_without_history_async(chain, chain_input)
            
            # STEP 6: Store in sahasra_history for persistence (async)
            # Store user question
            user_history_data = {
                "subject": subject,
                "message": question,
                "is_ai": False,
                "time": datetime.utcnow(),
                "session_id": session_id
            }
            await self._store_history_async(student_id, user_history_data)
            
            # Store AI response
            ai_history_data = {
                "subject": subject,
                "message": answer,
                "is_ai": True,
                "time": datetime.utcnow(),
                "session_id": session_id
            }
            await self._store_history_async(student_id, ai_history_data)
            
            return answer, 200
            
        except Exception as e:
            error_message = f"Error learning about {subject}: {str(e)}"
            print(error_message)
            return error_message, 500
    
    # Subject-specific convenience methods (now async)
    
    async def learn_science(self, question: str, student_id: str, session_id: str = None, include_pdfs: bool = True) -> Tuple[str, int]:
        """Learn about science (async).
        
        Args:
            question: Question about science
            student_id: ID of the student
            session_id: JWT token from X-Auth-Session (optional)
            include_pdfs: Whether to include user's PDFs in the answer
            
        Returns:
            Tuple of (answer, status_code)
        """
        return await self.learn_subject("science", question, student_id, session_id, include_pdfs)
    
    async def learn_social_science(self, question: str, student_id: str, session_id: str = None, include_pdfs: bool = True) -> Tuple[str, int]:
        """Learn about social science (async).
        
        Args:
            question: Question about social science
            student_id: ID of the student
            session_id: JWT token from X-Auth-Session (optional)
            include_pdfs: Whether to include user's PDFs in the answer
            
        Returns:
            Tuple of (answer, status_code)
        """
        return await self.learn_subject("social_science", question, student_id, session_id, include_pdfs)
    
    async def learn_mathematics(self, question: str, student_id: str, session_id: str = None, include_pdfs: bool = True) -> Tuple[str, int]:
        """Learn about mathematics (async).
        
        Args:
            question: Question about mathematics
            student_id: ID of the student
            session_id: JWT token from X-Auth-Session (optional)
            include_pdfs: Whether to include user's PDFs in the answer
            
        Returns:
            Tuple of (answer, status_code)
        """
        return await self.learn_subject("mathematics", question, student_id, session_id, include_pdfs)
    
    async def learn_english(self, question: str, student_id: str, session_id: str = None, include_pdfs: bool = True) -> Tuple[str, int]:
        """Learn about English (async).
        
        Args:
            question: Question about English
            student_id: ID of the student
            session_id: JWT token from X-Auth-Session (optional)
            include_pdfs: Whether to include user's PDFs in the answer
            
        Returns:
            Tuple of (answer, status_code)
        """
        return await self.learn_subject("english", question, student_id, session_id, include_pdfs)
    
    async def learn_hindi(self, question: str, student_id: str, session_id: str = None, include_pdfs: bool = True) -> Tuple[str, int]:
        """Learn about Hindi (async).
        
        Args:
            question: Question about Hindi
            student_id: ID of the student
            session_id: JWT token from X-Auth-Session (optional)
            include_pdfs: Whether to include user's PDFs in the answer
            
        Returns:
            Tuple of (answer, status_code)
        """
        return await self.learn_subject("hindi", question, student_id, session_id, include_pdfs) 
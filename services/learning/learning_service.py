import asyncio
import base64
import os
import uuid
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
from repositories.mongo_repository import HistoryRepository, QuotesRepository, QuestionRepository

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
        self.quotes_repository = QuotesRepository()
        self.question_repository = QuestionRepository()
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
                pdf_subject = (pdf.metadata.subject or "").lower() if pdf.metadata else ""
                
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

    def _process_base64_image(self, base64_data: str, field_name: str) -> str:
        """Process base64 image data and save it to static directory.
        
        Args:
            base64_data: Base64 encoded image data with format "data:image/format;base64,..."
            field_name: Name of the field (question_image or explaination_image)
            
        Returns:
            URL path to the saved image
            
        Raises:
            ValueError: If base64 data format is invalid
        """
        try:
            # Check if it's base64 image data
            if not base64_data.startswith("data:image/"):
                return base64_data  # Return as-is if not base64 image
            
            # Parse the base64 data
            header, image_data = base64_data.split(",", 1)
            image_format = header.split("/")[1].split(";")[0].lower()  # Extract format (png, jpg, etc.)
            
            # Validate image format
            allowed_formats = ["png", "jpg", "jpeg", "gif", "webp"]
            if image_format not in allowed_formats:
                raise ValueError(f"Unsupported image format: {image_format}")
            
            # Decode base64 data
            image_bytes = base64.b64decode(image_data)
            
            # Generate unique filename
            unique_id = str(uuid.uuid4())
            filename = f"{unique_id}.{image_format}"
            
            # Determine subdirectory based on field name
            if field_name == "question_image":
                subdirectory = "question_images"
            elif field_name == "explaination_image":
                subdirectory = "explaination_images"
            else:
                subdirectory = "question_images"  # fallback
            
            # Get the correct path to server/static directory
            # This file is in server/services/learning/learning_service.py
            # Use absolute path approach to be 100% sure
            current_file_path = os.path.abspath(__file__)
            print(f"Absolute file path: {current_file_path}")
            
            # Split path and find the actual server directory index
            path_parts = current_file_path.replace('\\', '/').split('/')
            print(f"Path parts: {path_parts}")
            
            # Find where 'server' appears in the path
            server_indices = [i for i, part in enumerate(path_parts) if part == 'server']
            print(f"Server indices found: {server_indices}")
            
            if server_indices:
                # Use the first occurrence of 'server' directory
                server_index = server_indices[0]
                server_path_parts = path_parts[:server_index + 1]
                server_dir = '/'.join(server_path_parts)
                # Convert back to Windows path format if needed
                server_dir = server_dir.replace('/', os.sep)
                print(f"Constructed server path: {server_dir}")
            else:
                # Ultimate fallback - use working directory + server
                working_dir = os.getcwd()
                server_dir = os.path.join(working_dir, 'server')
                print(f"Fallback server path: {server_dir}")
            
            # Construct static directory path
            static_dir = os.path.join(server_dir, "static")
            images_dir = os.path.join(static_dir, subdirectory)
            
            print(f"Final static directory: {static_dir}")
            print(f"Final images directory: {images_dir}")
            
            # Validate the server directory
            if os.path.exists(server_dir):
                server_contents = os.listdir(server_dir)
                print(f"Server directory contents: {server_contents}")
                
                # Check if this looks like the right server directory
                expected_dirs = ['services', 'models', 'repositories', 'routers']
                found_expected = [d for d in expected_dirs if d in server_contents]
                print(f"Expected server subdirs found: {found_expected}")
                
                if len(found_expected) < 2:
                    print(f"WARNING: This might not be the right server directory!")
                    print(f"Expected to find: {expected_dirs}")
                    print(f"Actually found: {server_contents}")
            else:
                print(f"WARNING: Server directory does not exist: {server_dir}")
            
            # Ensure directory exists
            os.makedirs(images_dir, exist_ok=True)
            print(f"Created directory: {images_dir}")
            
            # Double-check the directory was created in the right place
            if os.path.exists(images_dir):
                print(f"✓ SUCCESS: Directory exists at {images_dir}")
                # Also check if it's in the right place by verifying the parent structure
                parent_static = os.path.dirname(images_dir)
                parent_server = os.path.dirname(parent_static)
                print(f"✓ Parent static dir: {parent_static}")
                print(f"✓ Parent server dir: {parent_server}")
            else:
                print(f"✗ FAILED: Directory does not exist: {images_dir}")
                raise ValueError(f"Failed to create directory: {images_dir}")
            
            # Save image file
            file_path = os.path.join(images_dir, filename)
            with open(file_path, "wb") as f:
                f.write(image_bytes)
            
            print(f"Saved image to: {file_path}")
            print(f"File size: {len(image_bytes)} bytes")
            
            # Return URL path (relative to static directory)
            url_path = f"/static/{subdirectory}/{filename}"
            print(f"Generated URL: {url_path}")
            return url_path
            
        except Exception as e:
            print(f"Error processing base64 image for {field_name}: {str(e)}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Script directory: {os.path.dirname(__file__)}")
            raise ValueError(f"Invalid base64 image data for {field_name}: {str(e)}")

    async def _process_question_images_async(self, question_data: Dict) -> Dict:
        """Process base64 images in question data and replace with URLs (async).
        
        Args:
            question_data: Question data that may contain base64 images
            
        Returns:
            Question data with base64 images replaced by URLs
        """
        def _process_images_sync():
            # Create a copy to avoid modifying the original
            processed_data = question_data.copy()
            
            # Process questionimage if present and base64
            if "question_image" in processed_data and processed_data["question_image"]:
                if str(processed_data["question_image"]).startswith("data:image/"):
                    processed_data["question_image"] = self._process_base64_image(
                        processed_data["question_image"], "question_image"
                    )
            
            # Process explainationimage if present and base64  
            if "explaination_image" in processed_data and processed_data["explaination_image"]:
                if str(processed_data["explaination_image"]).startswith("data:image/"):
                    processed_data["explaination_image"] = self._process_base64_image(
                        processed_data["explaination_image"], "explaination_image"
                    )
            
            return processed_data
        
        # Run the image processing in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, _process_images_sync)

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

    async def get_learning_streak(self, student_id: str, subject: str = None, count_ai_messages: bool = False) -> Tuple[Dict, int]:
        """Get learning streak information for a student (async).
        
        Args:
            student_id: ID of the student
            subject: Optional subject to filter by (if None, counts all subjects)
            count_ai_messages: If True, counts both user and AI messages; if False, only user messages
            
        Returns:
            Tuple of (streak_info, status_code)
        """
        try:
            def _get_streak_sync():
                return self.history_repository.get_learning_streak(student_id, subject, count_ai_messages)
            
            # Run the database operation in a thread pool
            loop = asyncio.get_event_loop()
            streak_info = await loop.run_in_executor(self.thread_pool, _get_streak_sync)
            
            return streak_info, 200
            
        except Exception as e:
            error_message = f"Error getting learning streak: {str(e)}"
            print(error_message)
            return {"message": error_message}, 500 

    async def get_questions_answered_count(self, student_id: str, subject: str = None, from_date: datetime = None) -> Tuple[Dict, int]:
        """Get count of questions answered for a student (async).
        
        Args:
            student_id: ID of the student
            subject: Optional subject to filter by (if None, counts all subjects)
            from_date: Optional datetime to filter from this date onwards
            
        Returns:
            Tuple of (questions_count_info, status_code)
        """
        try:
            def _get_count_sync():
                return self.history_repository.get_questions_answered_count(student_id, subject, from_date)
            
            # Run the database operation in a thread pool
            loop = asyncio.get_event_loop()
            count_info = await loop.run_in_executor(self.thread_pool, _get_count_sync)
            
            return count_info, 200
            
        except Exception as e:
            error_message = f"Error getting questions answered count: {str(e)}"
            print(error_message)
            return {"message": error_message}, 500

    async def ensure_quotes_exist(self) -> bool:
        """Ensure that educational quotes exist in the database, generate if needed (async).
        
        Returns:
            True if quotes exist or were successfully generated
        """
        try:
            def _check_quotes_sync():
                return self.quotes_repository.get_quotes_count()
            
            # Check if quotes exist
            loop = asyncio.get_event_loop()
            quotes_count = await loop.run_in_executor(self.thread_pool, _check_quotes_sync)
            
            if quotes_count >= 50:
                return True
            
            # Generate quotes if not enough exist
            print(f"Only {quotes_count} quotes found, generating 50 educational quotes...")
            quotes_list = await self.generate_educational_quotes(50)
            
            if quotes_list:
                def _add_quotes_sync():
                    return self.quotes_repository.add_quotes_bulk(quotes_list)
                
                success = await loop.run_in_executor(self.thread_pool, _add_quotes_sync)
                print(f"Generated and stored {len(quotes_list)} quotes successfully: {success}")
                return success
            
            return False
            
        except Exception as e:
            print(f"Error ensuring quotes exist: {str(e)}")
            return False

    async def generate_educational_quotes(self, count: int = 50) -> List[Dict]:
        """Generate educational quotes using LLM (async).
        
        Args:
            count: Number of quotes to generate
            
        Returns:
            List of quote dictionaries with 'quote' and 'author' fields
        """
        try:
            prompt_template = """
            You are a curator of educational wisdom. Generate {count} inspiring and meaningful quotes about education, learning, knowledge, wisdom, personal growth, and academic achievement.
            
            Requirements:
            1. Include quotes from famous educators, philosophers, scientists, writers, and thought leaders
            2. Mix historical and contemporary figures
            3. Ensure quotes are genuinely inspiring and relevant to learning
            4. Include diverse perspectives and backgrounds
            5. Each quote should be meaningful and motivational for students
            
            Format your response as a JSON array of objects with this exact structure:
            [
              {{
                "quote": "The exact quote text here",
                "author": "Author Name"
              }},
              {{
                "quote": "Another inspiring quote about education",
                "author": "Another Author"
              }}
            ]
            
            Examples of the type of quotes to include:
            - "Education is the most powerful weapon which you can use to change the world." - Nelson Mandela
            - "The beautiful thing about learning is that no one can take it away from you." - B.B. King
            - "Live as if you were to die tomorrow. Learn as if you were to live forever." - Mahatma Gandhi
            
            Generate {count} unique, inspiring educational quotes now.
            """
            
            from langchain_core.prompts import PromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            
            prompt = PromptTemplate.from_template(prompt_template)
            
            def _generate_quotes_sync():
                chain = prompt | self.ug_llm | StrOutputParser()
                return chain.invoke({"count": count})
            
            # Run LLM generation in thread pool
            loop = asyncio.get_event_loop()
            quotes_json = await loop.run_in_executor(self.thread_pool, _generate_quotes_sync)
            
            # Parse the JSON response
            import json
            try:
                # Clean up the response
                quotes_json = quotes_json.replace("```json", "").replace("```", "").strip()
                quotes_list = json.loads(quotes_json)
                
                # Validate the structure
                valid_quotes = []
                for quote in quotes_list:
                    if isinstance(quote, dict) and "quote" in quote and "author" in quote:
                        if quote["quote"].strip() and quote["author"].strip():
                            valid_quotes.append({
                                "quote": quote["quote"].strip(),
                                "author": quote["author"].strip()
                            })
                
                print(f"Generated {len(valid_quotes)} valid quotes out of {len(quotes_list)} total")
                return valid_quotes
                
            except json.JSONDecodeError as e:
                print(f"Error parsing quotes JSON: {str(e)}")
                # Try to extract JSON using basic pattern matching
                try:
                    start_idx = quotes_json.find('[')
                    end_idx = quotes_json.rfind(']') + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        extracted_json = quotes_json[start_idx:end_idx]
                        quotes_list = json.loads(extracted_json)
                        
                        valid_quotes = []
                        for quote in quotes_list:
                            if isinstance(quote, dict) and "quote" in quote and "author" in quote:
                                if quote["quote"].strip() and quote["author"].strip():
                                    valid_quotes.append({
                                        "quote": quote["quote"].strip(),
                                        "author": quote["author"].strip()
                                    })
                        
                        return valid_quotes
                except:
                    pass
                
                return []
            
        except Exception as e:
            print(f"Error generating educational quotes: {str(e)}")
            return []

    async def get_random_educational_quote(self) -> Tuple[Dict, int]:
        """Get a random educational quote (async).
        
        Returns:
            Tuple of (quote_info, status_code)
        """
        try:
            def _get_quote_sync():
                return self.quotes_repository.get_random_quote()
            
            # Try to get a random quote first
            loop = asyncio.get_event_loop()
            quote = await loop.run_in_executor(self.thread_pool, _get_quote_sync)
            
            # If no quote found, then ensure quotes exist and try again
            if not quote:
                print("No quotes found, ensuring quotes exist...")
                quotes_exist = await self.ensure_quotes_exist()
                
                if not quotes_exist:
                    return {"message": "Unable to load educational quotes"}, 500
                
                # Try again after ensuring quotes exist
                quote = await loop.run_in_executor(self.thread_pool, _get_quote_sync)
            
            if quote:
                return quote, 200
            else:
                return {"message": "No quotes available"}, 404
                
        except Exception as e:
            error_message = f"Error getting random quote: {str(e)}"
            print(error_message)
            return {"message": error_message}, 500

    async def get_learning_info(self, student_id: str, subject: str = None, from_date: datetime = None) -> Tuple[Dict, int]:
        """Get comprehensive learning information for a student (async).
        
        Args:
            student_id: ID of the student
            subject: Optional subject to filter by (if None, gets all subjects)
            from_date: Optional datetime to filter data from this date onwards
            
        Returns:
            Tuple of (learning_info, status_code)
        """
        try:
            # Get all the information in parallel
            streak_task = self.get_learning_streak(student_id, subject, count_ai_messages=False)
            questions_task = self.get_questions_answered_count(student_id, subject, from_date)
            quote_task = self.get_random_educational_quote()
            
            # Wait for all tasks to complete
            streak_result, questions_result, quote_result = await asyncio.gather(
                streak_task, questions_task, quote_task, return_exceptions=True
            )
            
            # Process results
            learning_info = {}
            
            # Handle streak result
            if isinstance(streak_result, tuple) and streak_result[1] == 200:
                learning_info["learning_streak"] = streak_result[0]
            else:
                learning_info["learning_streak"] = {
                    "current_streak": 0,
                    "last_activity_date": None,
                    "longest_streak": 0,
                    "total_active_days": 0
                }
            
            # Handle questions result
            if isinstance(questions_result, tuple) and questions_result[1] == 200:
                learning_info["questions_answered"] = questions_result[0]
            else:
                learning_info["questions_answered"] = {
                    "total_questions_answered": 0,
                    "by_subject": {}
                }
            
            # Handle quote result
            if isinstance(quote_result, tuple) and quote_result[1] == 200:
                learning_info["educational_quote"] = quote_result[0]
            else:
                learning_info["educational_quote"] = {
                    "quote": "Education is the passport to the future, for tomorrow belongs to those who prepare for it today.",
                    "author": "Malcolm X"
                }
            
            # Add metadata
            learning_info["generated_at"] = datetime.utcnow().isoformat()
            learning_info["student_id"] = student_id
            if subject:
                learning_info["subject_filter"] = subject
            
            return learning_info, 200
            
        except Exception as e:
            error_message = f"Error getting learning info: {str(e)}"
            print(error_message)
            return {"message": error_message}, 500 

    async def fetch_questions_by_topic_subject(self, subject: str, topic: str) -> Tuple[Dict, int]:
        """Fetch all questions from question_bank collection by topic and subject (async).
        
        Args:
            subject: Subject to filter by
            topic: Topic to filter by
            
        Returns:
            Tuple of (questions_info, status_code)
        """
        try:
            # Validate required parameters
            if not subject or not topic:
                return {"message": "Both subject and topic are required"}, 400

            def _fetch_questions_sync():
                return self.question_repository.get_questions_by_topic_subject(
                    subject=subject,
                    topic=topic
                )

            # Run the database operation in a thread pool
            loop = asyncio.get_event_loop()
            questions = await loop.run_in_executor(self.thread_pool, _fetch_questions_sync)

            # Prepare response
            response_data = {
                "questions": questions,
                "total_count": len(questions),
                "subject": subject,
                "topic": topic
            }

            return response_data, 200
            
        except Exception as e:
            error_message = f"Error fetching questions: {str(e)}"
            print(error_message)
            return {"message": error_message}, 500

    async def update_question_document(self, question_data: Dict) -> Tuple[Dict, int]:
        """Update an entire document in the question_bank collection (async).
        
        Args:
            question_data: Complete question document with _id field
            
        Returns:
            Tuple of (update_result, status_code)
        """
        try:
            # Validate that question_data is provided
            if not question_data:
                return {
                    "success": False,
                    "message": "question_data is required"
                }, 400

            # Validate that _id is present
            if "_id" not in question_data:
                return {
                    "success": False,
                    "message": "question_data must contain an '_id' field"
                }, 400

            # Process base64 images and replace with URLs
            try:
                processed_question_data = await self._process_question_images_async(question_data)
            except ValueError as img_error:
                return {
                    "success": False,
                    "message": str(img_error)
                }, 400

            def _update_question_sync():
                return self.question_repository.update_question_document(processed_question_data)

            # Run the database operation in a thread pool
            loop = asyncio.get_event_loop()
            success, message, updated_at = await loop.run_in_executor(self.thread_pool, _update_question_sync)

            if success:
                response_data = {
                    "success": True,
                    "message": message,
                    "document_id": str(processed_question_data["_id"])
                }
                # Add updated_at to response if available
                if updated_at:
                    response_data["updated_at"] = updated_at.isoformat()
                
                return response_data, 200
            else:
                return {
                    "success": False,
                    "message": message
                }, 400
            
        except Exception as e:
            error_message = f"Error updating question document: {str(e)}"
            print(error_message)
            return {
                "success": False,
                "message": error_message
            }, 500 
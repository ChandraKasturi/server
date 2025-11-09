import asyncio
import base64
import os
import uuid
import tempfile
import shutil
import io
import time
from loguru import logger
from typing import List, Dict, Any, Optional, Tuple, BinaryIO
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_postgres import PGVectorStore
from langchain_postgres import PGEngine
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import concurrent.futures
from fastapi import UploadFile
from google import genai
from google.genai import types as genai_types
import fitz  # PyMuPDF
import PIL.Image

from config import settings
from repositories.pdf_repository import PDFRepository
from repositories.mongo_repository import HistoryRepository, QuotesRepository, QuestionRepository
from services.learning.learning_achievement_service import LearningAchievementService

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
        '''self.llm = ChatGoogleGenerativeAI(google_api_key=settings.GOOGLE_API_KEY, model="gemini-2.5-pro")
        self.ug_llm = ChatGoogleGenerativeAI(google_api_key=settings.GOOGLE_API_KEY, model="gemini-2.5-pro")'''
        self.pdf_repository = PDFRepository()
        self.history_repository = HistoryRepository()
        self.quotes_repository = QuotesRepository()
        self.question_repository = QuestionRepository()
        self.learning_achievement_service = LearningAchievementService()
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
            history_size=settings.MONGO_HISTORY_SIZE
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
                        ug = PGEngine.from_connection_string(url=connection_string)
                        pdf_vector_store = PGVectorStore.create_sync(
                            engine=ug,
                            embedding_service=self.embeddings,
                            table_name=collection_name,
                        )
                        
                        # Create a retriever from the vector store
                        """pdf_retriever = pdf_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2})"""
                        
                        # Get relevant documents from this PDF
                        pdf_results = pdf_vector_store.similarity_search(question, k=2)
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
    
    def _get_single_pdf_content(self, pdf, question: str, connection_string: str) -> List[Dict]:
        """Retrieve content from a single PDF (helper for parallel processing).
        
        Args:
            pdf: PDF document object
            question: Question to retrieve relevant content for
            connection_string: Database connection string
            
        Returns:
            List of documents with relevant content from this PDF
        """
        try:
            # Create a collection name specific to this PDF
            collection_name = f"pdf_{pdf.id}"
            
            # Initialize PGVector with the student-specific connection
            ug = PGEngine.from_connection_string(url=connection_string)
            pdf_vector_store = PGVectorStore.create_sync(
                engine=ug,
                embedding_service=self.embeddings,
                table_name=collection_name,
            )
            
            # Get relevant documents from this PDF
            pdf_results = pdf_vector_store.similarity_search(question, k=2)
            
            # Add source information to each document
            for doc in pdf_results:
                if not hasattr(doc, "metadata"):
                    doc.metadata = {}
                doc.metadata["source"] = f"From your document: {pdf.title}"
            
            return pdf_results
            
        except Exception as e:
            print(f"Error retrieving from PDF {pdf.id}: {str(e)}")
            return []
    
    async def _get_subject_pdf_content_async(self, student_id: str, subject: str, question: str) -> List[Dict]:
        """Async version of getting content from user's PDFs related to the specified subject.
        Now with PARALLEL processing of multiple PDFs for better performance.
        
        Args:
            student_id: ID of the student
            subject: Subject to filter PDFs by
            question: Question to retrieve relevant content for
            
        Returns:
            List of documents with relevant content
        """
        try:
            # Get all PDFs for this student that have been processed successfully
            def _get_user_pdfs():
                return self.pdf_repository.get_user_pdf_documents(student_id)
            
            loop = asyncio.get_event_loop()
            user_pdfs = await loop.run_in_executor(self.thread_pool, _get_user_pdfs)
            
            completed_pdfs = [pdf for pdf in user_pdfs if pdf.processing_status == "completed"]
            
            # Filter PDFs by subject if specified
            subject_pdfs = []
            for pdf in completed_pdfs:
                pdf_subject = (pdf.metadata.subject or "").lower() if pdf.metadata else ""
                
                # Match PDFs with this subject or with no subject specified
                if subject.lower() in pdf_subject or not pdf_subject:
                    subject_pdfs.append(pdf)
            
            if not subject_pdfs:
                return []
            
            # Create student-specific connection string
            connection_string = self._get_student_specific_connection_string(student_id)
            
            # Process PDFs in PARALLEL - MAJOR PERFORMANCE IMPROVEMENT for multi-PDF queries
            pdf_tasks = []
            for pdf in subject_pdfs[:5]:  # Limit to 5 PDFs for performance
                task = loop.run_in_executor(
                    self.thread_pool,
                    self._get_single_pdf_content,
                    pdf,
                    question,
                    connection_string
                )
                pdf_tasks.append(task)
            
            # Wait for all PDF retrievals to complete in parallel
            pdf_results_list = await asyncio.gather(*pdf_tasks, return_exceptions=True)
            
            # Flatten results and filter out errors
            pdf_docs = []
            for results in pdf_results_list:
                if isinstance(results, list):
                    pdf_docs.extend(results)
                elif isinstance(results, Exception):
                    print(f"Error in parallel PDF retrieval: {str(results)}")
            
            return pdf_docs
            
        except Exception as e:
            print(f"Error in async PDF content retrieval: {str(e)}")
            return []

    async def _get_vector_store_results_async(self, subject_collection: str, question: str) -> List:
        """Async version of getting results from vector store.
        
        Args:
            subject_collection: Subject collection name
            question: Question to search for
            
        Returns:
            List of relevant documents
        """
        def _get_results_sync():
            ug = PGEngine.from_connection_string(url=settings.PGVECTOR_CONNECTION_STRING)
            # Initialize PGVector with the subject collection
            subject_vector_store = PGVectorStore.create_sync(
                engine=ug,
                embedding_service=self.embeddings,
                table_name=subject_collection,
            )
            
            # Create a retriever from the vector store
            """subject_retriever = subject_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})"""
            
            # Get relevant documents from subject collection
            return subject_vector_store.similarity_search(question, k=5)
        
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

    async def _return_empty_list_async(self) -> List:
        """Helper method to return an empty list asynchronously.
        
        Used for parallel execution when certain operations are disabled.
        
        Returns:
            Empty list
        """
        return []

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

    async def _process_learning_achievements_async(self, student_id: str, interaction_data: Dict) -> None:
        """Async version of processing learning achievements.
        
        Args:
            student_id: ID of the student
            interaction_data: Learning interaction data
        """
        def _process_achievements_sync():
            return self.learning_achievement_service.process_learning_interaction(student_id, interaction_data)
        
        # Run the achievement processing in a thread pool
        loop = asyncio.get_event_loop()
        try:
            achievement_results = await loop.run_in_executor(self.thread_pool, _process_achievements_sync)
            # Log achievement results for debugging
            if achievement_results.get("achievements_earned") or achievement_results.get("badges_updated"):
                print(f"Learning achievements processed for {student_id}: {achievement_results}")
        except Exception as e:
            print(f"Error processing learning achievements: {str(e)}")

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
                    include_pdfs: bool = True,
                    include_images: bool = True) -> Tuple[Dict, int]:
        """Learn about a specific subject with enhanced image support (async version).
        
        Args:
            subject: Subject to learn about (science, social_science, mathematics, english, hindi)
            question: Question about the subject
            student_id: ID of the student
            session_id: JWT token from X-Auth-Session (optional)
            include_pdfs: Whether to include user's PDFs in the answer
            include_images: Whether to search for relevant images from learning PDFs
            
        Returns:
            Tuple of (response_dict, status_code) containing answer and image information
        """
        # ⏱️ START TIMING
        start_time = time.time()
        logger.info(f"🚀 [TIMING] learn_subject started for {subject} - Question: {question[:50]}...")
        
        try:
            # Validate subject
            if subject not in self.SUBJECT_COLLECTIONS:
                return {
                    "answer": f"Invalid subject: {subject}. Valid subjects are: {', '.join(self.SUBJECT_COLLECTIONS.keys())}",
                    "has_image": False
                }, 400
            
            # STEP 1-3: Run all retrievals in PARALLEL for better performance (async)
            subject_collection = self.SUBJECT_COLLECTIONS[subject]
            
            # Create tasks for parallel execution
            retrieval_tasks = [
                self._get_vector_store_results_async(subject_collection, question),
            ]
            
            # Add PDF retrieval task if requested
            if include_pdfs:
                retrieval_tasks.append(
                    self._get_subject_pdf_content_async(student_id, subject, question)
                )
            else:
                # Add empty result if PDFs not requested
                retrieval_tasks.append(self._return_empty_list_async())
            
            # Add image search task if requested
            if include_images:
                retrieval_tasks.append(
                    self._find_relevant_learning_images_async(
                        user_id=student_id,
                        subject=subject, 
                        query=question,
                        similarity_threshold=0.3,
                        max_images=3
                    )
                )
            else:
                # Add empty result if images not requested
                retrieval_tasks.append(self._return_empty_list_async())
            
            # Execute all retrievals in parallel - MAJOR PERFORMANCE IMPROVEMENT
            retrieval_start = time.time()
            subject_docs, pdf_docs, relevant_images = await asyncio.gather(*retrieval_tasks)
            retrieval_time = time.time() - retrieval_start
            logger.info(f"⏱️ [TIMING] Parallel retrieval completed in {retrieval_time:.2f}s")
            logger.info(f"   - Retrieved {len(subject_docs)} subject docs, {len(pdf_docs)} PDF docs, {len(relevant_images)} images")
            
            # Extract content from subject knowledge documents
            subject_context = [doc.page_content for doc in subject_docs]
            logger.info(f"Subject context: {subject_context}")
            
            # Extract content from PDF documents with source info
            pdf_context = [f"{doc.metadata.get('source', '')}: {doc.page_content}" for doc in pdf_docs]
            
            # STEP 4: Combine both contexts
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
            context = context + "\n\n Important: Make sure to Answer in Markdown and latex where ever applicable and format them"
            
            # ⏱️ Log context size
            context_chars = len(context)
            context_tokens_estimate = context_chars // 4  # Rough estimate: 1 token ≈ 4 chars
            logger.info(f"⏱️ [TIMING] Context prepared: {context_chars} chars (~{context_tokens_estimate} tokens)")
            
            # STEP 5: Create enhanced prompt with subject-specific system message and image references
            system_prompt = self.SUBJECT_PROMPTS.get(subject, "You are an educational assistant.")
            
            image_context = ""
            if relevant_images:
                image_context = "\n\nRELEVANT EDUCATIONAL IMAGES:\n"
                for i, img in enumerate(relevant_images, 1):
                    full_image_url = f"https://aigenix.in{img['image_url']}"
                    image_context += f"\nImage {i}: ![{img['caption']}]({full_image_url})\n"
                    image_context += f"Caption: {img['caption']}\n"
                    image_context += f"Page: {img.get('page_number', 'Unknown')}\n"
                
                image_context += "\nPlease include these relevant images in your response using markdown format with the full URLs provided above. Reference them appropriately to enhance your explanation."
            
            prompt_messages = [
                ("system", f"{system_prompt}\n\n"
                          "Use the provided context to give accurate answers. "
                          "If your answer includes information from the student's own documents, clearly indicate this. "
                          "If relevant images are available, include them in your response using markdown format and explain how they relate to the topic. "
                          "Always use the full image URLs provided in the image context. "
                          "If you're unsure or the answer is not in the context, be honest about it.\n\n"
                          f"Context: {context}{image_context}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}")
            ]
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages(prompt_messages)
            
            # STEP 6: Create chain with history integration
            chain = prompt | self.ug_llm | StrOutputParser()
            
            # Prepare input for the chain
            chain_input = {
                "context": context,
                "question": question
            }
            
            # Create chain with message history if session_id is provided and run async
            # ⏱️ THIS IS LIKELY THE SLOWEST STEP
            llm_start = time.time()
            logger.info(f"⏱️ [TIMING] Starting LLM generation (model: {self.ug_llm})...")
            
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
            
            llm_time = time.time() - llm_start
            logger.info(f"⏱️ [TIMING] LLM generation completed in {llm_time:.2f}s")
            logger.info(f"   - Generated answer length: {len(answer)} chars")
            
            # STEP 7: Create structured response
            response = {
                "answer": answer,
                "has_image": len(relevant_images) > 0,
                "subject": subject
            }
            
            # Add image information if available
            if relevant_images:
                # For backward compatibility, use the first image for single image fields
                first_image = relevant_images[0]
                response.update({
                    "image_url": first_image["image_url"],
                    "image_caption": first_image["caption"],
                    "image_page": first_image.get("page_number"),
                    "image_score": first_image.get("score"),
                    "image_pdf_id": first_image.get("pdf_id"),
                    "images": relevant_images  # All images for enhanced response
                })
            
            # STEP 8: Store in sahasra_history for persistence (BACKGROUND - non-blocking)
            # Store user question
            user_history_data = {
                "subject": subject,
                "message": question,
                "is_ai": False,
                "time": datetime.utcnow(),
                "session_id": session_id
            }
            
            # Store AI response with image info
            ai_response_text = answer
            if relevant_images:
                image_refs = ", ".join([img['image_url'] for img in relevant_images])
                ai_response_text += f" [Image references: {image_refs}]"
            
            ai_history_data = {
                "subject": subject,
                "message": ai_response_text,
                "is_ai": True,
                "time": datetime.utcnow(),
                "session_id": session_id
            }
            
            # Run history storage and achievements in background - DON'T WAIT FOR THEM
            # This improves response time by 100-300ms
            background_start = time.time()
            asyncio.create_task(self._store_history_async(student_id, user_history_data))
            asyncio.create_task(self._store_history_async(student_id, ai_history_data))
            asyncio.create_task(self._process_learning_achievements_async(student_id, user_history_data))
            background_time = time.time() - background_start
            logger.info(f"⏱️ [TIMING] Background tasks queued in {background_time:.3f}s")
            
            # ⏱️ TOTAL TIME
            total_time = time.time() - start_time
            logger.info(f"✅ [TIMING] TOTAL learn_subject time: {total_time:.2f}s")
            logger.info(f"   Breakdown: Retrieval={retrieval_time:.2f}s, LLM={llm_time:.2f}s, Other={total_time - retrieval_time - llm_time:.2f}s")
            
            return response, 200
            
        except Exception as e:
            error_time = time.time() - start_time
            error_message = f"Error learning about {subject}: {str(e)}"
            logger.error(f"❌ [TIMING] Error after {error_time:.2f}s: {error_message}")
            print(error_message)
            return {
                "answer": error_message,
                "has_image": False,
                "subject": subject,
                "images": []
            }, 500
    
    # Subject-specific convenience methods (now async)
    
    async def learn_science(self, question: str, student_id: str, session_id: str = None, include_pdfs: bool = True) -> Tuple[Dict, int]:
        """Learn about science with enhanced image support (async).
        
        Args:
            question: Question about science
            student_id: ID of the student
            session_id: JWT token from X-Auth-Session (optional)
            include_pdfs: Whether to include user's PDFs in the answer
            
        Returns:
            Tuple of (response_dict, status_code) containing answer and image information
        """
        return await self.learn_subject("science", question, student_id, session_id, include_pdfs, include_images=True)
    
    async def learn_social_science(self, question: str, student_id: str, session_id: str = None, include_pdfs: bool = True) -> Tuple[Dict, int]:
        """Learn about social science with enhanced image support (async).
        
        Args:
            question: Question about social science
            student_id: ID of the student
            session_id: JWT token from X-Auth-Session (optional)
            include_pdfs: Whether to include user's PDFs in the answer
            
        Returns:
            Tuple of (response_dict, status_code) containing answer and image information
        """
        return await self.learn_subject("social_science", question, student_id, session_id, include_pdfs, include_images=True)
    
    async def learn_mathematics(self, question: str, student_id: str, session_id: str = None, include_pdfs: bool = True) -> Tuple[Dict, int]:
        """Learn about mathematics with enhanced image support (async).
        
        Args:
            question: Question about mathematics
            student_id: ID of the student
            session_id: JWT token from X-Auth-Session (optional)
            include_pdfs: Whether to include user's PDFs in the answer
            
        Returns:
            Tuple of (response_dict, status_code) containing answer and image information
        """
        return await self.learn_subject("mathematics", question, student_id, session_id, include_pdfs, include_images=True)
    
    async def learn_english(self, question: str, student_id: str, session_id: str = None, include_pdfs: bool = True) -> Tuple[Dict, int]:
        """Learn about English with enhanced image support (async).
        
        Args:
            question: Question about English
            student_id: ID of the student
            session_id: JWT token from X-Auth-Session (optional)
            include_pdfs: Whether to include user's PDFs in the answer
            
        Returns:
            Tuple of (response_dict, status_code) containing answer and image information
        """
        return await self.learn_subject("english", question, student_id, session_id, include_pdfs, include_images=True)
    
    async def learn_hindi(self, question: str, student_id: str, session_id: str = None, include_pdfs: bool = True) -> Tuple[Dict, int]:
        """Learn about Hindi with enhanced image support (async).
        
        Args:
            question: Question about Hindi
            student_id: ID of the student
            session_id: JWT token from X-Auth-Session (optional)
            include_pdfs: Whether to include user's PDFs in the answer
            
        Returns:
            Tuple of (response_dict, status_code) containing answer and image information
        """
        return await self.learn_subject("hindi", question, student_id, session_id, include_pdfs, include_images=True)

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

    async def get_weekly_questions_change(self, student_id: str, subject: str = None) -> Tuple[Dict, int]:
        """Get weekly change in questions answered for a student (async).
        
        Args:
            student_id: ID of the student
            subject: Optional subject to filter by (if None, counts all subjects)
            
        Returns:
            Tuple of (weekly_change_info, status_code)
        """
        try:
            def _get_change_sync():
                return self.history_repository.get_weekly_questions_change(student_id, subject)
            
            # Run the database operation in a thread pool
            loop = asyncio.get_event_loop()
            change_info = await loop.run_in_executor(self.thread_pool, _get_change_sync)
            
            return change_info, 200
            
        except Exception as e:
            error_message = f"Error getting weekly questions change: {str(e)}"
            print(error_message)
            return {"message": error_message}, 500

    async def get_learning_hours(self, student_id: str, subject: str = None) -> Tuple[Dict, int]:
        """Get learning hours for a student (async).
        
        Args:
            student_id: ID of the student
            subject: Optional subject to filter by (if None, counts all subjects)
            
        Returns:
            Tuple of (learning_hours_info, status_code)
        """
        try:
            def _get_hours_sync():
                return self.history_repository.get_learning_hours(student_id, subject)
            
            # Run the database operation in a thread pool
            loop = asyncio.get_event_loop()
            hours_info = await loop.run_in_executor(self.thread_pool, _get_hours_sync)
            
            return hours_info, 200
            
        except Exception as e:
            error_message = f"Error getting learning hours: {str(e)}"
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
            weekly_change_task = self.get_weekly_questions_change(student_id, subject)
            learning_hours_task = self.get_learning_hours(student_id, subject)
            quote_task = self.get_random_educational_quote()
            
            # Wait for all tasks to complete
            streak_result, questions_result, weekly_change_result, learning_hours_result, quote_result = await asyncio.gather(
                streak_task, questions_task, weekly_change_task, learning_hours_task, quote_task, return_exceptions=True
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
            
            # Handle weekly change result
            if isinstance(weekly_change_result, tuple) and weekly_change_result[1] == 200:
                # Add weekly change data to questions_answered section
                weekly_change_data = weekly_change_result[0]
                learning_info["questions_answered"]["weekly_change"] = weekly_change_data["weekly_change"]
                learning_info["questions_answered"]["this_week_count"] = weekly_change_data["this_week_count"]
                learning_info["questions_answered"]["last_week_count"] = weekly_change_data["last_week_count"]
                learning_info["questions_answered"]["is_increase"] = weekly_change_data["is_increase"]
                learning_info["questions_answered"]["is_decrease"] = weekly_change_data["is_decrease"]
            else:
                learning_info["questions_answered"]["weekly_change"] = 0
                learning_info["questions_answered"]["this_week_count"] = 0
                learning_info["questions_answered"]["last_week_count"] = 0
                learning_info["questions_answered"]["is_increase"] = False
                learning_info["questions_answered"]["is_decrease"] = False
            
            # Handle learning hours result
            if isinstance(learning_hours_result, tuple) and learning_hours_result[1] == 200:
                learning_info["learning_hours"] = learning_hours_result[0]
            else:
                learning_info["learning_hours"] = {
                    "total_learning_hours": 0.0,
                    "learning_hours_today": 0.0,
                    "sessions_analyzed": 0
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

    async def upload_learning_pdf(self, 
                                  file: UploadFile, 
                                  user_id: str, 
                                  title: str,
                                  subject: str,
                                  description: Optional[str] = None,
                                  topic: Optional[str] = None,
                                  grade: Optional[str] = None) -> Tuple[Dict, int]:
        """Upload and process a PDF for learning purposes with subject-specific storage.
        
        Args:
            file: Uploaded PDF file
            user_id: ID of the student
            title: Title of the PDF document
            subject: Subject category (must be one of the valid subjects)
            description: Optional description
            topic: Optional topic within the subject
            grade: Optional grade level
            
        Returns:
            Tuple of (response_data, status_code)
        """
        try:
            # Validate subject
            if subject not in self.SUBJECT_COLLECTIONS:
                return {
                    "success": False,
                    "message": f"Invalid subject: {subject}. Valid subjects are: {', '.join(self.SUBJECT_COLLECTIONS.keys())}"
                }, 400
            
            # Validate file type
            if not file.filename.lower().endswith('.pdf'):
                return {
                    "success": False,
                    "message": "Uploaded file must be a PDF"
                }, 400
            
            # Generate unique ID for the PDF
            pdf_id = str(uuid.uuid4())
            
            # Save the uploaded file temporarily
            file_contents = await file.read()
            
            # Process the PDF immediately (in-memory processing)
            result = await self._process_learning_pdf_async(
                pdf_id=pdf_id,
                user_id=user_id,
                title=title,
                subject=subject,
                description=description,
                topic=topic,
                grade=grade,
                file_contents=file_contents,
                filename=file.filename
            )
            
            return result
            
        except Exception as e:
            error_message = f"Error uploading learning PDF: {str(e)}"
            print(error_message)
            return {
                "success": False,
                "message": error_message
            }, 500

    async def _process_learning_pdf_async(self,
                                          pdf_id: str,
                                          user_id: str,
                                          title: str,
                                          subject: str,
                                          description: Optional[str],
                                          topic: Optional[str],
                                          grade: Optional[str],
                                          file_contents: bytes,
                                          filename: str) -> Tuple[Dict, int]:
        """Process a learning PDF by extracting text, creating embeddings, and storing in subject-specific vector store.
        
        Args:
            pdf_id: Unique identifier for the PDF
            user_id: ID of the student
            title: Title of the PDF document
            subject: Subject category
            description: Optional description
            topic: Optional topic
            grade: Optional grade level
            file_contents: Raw PDF file content
            filename: Original filename
            
        Returns:
            Tuple of (response_data, status_code)
        """
        try:
            # Create a temporary file for processing
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(file_contents)
                temp_file_path = temp_file.name
            
            try:
                # Extract text from PDF using Gemini OCR (async)
                chunks, page_count = await self._extract_text_from_pdf_async(temp_file_path)
                
                # Extract images from PDF (async)
                image_data = await self._extract_images_from_learning_pdf_async(temp_file_path, user_id, pdf_id)
                
                # Store chunks in subject-specific vector database (async)
                chunks_created = await self._store_learning_chunks_in_vector_db_async(
                    pdf_id=pdf_id,
                    user_id=user_id,
                    subject=subject,
                    chunks=chunks,
                    title=title,
                    description=description,
                    topic=topic,
                    grade=grade
                )
                
                # Store image captions in vector database (async)
                images_processed = 0
                if image_data:
                    images_processed = await self._store_learning_image_captions_async(
                        pdf_id=pdf_id,
                        user_id=user_id,
                        subject=subject,
                        image_data=image_data
                    )
                
                # Create response
                response_data = {
                    "success": True,
                    "pdf_id": pdf_id,
                    "title": title,
                    "subject": subject,
                    "processing_status": "completed",
                    "chunks_created": chunks_created,
                    "images_extracted": len(image_data) if image_data else 0,
                    "images_processed": images_processed,
                    "upload_date": datetime.utcnow().isoformat(),
                    "file_size": len(file_contents),
                    "page_count": page_count,
                    "message": f"PDF processed successfully for {subject} learning"
                }
                
                return response_data, 200
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            error_message = f"Error processing learning PDF: {str(e)}"
            print(error_message)
            return {
                "success": False,
                "message": error_message
            }, 500

    async def _extract_text_from_pdf_async(self, pdf_path: str) -> Tuple[List[Dict], int]:
        """Extract text from PDF using Gemini OCR (async).
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (chunks, page_count)
        """
        def _extract_text_sync():
            return asyncio.run(self._extract_text_from_pdf(pdf_path))
        
        # Run the synchronous operation in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, _extract_text_sync)

    async def _extract_text_from_pdf(self, pdf_path: str) -> Tuple[List[Dict], int]:
        """Extract text from PDF using Gemini OCR.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (chunks, page_count)
        """
        chunks = []
        
        try:
            # Configure Gemini client
            gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
            
            # Read PDF file and upload to Gemini
            with open(pdf_path, 'rb') as file:
                pdf_content = file.read()
                pdf_io = io.BytesIO(pdf_content)
                pdf_io.name = os.path.basename(pdf_path)
                
                # Upload PDF using Gemini File API
                uploaded_file = gemini_client.files.upload(
                    file=pdf_io,
                    config=dict(mime_type='application/pdf')
                )
            
            # Create prompt for text extraction with page information
            extraction_prompt = f"""
            Extract all text content from this PDF document. 
            For each page, provide the text content along with the page number.
            Format your response as follows for each page:
            
            PAGE [page_number]:
            [text content for that page]
            
            PAGE [next_page_number]:
            [text content for that page]
            
            Continue this format for all pages in the document.
            Make sure to preserve the original formatting and structure as much as possible.
            If a page has no readable text, still include it as "PAGE [page_number]: [No readable text]"
            """
            
            # Generate content using Gemini
            response = gemini_client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[uploaded_file, extraction_prompt]
            )
            
            extracted_text = response.text
            
            # Parse the response to create chunks
            pages = self._parse_gemini_text_response(extracted_text)
            
            # Convert to the expected format and create chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            for page_num, page_text in pages.items():
                if page_text.strip():  # Only add pages with content
                    # Split page text into smaller chunks if needed
                    page_chunks = text_splitter.split_text(page_text.strip())
                    
                    for i, chunk_text in enumerate(page_chunks):
                        chunks.append({
                            'text': chunk_text,
                            'page': page_num,
                            'chunk_index': i,
                            'metadata': {
                                'page': page_num,
                                'chunk_index': i,
                                'total_pages': len(pages),
                                'extraction_method': 'gemini_ocr',
                                'source': 'learning_pdf'
                            }
                        })
            
            num_pages = len(pages)
            
            # Clean up - delete the uploaded file from Gemini
            try:
                gemini_client.files.delete(name=uploaded_file.name)
                print(f"Successfully deleted uploaded file {uploaded_file.name} from Gemini.")
            except Exception as del_e:
                print(f"Could not delete uploaded file {uploaded_file.name} from Gemini: {del_e}")
            
            return chunks, num_pages
            
        except Exception as e:
            print(f"Error extracting text with Gemini OCR: {str(e)}")
            # Fallback to pypdf if Gemini fails
            return await self._extract_text_fallback(pdf_path)

    def _parse_gemini_text_response(self, text_response: str) -> Dict[int, str]:
        """Parse Gemini's text extraction response to extract page-wise content.
        
        Args:
            text_response: Raw text response from Gemini
            
        Returns:
            Dictionary mapping page numbers to their text content
        """
        pages = {}
        current_page = None
        current_text = []
        
        lines = text_response.split('\n')
        
        for line in lines:
            # Check if line indicates a new page
            if line.strip().startswith('PAGE ') and ':' in line:
                # Save previous page if exists
                if current_page is not None:
                    pages[current_page] = '\n'.join(current_text)
                
                # Extract page number
                try:
                    page_part = line.strip().split(':')[0]  # Get "PAGE X" part
                    page_num_str = page_part.replace('PAGE', '').strip()
                    current_page = int(page_num_str)
                    current_text = []
                    
                    # Add any text after the colon on the same line
                    remaining_text = ':'.join(line.strip().split(':')[1:]).strip()
                    if remaining_text:
                        current_text.append(remaining_text)
                        
                except (ValueError, IndexError):
                    # If page number parsing fails, treat as regular text
                    if current_page is not None:
                        current_text.append(line)
            else:
                # Regular text line
                if current_page is not None:
                    current_text.append(line)
                elif not pages:  # If no page marker found yet, assume page 1
                    current_page = 1
                    current_text.append(line)
        
        # Save the last page
        if current_page is not None:
            pages[current_page] = '\n'.join(current_text)
        
        # If no pages were found, treat entire response as page 1
        if not pages:
            pages[1] = text_response
        
        return pages

    async def _extract_text_fallback(self, pdf_path: str) -> Tuple[List[Dict], int]:
        """Fallback text extraction method using pypdf.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (chunks, num_pages)
        """
        import pypdf
        chunks = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    # Split page text into smaller chunks
                    page_chunks = text_splitter.split_text(text.strip())
                    
                    for i, chunk_text in enumerate(page_chunks):
                        chunks.append({
                            'text': chunk_text,
                            'page': page_num,
                            'chunk_index': i,
                            'metadata': {
                                'page': page_num,
                                'chunk_index': i,
                                'total_pages': num_pages,
                                'extraction_method': 'pypdf_fallback',
                                'source': 'learning_pdf'
                            }
                        })
        
        return chunks, num_pages

    async def _extract_images_from_learning_pdf_async(self, pdf_path: str, user_id: str, pdf_id: str) -> List[Dict]:
        """Extract images from a learning PDF and generate captions (async).
        
        Args:
            pdf_path: Path to the PDF file
            user_id: ID of the user
            pdf_id: ID of the PDF document
            
        Returns:
            List of dictionaries with image information
        """
        def _extract_images_sync():
            return asyncio.run(self._extract_images_from_learning_pdf(pdf_path, user_id, pdf_id))
        
        # Run the synchronous operation in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, _extract_images_sync)

    async def _extract_images_from_learning_pdf(self, pdf_path: str, user_id: str, pdf_id: str) -> List[Dict]:
        """Extract images from a learning PDF and generate captions using Gemini.
        
        Args:
            pdf_path: Path to the PDF file
            user_id: ID of the user
            pdf_id: ID of the PDF document
            
        Returns:
            List of dictionaries with image information
        """
        image_data = []
        
        # Create output folder for images
        images_folder = os.path.join(settings.static_dir_path, "pdf_images", f"learning_{pdf_id}")
        os.makedirs(images_folder, exist_ok=True)
        
        # Create Gemini client
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        
        # Open the PDF
        doc = fitz.open(pdf_path)
        image_count = 0
        
        try:
            # Loop through all the pages
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Get the images on the page
                image_list = page.get_images(full=True)

                for img in image_list:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_filename = os.path.join(images_folder, f"learning_image_{page_num + 1}_{image_count}.png")
                    
                    # Write the image to a file
                    with open(image_filename, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    # Generate a URL for the image (relative to static directory)
                    image_url = f"/static/pdf_images/learning_{pdf_id}/learning_image_{page_num + 1}_{image_count}.png"
                    
                    # Generate caption with Gemini API
                    try:
                        # Open the image with PIL
                        pil_image = PIL.Image.open(image_filename)
                        
                        # Generate educational caption
                        response = client.models.generate_content(
                            model="gemini-2.5-pro",
                            contents=["Write a detailed educational caption for this image, focusing on what students can learn from it", pil_image]
                        )
                        
                        caption = response.text.strip()
                    except Exception as e:
                        caption = f"Educational image from page {page_num + 1}"
                        print(f"Error generating caption with Gemini: {str(e)}")
                    
                    # Store image data
                    image_data.append({
                        "image_path": image_filename,
                        "image_url": image_url,
                        "page_number": page_num + 1,
                        "caption": caption,
                        "image_index": image_count,
                        "source": "learning_pdf"
                    })
                    
                    image_count += 1
            
            print(f"Extracted {image_count} images from learning PDF {pdf_id}")
            return image_data
            
        except Exception as e:
            print(f"Error extracting images from learning PDF: {str(e)}")
            return []
        finally:
            doc.close()

    async def _store_learning_chunks_in_vector_db_async(self,
                                                        pdf_id: str,
                                                        user_id: str,
                                                        subject: str,
                                                        chunks: List[Dict],
                                                        title: str,
                                                        description: Optional[str] = None,
                                                        topic: Optional[str] = None,
                                                        grade: Optional[str] = None) -> int:
        """Store learning PDF chunks in subject-specific vector database (async).
        
        Args:
            pdf_id: ID of the PDF document
            user_id: ID of the user
            subject: Subject category
            chunks: List of text chunks
            title: Title of the PDF
            description: Optional description
            topic: Optional topic
            grade: Optional grade level
            
        Returns:
            Number of chunks successfully stored
        """
        def _store_chunks_sync():
            return asyncio.run(self._store_learning_chunks_in_vector_db(
                pdf_id, user_id, subject, chunks, title, description, topic, grade
            ))
        
        # Run the synchronous operation in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, _store_chunks_sync)

    async def _store_learning_chunks_in_vector_db(self,
                                                  pdf_id: str,
                                                  user_id: str,
                                                  subject: str,
                                                  chunks: List[Dict],
                                                  title: str,
                                                  description: Optional[str] = None,
                                                  topic: Optional[str] = None,
                                                  grade: Optional[str] = None) -> int:
        """Store learning PDF chunks in subject-specific vector database.
        
        Args:
            pdf_id: ID of the PDF document
            user_id: ID of the user
            subject: Subject category
            chunks: List of text chunks
            title: Title of the PDF
            description: Optional description
            topic: Optional topic
            grade: Optional grade level
            
        Returns:
            Number of chunks successfully stored
        """
        try:
            # Create subject-specific collection name for learning PDFs
            collection_name = self.SUBJECT_COLLECTIONS[subject]
            
            # Use the main PGVector connection (not student-specific for learning content)
            connection_string = settings.PGVECTOR_CONNECTION_STRING
            
            # Initialize embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
            ug = PGEngine.from_connection_string(url=connection_string)
            # Use PGVector with subject-specific collection
            vector_store = PGVectorStore.create_sync(
                engine=ug,
                embedding_service=embeddings,
                table_name=collection_name,
            )
            
            # Convert chunks to documents for vector storage
            documents = []
            for i, chunk in enumerate(chunks):
                # Create document with comprehensive metadata
                doc = Document(
                    page_content=chunk['text'],
                    metadata={
                        'pdf_id': pdf_id,
                        'user_id': user_id,
                        'subject': subject,
                        'topic': topic or '',
                        'grade': grade or '',
                        'title': title,
                        'description': description or '',
                        'page': chunk.get('page', 0),
                        'chunk_index': chunk.get('chunk_index', i),
                        'extraction_method': chunk.get('metadata', {}).get('extraction_method', 'unknown'),
                        'source': 'learning_pdf',
                        'upload_date': datetime.utcnow().isoformat()
                    }
                )
                documents.append(doc)
            
            # Add documents to vector store
            if documents:
                vector_store.add_documents(documents)
                print(f"Successfully stored {len(documents)} chunks in learning vector database for {subject}")
                return len(documents)
            else:
                return 0
                
        except Exception as e:
            print(f"Error storing learning chunks in vector database: {str(e)}")
            return 0

    async def _store_learning_image_captions_async(self,
                                                   pdf_id: str,
                                                   user_id: str,
                                                   subject: str,
                                                   image_data: List[Dict]) -> int:
        """Store learning PDF image captions in vector database (async).
        
        Args:
            pdf_id: ID of the PDF document
            user_id: ID of the user
            subject: Subject category
            image_data: List of image data with captions
            
        Returns:
            Number of image captions successfully stored
        """
        def _store_captions_sync():
            return asyncio.run(self._store_learning_image_captions(pdf_id, user_id, subject, image_data))
        
        # Run the synchronous operation in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, _store_captions_sync)

    async def _store_learning_image_captions(self,
                                             pdf_id: str,
                                             user_id: str,
                                             subject: str,
                                             image_data: List[Dict]) -> int:
        """Store learning PDF image captions in vector database.
        
        Args:
            pdf_id: ID of the PDF document
            user_id: ID of the user
            subject: Subject category
            image_data: List of image data with captions
            
        Returns:
            Number of image captions successfully stored
        """
        try:
            # Create subject-specific collection name for learning PDF images
            connection_string = settings.PGVECTOR_CONNECTION_STRING
            ug = PGEngine.from_connection_string(url=connection_string)
            collection_name = f"learning_{subject}_images_{user_id}"
            # Use the main PGVector connection
            
            
            # Initialize embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
            
            # Use PGVector with subject-specific collection
            vector_store = PGVectorStore.create_sync(
                engine=ug,
                embedding_service=embeddings,
                table_name=collection_name,
            )
            
            # Convert image captions to documents for vector storage
            documents = []
            for i, img in enumerate(image_data):
                # Create document with comprehensive metadata
                doc = Document(
                    page_content=img["caption"],
                    metadata={
                        'pdf_id': pdf_id,
                        'user_id': user_id,
                        'subject': subject,
                        'image_id': f"learning_image_{i}",
                        'page_number': img.get("page_number"),
                        'image_url': img["image_url"],
                        'image_path': img["image_path"],
                        'image_index': img.get("image_index", i),
                        'type': 'learning_image_caption',
                        'source': 'learning_pdf',
                        'upload_date': datetime.utcnow().isoformat()
                    }
                )
                documents.append(doc)
            
            # Add documents to vector store
            if documents:
                vector_store.add_documents(documents)
                print(f"Successfully stored {len(documents)} image captions in learning vector database for {subject}")
                return len(documents)
            else:
                return 0
                
        except Exception as e:
            print(f"Error storing learning image captions in vector database: {str(e)}")
            return 0

    async def _find_relevant_learning_images_async(self, user_id: str, subject: str, query: str, similarity_threshold: float = 0.4, max_images: int = 3) -> List[Dict]:
        """Find multiple relevant images from learning PDFs for a query (async).
        
        Args:
            user_id: ID of the user
            subject: Subject category
            query: Query to search for
            similarity_threshold: Maximum similarity score threshold (lower is more similar)
            max_images: Maximum number of images to return
            
        Returns:
            List of dictionaries with image information
        """
        def _find_images_sync():
            return self._find_relevant_learning_images(user_id, subject, query, similarity_threshold, max_images)
        
        # Run the synchronous operation in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, _find_images_sync)

    def _find_relevant_learning_images(self, user_id: str, subject: str, query: str, similarity_threshold: float = 0.4, max_images: int = 3) -> List[Dict]:
        """Find multiple relevant images from learning PDFs for a query.
        
        Args:
            user_id: ID of the user
            subject: Subject category
            query: Query to search for
            similarity_threshold: Maximum similarity score threshold (lower is more similar)
            max_images: Maximum number of images to return
            
        Returns:
            List of dictionaries with image information
        """
        relevant_images = []
        
        try:
            # Create collection name for learning PDF images
            collection_name = f"learning_{subject}_images_{user_id}"
            connection_string = settings.PGVECTOR_CONNECTION_STRING
            ug = PGEngine.from_connection_string(url=connection_string)
            # Use the main PGVector connection
           
            
            # Initialize vector store
            image_vector_store = PGVectorStore.create_sync(
                engine=ug,
                embedding_service=self.embeddings,
                table_name=collection_name,
            )
            
            # Perform similarity search to find relevant images
            results = image_vector_store.similarity_search_with_score(
                query, 
                k=max_images * 2  # Get more results to filter
            )
            
            logger.info(f"Learning image search results count: {len(results)}")
            
            for doc, score in results:
                logger.info(f"Learning image found with score: {score}, threshold: {similarity_threshold}")
                
                # Check if the similarity score meets the threshold (lower score = more similar)
                if score <= similarity_threshold:
                    if doc.metadata and "image_url" in doc.metadata:
                        image_info = {
                            "image_url": doc.metadata["image_url"],
                            "caption": doc.page_content,
                            "score": score,
                            "page_number": doc.metadata.get("page_number"),
                            "pdf_id": doc.metadata.get("pdf_id"),
                            "subject": doc.metadata.get("subject")
                        }
                        relevant_images.append(image_info)
                        logger.info(f"Added relevant image: {image_info['image_url']}")
                        
                        # Stop if we have enough images
                        if len(relevant_images) >= max_images:
                            break
                else:
                    logger.info(f"Learning image found but similarity score {score} doesn't meet threshold {similarity_threshold}")
                    
        except Exception as e:
            print(f"Error finding relevant learning images: {str(e)}")
            
        return relevant_images

    async def upload_learning_image(self,
                                    file: UploadFile,
                                    user_id: str,
                                    caption: str,
                                    subject: str,
                                    topic: Optional[str] = None,
                                    grade: Optional[str] = None,
                                    page_number: Optional[int] = None) -> Tuple[Dict, int]:
        """Upload and store an image with caption for learning purposes.
        
        Args:
            file: Uploaded image file
            user_id: ID of the user
            caption: Caption or description for the image
            subject: Subject category (must be one of the valid subjects)
            topic: Optional topic within the subject
            grade: Optional grade level
            page_number: Optional page number reference
            
        Returns:
            Tuple of (response_data, status_code)
        """
        try:
            # Validate subject
            if subject not in self.SUBJECT_COLLECTIONS:
                return {
                    "success": False,
                    "message": f"Invalid subject: {subject}. Valid subjects are: {', '.join(self.SUBJECT_COLLECTIONS.keys())}"
                }, 400
            
            # Validate file type (allow common image formats)
            allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']
            file_extension = None
            if file.filename:
                file_extension = '.' + file.filename.lower().split('.')[-1]
                if file_extension not in allowed_extensions:
                    return {
                        "success": False,
                        "message": f"Invalid image format. Allowed formats: {', '.join(allowed_extensions)}"
                    }, 400
            else:
                return {
                    "success": False,
                    "message": "No filename provided"
                }, 400
            
            # Validate file size (max 10MB for images)
            if file.size and file.size > 10 * 1024 * 1024:
                return {
                    "success": False,
                    "message": "Image size too large. Maximum size is 10MB."
                }, 400
            
            # Generate unique ID and filename for the image
            image_id = str(uuid.uuid4())
            filename = f"learning_image_{image_id}{file_extension}"
            
            # Save the uploaded file to static directory
            file_contents = await file.read()
            
            # Process and save the image
            result = await self._process_learning_image_async(
                image_id=image_id,
                user_id=user_id,
                caption=caption,
                subject=subject,
                topic=topic,
                grade=grade,
                page_number=page_number,
                file_contents=file_contents,
                filename=filename,
                original_filename=file.filename
            )
            
            return result
            
        except Exception as e:
            error_message = f"Error uploading learning image: {str(e)}"
            print(error_message)
            return {
                "success": False,
                "message": error_message
            }, 500

    async def _process_learning_image_async(self,
                                            image_id: str,
                                            user_id: str,
                                            caption: str,
                                            subject: str,
                                            topic: Optional[str],
                                            grade: Optional[str],
                                            page_number: Optional[int],
                                            file_contents: bytes,
                                            filename: str,
                                            original_filename: str) -> Tuple[Dict, int]:
        """Process and store a learning image with its caption in vector database.
        
        Args:
            image_id: Unique identifier for the image
            user_id: ID of the user
            caption: Caption or description for the image
            subject: Subject category
            topic: Optional topic
            grade: Optional grade level
            page_number: Optional page number reference
            file_contents: Raw image file content
            filename: Generated filename
            original_filename: Original uploaded filename
            
        Returns:
            Tuple of (response_data, status_code)
        """
        try:
            # Save image to static directory
            image_url = await self._save_learning_image_to_static_async(
                file_contents, filename, subject, user_id
            )
            
            # Store image caption in vector database
            success = await self._store_single_learning_image_caption_async(
                image_id=image_id,
                user_id=user_id,
                subject=subject,
                caption=caption,
                image_url=image_url,
                topic=topic,
                grade=grade,
                page_number=page_number,
                original_filename=original_filename
            )
            
            if success:
                response_data = {
                    "success": True,
                    "message": "Image uploaded and processed successfully",
                    "image_id": image_id,
                    "image_url": image_url,
                    "subject": subject,
                    "caption": caption,
                    "upload_date": datetime.utcnow().isoformat()
                }
                return response_data, 200
            else:
                return {
                    "success": False,
                    "message": "Failed to store image caption in vector database"
                }, 500
                
        except Exception as e:
            error_message = f"Error processing learning image: {str(e)}"
            print(error_message)
            return {
                "success": False,
                "message": error_message
            }, 500

    async def _save_learning_image_to_static_async(self,
                                                   file_contents: bytes,
                                                   filename: str,
                                                   subject: str,
                                                   user_id: str) -> str:
        """Save learning image to static directory (async).
        
        Args:
            file_contents: Raw image file content
            filename: Filename to save as
            subject: Subject category
            user_id: ID of the user
            
        Returns:
            URL path to the saved image
        """
        def _save_image_sync():
            return self._save_learning_image_to_static(file_contents, filename, subject, user_id)
        
        # Run the file operation in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, _save_image_sync)

    def _save_learning_image_to_static(self,
                                       file_contents: bytes,
                                       filename: str,
                                       subject: str,
                                       user_id: str) -> str:
        """Save learning image to static directory.
        
        Args:
            file_contents: Raw image file content
            filename: Filename to save as
            subject: Subject category
            user_id: ID of the user
            
        Returns:
            URL path to the saved image
        """
        # Create directory structure: static/learning_images/{subject}/{user_id}/
        current_file_path = os.path.abspath(__file__)
        path_parts = current_file_path.replace('\\', '/').split('/')
        server_indices = [i for i, part in enumerate(path_parts) if part == 'server']
        
        if server_indices:
            server_index = server_indices[0]
            server_path_parts = path_parts[:server_index + 1]
            server_dir = '/'.join(server_path_parts)
            server_dir = server_dir.replace('/', os.sep)
        else:
            working_dir = os.getcwd()
            server_dir = os.path.join(working_dir, 'server')
        
        # Create the learning images directory structure
        static_dir = os.path.join(server_dir, "static")
        learning_images_dir = os.path.join(static_dir, "learning_images", subject, user_id)
        
        # Ensure directory exists
        os.makedirs(learning_images_dir, exist_ok=True)
        
        # Save image file
        file_path = os.path.join(learning_images_dir, filename)
        with open(file_path, "wb") as f:
            f.write(file_contents)
        
        # Return URL path (relative to static directory)
        url_path = f"/static/learning_images/{subject}/{user_id}/{filename}"
        print(f"Saved learning image to: {file_path}")
        print(f"Generated URL: {url_path}")
        
        return url_path

    async def _store_single_learning_image_caption_async(self,
                                                         image_id: str,
                                                         user_id: str,
                                                         subject: str,
                                                         caption: str,
                                                         image_url: str,
                                                         topic: Optional[str] = None,
                                                         grade: Optional[str] = None,
                                                         page_number: Optional[int] = None,
                                                         original_filename: Optional[str] = None) -> bool:
        """Store a single learning image caption in vector database (async).
        
        Args:
            image_id: Unique identifier for the image
            user_id: ID of the user
            subject: Subject category
            caption: Image caption
            image_url: URL path to the image
            topic: Optional topic
            grade: Optional grade level
            page_number: Optional page number reference
            original_filename: Original uploaded filename
            
        Returns:
            True if successfully stored
        """
        def _store_caption_sync():
            return asyncio.run(self._store_single_learning_image_caption(
                image_id, user_id, subject, caption, image_url, topic, grade, page_number, original_filename
            ))
        
        # Run the synchronous operation in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, _store_caption_sync)

    async def _store_single_learning_image_caption(self,
                                                   image_id: str,
                                                   user_id: str,
                                                   subject: str,
                                                   caption: str,
                                                   image_url: str,
                                                   topic: Optional[str] = None,
                                                   grade: Optional[str] = None,
                                                   page_number: Optional[int] = None,
                                                   original_filename: Optional[str] = None) -> bool:
        """Store a single learning image caption in vector database.
        
        Args:
            image_id: Unique identifier for the image
            user_id: ID of the user
            subject: Subject category
            caption: Image caption
            image_url: URL path to the image
            topic: Optional topic
            grade: Optional grade level
            page_number: Optional page number reference
            original_filename: Original uploaded filename
            
        Returns:
            True if successfully stored
        """
        try:
            # Create subject-specific collection name for learning images
            collection_name = f"learning_{subject}_images_{user_id}"
            connection_string = settings.PGVECTOR_CONNECTION_STRING
            ug = PGEngine.from_connection_string(url=connection_string)
            
            # Use the main PGVector connection
            
            
            # Initialize embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
            
            # Use PGVector with subject-specific collection
            vector_store = PGVectorStore.create_sync(
                engine=ug,
                embedding_service=embeddings,
                table_name=collection_name,
            )
            
            # Create document with comprehensive metadata
            doc = Document(
                page_content=caption,
                metadata={
                    'image_id': image_id,
                    'user_id': user_id,
                    'subject': subject,
                    'topic': topic or '',
                    'grade': grade or '',
                    'page_number': page_number,
                    'image_url': image_url,
                    'original_filename': original_filename or '',
                    'type': 'user_uploaded_learning_image',
                    'source': 'user_upload',
                    'upload_date': datetime.utcnow().isoformat()
                }
            )
            
            # Add document to vector store
            vector_store.add_documents([doc])
            print(f"Successfully stored learning image caption in vector database for {subject}")
            return True
                
        except Exception as e:
            print(f"Error storing learning image caption in vector database: {str(e)}")
            return False

    async def get_student_learning_achievements(self, student_id: str, achievement_type: str = None) -> Tuple[Dict, int]:
        """Get learning achievements for a student (async).
        
        Args:
            student_id: ID of the student
            achievement_type: Optional filter by achievement type
            
        Returns:
            Tuple of (achievements_data, status_code)
        """
        try:
            def _get_achievements_sync():
                return self.learning_achievement_service.get_student_learning_achievements(student_id, achievement_type)
            
            # Run the database operation in a thread pool
            loop = asyncio.get_event_loop()
            achievements_data, status_code = await loop.run_in_executor(self.thread_pool, _get_achievements_sync)
            
            return achievements_data, status_code
            
        except Exception as e:
            error_message = f"Error getting learning achievements: {str(e)}"
            print(error_message)
            return {"message": error_message}, 500

    async def get_student_learning_badges(self, student_id: str, subject: str = None) -> Tuple[Dict, int]:
        """Get learning badges for a student (async).
        
        Args:
            student_id: ID of the student
            subject: Optional filter by subject
            
        Returns:
            Tuple of (badges_data, status_code)
        """
        try:
            def _get_badges_sync():
                return self.learning_achievement_service.get_student_learning_badges(student_id, subject)
            
            # Run the database operation in a thread pool
            loop = asyncio.get_event_loop()
            badges_data, status_code = await loop.run_in_executor(self.thread_pool, _get_badges_sync)
            
            return badges_data, status_code
            
        except Exception as e:
            error_message = f"Error getting learning badges: {str(e)}"
            print(error_message)
            return {"message": error_message}, 500

    async def get_student_learning_streaks(self, student_id: str) -> Tuple[Dict, int]:
        """Get learning streaks for a student (async).
        
        Args:
            student_id: ID of the student
            
        Returns:
            Tuple of (streaks_data, status_code)
        """
        try:
            def _get_streaks_sync():
                return self.learning_achievement_service.get_student_learning_streaks(student_id)
            
            # Run the database operation in a thread pool
            loop = asyncio.get_event_loop()
            streaks_data, status_code = await loop.run_in_executor(self.thread_pool, _get_streaks_sync)
            
            return streaks_data, status_code
            
        except Exception as e:
            error_message = f"Error getting learning streaks: {str(e)}"
            print(error_message)
            return {"message": error_message}, 500
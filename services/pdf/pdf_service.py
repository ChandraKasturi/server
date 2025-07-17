import os
import uuid
import asyncio
import json
from typing import Dict, List, Optional, Any, Union, Tuple, BinaryIO
from datetime import datetime
import pypdf
from fastapi import UploadFile
import tempfile
import shutil
import redis.asyncio as redis
import io
from google import genai
from google.genai import types as genai_types
import concurrent.futures

from config import settings
from models.pdf_models import (
    ProcessingStatus, PDFDocument, PDFChunk, PDFUploadRequest
)
from repositories.pdf_repository import PDFRepository
from repositories.pgvector_repository import LangchainVectorRepository
from repositories.postgres_text_repository import PostgresTextRepository
from repositories.mongo_repository import QuestionRepository
from langchain_postgres.vectorstores import PGVector
from langchain.embeddings import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class PDFUploadService:
    """Service for handling PDF uploads."""
    
    def __init__(self, pdf_repository: Optional[PDFRepository] = None, redis_client: Optional[redis.Redis] = None):
        """Initialize the PDF upload service.
        
        Args:
            pdf_repository: Repository for PDF storage operations
            redis_client: Redis client for queue management
        """
        self.pdf_repository = pdf_repository or PDFRepository()
        self.redis_client = redis_client or redis.from_url(settings.REDIS_URL)
        self.upload_dir = os.path.join(settings.static_dir_path, "pdfs")
        
        # Ensure the upload directory exists
        if not os.path.exists(self.upload_dir):
            os.makedirs(self.upload_dir, exist_ok=True)
    
    def _get_user_upload_dir(self, user_id: str) -> str:
        """Get the upload directory for a specific user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Path to the user's upload directory
        """
        user_dir = os.path.join(self.upload_dir, user_id)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir, exist_ok=True)
        return user_dir
    
    async def upload_pdf(self, 
                         file: UploadFile, 
                         user_id: str, 
                         metadata: PDFUploadRequest) -> PDFDocument:
        """Upload a PDF file and store metadata.
        
        Args:
            file: Uploaded PDF file
            user_id: ID of the user uploading the file
            metadata: Additional metadata for the PDF
            
        Returns:
            PDFDocument with metadata
        """
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise ValueError("Uploaded file must be a PDF")
        
        # Generate unique ID for the PDF
        pdf_id = str(uuid.uuid4())
        
        # Prepare upload directory
        user_dir = self._get_user_upload_dir(user_id)
        
        # Create a unique filename to avoid collisions
        safe_filename = f"{pdf_id}_{file.filename.replace(' ', '_')}"
        file_path = os.path.join(user_dir, safe_filename)
        
        # Save the file
        with open(file_path, "wb") as pdf_file:
            # Read file in chunks to avoid loading large files into memory
            contents = await file.read()
            pdf_file.write(contents)
            file_size = len(contents)
        
        # Create PDF document record
        from models.pdf_models import PDFDocumentMetadata
        pdf_metadata = PDFDocumentMetadata(
            subject=metadata.subject,
            grade=metadata.grade
        )
        
        pdf_document = PDFDocument(
            id=pdf_id,
            user_id=user_id,
            file_name=file.filename,
            file_path=file_path,
            file_size=file_size,
            title=metadata.title,
            description=metadata.description,
            upload_date=datetime.utcnow(),
            processing_status=ProcessingStatus.PENDING,
            metadata=pdf_metadata
        )
        
        # Save to repository
        self.pdf_repository.save_pdf_document(pdf_document)
        
        # Add to Redis processing queue
        task = {
            "pdf_id": pdf_id,
            "user_id": user_id,
            "priority": 1,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add to Redis queue with priority
        await self.redis_client.zadd(
            "pdf_processing_queue",
            {json.dumps(task): 1}  # Score 1 = standard priority
        )
        print(f"PDF added to Redis queue: {pdf_id}")
        return pdf_document
    
    def get_user_pdfs(self, user_id: str, subject: Optional[str] = None) -> List[PDFDocument]:
        """Get all PDFs uploaded by a user.
        
        Args:
            user_id: ID of the user
            subject: Optional subject to filter PDFs by
            
        Returns:
            List of PDF documents
        """
        return self.pdf_repository.get_user_pdf_documents(user_id, subject)
    
    def get_pdf(self, pdf_id: str) -> Optional[PDFDocument]:
        """Get a PDF document by ID.
        
        Args:
            pdf_id: ID of the PDF document
            
        Returns:
            PDF document if found, None otherwise
        """
        return self.pdf_repository.get_pdf_document(pdf_id)
    
    def delete_pdf(self, pdf_id: str, user_id: str) -> bool:
        """Delete a PDF document and its file.
        
        Args:
            pdf_id: ID of the PDF document
            user_id: ID of the user (for authorization)
            
        Returns:
            True if deletion was successful
        """
        pdf_document = self.pdf_repository.get_pdf_document(pdf_id)
        
        if not pdf_document:
            return False
            
        # Check authorization
        if pdf_document.user_id != user_id:
            raise PermissionError("You don't have permission to delete this PDF")
            
        # Delete the file
        if os.path.exists(pdf_document.file_path):
            os.remove(pdf_document.file_path)
            
        # Delete from repository
        doc_deleted, chunks_deleted = self.pdf_repository.delete_pdf_document(pdf_id)
        
        return doc_deleted


class PDFProcessingService:
    """Service for processing PDF files asynchronously with Redis queue."""
    
    def __init__(self, 
                 pdf_repository: Optional[PDFRepository] = None,
                 redis_client: Optional[redis.Redis] = None,
                 websocket_manager=None,
                 openai_api_key: Optional[str] = None,
                 max_workers: int = 5):
        """Initialize the PDF processing service.
        
        Args:
            pdf_repository: Repository for PDF storage operations
            redis_client: Redis client for queue management
            websocket_manager: WebSocket manager for real-time updates
            openai_api_key: API key for OpenAI. If None, uses the one from settings.
            max_workers: Maximum number of concurrent PDF processing workers
        """
        self.pdf_repository = pdf_repository or PDFRepository()
        self.redis_client = redis_client or redis.from_url(settings.REDIS_URL)
        self.websocket_manager = websocket_manager
        self.api_key = openai_api_key or settings.OPENAI_API_KEY
        self.max_workers = max_workers
        self.worker_semaphore = asyncio.Semaphore(max_workers)
        self.postgres_text_repository = PostgresTextRepository()
        # Thread pool for blocking operations
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        
    async def process_queued_pdfs(self):
        """Process all queued PDFs asynchronously using Redis queue."""
        while True:
            try:
                print(f"Redis Client: {self.redis_client}")
                # Get the highest priority task from Redis sorted set
                tasks = await self.redis_client.zpopmax("pdf_processing_queue", 1)
                
                if not tasks:
                    # No PDFs in queue, wait before checking again
                    await asyncio.sleep(5)
                    continue
                
                # Parse the task
                task_json, priority = tasks[0]
                task = json.loads(task_json)
                
                pdf_id = task.get("pdf_id")
                user_id = task.get("user_id")
                
                if not pdf_id or not user_id:
                    continue
                
                # Get the PDF document
                pdf_document = self.pdf_repository.get_pdf_document(pdf_id)
                
                if not pdf_document:
                    await self.redis_client.hset(
                        f"pdf_processing_errors:{pdf_id}",
                        mapping={
                            "error": "PDF document not found",
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                    continue
                
                # Update status to processing
                self.pdf_repository.update_pdf_status(
                    pdf_id,
                    ProcessingStatus.PROCESSING
                )
                
                # Process the PDF with concurrency control
                # Spawn a new task for processing
                asyncio.create_task(self._process_with_semaphore(pdf_document, user_id))
                
            except Exception as e:
                print(f"Error in PDF processing worker: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _process_with_semaphore(self, pdf_document: PDFDocument, user_id: str):
        """Process a PDF document with worker concurrency control.
        
        Args:
            pdf_document: PDF document to process
            user_id: User ID for student-specific storage
        """
        async with self.worker_semaphore:
            try:
                success = await self.process_pdf(pdf_document, user_id)
                
                if success:
                    # Mark as complete
                    self.pdf_repository.update_pdf_status(
                        pdf_document.id,
                        ProcessingStatus.COMPLETED
                    )
                    
                    # Record success in Redis
                    success_status = {
                        "status": "completed",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    await self.redis_client.hset(
                        f"pdf_processing_status:{pdf_document.id}",
                        mapping=success_status
                    )
                    
                    # Publish success event
                    await self._publish_status_update(pdf_document.id, success_status)
                else:
                    # Mark as failed
                    self.pdf_repository.update_pdf_status(
                        pdf_document.id,
                        ProcessingStatus.FAILED,
                        "Error processing PDF"
                    )
                    
                    # Record failure in Redis
                    failure_status = {
                        "status": "failed",
                        "error": "Error processing PDF",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    await self.redis_client.hset(
                        f"pdf_processing_status:{pdf_document.id}",
                        mapping=failure_status
                    )
                    
                    # Publish failure event
                    await self._publish_status_update(pdf_document.id, failure_status)
            
            except Exception as e:
                error_message = str(e)
                print(f"Error processing PDF {pdf_document.id}: {error_message}")
                
                # Update status in MongoDB
                self.pdf_repository.update_pdf_status(
                    pdf_document.id,
                    ProcessingStatus.FAILED,
                    error_message
                )
                
                # Record error in Redis
                error_status = {
                    "status": "failed",
                    "error": error_message,
                    "timestamp": datetime.utcnow().isoformat()
                }
                await self.redis_client.hset(
                    f"pdf_processing_status:{pdf_document.id}",
                    mapping=error_status
                )
                
                # Publish error event
                await self._publish_status_update(pdf_document.id, error_status)
    
    async def process_pdf(self, pdf_document: PDFDocument, user_id: str) -> bool:
        """Process a PDF document and store its contents (async version).
        
        Args:
            pdf_document: PDF document to process
            user_id: ID of the user who owns the document
            
        Returns:
            True if processing was successful
        """
        pdf_id = pdf_document.id
        file_path = pdf_document.file_path
        
        try:
            # Update processing status (async)
            await self._update_pdf_status_async(pdf_id, ProcessingStatus.PROCESSING)
            
            # Update Redis status
            await self.redis_client.hset(
                f"pdf_processing_status:{pdf_id}",
                mapping={
                    "status": "processing",
                    "start_time": datetime.utcnow().isoformat()
                }
            )
            
            # Publish processing started event
            await self._publish_status_update(pdf_id, {
                "status": "processing",
                "start_time": datetime.utcnow().isoformat(),
                "step": "started"
            })
            
            # Extract images from PDF first (async)
            await self.redis_client.hset(
                f"pdf_processing_status:{pdf_id}",
                mapping={"step": "extracting_images"}
            )
            
            # Publish image extraction step
            await self._publish_status_update(pdf_id, {
                "status": "processing",
                "step": "extracting_images"
            })
            
            image_data = await self._extract_images_from_pdf_async(file_path, user_id, pdf_id)
            
            # Extract text from PDF (async)
            try:
                await self.redis_client.hset(
                    f"pdf_processing_status:{pdf_id}",
                    mapping={"step": "extracting_text"}
                )
                
                # Publish text extraction step
                await self._publish_status_update(pdf_id, {
                    "status": "processing",
                    "step": "extracting_text"
                })
                
                chunks, page_count = await self._extract_text_async(file_path)
                print(f"Chunks: {len(chunks)}")
                print(f"Page count: {page_count}")
                
                # Store the full text in PostgreSQL for the specific user (async)
                full_text = "\n\n".join([chunk["text"] for chunk in chunks])
                full_text = full_text.replace("\x00", "")
                
                # Store in Postgres text repository (async)
                await self._store_pdf_text_async(
                    student_id=user_id,
                    pdf_id=pdf_id,
                    title=pdf_document.title,
                    content=full_text,
                    page_count=page_count,
                    metadata=pdf_document.metadata.dict() if pdf_document.metadata else {}
                )
                
                # Store chunks in the user's database (async)
                chunk_data = [{
                    "chunk_index": i,
                    "page_number": chunk.get("page", 0),
                    "content": chunk["text"].replace("\x00", "")
                } for i, chunk in enumerate(chunks)]
                
                await self._store_pdf_chunks_async(user_id, pdf_id, chunk_data)
                
                # Convert chunks to PDFChunk objects and store in MongoDB (async)
                await self.redis_client.hset(
                    f"pdf_processing_status:{pdf_id}",
                    mapping={"step": "storing_chunks_in_mongodb"}
                )
                
                # Publish MongoDB storage step
                await self._publish_status_update(pdf_id, {
                    "status": "processing",
                    "step": "storing_chunks_in_mongodb"
                })
                
                pdf_chunks = []
                for i, chunk in enumerate(chunks):
                    chunk_obj = PDFChunk(
                        id=str(uuid.uuid4()),
                        pdf_id=pdf_id,
                        chunk_index=i,
                        page_number=chunk.get("page", 0),
                        content=chunk["text"].replace("\x00", ""),
                        embedding=None  # We'll compute this in the vector DB step
                    )
                    await self._save_pdf_chunk_async(chunk_obj)
                    pdf_chunks.append(chunk_obj)
                
                # Process image captions (async)
                if image_data:
                    await self.redis_client.hset(
                        f"pdf_processing_status:{pdf_id}",
                        mapping={"step": "processing_image_captions"}
                    )
                    
                    # Publish image caption processing step
                    await self._publish_status_update(pdf_id, {
                        "status": "processing",
                        "step": "processing_image_captions",
                        "images_extracted": len(image_data)
                    })
                    
                    # Store image captions in PGVector with a separate collection (async)
                    await self._store_image_captions_in_vector_db_async(pdf_id, image_data, user_id)
                
                # Store chunks in vector DB for semantic search (async)
                await self.redis_client.hset(
                    f"pdf_processing_status:{pdf_id}",
                    mapping={"step": "storing_in_vector_db"}
                )
                
                # Publish vector DB storage step
                await self._publish_status_update(pdf_id, {
                    "status": "processing",
                    "step": "storing_in_vector_db"
                })
                
                await self._store_chunks_in_vector_db_async(pdf_id, pdf_chunks, user_id)
                
                # Update processing status to completed (async)
                await self._update_pdf_status_async(pdf_id, ProcessingStatus.COMPLETED)
                
                # Update PDF metadata with image count (async)
                if image_data:
                    # Get current metadata
                    pdf_doc = self.pdf_repository.get_pdf_document(pdf_id)
                    if pdf_doc:
                        from models.pdf_models import PDFDocumentMetadata
                        # Convert existing metadata to dict and add image information
                        current_metadata_dict = pdf_doc.metadata.dict() if pdf_doc.metadata else {}
                        current_metadata_dict.update({
                            "image_count": len(image_data),
                            "has_images": len(image_data) > 0,
                            "images_processed": True
                        })
                        # Update metadata (async) - pass dict directly to MongoDB
                        await self._update_pdf_document_async(pdf_id, {"metadata": current_metadata_dict})
                
                # Update Redis status
                completion_status = {
                    "status": "completed",
                    "end_time": datetime.utcnow().isoformat(),
                    "step": "completed",
                    "images_extracted": len(image_data)
                }
                
                await self.redis_client.hset(
                    f"pdf_processing_status:{pdf_id}",
                    mapping=completion_status
                )
                
                # Publish completion event
                await self._publish_status_update(pdf_id, completion_status)
                
                return True
                
            except Exception as e:
                error_message = f"Error processing PDF: {str(e)}"
                print(error_message)
                
                # Update processing status to failed (async)
                await self._update_pdf_status_async(pdf_id, ProcessingStatus.FAILED, error_message)
                
                # Update Redis status
                failure_status = {
                    "status": "failed",
                    "error": error_message,
                    "end_time": datetime.utcnow().isoformat()
                }
                
                await self.redis_client.hset(
                    f"pdf_processing_status:{pdf_id}",
                    mapping=failure_status
                )
                
                # Publish failure event
                await self._publish_status_update(pdf_id, failure_status)
                
                # Log error details
                await self.redis_client.hset(
                    f"pdf_processing_errors:{pdf_id}",
                    mapping={
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                
                return False
                
        except Exception as e:
            error_message = f"Unexpected error in PDF processing: {str(e)}"
            print(error_message)
            
            # Update processing status to failed (async)
            await self._update_pdf_status_async(pdf_id, ProcessingStatus.FAILED, error_message)
            
            # Update Redis status
            unexpected_failure_status = {
                "status": "failed",
                "error": error_message,
                "end_time": datetime.utcnow().isoformat()
            }
            
            await self.redis_client.hset(
                f"pdf_processing_status:{pdf_id}",
                mapping=unexpected_failure_status
            )
            
            # Publish unexpected failure event
            await self._publish_status_update(pdf_id, unexpected_failure_status)
            
            return False
    
    async def _extract_text_async(self, file_path: str) -> Tuple[List[Dict], int]:
        """Async version of text extraction from a PDF file using Gemini OCR.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (chunks, num_pages)
        """
        def _extract_text_sync():
            return asyncio.run(self._extract_text(file_path))
        
        # Run the synchronous operation in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, _extract_text_sync)
    
    async def _store_chunks_in_vector_db_async(self, pdf_id: str, chunks: List[PDFChunk], user_id: str):
        """Async version of storing chunks in the vector database.
        
        Args:
            pdf_id: ID of the PDF document
            chunks: List of text chunks to store
            user_id: User ID for student-specific storage
        """
        def _store_chunks_sync():
            # This is a synchronous wrapper for the existing method
            asyncio.run(self._store_chunks_in_vector_db(pdf_id, chunks, user_id))
        
        # Run the synchronous operation in a thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.thread_pool, _store_chunks_sync)
    
    async def _store_image_captions_in_vector_db_async(self, pdf_id: str, image_data: List[Dict], user_id: str):
        """Async version of storing image captions in the vector database.
        
        Args:
            pdf_id: ID of the PDF document
            image_data: List of image data with captions
            user_id: User ID for student-specific storage
        """
        def _store_captions_sync():
            asyncio.run(self._store_image_captions_in_vector_db(pdf_id, image_data, user_id))
        
        # Run the synchronous operation in a thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.thread_pool, _store_captions_sync)
    
    async def _extract_images_from_pdf_async(self, pdf_path: str, user_id: str, pdf_id: str) -> List[Dict]:
        """Async version of extracting images from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            user_id: ID of the user who owns the document
            pdf_id: ID of the PDF document
            
        Returns:
            List of dictionaries with image information
        """
        def _extract_images_sync():
            return asyncio.run(self._extract_images_from_pdf(pdf_path, user_id, pdf_id))
        
        # Run the synchronous operation in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, _extract_images_sync)
    
    async def _update_pdf_status_async(self, pdf_id: str, status: ProcessingStatus, error: str = None):
        """Async version of updating PDF status.
        
        Args:
            pdf_id: ID of the PDF document
            status: New processing status
            error: Error message if any
        """
        def _update_status_sync():
            return self.pdf_repository.update_pdf_status(pdf_id, status, error)
        
        # Run the database operation in a thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.thread_pool, _update_status_sync)
    
    async def _publish_status_update(self, pdf_id: str, status_data: Dict[str, Any]):
        """Publish status update to WebSocket connections for real-time notifications.
        
        Args:
            pdf_id: ID of the PDF document
            status_data: Status data to publish
        """
        try:
            # Broadcast to WebSocket connections
            if self.websocket_manager:
                await self.websocket_manager.send_to_pdf_connections(pdf_id, {
                    "type": "status",
                    "data": status_data
                })
                print(f"Broadcast status update to WebSocket connections for PDF {pdf_id}")
            else:
                print(f"No WebSocket manager available for PDF {pdf_id}")
                
        except Exception as e:
            print(f"Error publishing status update: {str(e)}")
    
    async def _store_pdf_text_async(self, student_id: str, pdf_id: str, title: str, content: str, page_count: int, metadata: Dict):
        """Async version of storing PDF text.
        
        Args:
            student_id: ID of the student
            pdf_id: ID of the PDF document
            title: Title of the PDF
            content: Full text content
            page_count: Number of pages
            metadata: PDF metadata
        """
        def _store_text_sync():
            return self.postgres_text_repository.store_pdf_text(
                student_id=student_id,
                pdf_id=pdf_id,
                title=title,
                content=content,
                page_count=page_count,
                metadata=metadata
            )
        
        # Run the database operation in a thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.thread_pool, _store_text_sync)
    
    async def _store_pdf_chunks_async(self, student_id: str, pdf_id: str, chunks: List[Dict]):
        """Async version of storing PDF chunks.
        
        Args:
            student_id: ID of the student
            pdf_id: ID of the PDF document
            chunks: List of chunks to store
        """
        def _store_chunks_sync():
            return self.postgres_text_repository.store_pdf_chunks(student_id=student_id, pdf_id=pdf_id, chunks=chunks)
        
        # Run the database operation in a thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.thread_pool, _store_chunks_sync)
    
    async def _save_pdf_chunk_async(self, chunk_obj: PDFChunk):
        """Async version of saving PDF chunk.
        
        Args:
            chunk_obj: PDF chunk to save
        """
        def _save_chunk_sync():
            return self.pdf_repository.save_pdf_chunk(chunk_obj)
        
        # Run the database operation in a thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.thread_pool, _save_chunk_sync)
    
    async def _update_pdf_document_async(self, pdf_id: str, update_data: Dict):
        """Async version of updating PDF document.
        
        Args:
            pdf_id: ID of the PDF document
            update_data: Data to update
        """
        def _update_doc_sync():
            return self.pdf_repository.update_pdf_document(pdf_id, update_data)
        
        # Run the database operation in a thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.thread_pool, _update_doc_sync)

    async def _extract_text(self, file_path: str) -> Tuple[List[Dict], int]:
        """Extract text from a PDF file using Gemini OCR.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (chunks, num_pages)
        """
        chunks = []
        
        try:
            # Configure Gemini client
            gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
            
            # Read PDF file and upload to Gemini
            with open(file_path, 'rb') as file:
                pdf_content = file.read()
                pdf_io = io.BytesIO(pdf_content)
                pdf_io.name = os.path.basename(file_path)
                
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
                model="gemini-2.0-flash",
                contents=[uploaded_file, extraction_prompt]
            )
            
            extracted_text = response.text
            
            # Parse the response to create chunks
            pages = self._parse_gemini_text_response(extracted_text)
            
            # Convert to the expected format
            for page_num, page_text in pages.items():
                if page_text.strip():  # Only add pages with content
                    chunks.append({
                        'text': page_text.strip(),
                        'page': page_num,
                        'metadata': {
                            'page': page_num,
                            'total_pages': len(pages),
                            'extraction_method': 'gemini_ocr'
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
            # Fallback to original pypdf method if Gemini fails
            return await self._extract_text_fallback(file_path)
    
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
    
    async def _extract_text_fallback(self, file_path: str) -> Tuple[List[Dict], int]:
        """Fallback text extraction method using pypdf.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (chunks, num_pages)
        """
        chunks = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            num_pages = min(len(pdf_reader.pages), settings.PDF_MAX_PAGES)
            
            for page_num, page in enumerate(pdf_reader.pages[:num_pages], 1):
                text = page.extract_text()
                if text.strip():
                    # Simple chunking by page for now
                    chunks.append({
                        'text': text,
                        'page': page_num,
                        'metadata': {
                            'page': page_num,
                            'total_pages': num_pages,
                            'extraction_method': 'pypdf_fallback'
                        }
                    })
        
        return chunks, num_pages
    
    async def _store_chunks_in_vector_db(self, pdf_id: str, chunks: List[PDFChunk], user_id: str):
        """Store chunks in the vector database using student-specific database.
        
        Args:
            pdf_id: ID of the PDF document
            chunks: List of text chunks to store
            user_id: User ID for student-specific storage
        """
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
                student_db = f"student_{user_id}"
                
                # Reconstruct the connection string
                connection_string = f"{prefix}{host_port}/{student_db}"
            else:
                # If no database name in original connection, just append it
                connection_string = f"{base_connection}/student_{user_id}"
        else:
            # Fallback: just use base connection
            connection_string = base_connection
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        '''embeddings = GoogleGenerativeAIEmbeddings(google_api_key=settings.GOOGLE_API_KEY,model="models/gemini-embedding-exp-03-07")'''
        
        # Create collection name for the PDF within the student's database
        collection_name = f"pdf_{pdf_id}"
        
        # Use PGVector with student-specific connection
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=connection_string,
            use_jsonb=True
        )
        
        # Convert chunks to documents for vector storage
        from langchain_core.documents import Document
        
        documents = []
        for chunk in chunks:
            # Create document with metadata
            # Convert Pydantic metadata to dict if it exists
            chunk_metadata_dict = chunk.metadata.dict() if chunk.metadata else {}
            
            doc = Document(
                page_content=chunk.content,
                metadata={
                    'pdf_id': pdf_id,
                    'chunk_id': chunk.id,
                    'page': chunk.page_number,
                    'user_id': user_id,  # Include user_id in metadata
                    **chunk_metadata_dict
                }
            )
            documents.append(doc)
        
        # Add documents to vector store
        if documents:
            vector_store.add_documents(documents)

    async def process_specific_pdf(self, pdf_id: str, user_id: str) -> bool:
        """Process a specific PDF document immediately.
        
        Args:
            pdf_id: ID of the PDF document
            user_id: User ID for student-specific storage
            
        Returns:
            True if processing was started
        """
        pdf_document = self.pdf_repository.get_pdf_document(pdf_id)
        
        if not pdf_document:
            return False
            
        # Update status to processing
        self.pdf_repository.update_pdf_status(
            pdf_id, 
            ProcessingStatus.PROCESSING
        )
        
        # Add to high priority queue
        task = {
            "pdf_id": pdf_id,
            "user_id": user_id,
            "priority": 10,  # Higher priority
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add to Redis queue with high priority
        await self.redis_client.zadd(
            "pdf_processing_queue",
            {json.dumps(task): 10}  # Score 10 = higher priority
        )
        
        return True
    
    async def _extract_images_from_pdf(self, pdf_path: str, user_id: str, pdf_id: str) -> List[Dict]:
        """Extract images from a PDF file and generate captions using Gemini.
        
        Args:
            pdf_path: Path to the PDF file
            user_id: ID of the user who owns the document
            pdf_id: ID of the PDF document
            
        Returns:
            List of dictionaries with image information
        """
        import fitz  # PyMuPDF
        from google import genai
        import PIL.Image
        
        image_data = []
        
        # Create output folder for images
        images_folder = os.path.join(settings.static_dir_path, "pdf_images", pdf_id)
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
                    image_filename = os.path.join(images_folder, f"image_{page_num + 1}_{image_count}.png")
                    
                    # Write the image to a file
                    with open(image_filename, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    # Generate a URL for the image (relative to static directory)
                    image_url = f"/static/pdf_images/{pdf_id}/image_{page_num + 1}_{image_count}.png"
                    
                    # Generate caption with Gemini API
                    try:
                        # Open the image with PIL
                        pil_image = PIL.Image.open(image_filename)
                        
                        # Generate caption
                        response = client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=["Write a concise and accurate caption for this image", pil_image]
                        )
                        
                        caption = response.text.strip()
                    except Exception as e:
                        caption = f"Image from page {page_num + 1}"
                        print(f"Error generating caption with Gemini: {str(e)}")
                    
                    # Store image data
                    image_data.append({
                        "image_path": image_filename,
                        "image_url": image_url,
                        "page_number": page_num + 1,
                        "caption": caption
                    })
                    
                    image_count += 1
            
            print(f"Extracted {image_count} images from PDF {pdf_id}")
            return image_data
            
        except Exception as e:
            print(f"Error extracting images from PDF: {str(e)}")
            return []
    
    async def _store_image_captions_in_vector_db(self, pdf_id: str, image_data: List[Dict], user_id: str):
        """Store image captions in the vector database using a separate collection.
        
        Args:
            pdf_id: ID of the PDF document
            image_data: List of image data with captions
            user_id: User ID for student-specific storage
        """
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
                student_db = f"student_{user_id}"
                
                # Reconstruct the connection string
                connection_string = f"{prefix}{host_port}/{student_db}"
            else:
                # If no database name in original connection, just append it
                connection_string = f"{base_connection}/student_{user_id}"
        else:
            # Fallback: just use base connection
            connection_string = base_connection
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        '''embeddings = GoogleGenerativeAIEmbeddings(google_api_key=settings.GOOGLE_API_KEY,model="models/gemini-embedding-exp-03-07")'''
        
        # Create collection name for the images within the student's database
        collection_name = f"pdf_{pdf_id}_images"
        
        # Use PGVector with student-specific connection
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=connection_string,
            use_jsonb=True
        )
        
        # Convert image captions to documents for vector storage
        from langchain_core.documents import Document
        
        documents = []
        for i, img in enumerate(image_data):
            # Create document with metadata
            doc = Document(
                page_content=img["caption"],
                metadata={
                    'pdf_id': pdf_id,
                    'image_id': f"image_{i}",
                    'page_number': img.get("page_number"),
                    'image_url': img["image_url"],
                    'image_path': img["image_path"],
                    'user_id': user_id,
                    'type': 'image_caption',
                    'source': 'image_extraction'
                }
            )
            documents.append(doc)
        
        # Add documents to vector store
        if documents:
            vector_store.add_documents(documents)


class PDFQuestionGenerationService:
    """Service for generating questions from PDF documents using OpenAI."""
    
    def __init__(self, question_repository: Optional[QuestionRepository] = None):
        """Initialize the PDF question generation service.
        
        Args:
            question_repository: Repository for question storage operations
        """
        self.question_repository = question_repository or QuestionRepository()
    
    async def generate_questions_from_pdf(
        self, 
        pdf_content: bytes, 
        subject: str, 
        topic: str, 
        subtopic: str
    ) -> Dict[str, Any]:
        """Generate questions from a PDF using OpenAI and insert them into question bank.
        
        Args:
            pdf_content: Raw PDF file content as bytes
            subject: Subject for the questions
            topic: Topic for the questions
            subtopic: Subtopic for the questions
            
        Returns:
            Dictionary with generated questions data
            
        Raises:
            Exception: If question generation fails
        """
        try:
            # Create BytesIO object from PDF content
            pdf_io = io.BytesIO(pdf_content)
            pdf_io.name = "input_document.pdf"

            # Configure Gemini client
            gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)

            # Upload PDF using Gemini File API
            uploaded_file = gemini_client.files.upload(
                file=pdf_io,
                config=dict(mime_type='application/pdf')
            )
            
            # Get current timestamp for all questions
            current_timestamp = datetime.now().isoformat()
            
            # Create prompt for question generation
            prompt = self._create_question_generation_prompt(subject, topic, subtopic, current_timestamp)
            
            # Generate content using Gemini
            response = gemini_client.models.generate_content(
                model="gemini-2.5-pro-preview-05-06",
                contents=[uploaded_file, prompt],
            )
            raw_response_content = response.text
            
            # Clean and format the JSON response using Gemini
            cleaned_json = await self._clean_json_with_gemini(raw_response_content)
            
            # Parse and process the response
            questions = self._parse_openai_response(cleaned_json, subject, topic, subtopic, current_timestamp)
            
            # Insert questions into the database
            successful_questions = []
            for question in questions:
                success = self.question_repository.insert_question(question)
                if success:
                    question.pop("_id", None)
                    successful_questions.append(question)
                else:
                    print(f"Failed to insert question: {question.get('question', 'Unknown question')}")
            
            return {
                "total_questions_generated": len(successful_questions),
                "questions": successful_questions
            }
            
        except Exception as e:
            raise Exception(f"Error generating questions from PDF: {str(e)}")
        finally:
            # Close the question repository connection
            if hasattr(self.question_repository, 'close'):
                self.question_repository.close()
            # Delete the uploaded file from Gemini to free up resources
            if 'uploaded_file' in locals() and uploaded_file:
                try:
                    gemini_client.files.delete(name=uploaded_file.name)
                    print(f"Successfully deleted uploaded file {uploaded_file.name} from Gemini.")
                except Exception as del_e:
                    print(f"Could not delete uploaded file {uploaded_file.name} from Gemini: {del_e}")
    
    def _create_question_generation_prompt(self, subject: str, topic: str, subtopic: str, current_timestamp: str) -> str:
        """Create the prompt for question generation.
        
        Args:
            subject: Subject for the questions
            topic: Topic for the questions  
            subtopic: Subtopic for the questions
            current_timestamp: Current timestamp for questions
            
        Returns:
            Formatted prompt string
        """
        return f"""
Analyze the provided PDF document and EXTRACT questions from its content.
For EACH question identified in the document, generate a JSON object.
Determine the question_type from: "multiple_choice", "VERY_SHORT_ANSWER", "SHORT_ANSWER", "LONG_ANSWER", "CASE_STUDY".
The output should be a JSON array containing these objects.
Write questions and options in LaTeX if needed. Do NOT include an "_id" field in your response.
IMPORTANT: Make sure to extract all the questions from the document and classify them appropriately:
- multiple_choice: Has distinct options (A, B, C, D)
- VERY_SHORT_ANSWER: Requires 1-3 words (definitions, terms)
- SHORT_ANSWER: Requires 1-3 sentences (brief explanations)
- LONG_ANSWER: Requires detailed explanations (multiple paragraphs)
- CASE_STUDY: Scenario-based application questions

**General Instructions for ALL questions:**
- "questionset" MUST be "scanned".
- "created_at" MUST use the provided timestamp: {current_timestamp}.
- "question_image" and "explaination_image" MUST be empty strings "" for now.
- `subject`, `topic`, and `subtopic` will be provided as context variables; use their values.
- `level`: Determine the difficulty level for this question [1-3] (1 Easy, 2 Medium, 3 Hard).
- `marks`: Extract the marks allocated for this question, if visible. If not visible, suggest a reasonable default (e.g., 1 for simple MCQs, 2-5 for descriptive, based on complexity).
- `grading_criteria`: Must not be empty.
- `model_answer`: Must not be empty.

If the question is primarily multiple-choice (has distinct options), use this format:
{{
  "question": "Extract the main question text from the document for this specific question (in LaTeX if needed)",
  "option1": "You must extract the first multiple-choice option for this question (in LaTeX if needed)",
  "option2": "You must extract the second multiple-choice option for this question (in LaTeX if needed)",
  "option3": "You must extract the third multiple-choice option for this question (in LaTeX if needed)",
  "option4": "You must extract the fourth multiple-choice option for this question (in LaTeX if needed)",
  "correctanswer": "Identify and provide the correct answer label (e.g., 'option1', 'option2', 'option3', or 'option4'). This field MUST be filled.",
  "explaination": "Provide a detailed explanation for why the correct answer is correct and, if applicable, why the other options are incorrect.  or if present in the document, provide the explanation from the document. This field MUST be filled.",
  "grading_criteria": "Typically for MCQs: 'Full marks if the correct option is selected, zero otherwise.' or similar based on the question's nature. This field MUST be filled.",
  "question_image": "",
  "question_type": "multiple_choice",
  "explaination_image": "",
  "subject": "{subject}",
  "topic": "{topic}",
  "subtopic": "{subtopic}",
  "level": "Determine the difficulty level for this question [1-3] 1 Easy , 2 Medium, 3 Hard",
  "questionset": "scanned",
  "marks": "Extract the marks allocated for this question, if visible. If not, suggest a default like 1.",
  "created_at": "{current_timestamp}",
  "ignore":True
}}

If the question requires a VERY SHORT ANSWER (1-3 words), use this format:
{{
  "question": "Extract the main question text from the document for this specific question (in LaTeX if needed)",
  "model_answer": "Expected very short answer (1-3 words or brief phrase)",
  "grading_criteria": "Full marks for exact/equivalent term, partial marks for close answers, zero for incorrect",
  "explaination": "Brief explanation about the expected answer and key concept",
  "question_image": "",
  "explaination_image": "",
  "question_type": "VERY_SHORT_ANSWER",
  "subject": "{subject}",
  "topic": "{topic}",
  "subtopic": "{subtopic}",
  "level": "Determine the difficulty level for this question [1-3] 1 Easy , 2 Medium, 3 Hard",
  "questionset": "scanned",
  "marks": "Extract the marks allocated for this question, if visible. If not, suggest default like 1.",
  "created_at": "{current_timestamp}",
  "ignore":True
}}

If the question requires a SHORT ANSWER (1-3 sentences), use this format:
{{
  "question": "Extract the main question text from the document for this specific question (in LaTeX if needed)",
  "model_answer": "Brief but complete answer (1-3 sentences covering key points)",
  "grading_criteria": "Key points breakdown: main concept (X marks), supporting detail (Y marks), clarity (Z marks)",
  "explaination": "Brief explanation about the question and key concepts tested",
  "question_image": "",
  "explaination_image": "",
  "question_type": "SHORT_ANSWER",
  "subject": "{subject}",
  "topic": "{topic}",
  "subtopic": "{subtopic}",
  "level": "Determine the difficulty level for this question [1-3] 1 Easy , 2 Medium, 3 Hard",
  "questionset": "scanned",
  "marks": "Extract the marks allocated for this question, if visible. If not, suggest default like 3.",
  "created_at": "{current_timestamp}",
  "ignore":True
}}

If the question requires a LONG ANSWER (detailed explanation), use this format:
{{
  "question": "Extract the main question text from the document for this specific question (in LaTeX if needed)",
  "model_answer": "Comprehensive detailed answer with multiple key points and examples",
  "grading_criteria": "Detailed breakdown: concept understanding (X marks), examples/evidence (Y marks), analysis/evaluation (Z marks), structure/clarity (W marks)",
  "explaination": "Explanation of the depth and scope expected in the answer",
  "question_image": "",
  "explaination_image": "",
  "question_type": "LONG_ANSWER",
  "subject": "{subject}",
  "topic": "{topic}",
  "subtopic": "{subtopic}",
  "level": "Determine the difficulty level for this question [1-3] 1 Easy , 2 Medium, 3 Hard",
  "questionset": "scanned",
  "marks": "Extract the marks allocated for this question, if visible. If not, suggest default like 5.",
  "created_at": "{current_timestamp}",
  "ignore":True
}}

If the question is a CASE STUDY (scenario-based application), use this format:
{{
  "question": "Extract the scenario and question text from the document (in LaTeX if needed)",
  "model_answer": "Comprehensive case analysis with problem identification, theoretical application, and practical solutions",
  "grading_criteria": "Case analysis (X marks), theoretical application (Y marks), practical solutions (Z marks), justification (W marks)",
  "explaination": "Overview of the case scenario and key learning objectives",
  "question_image": "",
  "explaination_image": "",
  "question_type": "CASE_STUDY",
  "subject": "{subject}",
  "topic": "{topic}",
  "subtopic": "{subtopic}",
  "level": "Determine the difficulty level for this question [1-3] 1 Easy , 2 Medium, 3 Hard",
  "questionset": "scanned",
  "marks": "Extract the marks allocated for this question, if visible. If not, suggest default like 10.",
  "created_at": "{current_timestamp}",
  "ignore":True
}}

Return a JSON array where each element is a JSON object formatted as described above.
If no questions are found, return an empty array [].
Ensure the output is a single valid JSON array string.
RETURN ONLY JSON NO ADDITIONAL TEXT OR COMMENTS MAKE SURE YOU GET ALL THE QUESTIONS AT ONCe
IMPORTANT: Make sure the FINAL JSON STRUCTURE is PARSABLE with json.loads()
IMPORTANT: Make sure the final JSON has only utf-8 encoded characters.
IMPORTANT: Make sure to Escape all backslashes and other charecters which could cause errors in JSON parsing.
IMPORTANT: FOR MCQ Do not include the options in the question keep them in the options field.
IMPORTANT: FOR all text-based questions (VERY_SHORT_ANSWER, SHORT_ANSWER, LONG_ANSWER, CASE_STUDY) model_answer and grading_criteria MUST NOT BE EMPTY.
"""
    
    async def _clean_json_with_gemini(self, raw_json_text: str) -> str:
        """Clean and format JSON response using Gemini to ensure it's parsable.
        
        Args:
            raw_json_text: Raw JSON text from OpenAI that may have formatting issues
            
        Returns:
            Cleaned and properly formatted JSON string
            
        Raises:
            Exception: If Gemini cleaning fails
        """
        try:
            # Configure Gemini client
            client = genai.Client(api_key=settings.GEMINI_API_KEY)
            
            # Create prompt for JSON cleaning
            cleaning_prompt = f"""
You are a JSON formatter and validator. Your task is to take the provided text and:

1. Extract ONLY the JSON array from the text (ignore any surrounding text, explanations, or markdown)
2. Fix any JSON formatting issues (missing commas, quotes, brackets, etc.)
3. Properly escape all special characters (backslashes, quotes, etc.) that could cause JSON parsing errors
4. Ensure all strings are properly quoted
5. Remove any invalid characters or syntax errors
6. Return ONLY the clean, valid JSON array - no additional text, explanations, or markdown formatting

Input text to clean:
{raw_json_text}

Return only the cleaned JSON array:
"""
            
            # Generate cleaned JSON using Gemini
            response = client.models.generate_content(
                model="gemini-2.5-pro-preview-05-06",
                contents=[cleaning_prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1  # Low temperature for consistent formatting
                )
            )
            
            cleaned_json = response.text.strip()
            
            # Additional cleanup - remove any markdown formatting that might remain
            if cleaned_json.startswith("```json"):
                cleaned_json = cleaned_json[7:]
            if cleaned_json.startswith("```"):
                cleaned_json = cleaned_json[3:]
            if cleaned_json.endswith("```"):
                cleaned_json = cleaned_json[:-3]
            
            cleaned_json = cleaned_json.strip()
            
            # Log the cleaning process
            print(f"ORIGINAL JSON LENGTH: {len(raw_json_text)}")
            print(f"CLEANED JSON LENGTH: {len(cleaned_json)}")
            
            # Save cleaned response for debugging

            
            return cleaned_json
            
        except Exception as e:
            print(f"Error cleaning JSON with Gemini: {str(e)}")
            # Fallback to original text if Gemini cleaning fails
            return raw_json_text
    
    def _parse_openai_response(
        self, 
        response_text: str, 
        subject: str, 
        topic: str, 
        subtopic: str, 
        current_timestamp: str
    ) -> List[Dict[str, Any]]:
        """Parse the OpenAI response and process the questions.
        
        Args:
            response_text: Raw response text from OpenAI
            subject: Subject for the questions
            topic: Topic for the questions
            subtopic: Subtopic for the questions
            current_timestamp: Current timestamp for questions
            
        Returns:
            List of processed question dictionaries
            
        Raises:
            Exception: If JSON parsing fails
        """
        try:
            # Clean up the response text
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            print(f"RAW RESPONSE: {response_text}")

            # Parse JSON response
            list_of_generated_json = json.loads(response_text)
            
            processed_questions = []
            
            if isinstance(list_of_generated_json, list):
                for item_json in list_of_generated_json:
                    # Common fields enforced for all types
                    item_json["subject"] = subject
                    item_json["topic"] = topic
                    item_json["subtopic"] = subtopic
                    item_json["questionset"] = "scanned"
                    item_json["created_at"] = current_timestamp
                    item_json["explaination"] = ""  # Always empty as per requirement
                    
                    question_type = str(item_json.get("question_type", "")).upper()
                    
                    # Handle text-based answer types
                    if question_type in ["VERY_SHORT_ANSWER", "SHORT_ANSWER", "LONG_ANSWER", "CASE_STUDY"]:
                        # Ensure these are treated as text-based questions
                        item_json["question_type"] = question_type
                        
                        # Ensure required fields exist for text-based questions
                        if "model_answer" not in item_json:
                            item_json["model_answer"] = ""
                        if "grading_criteria" not in item_json:
                            item_json["grading_criteria"] = ""
                        
                        # Remove MCQ specific fields if present
                        mcq_keys = ["option1", "option2", "option3", "option4", "correctanswer"]
                        for key in mcq_keys:
                            item_json.pop(key, None)
                            
                        # Set appropriate marks based on question type
                        if question_type == "VERY_SHORT_ANSWER" and item_json.get("marks", "") == "":
                            item_json["marks"] = "1"
                        elif question_type == "SHORT_ANSWER" and item_json.get("marks", "") == "":
                            item_json["marks"] = "3"
                        elif question_type == "LONG_ANSWER" and item_json.get("marks", "") == "":
                            item_json["marks"] = "5"
                        elif question_type == "CASE_STUDY" and item_json.get("marks", "") == "":
                            item_json["marks"] = "10"
                            
                    else:
                        # Handle MCQ questions
                        if question_type not in ["MULTIPLE_CHOICE", "SINGLE_SELECT_MCQ"]:
                            if any(opt_key in item_json for opt_key in ["option1", "option2", "option3", "option4"]):
                                item_json["question_type"] = "multiple_choice"
                            else:
                                # Default to multiple choice if type is unclear
                                item_json["question_type"] = "multiple_choice"
                        
                        # For MCQ questions, ensure correctanswer field exists
                        if "correctanswer" not in item_json:
                            item_json["correctanswer"] = ""
                        
                        # Remove text-based question specific fields if present
                        text_based_keys = ["model_answer", "grading_criteria"]
                        for key in text_based_keys:
                            item_json.pop(key, None)
                    
                    processed_questions.append(item_json)
                
            return processed_questions
            
        except json.JSONDecodeError as e:
            raise Exception(f"Error parsing OpenAI response: {str(e)}") 
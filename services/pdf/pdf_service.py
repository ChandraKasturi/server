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

from config import settings
from models.pdf_models import (
    ProcessingStatus, PDFDocument, PDFChunk, PDFUploadRequest
)
from repositories.pdf_repository import PDFRepository
from repositories.pgvector_repository import LangchainVectorRepository
from repositories.postgres_text_repository import PostgresTextRepository
from langchain_postgres.vectorstores import PGVector
from langchain.embeddings import OpenAIEmbeddings


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
            metadata={
                "subject": metadata.subject,
                "grade": metadata.grade
            }
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
    
    def get_user_pdfs(self, user_id: str) -> List[PDFDocument]:
        """Get all PDFs uploaded by a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of PDF documents
        """
        return self.pdf_repository.get_user_pdf_documents(user_id)
    
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
                 openai_api_key: Optional[str] = None,
                 max_workers: int = 5):
        """Initialize the PDF processing service.
        
        Args:
            pdf_repository: Repository for PDF storage operations
            redis_client: Redis client for queue management
            openai_api_key: API key for OpenAI. If None, uses the one from settings.
            max_workers: Maximum number of concurrent PDF processing workers
        """
        self.pdf_repository = pdf_repository or PDFRepository()
        self.redis_client = redis_client or redis.from_url(settings.REDIS_URL)
        self.api_key = openai_api_key or settings.OPENAI_API_KEY
        self.max_workers = max_workers
        self.worker_semaphore = asyncio.Semaphore(max_workers)
        self.postgres_text_repository = PostgresTextRepository()
        
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
                    await self.redis_client.hset(
                        f"pdf_processing_status:{pdf_document.id}",
                        mapping={
                            "status": "completed",
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                else:
                    # Mark as failed
                    self.pdf_repository.update_pdf_status(
                        pdf_document.id,
                        ProcessingStatus.FAILED,
                        "Error processing PDF"
                    )
                    
                    # Record failure in Redis
                    await self.redis_client.hset(
                        f"pdf_processing_status:{pdf_document.id}",
                        mapping={
                            "status": "failed",
                            "error": "Error processing PDF",
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
            
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
                await self.redis_client.hset(
                    f"pdf_processing_status:{pdf_document.id}",
                    mapping={
                        "status": "failed",
                        "error": error_message,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
    
    async def process_pdf(self, pdf_document: PDFDocument, user_id: str) -> bool:
        """Process a PDF document and store its contents.
        
        Args:
            pdf_document: PDF document to process
            user_id: ID of the user who owns the document
            
        Returns:
            True if processing was successful
        """
        pdf_id = pdf_document.id
        file_path = pdf_document.file_path
        
        try:
            # Update processing status
            self.pdf_repository.update_pdf_status(
                pdf_id,
                ProcessingStatus.PROCESSING
            )
            
            # Update Redis status
            await self.redis_client.hset(
                f"pdf_processing_status:{pdf_id}",
                mapping={
                    "status": "processing",
                    "start_time": datetime.utcnow().isoformat()
                }
            )
            
            # Extract text from PDF
            try:
                await self.redis_client.hset(
                    f"pdf_processing_status:{pdf_id}",
                    mapping={"step": "extracting_text"}
                )
                
                chunks, page_count = await self._extract_text(file_path)
                print(f"Chunks: {chunks}")
                print(f"Page count: {page_count}")
                # Store the full text in PostgreSQL for the specific user
                full_text = "\n\n".join([chunk["text"] for chunk in chunks])
                full_text = full_text.replace("\x00", "")
                # Store in Postgres text repository
                await self.redis_client.hset(
                    f"pdf_processing_status:{pdf_id}",
                    mapping={"step": "storing_text_in_postgres"}
                )
                
                # Store the full extracted text in the user's database
                self.postgres_text_repository.store_pdf_text(
                    student_id=user_id,
                    pdf_id=pdf_id,
                    title=pdf_document.title,
                    content=full_text,
                    page_count=page_count,
                    metadata=pdf_document.metadata
                )
                
                # Store chunks in the user's database
                self.postgres_text_repository.store_pdf_chunks(
                    student_id=user_id,
                    pdf_id=pdf_id,
                    chunks=[{
                        "chunk_index": i,
                        "page_number": chunk.get("page", 0),
                        "content": chunk["text"].replace("\x00", "")
                    } for i, chunk in enumerate(chunks)]
                )
                
                # Convert chunks to PDFChunk objects and store in MongoDB
                await self.redis_client.hset(
                    f"pdf_processing_status:{pdf_id}",
                    mapping={"step": "storing_chunks_in_mongodb"}
                )
                
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
                    self.pdf_repository.save_pdf_chunk(chunk_obj)
                    pdf_chunks.append(chunk_obj)
                
                # Store chunks in vector DB for semantic search
                await self.redis_client.hset(
                    f"pdf_processing_status:{pdf_id}",
                    mapping={"step": "storing_in_vector_db"}
                )
                
                await self._store_chunks_in_vector_db(pdf_id, pdf_chunks, user_id)
                
                # Update processing status to completed
                self.pdf_repository.update_pdf_status(
                    pdf_id,
                    ProcessingStatus.COMPLETED
                )
                
                # Update Redis status
                await self.redis_client.hset(
                    f"pdf_processing_status:{pdf_id}",
                    mapping={
                        "status": "completed",
                        "end_time": datetime.utcnow().isoformat(),
                        "step": "completed"
                    }
                )
                
                return True
                
            except Exception as e:
                error_message = f"Error processing PDF: {str(e)}"
                print(error_message)
                
                # Update processing status to failed
                self.pdf_repository.update_pdf_status(
                    pdf_id,
                    ProcessingStatus.FAILED,
                    error=error_message
                )
                
                # Update Redis status
                await self.redis_client.hset(
                    f"pdf_processing_status:{pdf_id}",
                    mapping={
                        "status": "failed",
                        "error": error_message,
                        "end_time": datetime.utcnow().isoformat()
                    }
                )
                
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
            
            # Update processing status to failed
            self.pdf_repository.update_pdf_status(
                pdf_id,
                ProcessingStatus.FAILED,
                error=error_message
            )
            
            # Update Redis status
            await self.redis_client.hset(
                f"pdf_processing_status:{pdf_id}",
                mapping={
                    "status": "failed",
                    "error": error_message,
                    "end_time": datetime.utcnow().isoformat()
                }
            )
            
            return False
    
    async def _extract_text(self, file_path: str) -> Tuple[List[Dict], int]:
        """Extract text from a PDF file.
        
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
                            'total_pages': num_pages
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
            doc = Document(
                page_content=chunk.content,
                metadata={
                    'pdf_id': pdf_id,
                    'chunk_id': chunk.id,
                    'page': chunk.page_number,
                    'user_id': user_id,  # Include user_id in metadata
                    **chunk.metadata
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
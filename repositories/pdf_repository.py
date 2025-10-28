import pymongo
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from bson.objectid import ObjectId

from config import settings
from models.pdf_models import (
    ProcessingStatus, PDFDocument, PDFChunk, 
    GeneratedQuestion, QuestionOption, QuestionAnswer,
    LearningSession, LearningInteraction
)
from repositories.mongo_repository import MongoRepository
from repositories.decorators import mongo_retry


class PDFRepository(MongoRepository):
    """Repository for PDF document operations."""
    
    def __init__(self, mongo_uri: str = None):
        """Initialize repository with MongoDB connection."""
        super().__init__(mongo_uri)
        
        # Create collections for PDF documents
        self.pdf_documents = self.get_collection("pdf_storage", "documents")
        self.pdf_chunks = self.get_collection("pdf_storage", "chunks")
        self.processing_queue = self.get_collection("pdf_storage", "processing_queue")
        
        # Ensure indexes
        self.pdf_documents.create_index([("user_id", pymongo.ASCENDING)])
        self.pdf_documents.create_index([("processing_status", pymongo.ASCENDING)])
        self.pdf_chunks.create_index([("pdf_id", pymongo.ASCENDING)])
        self.pdf_chunks.create_index([("page_number", pymongo.ASCENDING)])
        self.processing_queue.create_index([("status", pymongo.ASCENDING)])
        self.processing_queue.create_index([("priority", pymongo.DESCENDING)])
        
    @mongo_retry(max_retries=3, delay=1)
    def save_pdf_document(self, pdf_document: PDFDocument) -> str:
        """Save a PDF document to the database.
        
        Args:
            pdf_document: PDF document data
            
        Returns:
            ID of the created document
        """
        pdf_dict = pdf_document.dict()
        result = self.pdf_documents.insert_one(pdf_dict)
        return str(result.inserted_id)
    
    def get_pdf_document(self, pdf_id: str) -> Optional[PDFDocument]:
        """Get a PDF document by ID.
        
        Args:
            pdf_id: ID of the PDF document
            
        Returns:
            PDF document data if found, None otherwise
        """
        doc = self.pdf_documents.find_one({"id": pdf_id})
        if doc:
            return PDFDocument(**doc)
        return None
    
    def get_user_pdf_documents(self, user_id: str, subject: Optional[str] = None) -> List[PDFDocument]:
        """Get all PDF documents for a user.
        
        Args:
            user_id: ID of the user
            subject: Optional subject to filter PDFs by    
            
        Returns:
            List of PDF documents
        """
        query = {"user_id": user_id}
        
        # Add subject filter if provided
        if subject:
            query["metadata.subject"] = subject
            
        docs = list(self.pdf_documents.find(query))
        return [PDFDocument(**doc) for doc in docs]
    
    @mongo_retry(max_retries=3, delay=1)
    def update_pdf_status(self, pdf_id: str, status: ProcessingStatus, 
                          error: Optional[str] = None) -> bool:
        """Update the processing status of a PDF document.
        
        Args:
            pdf_id: ID of the PDF document
            status: New processing status
            error: Error message if any
            
        Returns:
            True if update was successful, False otherwise
        """
        update_data = {"processing_status": status}
        
        if status == ProcessingStatus.PROCESSING:
            update_data["process_start_time"] = datetime.utcnow()
        elif status == ProcessingStatus.COMPLETED or status == ProcessingStatus.FAILED:
            update_data["process_end_time"] = datetime.utcnow()
            
        if error and status == ProcessingStatus.FAILED:
            update_data["processing_error"] = error
            
        result = self.pdf_documents.update_one(
            {"id": pdf_id},
            {"$set": update_data}
        )
        return result.modified_count > 0
    
    def add_to_processing_queue(self, pdf_id: str, priority: int = 1) -> str:
        """Add a PDF document to the processing queue.
        
        Args:
            pdf_id: ID of the PDF document
            priority: Priority level (higher numbers = higher priority)
            
        Returns:
            ID of the queue entry
        """
        queue_id = str(uuid.uuid4())
        entry = {
            "id": queue_id,
            "pdf_id": pdf_id,
            "queue_time": datetime.utcnow(),
            "priority": priority,
            "attempts": 0,
            "max_attempts": 3,
            "status": "queued",
            "worker_id": None,
            "error_message": None,
            "next_retry": None
        }
        self.processing_queue.insert_one(entry)
        return queue_id
    
    def get_next_queued_pdf(self) -> Optional[Dict]:
        """Get the next PDF document from the processing queue.
        
        Returns:
            Queue entry for the next PDF to process
        """
        # Find and update in one atomic operation
        return self.processing_queue.find_one_and_update(
            {"status": "queued", "attempts": {"$lt": 3}},
            {"$set": {"status": "processing", "worker_id": str(uuid.uuid4())},
             "$inc": {"attempts": 1}},
            sort=[("priority", pymongo.DESCENDING), ("queue_time", pymongo.ASCENDING)],
            return_document=pymongo.ReturnDocument.AFTER
        )
    
    def mark_queue_entry_complete(self, queue_id: str) -> bool:
        """Mark a processing queue entry as completed.
        
        Args:
            queue_id: ID of the queue entry
            
        Returns:
            True if update was successful, False otherwise
        """
        result = self.processing_queue.update_one(
            {"id": queue_id},
            {"$set": {"status": "completed"}}
        )
        return result.modified_count > 0
    
    def mark_queue_entry_failed(self, queue_id: str, error_message: str) -> bool:
        """Mark a processing queue entry as failed.
        
        Args:
            queue_id: ID of the queue entry
            error_message: Error message
            
        Returns:
            True if update was successful, False otherwise
        """
        result = self.processing_queue.update_one(
            {"id": queue_id},
            {"$set": {
                "status": "failed",
                "error_message": error_message,
                "next_retry": datetime.utcnow()
            }}
        )
        return result.modified_count > 0
    
    @mongo_retry(max_retries=3, delay=1)
    def save_pdf_chunk(self, chunk: PDFChunk) -> str:
        """Save a PDF chunk to the database.
        
        Args:
            chunk: PDF chunk data
            
        Returns:
            ID of the created chunk
        """
        chunk_dict = chunk.dict()
        # Don't store embedding vectors in MongoDB - these should go in PGVector
        if 'embedding' in chunk_dict:
            del chunk_dict['embedding']
            
        result = self.pdf_chunks.insert_one(chunk_dict)
        return str(result.inserted_id)
    
    def get_pdf_chunks(self, pdf_id: str) -> List[PDFChunk]:
        """Get all chunks for a PDF document.
        
        Args:
            pdf_id: ID of the PDF document
            
        Returns:
            List of PDF chunks
        """
        chunks = list(self.pdf_chunks.find({"pdf_id": pdf_id}).sort("chunk_index", pymongo.ASCENDING))
        return [PDFChunk(**chunk) for chunk in chunks]
    
    def delete_pdf_document(self, pdf_id: str) -> Tuple[bool, bool]:
        """Delete a PDF document and its chunks.
        
        Args:
            pdf_id: ID of the PDF document
            
        Returns:
            Tuple of (document_deleted, chunks_deleted)
        """
        doc_result = self.pdf_documents.delete_one({"id": pdf_id})
        chunk_result = self.pdf_chunks.delete_many({"pdf_id": pdf_id})
        return doc_result.deleted_count > 0, chunk_result.deleted_count > 0
    
    @mongo_retry(max_retries=3, delay=1)
    def update_pdf_document(self, pdf_id: str, update_data: Dict[str, Any]) -> bool:
        """Update a PDF document with the provided data.
        
        Args:
            pdf_id: ID of the PDF document
            update_data: Dictionary of fields to update
            
        Returns:
            True if update was successful
        """
        result = self.pdf_documents.update_one(
            {"id": pdf_id},
            {"$set": update_data}
        )
        return result.modified_count > 0


class QuestionRepository(MongoRepository):
    """Repository for PDF-generated questions."""
    
    def __init__(self, mongo_uri: str = None):
        """Initialize repository with MongoDB connection."""
        super().__init__(mongo_uri)
        
        # Create collections for questions
        self.questions = self.get_collection("pdf_storage", "questions")
        self.options = self.get_collection("pdf_storage", "question_options")
        self.answers = self.get_collection("pdf_storage", "question_answers")
        
        # Ensure indexes
        self.questions.create_index([("pdf_id", pymongo.ASCENDING)])
        self.questions.create_index([("question_type", pymongo.ASCENDING)])
        self.options.create_index([("question_id", pymongo.ASCENDING)])
        self.answers.create_index([("question_id", pymongo.ASCENDING)])
    
    def save_question(self, question: GeneratedQuestion) -> str:
        """Save a generated question to the database.
        
        Args:
            question: Question data
            
        Returns:
            ID of the created question
        """
        question_dict = question.dict()
        # Don't store embedding vectors in MongoDB - these should go in PGVector
        if 'embedding' in question_dict:
            del question_dict['embedding']
            
        result = self.questions.insert_one(question_dict)
        return str(result.inserted_id)
    
    def save_question_option(self, option: QuestionOption) -> str:
        """Save a question option to the database.
        
        Args:
            option: Option data
            
        Returns:
            ID of the created option
        """
        option_dict = option.dict()
        result = self.options.insert_one(option_dict)
        return str(result.inserted_id)
    
    def save_question_answer(self, answer: QuestionAnswer) -> str:
        """Save a question answer to the database.
        
        Args:
            answer: Answer data
            
        Returns:
            ID of the created answer
        """
        answer_dict = answer.dict()
        result = self.answers.insert_one(answer_dict)
        return str(result.inserted_id)
    
    def get_pdf_questions(self, pdf_id: str, question_type: Optional[str] = None) -> List[Dict]:
        """Get all questions for a PDF document.
        
        Args:
            pdf_id: ID of the PDF document
            question_type: Type of questions to retrieve (optional filter)
            
        Returns:
            List of questions with options and answers
        """
        query = {"pdf_id": pdf_id}
        if question_type:
            query["question_type"] = question_type
            
        questions = list(self.questions.find(query))
        
        # For each question, include options and answers
        enriched_questions = []
        for q in questions:
            q_id = q["id"]
            
            # Get options
            options = list(self.options.find({"question_id": q_id}).sort("option_order", pymongo.ASCENDING))
            
            # Get answer
            answer = self.answers.find_one({"question_id": q_id})
            
            # Add to enriched question
            q["options"] = options
            if answer:
                q["answer"] = answer
                
            enriched_questions.append(q)
            
        return enriched_questions
    
    def get_question_by_id(self, question_id: str) -> Optional[Dict]:
        """Get a question by ID with its options and answer.
        
        Args:
            question_id: ID of the question
            
        Returns:
            Question data with options and answer if found
        """
        question = self.questions.find_one({"id": question_id})
        if not question:
            return None
            
        # Get options
        options = list(self.options.find({"question_id": question_id}).sort("option_order", pymongo.ASCENDING))
        
        # Get answer
        answer = self.answers.find_one({"question_id": question_id})
        
        # Add to enriched question
        question["options"] = options
        if answer:
            question["answer"] = answer
            
        return question


class LearningRepository(MongoRepository):
    """Repository for learning sessions and interactions."""
    
    def __init__(self, mongo_uri: str = None):
        """Initialize repository with MongoDB connection."""
        super().__init__(mongo_uri)
        
        # Create collections for learning
        self.sessions = self.get_collection("pdf_storage", "learning_sessions")
        self.interactions = self.get_collection("pdf_storage", "learning_interactions")
        
        # Ensure indexes
        self.sessions.create_index([("user_id", pymongo.ASCENDING)])
        self.sessions.create_index([("pdf_id", pymongo.ASCENDING)])
        self.interactions.create_index([("session_id", pymongo.ASCENDING)])
    
    def create_session(self, session: LearningSession) -> str:
        """Create a new learning session.
        
        Args:
            session: Session data
            
        Returns:
            ID of the created session
        """
        session_dict = session.dict()
        result = self.sessions.insert_one(session_dict)
        return str(result.inserted_id)
    
    def get_session(self, session_id: str) -> Optional[LearningSession]:
        """Get a learning session by ID.
        
        Args:
            session_id: ID of the session
            
        Returns:
            Session data if found
        """
        session = self.sessions.find_one({"id": session_id})
        if session:
            return LearningSession(**session)
        return None
    
    def get_user_sessions(self, user_id: str, pdf_id: Optional[str] = None) -> List[LearningSession]:
        """Get learning sessions for a user, optionally filtered by PDF.
        
        Args:
            user_id: ID of the user
            pdf_id: ID of the PDF document (optional filter)
            
        Returns:
            List of learning sessions
        """
        query = {"user_id": user_id}
        if pdf_id:
            query["pdf_id"] = pdf_id
            
        sessions = list(self.sessions.find(query).sort("start_time", pymongo.DESCENDING))
        return [LearningSession(**session) for session in sessions]
    
    def add_interaction(self, interaction: LearningInteraction) -> str:
        """Add a learning interaction to a session.
        
        Args:
            interaction: Interaction data
            
        Returns:
            ID of the created interaction
        """
        interaction_dict = interaction.dict()
        # Don't store embedding vectors in MongoDB - these should go in PGVector
        if 'query_embedding' in interaction_dict:
            del interaction_dict['query_embedding']
            
        result = self.interactions.insert_one(interaction_dict)
        
        # Update session with question count
        self.sessions.update_one(
            {"id": interaction.session_id},
            {"$inc": {"total_questions": 1}}
        )
        
        return str(result.inserted_id)
    
    def get_session_interactions(self, session_id: str) -> List[LearningInteraction]:
        """Get all interactions for a learning session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            List of learning interactions
        """
        interactions = list(
            self.interactions.find({"session_id": session_id})
            .sort("interaction_time", pymongo.ASCENDING)
        )
        return [LearningInteraction(**interaction) for interaction in interactions]
    
    def end_session(self, session_id: str, summary: Optional[str] = None) -> bool:
        """End a learning session.
        
        Args:
            session_id: ID of the session
            summary: Optional summary of the session
            
        Returns:
            True if update was successful
        """
        update_data = {"end_time": datetime.utcnow()}
        if summary:
            update_data["session_summary"] = summary
            
        result = self.sessions.update_one(
            {"id": session_id},
            {"$set": update_data}
        )
        return result.modified_count > 0 
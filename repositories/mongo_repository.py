import pymongo
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from bson.objectid import ObjectId

from config import settings

class MongoRepository:
    """Base repository for MongoDB operations."""
    
    def __init__(self, mongo_uri: str = None):
        """Initialize MongoDB client.
        
        Args:
            mongo_uri: MongoDB connection string. If None, uses the one from settings.
        """
        self.mongo_uri = mongo_uri or settings.MONGO_URI
        self.client = pymongo.MongoClient(self.mongo_uri)
        
    def get_db(self, db_name: str):
        """Get a database by name."""
        return self.client[db_name]
        
    def get_collection(self, db_name: str, collection_name: str):
        """Get a collection from a database."""
        return self.client[db_name][collection_name]
    
    def close(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()

class UserRepository(MongoRepository):
    """Repository for user-related operations."""
    
    def __init__(self, mongo_uri: str = None):
        super().__init__(mongo_uri)
        self.users_collection = self.get_collection(
            settings.MONGO_DATABASE_USERS, 
            settings.MONGO_COLLECTION_USERS
        )
    
    def find_by_email_or_mobile(self, email_or_mobile: str) -> Optional[Dict]:
        """Find a user by email or mobile number."""
        if not email_or_mobile:
            return None
            
        # Check if it's an email (lowercase) or keep as is for mobile
        query_value = email_or_mobile.lower() if '@' in email_or_mobile else email_or_mobile
        
        return self.users_collection.find_one({
            "$or": [
                {"email": {"$eq": query_value}},
                {"mobilenumber": {"$eq": email_or_mobile}}
            ]
        })
    
    def find_by_credentials(self, email_or_mobile: str, password: str) -> Optional[Dict]:
        """Find a user by email/mobile and password."""
        if not email_or_mobile or not password:
            return None
            
        # Check if it's an email (lowercase) or keep as is for mobile
        query_value = email_or_mobile.lower() if '@' in email_or_mobile else email_or_mobile
        
        return self.users_collection.find_one({
            "$and": [
                {"$or": [
                    {"email": {"$eq": query_value}},
                    {"mobilenumber": {"$eq": email_or_mobile}}
                ]},
                {"password": {"$eq": password}}
            ]
        })
    
    def insert_user(self, user_data: Dict) -> str:
        """Insert a new user and return the user_id."""
        result = self.users_collection.insert_one(user_data)
        return str(result.inserted_id)
    
    def update_user(self, user_id: str, update_data: Dict) -> bool:
        """Update user data by user_id."""
        result = self.users_collection.update_one(
            {"student_id": user_id},
            {"$set": update_data}
        )
        return result.modified_count > 0
    
    def update_password(self, email_or_mobile: str, new_password: str) -> bool:
        """Update a user's password by email or mobile."""
        # Check if it's an email (lowercase) or keep as is for mobile
        query_value = email_or_mobile.lower() if '@' in email_or_mobile else email_or_mobile
        
        result = self.users_collection.update_one(
            {"$or": [
                {"email": {"$eq": query_value}},
                {"mobilenumber": {"$eq": email_or_mobile}}
            ]},
            {"$set": {"password": new_password}}
        )
        return result.modified_count > 0

class TokenRepository(MongoRepository):
    """Repository for token-related operations."""
    
    def __init__(self, mongo_uri: str = None):
        super().__init__(mongo_uri)
        self.auth_tokens_collection = self.get_collection(
            settings.MONGO_DATABASE_TOKENS, 
            settings.MONGO_COLLECTION_AUTH_TOKENS
        )
        self.register_tokens_collection = self.get_collection(
            settings.MONGO_DATABASE_TOKENS, 
            settings.MONGO_COLLECTION_REGISTER_TOKENS
        )
        self.password_tokens_collection = self.get_collection(
            settings.MONGO_DATABASE_TOKENS, 
            settings.MONGO_COLLECTION_PASSWORD_TOKENS
        )
        
        # Ensure indexes for token expiration
        self.auth_tokens_collection.create_index(
            "ExpiresAt", 
            expireAfterSeconds=settings.AUTH_TOKEN_EXPIRE_SECONDS
        )
        self.register_tokens_collection.create_index(
            "ExpiresAt", 
            expireAfterSeconds=settings.REGISTER_TOKEN_EXPIRE_SECONDS
        )
        self.password_tokens_collection.create_index(
            "ExpiresAt", 
            expireAfterSeconds=settings.PASSWORD_TOKEN_EXPIRE_SECONDS
        )
    
    def store_auth_token(self, student_id: str, token: str) -> bool:
        """Store an authentication token."""
        result = self.auth_tokens_collection.insert_one({
            "student_id": student_id,
            "token": token,
            "ExpiresAt": datetime.utcnow()
        })
        return result.acknowledged
    
    def get_auth_token(self, student_id: str) -> Optional[str]:
        """Get authentication token for a student."""
        token_doc = self.auth_tokens_collection.find_one({"student_id": student_id})
        return token_doc.get("token") if token_doc else None
    
    def delete_auth_token(self, student_id: str) -> bool:
        """Delete an authentication token."""
        result = self.auth_tokens_collection.delete_one({"student_id": student_id})
        return result.deleted_count > 0
    
    def store_register_token(self, register_data: Dict) -> bool:
        """Store a registration token with associated data."""
        register_data["ExpiresAt"] = datetime.utcnow()
        result = self.register_tokens_collection.insert_one(register_data)
        return result.acknowledged
    
    def get_register_token_data(self, token: str) -> Optional[Dict]:
        """Get registration data by token."""
        return self.register_tokens_collection.find_one({"token": token})
    
    def store_password_token(self, email_or_mobile: str, token: str) -> bool:
        """Store a password reset token."""
        result = self.password_tokens_collection.insert_one({
            "email": email_or_mobile,
            "token": token,
            "ExpiresAt": datetime.utcnow()
        })
        return result.acknowledged
    
    def get_password_token_data(self, token: str) -> Optional[Dict]:
        """Get password reset data by token."""
        return self.password_tokens_collection.find_one({"token": token})

class QuestionRepository(MongoRepository):
    """Repository for question bank operations."""
    
    def __init__(self, mongo_uri: str = None):
        super().__init__(mongo_uri)
        self.questions_collection = self.get_collection(
            settings.MONGO_DATABASE_QUESTIONS, 
            settings.MONGO_COLLECTION_QUESTION_BANK
        )
        self.topic_subtopic_collection = self.get_collection(
            settings.MONGO_DATABASE_SUBJECTDATA, 
            settings.MONGO_COLLECTION_TOPIC_SUBTOPIC
        )
    
    def insert_question(self, question_data: Dict) -> bool:
        """Insert a new question."""
        result = self.questions_collection.insert_one(question_data)
        return result.acknowledged
    
    def find_questions(self, query: Dict, limit: int = 0) -> List[Dict]:
        """Find questions based on query criteria."""
        return list(self.questions_collection.find(query).limit(limit))
    
    def get_all_topics_subtopics(self) -> List[Dict]:
        """Get all subject topics and subtopics."""
        return list(self.topic_subtopic_collection.find({}, {"_id": 0}))

class FeedbackRepository(MongoRepository):
    """Repository for user feedback operations."""
    
    def add_feedback(self, student_id: str, feedback_text: str) -> bool:
        """Add feedback for a student."""
        collection = self.get_collection(student_id, "feedback")
        result = collection.insert_one({
            "Date": datetime.utcnow(),
            "FeedBack": feedback_text
        })
        return result.acknowledged

class HistoryRepository(MongoRepository):
    """Repository for user history and assessment operations."""
    
    def get_history(self, student_id: str, from_date: datetime = None) -> List[Dict]:
        """Get user history from a specific date."""
        collection = self.get_collection(student_id, "sahasra_history")
        query = {"time": {"$gte": from_date}} if from_date else {}
        return list(collection.find(query))
    
    def get_assessments(self, student_id: str, from_date: datetime = None) -> List[Dict]:
        """Get user assessments from a specific date."""
        collection = self.get_collection(student_id, "sahasra_assessments")
        query = {"created_at": {"$gte": from_date}} if from_date else {}
        return list(collection.find(query))
    
    def get_assessment_by_id(self, student_id: str, assessment_id: str) -> Optional[Dict]:
        """Get a specific assessment by ID."""
        collection = self.get_collection(student_id, "sahasra_assessments")
        try:
            # First try to find by ObjectId (legacy assessments)
            try:
                assessment_obj_id = ObjectId(assessment_id)
                assessment = collection.find_one({"_id": assessment_obj_id})
                if assessment:
                    return assessment
            except:
                # Invalid ObjectId format or not found
                pass
            
            # Try to find by string ID (new assessments)
            assessment = collection.find_one({"id": assessment_id})
            return assessment
        except Exception as e:
            print(f"Error getting assessment: {str(e)}")
            return None
    
    def get_pdf_assessments(self, student_id: str, pdf_id: str) -> List[Dict]:
        """Get all assessments for a specific PDF.
        
        Args:
            student_id: ID of the student
            pdf_id: ID of the PDF document
            
        Returns:
            List of assessments for the PDF
        """
        collection = self.get_collection(student_id, "sahasra_assessments")
        return list(collection.find({"pdf_id": pdf_id}))
    
    def add_assessment(self, student_id: str, assessment_data: Dict) -> str:
        """Add a new assessment and return its ID.
        
        Args:
            student_id: ID of the student
            assessment_data: Assessment data to store
            
        Returns:
            ID of the created assessment
        """
        # Ensure created_at exists
        if "created_at" not in assessment_data:
            assessment_data["created_at"] = datetime.utcnow()
            
        # For backward compatibility, also include date field
        assessment_data["date"] = assessment_data.get("created_at")
            
        collection = self.get_collection(student_id, "sahasra_assessments")
        result = collection.insert_one(assessment_data)
        
        # If the assessment has an 'id' field, return it, otherwise use ObjectId
        return assessment_data.get("id") or str(result.inserted_id)
    
    def update_assessment(self, student_id: str, assessment_id: str, 
                         update_data: Dict) -> bool:
        """Update an assessment.
        
        Args:
            student_id: ID of the student
            assessment_id: ID of the assessment
            update_data: Data to update
            
        Returns:
            True if update was successful
        """
        collection = self.get_collection(student_id, "sahasra_assessments")
        
        # Try both ObjectId and string ID
        try:
            # First try by ObjectId
            try:
                assessment_obj_id = ObjectId(assessment_id)
                result = collection.update_one(
                    {"_id": assessment_obj_id}, 
                    {"$set": update_data}
                )
                if result.modified_count > 0:
                    return True
            except:
                # Invalid ObjectId format or not updated
                pass
            
            # Try by string ID
            result = collection.update_one(
                {"id": assessment_id}, 
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating assessment: {str(e)}")
            return False
    
    def add_history_item(self, student_id: str, history_data: Dict) -> str:
        """Add a history item and return its ID."""
        collection = self.get_collection(student_id, "sahasra_history")
        # Ensure time field exists
        if "time" not in history_data:
            history_data["time"] = datetime.utcnow()
        result = collection.insert_one(history_data)
        return str(result.inserted_id) 
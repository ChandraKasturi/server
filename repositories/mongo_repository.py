import pymongo
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Any, Union, Tuple
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
        self.mobile_verification_tokens_collection = self.get_collection(
            settings.MONGO_DATABASE_TOKENS, 
            settings.MONGO_COLLECTION_MOBILE_VERIFICATION_TOKENS
        )
        self.email_verification_tokens_collection = self.get_collection(
            settings.MONGO_DATABASE_TOKENS, 
            settings.MONGO_COLLECTION_EMAIL_VERIFICATION_TOKENS
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
        self.mobile_verification_tokens_collection.create_index(
            "ExpiresAt", 
            expireAfterSeconds=settings.MOBILE_VERIFICATION_TOKEN_EXPIRE_SECONDS
        )
        self.email_verification_tokens_collection.create_index(
            "ExpiresAt", 
            expireAfterSeconds=settings.EMAIL_VERIFICATION_TOKEN_EXPIRE_SECONDS
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
    
    def store_mobile_verification_token(self, student_id: str, old_mobile: str, new_mobile: str, token: str) -> bool:
        """Store a mobile verification token.
        
        Args:
            student_id: ID of the student requesting mobile change
            old_mobile: Current mobile number
            new_mobile: New mobile number to be verified
            token: OTP token
            
        Returns:
            True if successful, False otherwise
        """
        # Remove any existing mobile verification token for this student
        self.mobile_verification_tokens_collection.delete_many({"student_id": student_id})
        
        result = self.mobile_verification_tokens_collection.insert_one({
            "student_id": student_id,
            "old_mobile": old_mobile,
            "new_mobile": new_mobile,
            "token": token,
            "created_at": datetime.utcnow(),
            "ExpiresAt": datetime.utcnow()
        })
        return result.acknowledged
    
    def get_mobile_verification_token_data(self, student_id: str, token: str) -> Optional[Dict]:
        """Get mobile verification data by student ID and token.
        
        Args:
            student_id: ID of the student
            token: OTP token to verify
            
        Returns:
            Token data if found and valid, None otherwise
        """
        return self.mobile_verification_tokens_collection.find_one({
            "student_id": student_id,
            "token": token
        })
    
    def delete_mobile_verification_token(self, student_id: str) -> bool:
        """Delete mobile verification token for a student.
        
        Args:
            student_id: ID of the student
            
        Returns:
            True if deleted, False otherwise
        """
        result = self.mobile_verification_tokens_collection.delete_many({"student_id": student_id})
        return result.deleted_count > 0
    
    def has_pending_mobile_verification(self, student_id: str) -> bool:
        """Check if student has pending mobile verification.
        
        Args:
            student_id: ID of the student
            
        Returns:
            True if pending verification exists, False otherwise
        """
        return self.mobile_verification_tokens_collection.find_one({"student_id": student_id}) is not None
    
    def store_email_verification_token(self, student_id: str, old_email: str, new_email: str, token: str) -> bool:
        """Store an email verification token.
        
        Args:
            student_id: ID of the student requesting email change
            old_email: Current email address
            new_email: New email address to be verified
            token: OTP token
            
        Returns:
            True if successful, False otherwise
        """
        # Remove any existing email verification token for this student
        self.email_verification_tokens_collection.delete_many({"student_id": student_id})
        
        result = self.email_verification_tokens_collection.insert_one({
            "student_id": student_id,
            "old_email": old_email,
            "new_email": new_email,
            "token": token,
            "created_at": datetime.utcnow(),
            "ExpiresAt": datetime.utcnow()
        })
        return result.acknowledged
    
    def get_email_verification_token_data(self, student_id: str, token: str) -> Optional[Dict]:
        """Get email verification data by student ID and token.
        
        Args:
            student_id: ID of the student
            token: OTP token to verify
            
        Returns:
            Token data if found and valid, None otherwise
        """
        return self.email_verification_tokens_collection.find_one({
            "student_id": student_id,
            "token": token
        })
    
    def delete_email_verification_token(self, student_id: str) -> bool:
        """Delete email verification token for a student.
        
        Args:
            student_id: ID of the student
            
        Returns:
            True if deleted, False otherwise
        """
        result = self.email_verification_tokens_collection.delete_many({"student_id": student_id})
        return result.deleted_count > 0
    
    def has_pending_email_verification(self, student_id: str) -> bool:
        """Check if student has pending email verification.
        
        Args:
            student_id: ID of the student
            
        Returns:
            True if pending verification exists, False otherwise
        """
        return self.email_verification_tokens_collection.find_one({"student_id": student_id}) is not None

class QuestionRepository(MongoRepository):
    """Repository for question bank operations."""
    
    def __init__(self, mongo_uri: str = None):
        super().__init__(mongo_uri)
        # Keep the old collection for backward compatibility
        self.questions_collection = self.get_collection(
            settings.MONGO_DATABASE_QUESTIONS, 
            settings.MONGO_COLLECTION_QUESTION_BANK
        )
        self.topic_subtopic_collection = self.get_collection(
            settings.MONGO_DATABASE_SUBJECTDATA, 
            settings.MONGO_COLLECTION_TOPIC_SUBTOPIC
        )
    
    def _get_subject_topic_collection(self, subject: str, topic: str):
        """Get collection for specific subject and topic using pattern x_{subject_name}_{topic_name}."""
        # Normalize subject and topic names (lowercase, replace spaces/hyphens with underscores)
        normalized_subject = subject.lower().replace(" ", "_").replace("-", "_")
        normalized_topic = topic.lower().replace(" ", "_").replace("-", "_")
        print(f"Normalized subject: {normalized_subject}")
        print(f"Normalized topic: {normalized_topic}")
        collection_name = f"x_{normalized_subject}_{normalized_topic}" if not normalized_subject.startswith("x_") else f"{normalized_subject}_{normalized_topic}"
        return self.get_collection(settings.MONGO_DATABASE_QUESTIONS, collection_name)
    
    def insert_question(self, question_data: Dict) -> bool:
        """Insert a new question."""
        # If subject and topic are available, use the new collection pattern
        if "subject" in question_data and "topic" in question_data:
            collection = self._get_subject_topic_collection(
                question_data["subject"], 
                question_data["topic"]
            )
        else:
            # Fall back to the default collection for backward compatibility
            collection = self.questions_collection
            
        result = collection.insert_one(question_data)
        return result.acknowledged
    
    def find_questions(self, query: Dict, limit: int = 0, subject: str = None, topic: str = None) -> List[Dict]:
        """Find questions based on query criteria."""
        # If subject and topic are provided, use the new collection pattern
        if subject and topic:
            collection = self._get_subject_topic_collection(subject, topic)
            # Remove subject and topic from query since they're implicit in the collection
            filtered_query = query
        else:
            # Use the default collection and include subject/topic in query if present
            collection = self.questions_collection
            filtered_query = query
            
        return list(collection.find(filtered_query).limit(limit))
    
    def get_all_topics_subtopics(self) -> List[Dict]:
        """Get all subject topics and subtopics."""
        return list(self.topic_subtopic_collection.find({}, {"_id": 0}))
    
    def get_questions_by_topic_subject(self, subject: str, topic: str) -> List[Dict]:
        """Get all questions filtered by subject and topic.
        
        Args:
            subject: Subject to filter by
            topic: Topic to filter by
            
        Returns:
            List of all question dictionaries matching the criteria
        """
        # Use the new collection pattern for better performance
        if subject and topic:
            collection = self._get_subject_topic_collection(subject, topic)
            # No need to filter by subject/topic since they're implicit in the collection
            query = {}
            if subject:
                query["subject"] = {"$regex": f"^{subject}$", "$options": "i"}  # Case-insensitive exact match
            if topic:
                query["topic"] = {"$regex": f"^{topic}$", "$options": "i"}  # Case-insensitive exact match
        else:
            # Fall back to the old method for backward compatibility
            collection = self.questions_collection
            query = {}
            
            # Add subject filter if provided
            if subject:
                query["subject"] = {"$regex": f"^{subject}$", "$options": "i"}  # Case-insensitive exact match
                
            # Add topic filter if provided
            if topic:
                query["topic"] = {"$regex": f"^{topic}$", "$options": "i"}  # Case-insensitive exact match
        
        # Convert ObjectId to string for easier handling in API responses
        print(f"Query: {query}")
        print(f"Collection: {collection.name}")
        print(f"Database: {collection.database.name}")
        questions = list(collection.find(query))
        for question in questions:
            if "_id" in question:
                question["_id"] = str(question["_id"])
                
        return questions
    
    def update_question_document(self, question_data: Dict) -> Tuple[bool, str, Optional[datetime]]:
        """Update an entire document in the question_bank collection.
        
        Args:
            question_data: Complete question document with _id field
            
        Returns:
            Tuple of (success: bool, message: str, updated_at: datetime)
        """
        try:
            print(f"Question data: {question_data}")
            # Extract the _id from the document
            if "_id" not in question_data:
                return False, "Document must contain an '_id' field", None
            
            document_id = question_data["_id"]
            
            # Convert string ObjectId to ObjectId if needed
            if isinstance(document_id, str):
                try:
                    document_id = ObjectId(document_id)
                except Exception as e:
                    return False, f"Invalid ObjectId format: {str(e)}", None
            
            # Remove _id from the update data (we'll use it as filter)
            update_data = {k: v for k, v in question_data.items() if k != "_id"}
            
            # Add updated_at timestamp in Asia/Kolkata timezone
            kolkata_tz = ZoneInfo('Asia/Kolkata')
            updated_at = datetime.now(kolkata_tz)
            update_data["updated_at"] = updated_at
            
            # Determine which collection to use
            if "subject" in question_data and "topic" in question_data:
                collection = self._get_subject_topic_collection(
                    question_data["subject"], 
                    question_data["topic"]
                )
            else:
                # Fall back to the default collection
                collection = self.questions_collection
            
            # Perform the update
            result = collection.update_one(
                {"_id": document_id},
                {"$set": update_data},
                upsert=False  # Don't create if doesn't exist
            )
            
            if result.matched_count == 0:
                return False, f"No document found with _id: {document_id}", None
            elif result.modified_count == 0:
                return True, "Document found but no changes were needed", updated_at
            else:
                return True, f"Document updated successfully. Modified count: {result.modified_count}", updated_at
                
        except Exception as e:
            return False, f"Error updating document: {str(e)}", None

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
    
    def get_history(self, student_id: str, from_date: datetime = None, page: int = 1, page_size: int = 10, oldest_first: bool = False, subject: str = None) -> List[Dict]:
        """Get user history from a specific date with pagination and optional subject filtering.
        
        Args:
            student_id: ID of the student
            from_date: Optional datetime to filter history created after this date
            page: Page number (1-based indexing)
            page_size: Number of items per page (default: 10)
            oldest_first: If True, return oldest messages first; if False, return newest first (default: False)
            subject: Optional subject to filter by (e.g., 'science', 'mathematics', etc.)
            
        Returns:
            List of history items for the specified page
        """
        collection = self.get_collection(student_id, "sahasra_history")
        
        # Build query
        query = {}
        
        # Add date filter if provided
        if from_date:
            start_date = from_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = (from_date + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            query["time"] = {"$gte": start_date, "$lt": end_date}
        
        # Add subject filter if provided
        if subject and subject.lower() != "all":
            # Normalize subject name (replace hyphens with underscores)
            normalized_subject = subject.replace("-", "_").lower()
            query["subject"] = {"$regex": f"^{normalized_subject}$", "$options": "i"}  # Case-insensitive exact match
        
        # Calculate skip value for pagination (page is 1-based)
        skip = (page - 1) * page_size
        
        # Sort direction: 1 for ascending (oldest first), -1 for descending (newest first)
        sort_direction = 1 if oldest_first else -1
        
        # Get paginated results sorted by time
        return list(collection.find(query)
                   .sort("time", sort_direction)
                   .skip(skip)
                   .limit(page_size))
    
    def get_available_dates(self, student_id: str, from_date: datetime, limit: int = 5, subject: str = None) -> List[str]:
        """Get available dates where messages exist, starting from a given date.
        
        Args:
            student_id: ID of the student
            from_date: Date to start searching from (inclusive)
            limit: Maximum number of dates to return (default: 5)
            subject: Subject to filter history by (science, social_science, mathematics, english, hindi). 
                    If None, returns dates from all subjects.
        Returns:
            List of date strings in YYYY-MM-DD format, sorted by most recent first
        """
        collection = self.get_collection(student_id, "sahasra_history")
        
        # Build match query
        match_query = {
            "time": {"$lte": from_date}  # Messages on or before the given date
        }
        
        # Add subject filter if provided
        if subject and subject.lower() != "all":
            # Normalize subject name (replace hyphens with underscores)
            normalized_subject = subject.replace("-", "_").lower()
            match_query["subject"] = {"$regex": f"^{normalized_subject}$", "$options": "i"}  # Case-insensitive exact match
        
        # Create aggregation pipeline to get unique dates
        pipeline = [
            {
                "$match": match_query
            },
            {
                "$addFields": {
                    "date_only": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$time"
                        }
                    }
                }
            },
            {
                "$group": {
                    "_id": "$date_only",
                    "latest_time": {"$max": "$time"}  # Keep the latest time for sorting
                }
            },
            {
                "$sort": {"latest_time": -1}  # Sort by most recent first
            },
            {
                "$limit": limit
            },
            {
                "$project": {
                    "_id": 0,
                    "date": "$_id"
                }
            }
        ]
        
        # Execute aggregation pipeline
        results = list(collection.aggregate(pipeline))
        
        # Extract just the date strings
        return [result["date"] for result in results]
    
    def get_assessments(self, student_id: str, from_date: datetime = None, subject: str = None, topic: str = None) -> List[Dict]:
        """Get user assessments from a specific date and optionally filtered by subject or topic.
        
        Args:
            student_id: ID of the student
            from_date: Optional datetime to filter assessments created after this date
            subject: Optional subject to filter assessments by subject field
            topic: Optional topic to filter assessments that include this topic
            
        Returns:
            List of assessments matching the criteria
        """
        collection = self.get_collection(student_id, "sahasra_assessments")
        query = {}
        
        # Add date filter if provided
        if from_date:
            query["created_at"] = {"$gte": from_date}
            
        # Add subject filter if provided
        if subject:
            if subject.lower() == "all":
                # When subject is "all", only return assessments that have a subject field
                # This excludes PDF-related assessments that don't have a subject field
                query["subject"] = {"$exists": True}
            else:
                # Filter by specific subject
                query["subject"] = subject
            
        # Add topic filter if provided
        if topic:
            # Search both in 'topic' (legacy) field and 'topics' array
            query["$or"] = [
                {"topic": topic},  # Legacy field
                {"topics": topic}   # New array field
            ]
        print(f"Query: {query}")
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

    def get_learning_streak(self, student_id: str, subject: str = None, count_ai_messages: bool = False) -> Dict[str, Any]:
        """Calculate the current learning streak for a student.
        
        Args:
            student_id: ID of the student
            subject: Optional subject to filter by (if None, counts all subjects)
            count_ai_messages: If True, counts both user and AI messages; if False, only user messages
            
        Returns:
            Dictionary with streak information including:
            - current_streak: Number of consecutive days with activity (starting from today)
            - last_activity_date: Date of last activity
            - longest_streak: Longest streak in history
            - total_active_days: Total number of days with activity
        """
        collection = self.get_collection(student_id, "sahasra_history")
        
        # Build query
        query = {}
        if subject and subject.lower() != "all":
            normalized_subject = subject.replace("-", "_").lower()
            query["subject"] = {"$regex": f"^{normalized_subject}$", "$options": "i"}
        
        if not count_ai_messages:
            query["is_ai"] = False  # Only count user messages
        
        # Get all activity dates using aggregation to ensure no duplicate dates
        pipeline = [
            {"$match": query},
            {
                "$addFields": {
                    "date_only": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$time"
                        }
                    }
                }
            },
            {
                "$group": {
                    "_id": "$date_only",  # This ensures each date appears only once
                    "message_count": {"$sum": 1},
                    "latest_time": {"$max": "$time"}
                }
            },
            {
                "$sort": {"_id": -1}  # Sort by date descending (newest first)
            }
        ]
        
        activity_dates = list(collection.aggregate(pipeline))
        
        if not activity_dates:
            return {
                "current_streak": 0,
                "last_activity_date": None,
                "longest_streak": 0,
                "total_active_days": 0
            }
        
        # Convert to date objects (MongoDB aggregation already ensures uniqueness via $group)
        from datetime import datetime, timedelta
        active_dates = []
        
        for item in activity_dates:
            date_obj = datetime.strptime(item["_id"], "%Y-%m-%d").date()
            active_dates.append(date_obj)
        
        # Dates should already be sorted from MongoDB, but ensure newest first
        active_dates.sort(reverse=True)
        
        # Calculate current streak (consecutive days from today backwards)
        today = datetime.utcnow().date()
        current_streak = 0
        
        # Start checking from today
        check_date = today
        
        for active_date in active_dates:
            # Check if this active date is consecutive with our streak
            if active_date == check_date:
                current_streak += 1
                check_date = check_date - timedelta(days=1)
            elif active_date == check_date - timedelta(days=1):
                # Allow for 1-day gap (grace period)
                current_streak += 1
                check_date = active_date - timedelta(days=1)
            else:
                # Gap is too large, streak is broken
                break
        
        # Calculate longest streak in history
        longest_streak = 0
        temp_streak = 1
        
        if len(active_dates) > 1:
            for i in range(1, len(active_dates)):
                prev_date = active_dates[i-1]  # More recent date
                curr_date = active_dates[i]    # Older date
                
                # Calculate difference between consecutive dates
                diff = (prev_date - curr_date).days
                
                if diff == 1:  # Consecutive days
                    temp_streak += 1
                elif diff == 2:  # 1-day gap (grace period)
                    temp_streak += 1
                else:
                    # Gap is too large, start new streak
                    longest_streak = max(longest_streak, temp_streak)
                    temp_streak = 1
            
            # Don't forget the last streak
            longest_streak = max(longest_streak, temp_streak)
        else:
            longest_streak = 1 if active_dates else 0
        
        # Make sure current streak doesn't exceed longest streak
        longest_streak = max(longest_streak, current_streak)
        
        return {
            "current_streak": current_streak,
            "last_activity_date": active_dates[0].isoformat() if active_dates else None,
            "longest_streak": longest_streak,
            "total_active_days": len(active_dates)
        }

    def get_questions_answered_count(self, student_id: str, subject: str = None, from_date: datetime = None) -> Dict[str, int]:
        """Get count of questions answered (AI responses) for a student.
        
        Args:
            student_id: ID of the student
            subject: Optional subject to filter by (if None, counts all subjects)
            from_date: Optional datetime to filter messages from this date onwards
            
        Returns:
            Dictionary with questions answered counts
        """
        collection = self.get_collection(student_id, "sahasra_history")
        
        # Build query for AI messages (questions answered by AI)
        query = {"is_ai": True}
        
        if subject and subject.lower() != "all":
            normalized_subject = subject.replace("-", "_").lower()
            query["subject"] = {"$regex": f"^{normalized_subject}$", "$options": "i"}
        
        if from_date:
            query["time"] = {"$gte": from_date}
        
        # Get total count by finding all documents and getting length
        all_documents = list(collection.find(query))
        total_count = len(all_documents)
        
        # Debug logging
        print(f"Questions answered query: {query}")
        print(f"Found {total_count} AI messages for student {student_id}")
        
        # Additional debugging - check total documents in collection
        total_in_collection = len(list(collection.find({})))
        ai_true_total = len(list(collection.find({"is_ai": True})))
        print(f"Total documents in collection: {total_in_collection}")
        print(f"Total documents with is_ai=True: {ai_true_total}")
        if from_date:
            print(f"Filtering from date: {from_date}")
        
        # Get count by subject
        pipeline = [
            {"$match": query},
            {
                "$group": {
                    "_id": "$subject",
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"count": -1}}
        ]
        
        by_subject = {}
        for item in collection.aggregate(pipeline):
            subject_name = item["_id"] or "unknown"
            by_subject[subject_name] = item["count"]
        
        return {
            "total_questions_answered": total_count,
            "by_subject": by_subject
        }

class QuotesRepository(MongoRepository):
    """Repository for educational quotes operations."""
    
    def __init__(self, mongo_uri: str = None):
        super().__init__(mongo_uri)
        self.quotes_collection = self.get_collection(
            "educational_resources", 
            "quotes"
        )
    
    def get_quotes_count(self) -> int:
        """Get the total number of quotes in the collection."""
        return self.quotes_collection.count_documents({})
    
    def add_quotes_bulk(self, quotes_list: List[Dict]) -> bool:
        """Add multiple quotes at once.
        
        Args:
            quotes_list: List of quote dictionaries with 'quote' and 'author' fields
            
        Returns:
            True if insertion was successful
        """
        try:
            # Add created_at timestamp to each quote
            for quote in quotes_list:
                quote["created_at"] = datetime.utcnow()
            
            result = self.quotes_collection.insert_many(quotes_list)
            return len(result.inserted_ids) == len(quotes_list)
        except Exception as e:
            print(f"Error inserting quotes: {str(e)}")
            return False
    
    def get_random_quote(self) -> Optional[Dict]:
        """Get a random educational quote.
        
        Returns:
            Random quote dictionary or None if no quotes exist
        """
        try:
            # Use MongoDB's $sample aggregation to get a random document
            pipeline = [{"$sample": {"size": 1}}]
            result = list(self.quotes_collection.aggregate(pipeline))
            
            if result:
                quote = result[0]
                # Remove MongoDB ObjectId for cleaner response
                if "_id" in quote:
                    del quote["_id"]
                return quote
            return None
        except Exception as e:
            print(f"Error getting random quote: {str(e)}")
            return None 
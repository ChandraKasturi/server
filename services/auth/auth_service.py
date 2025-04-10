import secrets
import string
import jwt
import time
import re
import uuid
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, Any, List

from config import settings
from repositories.mongo_repository import UserRepository, TokenRepository
from services.notification.notification_service import NotificationService

class AuthService:
    """Service for handling authentication-related operations."""
    
    def __init__(self):
        """Initialize auth service with repositories."""
        self.user_repo = UserRepository()
        self.token_repo = TokenRepository()
        self.notification_service = NotificationService()
        self.phone_regex = re.compile(r"[0-9]+$")
        
    def login(self, email_or_mobile: str, password: str) -> Tuple[Optional[str], Optional[str]]:
        """Authenticate user and generate JWT token.
        
        Args:
            email_or_mobile: Email or mobile number of the user
            password: Password of the user
            
        Returns:
            Tuple of (student_id, token) if successful, (None, None) if unsuccessful
        """
        if not isinstance(email_or_mobile, str) or not isinstance(password, str):
            return None, None
            
        user = self.user_repo.find_by_credentials(email_or_mobile, password)
        
        if not user:
            return None, None
            
        student_id = user["student_id"]
        
        # Generate JWT token
        iat = int(time.time())
        jti = uuid.uuid4().hex
        token = jwt.encode(
            payload={"user": student_id, "iat": iat, "jti": jti},
            key=settings.JWT_SECRET,
            algorithm=settings.JWT_ALGORITHM
        )
        
        # Store token in database
        self.token_repo.store_auth_token(student_id, token)
        
        return student_id, token
        
    def logout(self, student_id: str) -> bool:
        """Logout user by deleting their auth token.
        
        Args:
            student_id: ID of the student to log out
            
        Returns:
            True if successful, False otherwise
        """
        return self.token_repo.delete_auth_token(student_id)
        
    def verify_token(self, token: str) -> Optional[str]:
        """Verify JWT token and return student ID if valid.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Student ID if token is valid, None otherwise
        """
        if not token:
            return None
            
        try:
            # Decode token
            payload = jwt.decode(
                jwt=token,
                key=settings.JWT_SECRET,
                algorithms=[settings.JWT_ALGORITHM]
            )
            
            student_id = payload["user"]
            
            # Check if token exists in database
            stored_token = self.token_repo.get_auth_token(student_id)
            if not stored_token:
                return None
                
            return student_id
        except Exception:
            return None
    
    def initiate_registration(self, email: str, phone_number: str) -> Tuple[str, int]:
        """Start the registration process by sending OTP.
        
        Args:
            email: Email of the user
            phone_number: Phone number of the user
            
        Returns:
            Tuple of (message, status_code)
        """
        email = email.lower()
        
        # Check if email or phone is already registered
        existing_user = self.user_repo.find_by_email_or_mobile(email) or self.user_repo.find_by_email_or_mobile(phone_number)
        
        if existing_user:
            return "Email or Phone Number Already Taken", 400
            
        # Generate OTP
        token = ''.join([secrets.choice(string.digits) for _ in range(6)])
        
        # Store registration data
        register_data = {
            "email": email,
            "phonenumber": phone_number,
            "token": token,
        }
        
        self.token_repo.store_register_token(register_data)
        
        # Send notifications
        self.notification_service.send_registration_otp_email(email, token)
        self.notification_service.send_registration_otp_sms(phone_number, token)
        
        return "Register Token Has been sent to your Email ID Or MobileNumber Please Use it to Confirm Your Registration", 200
    
    def complete_registration(self, register_data: Dict) -> Tuple[str, Optional[str], int]:
        """Complete registration with OTP verification.
        
        Args:
            register_data: Registration data including token and user details
            
        Returns:
            Tuple of (message, student_id, status_code)
        """
        token = register_data.get("token")
        
        # Verify token
        token_data = self.token_repo.get_register_token_data(token)
        
        if not token_data:
            return "Token Expired Try again", None, 400
            
        if token_data.get("token") != token:
            return "Wrong Token", None, 400
            
        # Generate student ID
        student_id = secrets.token_hex(16)
        
        # Prepare user data
        user_data = register_data.copy()
        user_data["student_id"] = student_id
        
        try:
            # Insert user into database
            self.user_repo.insert_user(user_data)
            return "User Registered Successfully You May Login Now", student_id, 200
        except Exception:
            return "Something Went Wrong", None, 400
    
    def initiate_password_reset(self, email_or_mobile: str) -> Tuple[str, int]:
        """Start the password reset process.
        
        Args:
            email_or_mobile: Email or mobile number of the user
            
        Returns:
            Tuple of (message, status_code)
        """
        if not isinstance(email_or_mobile, str):
            return "Email or mobile number must be a string", 400
            
        # Check if user exists
        user_email_or_mobile = self.user_repo.find_by_email_or_mobile(email_or_mobile)
        
        if not user_email_or_mobile:
            # Don't reveal if email/mobile exists for security
            return "If the Email or PhoneNumber You Provided Exists in our Database The reset Link Should be in Your Inbox Please Check your mail or Mobile", 200
            
        # Generate OTP
        token = ''.join([secrets.choice(string.digits) for _ in range(6)])
        
        # Store password reset token
        self.token_repo.store_password_token(email_or_mobile, token)
        
        try:
            # Send notifications based on whether it's an email or phone
            if '@' in email_or_mobile:
                self.notification_service.send_password_reset_email(email_or_mobile, token)
            else:
                self.notification_service.send_password_reset_sms(email_or_mobile, token)
                
            return "If the Email or PhoneNumber You Provided Exists in our Database The reset Link Should be in Your Inbox Please Check your mail or Mobile", 200
        except Exception:
            return "Something Went Wrong", 400
    
    def reset_password(self, password: str, token: str) -> Tuple[str, int]:
        """Reset user password using token.
        
        Args:
            password: New password
            token: Reset token
            
        Returns:
            Tuple of (message, status_code)
        """
        if not password or not token:
            return "Both password and token are required", 400
            
        # Verify token
        token_data = self.token_repo.get_password_token_data(token)
        
        if not token_data:
            return "Token Expired Try again", 400
            
        if token_data.get("token") != token:
            return "Wrong Token", 400
            
        try:
            # Update password
            email_or_mobile = token_data.get("email")
            self.user_repo.update_password(email_or_mobile, password)
            
            return "Password Updated Successfully You May Login Now", 200
        except Exception:
            return "Something Went Wrong", 400 
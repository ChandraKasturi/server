import os
import uuid
from typing import Dict, Tuple, Optional, Any, BinaryIO
from fastapi import UploadFile
from PIL import Image
from io import BytesIO

import secrets
import string
import re
from config import settings
from repositories.mongo_repository import UserRepository, TokenRepository
from services.notification.notification_service import NotificationService

class ProfileService:
    """Service for managing user profiles."""
    
    def __init__(self):
        """Initialize profile service."""
        self.user_repo = UserRepository()
        self.token_repo = TokenRepository()
        self.notification_service = NotificationService()
        self.phone_regex = re.compile(r"[0-9]+$")
        
    def get_profile(self, student_id: str) -> Tuple[Dict, int]:
        """Get a user's profile.
        
        Args:
            student_id: ID of the student
            
        Returns:
            Tuple of (profile_data, status_code)
        """
        try:
            user = self.user_repo.users_collection.find_one({"student_id": student_id})
            
            if not user:
                return {"Message": "User not found"}, 404
                
            # Filter sensitive fields
            filtered_user = {k: v for k, v in user.items() if k not in ['_id', 'password', 'token']}
            
            return filtered_user, 200
        except Exception:
            return {"Message": "Something went wrong"}, 500
    
    def update_profile(self, student_id: str, update_data: Dict) -> Tuple[Dict, int]:
        """Update a user's profile.
        
        Args:
            student_id: ID of the student
            update_data: New profile data
            
        Returns:
            Tuple of (message, status_code)
        """
        try:
            # Remove any fields that are None
            update_dict = {k: v for k, v in update_data.items() if v is not None}
            
            if not update_dict:
                return {"Message": "No fields to update"}, 400
            
            # Block mobile number updates - redirect to verification flow
            mobile_fields = ['phonenumber', 'mobile', 'mobilenumber', 'phone_number']
            for field in mobile_fields:
                if field in update_dict:
                    return {
                        "Message": "Mobile number updates require verification. Please use /updatemobile endpoint.",
                        "requires_verification": True,
                        "endpoint": "/updatemobile"
                    }, 400
            
            # Block email updates - redirect to verification flow
            email_fields = ['email', 'email_address', 'emailaddress']
            for field in email_fields:
                if field in update_dict:
                    return {
                        "Message": "Email address updates require verification. Please use /updateemail endpoint.",
                        "requires_verification": True,
                        "endpoint": "/updateemail"
                    }, 400
                
            success = self.user_repo.update_user(student_id, update_dict)
            
            if success:
                return {"Message": "Profile Updated Successfully"}, 200
            else:
                return {"Message": "User not found"}, 404
        except Exception:
            return {"Message": "Something went wrong"}, 500
    
    def get_profile_image_path(self, student_id: str) -> Tuple[str, int]:
        """Get the relative path to a user's profile image from /static.
        
        Args:
            student_id: ID of the student
            
        Returns:
            Tuple of (relative_image_path, status_code)
        """
        try:
            static_dir = settings.static_dir_path
            images_dir = settings.static_image_path
            
            # Ensure directories exist
            if not os.path.exists(static_dir):
                os.makedirs(static_dir)
                
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
            
            # Look for existing profile image with various extensions
            for ext in ['.jpg', '.jpeg', '.png', '.gif']:
                img_path = os.path.join(images_dir, f"{student_id}{ext}")
                if os.path.exists(img_path):
                    # Return relative path from /static
                    return f"/{settings.STATIC_DIR}/{settings.STATIC_IMAGE_DIR}/{student_id}{ext}", 200
            
            # If no image found, return default relative path
            return settings.default_profile_image_relative_path, 200
        except Exception:
            return settings.default_profile_image_relative_path, 200
    
    async def update_profile_image(self, student_id: str, file: UploadFile) -> Tuple[Dict, int]:
        """Update a user's profile image.
        
        Args:
            student_id: ID of the student
            file: Uploaded image file
            
        Returns:
            Tuple of (message, status_code)
        """
        try:
            static_dir = settings.static_dir_path
            images_dir = settings.static_image_path
            
            # Ensure directories exist
            if not os.path.exists(static_dir):
                os.makedirs(static_dir)
                
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
            
            # Read file content
            contents = await file.read()
            
            # Validate image
            try:
                image = Image.open(BytesIO(contents))
                image_format = image.format.lower()
                
                # Determine file extension
                if image_format not in ['jpeg', 'jpg', 'png', 'gif']:
                    return {"Message": "Invalid image format. Supported formats: JPEG, PNG, GIF"}, 400
                    
                ext = f".{image_format}" if image_format != 'jpeg' else '.jpg'
                
                # Delete any existing profile images for this student
                for old_ext in ['.jpg', '.jpeg', '.png', '.gif']:
                    old_path = os.path.join(images_dir, f"{student_id}{old_ext}")
                    if os.path.exists(old_path):
                        os.remove(old_path)
                
                # Create path for new image
                image_path = os.path.join(images_dir, f"{student_id}{ext}")
                
                # Save the image
                with open(image_path, "wb") as f:
                    f.write(contents)
                
                # Generate relative path from /static
                image_path = f"/{settings.STATIC_DIR}/{settings.STATIC_IMAGE_DIR}/{student_id}{ext}"
                
                return {"Message": "Image Uploaded Successfully", "image_url": image_path}, 200
            except Exception:
                return {"Message": "Invalid image file"}, 400
        except Exception:
            return {"Message": "Something went wrong"}, 500
    
    def request_mobile_update(self, student_id: str, new_mobile: str) -> Tuple[Dict, int]:
        """Request mobile number update with OTP verification.
        
        Args:
            student_id: ID of the student
            new_mobile: New mobile number to verify
            
        Returns:
            Tuple of (message, status_code)
        """
        try:
            # Get current user to get old mobile
            user = self.user_repo.users_collection.find_one({"student_id": student_id})
            
            if not user:
                return {"Message": "User not found"}, 404
                
            old_mobile = user.get("phonenumber", "")
            
            # Validate new mobile number
            if not isinstance(new_mobile, str) or not new_mobile.strip():
                return {"Message": "New mobile number is required"}, 400
                
            # Validate mobile number format (basic validation)
            if not self.phone_regex.match(new_mobile.replace('+', '').replace('-', '').replace(' ', '')):
                return {"Message": "Invalid mobile number format"}, 400
                
            # Check if new mobile is already taken by another user
            existing_user = self.user_repo.find_by_email_or_mobile(new_mobile)
            if existing_user and existing_user.get("student_id") != student_id:
                return {"Message": "Mobile number already registered to another account"}, 400
                
            # Check if there's already a pending verification for this user
            if self.token_repo.has_pending_mobile_verification(student_id):
                return {"Message": "Mobile verification already in progress. Please complete or wait for it to expire."}, 400
                
            # Generate OTP
            token = ''.join([secrets.choice(string.digits) for _ in range(6)])
            
            # Store mobile verification token
            success = self.token_repo.store_mobile_verification_token(student_id, old_mobile, new_mobile, token)
            
            if not success:
                return {"Message": "Failed to initiate mobile verification"}, 500
                
            # Send OTP to new mobile number
            try:
                self.notification_service.send_mobile_verification_otp_sms(new_mobile, token)
                return {"Message": "OTP sent to your new mobile number. Please verify to complete the change."}, 200
            except Exception:
                # Clean up token if SMS sending fails
                self.token_repo.delete_mobile_verification_token(student_id)
                return {"Message": "Failed to send OTP. Please try again."}, 500
                
        except Exception:
            return {"Message": "Something went wrong"}, 500
    
    def verify_mobile_update(self, student_id: str, otp_token: str) -> Tuple[Dict, int]:
        """Verify mobile number update with OTP.
        
        Args:
            student_id: ID of the student
            otp_token: OTP token to verify
            
        Returns:
            Tuple of (message, status_code)
        """
        try:
            if not otp_token or not otp_token.strip():
                return {"Message": "Verification token is required"}, 400
                
            # Get verification token data
            token_data = self.token_repo.get_mobile_verification_token_data(student_id, otp_token)
            
            if not token_data:
                return {"Message": "Invalid or expired verification token"}, 400
                
            if token_data.get("token") != otp_token:
                return {"Message": "Invalid verification token"}, 400
                
            new_mobile = token_data.get("new_mobile")
            
            # Double-check that the new mobile isn't taken by another user
            existing_user = self.user_repo.find_by_email_or_mobile(new_mobile)
            if existing_user and existing_user.get("student_id") != student_id:
                # Clean up token
                self.token_repo.delete_mobile_verification_token(student_id)
                return {"Message": "Mobile number already registered to another account"}, 400
                
            # Update user's mobile number
            update_success = self.user_repo.update_user(student_id, {"mobilenumber": new_mobile})
            
            if update_success:
                # Clean up verification token
                self.token_repo.delete_mobile_verification_token(student_id)
                return {"Message": "Mobile number updated successfully", "new_mobile": new_mobile}, 200
            else:
                return {"Message": "Failed to update mobile number"}, 500
                
        except Exception:
            return {"Message": "Something went wrong while updating mobile number"}, 500
    
    def get_pending_mobile_verification(self, student_id: str) -> Tuple[Dict, int]:
        """Get pending mobile verification details.
        
        Args:
            student_id: ID of the student
            
        Returns:
            Tuple of (verification_details, status_code)
        """
        try:
            token_data = self.token_repo.mobile_verification_tokens_collection.find_one(
                {"student_id": student_id},
                {"old_mobile": 1, "new_mobile": 1, "created_at": 1}
            )
            
            if token_data:
                return {
                    "has_pending_verification": True,
                    "old_mobile": token_data.get("old_mobile"),
                    "new_mobile": token_data.get("new_mobile"),
                    "requested_at": token_data.get("created_at")
                }, 200
            else:
                return {"has_pending_verification": False}, 200
        except Exception:
            return {"Message": "Something went wrong"}, 500
    
    def request_email_update(self, student_id: str, new_email: str) -> Tuple[Dict, int]:
        """Request email address update with OTP verification.
        
        Args:
            student_id: ID of the student
            new_email: New email address to verify
            
        Returns:
            Tuple of (message, status_code)
        """
        try:
            # Get current user to get old email
            user = self.user_repo.users_collection.find_one({"student_id": student_id})
            
            if not user:
                return {"Message": "User not found"}, 404
                
            old_email = user.get("email", "")
            
            # Validate new email address
            if not isinstance(new_email, str) or not new_email.strip():
                return {"Message": "New email address is required"}, 400
                
            # Basic email validation
            import re
            email_regex = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
            if not email_regex.match(new_email):
                return {"Message": "Invalid email address format"}, 400
                
            # Check if new email is already taken by another user
            existing_user = self.user_repo.find_by_email_or_mobile(new_email)
            if existing_user and existing_user.get("student_id") != student_id:
                return {"Message": "Email address already registered to another account"}, 400
                
            # Check if there's already a pending verification for this user
            if self.token_repo.has_pending_email_verification(student_id):
                return {"Message": "Email verification already in progress. Please complete or wait for it to expire."}, 400
                
            # Generate OTP
            token = ''.join([secrets.choice(string.digits) for _ in range(6)])
            
            # Store email verification token
            success = self.token_repo.store_email_verification_token(student_id, old_email, new_email, token)
            
            if not success:
                return {"Message": "Failed to initiate email verification"}, 500
                
            # Send OTP to new email address
            try:
                self.notification_service.send_email_verification_email(old_email, new_email, token)
                return {"Message": "Verification email sent to your new email address. Please check your inbox and enter the OTP to complete the change."}, 200
            except Exception:
                # Clean up token if email sending fails
                self.token_repo.delete_email_verification_token(student_id)
                return {"Message": "Failed to send verification email. Please try again."}, 500
                
        except Exception:
            return {"Message": "Something went wrong"}, 500
    
    def verify_email_update(self, student_id: str, otp_token: str) -> Tuple[Dict, int]:
        """Verify email address update with OTP.
        
        Args:
            student_id: ID of the student
            otp_token: OTP token to verify
            
        Returns:
            Tuple of (message, status_code)
        """
        try:
            if not otp_token or not otp_token.strip():
                return {"Message": "Verification token is required"}, 400
                
            # Get verification token data
            token_data = self.token_repo.get_email_verification_token_data(student_id, otp_token)
            
            if not token_data:
                return {"Message": "Invalid or expired verification token"}, 400
                
            if token_data.get("token") != otp_token:
                return {"Message": "Invalid verification token"}, 400
                
            new_email = token_data.get("new_email")
            
            # Double-check that the new email isn't taken by another user
            existing_user = self.user_repo.find_by_email_or_mobile(new_email)
            if existing_user and existing_user.get("student_id") != student_id:
                # Clean up token
                self.token_repo.delete_email_verification_token(student_id)
                return {"Message": "Email address already registered to another account"}, 400
                
            # Update user's email address
            update_success = self.user_repo.update_user(student_id, {"email": new_email})
            
            if update_success:
                # Clean up verification token
                self.token_repo.delete_email_verification_token(student_id)
                return {"Message": "Email address updated successfully", "new_email": new_email}, 200
            else:
                return {"Message": "Failed to update email address"}, 500
                
        except Exception:
            return {"Message": "Something went wrong while updating email address"}, 500
    
    def get_pending_email_verification(self, student_id: str) -> Tuple[Dict, int]:
        """Get pending email verification details.
        
        Args:
            student_id: ID of the student
            
        Returns:
            Tuple of (verification_details, status_code)
        """
        try:
            token_data = self.token_repo.email_verification_tokens_collection.find_one(
                {"student_id": student_id},
                {"old_email": 1, "new_email": 1, "created_at": 1}
            )
            
            if token_data:
                return {
                    "has_pending_verification": True,
                    "old_email": token_data.get("old_email"),
                    "new_email": token_data.get("new_email"),
                    "requested_at": token_data.get("created_at")
                }, 200
            else:
                return {"has_pending_verification": False}, 200
        except Exception:
            return {"Message": "Something went wrong"}, 500 
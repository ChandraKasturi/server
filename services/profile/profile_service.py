import os
import uuid
from typing import Dict, Tuple, Optional, Any, BinaryIO
from fastapi import UploadFile
from PIL import Image
from io import BytesIO

from config import settings
from repositories.mongo_repository import UserRepository

class ProfileService:
    """Service for managing user profiles."""
    
    def __init__(self):
        """Initialize profile service."""
        self.user_repo = UserRepository()
        
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
                
                return {"Message": "Image Uploaded Successfully", "image_path": image_path}, 200
            except Exception:
                return {"Message": "Invalid image file"}, 400
        except Exception:
            return {"Message": "Something went wrong"}, 500 
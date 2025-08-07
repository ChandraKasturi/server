import os
from fastapi import APIRouter, Depends, Request, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse

from models.u_models import (
    ProfileUmodel, ProfileGetResponse, ProfileUpdateResponse, ProfileImageUpdateResponse,
    MobileUpdateRequest, MobileVerificationRequest, MobileUpdateResponse, 
    MobileVerificationResponse, PendingMobileVerificationResponse,
    EmailUpdateRequest, EmailVerificationRequest, EmailUpdateResponse,
    EmailVerificationResponse, PendingEmailVerificationResponse
)
from services.profile.profile_service import ProfileService
from routers.auth import auth_middleware
from utils.json_response import UGJSONResponse
from config import settings

router = APIRouter(tags=["Profile"])

profile_service = ProfileService()

@router.get("/profile", response_model=ProfileGetResponse)
def get_profile(request: Request, student_id: str = Depends(auth_middleware)):
    """Get user profile information.
    
    Args:
        request: FastAPI request object
        student_id: ID of the student (from auth middleware)
        
    Returns:
        JSON response with profile data
    """
    profile_data, status_code = profile_service.get_profile(student_id)
    
    return JSONResponse(content=profile_data, status_code=status_code)

@router.post("/profile", response_model=ProfileUpdateResponse)
def update_profile(body: ProfileUmodel, student_id: str = Depends(auth_middleware)):
    """Update user profile information.
    
    Args:
        body: Profile data to update
        student_id: ID of the student (from auth middleware)
        
    Returns:
        JSON response with update status
    """
    result, status_code = profile_service.update_profile(student_id, dict(body))
    
    return JSONResponse(content=result, status_code=status_code)

@router.post("/updateprofileimage", response_model=ProfileImageUpdateResponse)
async def update_profile_image(student_id: str = Depends(auth_middleware), file: UploadFile = File(...)):
    """Update user profile image.
    
    Args:
        student_id: ID of the student (from auth middleware)
        file: Uploaded image file
        
    Returns:
        JSON response with update status
    """
    result, status_code = await profile_service.update_profile_image(student_id, file)
    
    return JSONResponse(content=result, status_code=status_code)

@router.get("/getprofileimage", response_class=FileResponse)
def get_profile_image(student_id: str = Depends(auth_middleware)):
    """Get user profile image.
    
    Args:
        student_id: ID of the student (from auth middleware)
        
    Returns:
        File response with profile image
    """
    relative_image_path, status_code = profile_service.get_profile_image_path(student_id)
    
    # Convert relative path to full file system path
    
    return UGJSONResponse(content={"image_url": relative_image_path}, status_code=status_code)

@router.post("/updatemobile", response_model=MobileUpdateResponse)
def request_mobile_update(body: MobileUpdateRequest, student_id: str = Depends(auth_middleware)):
    """Request mobile number update with OTP verification.
    
    Args:
        body: Request containing new mobile number
        student_id: ID of the student (from auth middleware)
        
    Returns:
        JSON response with request status
    """
    result, status_code = profile_service.request_mobile_update(student_id, body.new_mobile)
    
    return JSONResponse(content=result, status_code=status_code)

@router.post("/verifymobile", response_model=MobileVerificationResponse)
def verify_mobile_update(body: MobileVerificationRequest, student_id: str = Depends(auth_middleware)):
    """Verify mobile number update with OTP.
    
    Args:
        body: Request containing OTP token
        student_id: ID of the student (from auth middleware)
        
    Returns:
        JSON response with verification status
    """
    result, status_code = profile_service.verify_mobile_update(student_id, body.otp_token)
    
    return JSONResponse(content=result, status_code=status_code)

@router.get("/profile/pending-mobile-verification", response_model=PendingMobileVerificationResponse)
def get_pending_mobile_verification(student_id: str = Depends(auth_middleware)):
    """Get pending mobile verification details.
    
    Args:
        student_id: ID of the student (from auth middleware)
        
    Returns:
        JSON response with pending verification details
    """
    result, status_code = profile_service.get_pending_mobile_verification(student_id)
    
    return JSONResponse(content=result, status_code=status_code)

@router.post("/updateemail", response_model=EmailUpdateResponse)
def request_email_update(body: EmailUpdateRequest, student_id: str = Depends(auth_middleware)):
    """Request email address update with OTP verification.
    
    Args:
        body: Request containing new email address
        student_id: ID of the student (from auth middleware)
        
    Returns:
        JSON response with request status
    """
    result, status_code = profile_service.request_email_update(student_id, body.new_email)
    
    return JSONResponse(content=result, status_code=status_code)

@router.post("/verifyemail", response_model=EmailVerificationResponse)
def verify_email_update(body: EmailVerificationRequest, student_id: str = Depends(auth_middleware)):
    """Verify email address update with OTP.
    
    Args:
        body: Request containing OTP token
        student_id: ID of the student (from auth middleware)
        
    Returns:
        JSON response with verification status
    """
    result, status_code = profile_service.verify_email_update(student_id, body.otp_token)
    
    return JSONResponse(content=result, status_code=status_code)

@router.get("/profile/pending-email-verification", response_model=PendingEmailVerificationResponse)
def get_pending_email_verification(student_id: str = Depends(auth_middleware)):
    """Get pending email verification details.
    
    Args:
        student_id: ID of the student (from auth middleware)
        
    Returns:
        JSON response with pending verification details
    """
    result, status_code = profile_service.get_pending_email_verification(student_id)
    
    return JSONResponse(content=result, status_code=status_code) 
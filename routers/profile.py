from fastapi import APIRouter, Depends, Request, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse

from models.u_models import ProfileUmodel
from services.profile.profile_service import ProfileService
from routers.auth import auth_middleware
from utils.json_response import UGJSONResponse

router = APIRouter(tags=["Profile"])

profile_service = ProfileService()

@router.get("/profile", response_class=JSONResponse)
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

@router.post("/profile", response_class=JSONResponse)
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

@router.post("/updateprofileimage", response_class=UGJSONResponse)
async def update_profile_image(student_id: str = Depends(auth_middleware), file: UploadFile = File(...)):
    """Update user profile image.
    
    Args:
        student_id: ID of the student (from auth middleware)
        file: Uploaded image file
        
    Returns:
        JSON response with update status
    """
    result, status_code = await profile_service.update_profile_image(student_id, file)
    
    return UGJSONResponse(content=result, status_code=status_code)

@router.get("/getprofileimage", response_class=FileResponse)
def get_profile_image(student_id: str = Depends(auth_middleware)):
    """Get user profile image.
    
    Args:
        student_id: ID of the student (from auth middleware)
        
    Returns:
        File response with profile image
    """
    image_path, status_code = profile_service.get_profile_image_path(student_id)
    
    return FileResponse(image_path) 
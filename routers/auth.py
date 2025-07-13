from fastapi import APIRouter, Request, Response, Depends, HTTPException
from fastapi.responses import JSONResponse

from models.u_models import (
    loginUmodel, ForgotPasswordUmodel, UpdatePasswordUmodel, 
    registerUmodel, confirmRegisterUmodel, AuthMessageResponse
)
from services.auth.auth_service import AuthService

router = APIRouter(tags=["Authentication"])

auth_service = AuthService()

def auth_middleware(request: Request):
    """Authentication middleware to verify user session.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Student ID if session is valid
        
    Raises:
        HTTPException: If session is invalid
    """
    token = request.headers.get("X-Auth-Session")
    
    if not token:
        raise HTTPException(status_code=401, detail="No Session Found")
    
    student_id = auth_service.verify_token(token)
    
    if not student_id:
        raise HTTPException(status_code=401, detail="Session Expired")
    
    return student_id 

@router.post("/login", response_model=AuthMessageResponse)
def login(body: loginUmodel, request: Request, response: Response):
    """Login endpoint for user authentication.
    
    Args:
        body: Login credentials including mobile/email and password
        request: FastAPI request object
        response: FastAPI response object
        
    Returns:
        JSON response with login status
    """
    if not isinstance(body.mobilenumberoremail, str) or not isinstance(body.password, str):
        return JSONResponse(content={"Message": "Username and password Must be of type string"}, status_code=400)
    
    student_id, token = auth_service.login(body.mobilenumberoremail, body.password)
    
    if student_id and token:
        return JSONResponse(
            content={"Message": "Logged in Successfully"}, 
            status_code=200, 
            headers={"X-Auth-Session": token}
        )
    else:
        return JSONResponse(content={"Message": "Incorrect Username Or Password"}, status_code=400)

@router.post("/forgotpassword", response_model=AuthMessageResponse)
def forgot_password(body: ForgotPasswordUmodel):
    """Forgot password endpoint to initiate password reset.
    
    Args:
        body: Request containing mobile/email for password reset
        
    Returns:
        JSON response with password reset status
    """
    if not isinstance(body.mobilenumberoremail, str):
        return JSONResponse(content={"Message": "Email must be a string"}, status_code=400)
    
    message, status_code = auth_service.initiate_password_reset(body.mobilenumberoremail)
    
    return JSONResponse(content={"Message": message}, status_code=status_code)

@router.post("/updatepassword", response_model=AuthMessageResponse)
def reset_password(body: UpdatePasswordUmodel):
    """Update password endpoint for password reset.
    
    Args:
        body: Request containing new password and reset token
        
    Returns:
        JSON response with password update status
    """
    if not body.password and not body.token:
        return JSONResponse(content={"Message": "Provide a Token and a password"}, status_code=400)
    
    if not body.password:
        return JSONResponse(content={"Message": "Provide A new Password"}, status_code=400)
    
    if not body.token:
        return JSONResponse(content={"Message": "Provide a token"}, status_code=400)
    
    message, status_code = auth_service.reset_password(body.password, body.token)
    
    return JSONResponse(content={"Message": message}, status_code=status_code)

@router.post("/getotp", response_model=AuthMessageResponse)
def register_request(body: registerUmodel):
    """Registration request endpoint to get OTP.
    
    Args:
        body: Initial registration data including email and phone number
        
    Returns:
        JSON response with registration status
    """
    message, status_code = auth_service.initiate_registration(body.email, body.phonenumber)
    
    return JSONResponse(content={"Message": message}, status_code=status_code)

@router.post("/register", response_model=AuthMessageResponse)
def confirm_register(body: confirmRegisterUmodel, response: Response):
    """Complete registration endpoint with OTP verification.
    
    Args:
        body: Complete registration data with OTP verification token
        response: FastAPI response object
        
    Returns:
        JSON response with registration status
    """
    message, student_id, status_code = auth_service.complete_registration(dict(body))
    
    if student_id:
        # Login the user automatically after successful registration
        _, token = auth_service.login(body.email, body.password)
        
        if token:
            return JSONResponse(
                content={"Message": message}, 
                status_code=status_code, 
                headers={"X-Auth-Session": token}
            )
    
    return JSONResponse(content={"Message": "Unable To Register User"}, status_code=400)

@router.get("/logout", response_model=AuthMessageResponse)
def logout(student_id: str = Depends(auth_middleware)):
    """Logout endpoint to invalidate user session.
    
    Args:
        student_id: ID of the student (from auth middleware)
        
    Returns:
        JSON response with logout status
    """
    success = auth_service.logout(student_id)
    
    if success:
        return JSONResponse(content={"Message": "Logged Out Successfully"}, status_code=200)
    else:
        return JSONResponse(content={"Message": "Something Went Wrong"}, status_code=400)

# Auth middleware function to be imported by other routers

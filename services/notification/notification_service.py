import os
import requests
from jinja2 import Template
from typing import Dict, Optional, Any

from config import settings

class NotificationService:
    """Service for sending notifications via email and SMS."""
    
    def __init__(self):
        """Initialize notification service."""
        self.email_sender = settings.EMAIL_SENDER
        self.forwardemail_api_url = settings.FORWARDEMAIL_API_URL
        self.sms_api_url = settings.ADWINGSSMS_API_URL
        self.sms_api_key = settings.ADWINGSSMS_API_KEY
        self.sms_entity_id = settings.ADWINGSSMS_ENTITY_ID
        self.sms_sender_id = settings.ADWINGSSMS_SENDER_ID
        self.sms_template_register = settings.ADWINGSSMS_TEMPLATE_REGISTER
        self.sms_template_password = settings.ADWINGSSMS_TEMPLATE_PASSWORD
        self.sms_template_mobile_verification = settings.ADWINGSSMS_TEMPLATE_MOBILE_VERIFICATION
    
    def send_email(self, to_email: str, subject: str, html_content: str) -> bool:
        """Send an email using the ForwardEmail API.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            html_content: HTML content of the email
            
        Returns:
            True if successful, False otherwise
        """
        data = {
            "from": self.email_sender,
            "to": to_email,
            "subject": subject,
            "html": html_content
        }
        
        try:
            response = requests.post(self.forwardemail_api_url, data=data)
            print(f"Email sent to {to_email} with status code {response.status_code} and response {response.text}")
            return response.status_code == 200
        except Exception:
            return False
    
    def _render_template(self, template_path: str, context: Dict[str, Any]) -> str:
        """Render a template with given context.
        
        Args:
            template_path: Path to the template file
            context: Dictionary of context variables
            
        Returns:
            Rendered template as a string
        """
        try:
            with open(template_path, "r") as f:
                template_content = f.read()
            
            template = Template(template_content)
            return template.render(**context)
        except Exception:
            # Fallback template for robustness
            fallback_template = """
            <html>
            <body>
                <h1>{{ subject }}</h1>
                {% for key, value in items %}
                <p><strong>{{ key }}:</strong> {{ value }}</p>
                {% endfor %}
            </body>
            </html>
            """
            template = Template(fallback_template)
            return template.render(subject="Notification", items=context.items())
    
    def send_registration_otp_email(self, email: str, token: str) -> bool:
        """Send registration OTP email.
        
        Args:
            email: Recipient email address
            token: OTP token
            
        Returns:
            True if successful, False otherwise
        """
        try:
            html_content = self._render_template(
                settings.register_template_path,
                {"token": token}
            )
            
            subject = "Welcome to Sahasra AI - Use This OTP to Get Started"
            
            return self.send_email(email, subject, html_content)
        except Exception:
            return False
    
    def send_password_reset_email(self, email: str, token: str) -> bool:
        """Send password reset email.
        
        Args:
            email: Recipient email address
            token: Reset token
            
        Returns:
            True if successful, False otherwise
        """
        try:
            html_content = self._render_template(
                settings.forgot_password_template_path,
                {"otp": token}
            )
            
            subject = "Forgot Password"
            
            return self.send_email(email, subject, html_content)
        except Exception:
            return False
    
    def send_sms(self, phone_number: str, message: str, template_id: str) -> bool:
        """Send SMS using the AdwingsSMS API.
        
        Args:
            phone_number: Recipient phone number
            message: SMS message content
            template_id: SMS template ID
            
        Returns:
            True if successful, False otherwise
        """
        params = {
            "apikey": self.sms_api_key,
            "entityid": self.sms_entity_id,
            "senderid": self.sms_sender_id,
            "templateid": template_id,
            "number": phone_number,
            "message": message,
            "format": "json"
        }
        
        try:
            print(f"Params: {params}")
            response = requests.get(self.sms_api_url, params=params, verify=False)
            print(f"SMS sent to {phone_number} with status code {response.status_code} and response {response.text}")
            return response.status_code == 200
        except Exception:
            return False
    
    def send_registration_otp_sms(self, phone_number: str, token: str) -> bool:
        """Send registration OTP SMS.
        
        Args:
            phone_number: Recipient phone number
            token: OTP token
            
        Returns:
            True if successful, False otherwise
        """
        message = f"Your Sahasra signup OTP is {token}. Please enter this code to verify your mobile number. The code is valid for 10 minutes. Do not share"
        
        return self.send_sms(
            phone_number, 
            message, 
            self.sms_template_register
        )
    
    def send_password_reset_sms(self, phone_number: str, token: str) -> bool:
        """Send password reset SMS.
        
        Args:
            phone_number: Recipient phone number
            token: Reset token
            
        Returns:
            True if successful, False otherwise
        """
        message = f"Your Sahasra password reset OTP is {token}. Please enter this code to reset your password. The code is valid for 10 minutes."
        
        return self.send_sms(
            phone_number, 
            message, 
            self.sms_template_password
        )
    
    def send_mobile_verification_otp_sms(self, phone_number: str, token: str) -> bool:
        """Send mobile verification OTP SMS.
        
        Args:
            phone_number: Phone number to send SMS to
            token: OTP token
            
        Returns:
            True if successful, False otherwise
        """
        message = f"Your Sahasra OTP is {token}. Please enter this code to verify your mobile number. The code is valid for 10 minutes. Do not share"
        
        return self.send_sms(
            phone_number, 
            message, 
            self.sms_template_mobile_verification
        )
    
    def send_email_verification_email(self, old_email: str, new_email: str, token: str) -> bool:
        """Send email verification email to the new email address.
        
        Args:
            old_email: Current email address
            new_email: New email address to send verification to
            token: OTP token
            
        Returns:
            True if successful, False otherwise
        """
        try:
            html_content = self._render_template(
                settings.email_verification_template_path,
                {
                    "old_email": old_email,
                    "new_email": new_email,
                    "token": token
                }
            )
            
            return self.send_email(
                to_email=new_email,
                subject="Sahasra AI - Verify Your New Email Address",
                html_content=html_content
            )
        except Exception:
            return False 
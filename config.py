# server/config.py
import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

# Determine the base directory of the server application
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))

class Settings(BaseSettings):
    # Load .env file from the server directory
    model_config = SettingsConfigDict(env_file=os.path.join(SERVER_DIR, '.env'), extra='ignore')

    # MongoDB
    MONGO_URI: str
    MONGO_DATABASE_TOKENS: str = "sahasra_tokens"
    MONGO_COLLECTION_AUTH_TOKENS: str = "auth_tokens"
    MONGO_COLLECTION_REGISTER_TOKENS: str = "register_tokens"
    MONGO_COLLECTION_PASSWORD_TOKENS: str = "password_tokens"
    MONGO_COLLECTION_MOBILE_VERIFICATION_TOKENS: str = "mobile_verification_tokens"
    MONGO_COLLECTION_EMAIL_VERIFICATION_TOKENS: str = "email_verification_tokens"
    MONGO_DATABASE_USERS: str = "sahasra_users"
    MONGO_COLLECTION_USERS: str = "users"
    MONGO_DATABASE_QUESTIONS: str = "sahasra_questions"
    MONGO_COLLECTION_QUESTION_BANK: str = "question_bank"
    MONGO_DATABASE_SUBJECTDATA: str = "sahasra_subjectdata"
    MONGO_COLLECTION_TOPIC_SUBTOPIC: str = "topic_subtopic"
    MONGO_DATABASE_HISTORY: str = "history" # Base name for history DB/collection (used as collection name in LangChain)
    MONGO_HISTORY_SIZE: int = 1

    # JWT
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    AUTH_TOKEN_EXPIRE_SECONDS: int = 7200
    REGISTER_TOKEN_EXPIRE_SECONDS: int = 600 # Matches controller logic (10*60)
    PASSWORD_TOKEN_EXPIRE_SECONDS: int = 600 # Matches controller logic (10*60)
    MOBILE_VERIFICATION_TOKEN_EXPIRE_SECONDS: int = 600 # 10 minutes for mobile verification
    EMAIL_VERIFICATION_TOKEN_EXPIRE_SECONDS: int = 600 # 10 minutes for email verification

    # OpenAI
    OPENAI_API_KEY: str

    # Google AI
    GOOGLE_API_KEY: str
    
    # Gemini API (reusing Google API key by default)
    _GEMINI_API_KEY: str = ""

    # Supabase
    SUPABASE_URL: str
    SUPABASE_KEY: str
    SUPABASE_TABLE_BIOLOGY: str = "biology"
    SUPABASE_TABLE_UIMAGEURL: str = "uimageurl"
    SUPABASE_TABLE_CLASS_X: str = "class_x" # Default subject/table used in getUanswer
    SUPABASE_QUERY_MATCH_SUBJECT: str = "match_subject"
    SUPABASE_QUERY_MATCH_UIMAGEURL: str = "match_u_image_uurl"

    # PostgreSQL / PGVector
    POSTGRES_CONNECTION_STRING: str
    PGVECTOR_CONNECTION_STRING: str
    POSTGRES_POOL_SIZE: int = 20
    POSTGRES_POOL_RECYCLE: int = 3600
    POSTGRES_POOL_PRE_PING: bool = True
    POSTGRES_CONNECT_TIMEOUT: int = 10
    POSTGRES_STATEMENT_TIMEOUT: int = 30000
    # Email (ForwardEmail)
    EMAIL_SENDER: str
    FORWARDEMAIL_API_URL: str

    # SMS (AdwingsSMS)
    ADWINGSSMS_API_URL: str
    ADWINGSSMS_API_KEY: str
    ADWINGSSMS_ENTITY_ID: str
    ADWINGSSMS_SENDER_ID: str
    ADWINGSSMS_TEMPLATE_REGISTER: str
    ADWINGSSMS_TEMPLATE_PASSWORD: str
    ADWINGSSMS_TEMPLATE_MOBILE_VERIFICATION: str

    # Redis (for queue management and caching)
    REDIS_URL: str
    REDIS_QUEUE_PREFIX: str
    REDIS_MAX_WORKERS: int
    REDIS_TASK_TIMEOUT: int
    REDIS_RETRY_DELAY: int = 5
    REDIS_SOCKET_TIMEOUT: int = 5
    REDIS_SOCKET_CONNECT_TIMEOUT: int = 5
    REDIS_HEALTH_CHECK_INTERVAL: int = 30
    REDIS_MAX_RETRIES: int
    PDF_MAX_FILE_SIZE: int = 10 * 1024 * 1024 # 10MB
    PDF_UPLOAD_EXTENSIONS: str
    PDF_MAX_PAGES: int = 10
    
    # File Paths & URLs
    STATIC_DIR: str = "static"
    STATIC_ASSET_BASE_URL: str = "https://aigenix.in/static/" # Used in constructing URLs
    STATIC_IMAGE_DIR: str = "images" # Relative to STATIC_DIR
    DEFAULT_PROFILE_IMAGE_FILENAME: str = "defaultug.png" # Filename of default image
    EMAIL_TEMPLATE_FORGOT_PASSWORD: str = "forgotpasswordU.html"
    EMAIL_TEMPLATE_REGISTER: str = "emailU.html"
    EMAIL_TEMPLATE_EMAIL_VERIFICATION: str = "emailVerificationU.html"
    EMAIL_TEMPLATE_DIR: str = SERVER_DIR # Directory containing email templates

    # CORS
    CORS_ALLOWED_ORIGINS: str = "" # Comma-separated string

    # --- Properties to derive full paths ---

    @property
    def cors_origins_list(self) -> List[str]:
        """Parses the comma-separated string of origins into a list."""
        return [origin.strip() for origin in self.CORS_ALLOWED_ORIGINS.split(',') if origin.strip()]

    @property
    def default_profile_image_path(self) -> str:
        """Gets the full path to the default profile image."""
        return os.path.join(self.static_image_path, self.DEFAULT_PROFILE_IMAGE_FILENAME)
    
    @property
    def default_profile_image_relative_path(self) -> str:
        """Gets the relative path to the default profile image from /static."""
        # Returns path relative to /static directory
        return f"/{self.STATIC_DIR}/{self.STATIC_IMAGE_DIR}/{self.DEFAULT_PROFILE_IMAGE_FILENAME}"
    @property
    def static_dir_path(self) -> str:
        """Gets the full path to the static directory."""
        return os.path.join(SERVER_DIR, self.STATIC_DIR)

    @property
    def static_image_path(self) -> str:
        """Gets the full path to the static image directory."""
        return os.path.join(self.static_dir_path, self.STATIC_IMAGE_DIR)

    @property
    def forgot_password_template_path(self) -> str:
        """Gets the full path to the forgot password email template."""
        return os.path.join(self.EMAIL_TEMPLATE_DIR, self.EMAIL_TEMPLATE_FORGOT_PASSWORD)

    @property
    def register_template_path(self) -> str:
        """Gets the full path to the registration email template."""
        return os.path.join(self.EMAIL_TEMPLATE_DIR, self.EMAIL_TEMPLATE_REGISTER)

    @property
    def email_verification_template_path(self) -> str:
        """Gets the full path to the email verification template."""
        return os.path.join(self.EMAIL_TEMPLATE_DIR, self.EMAIL_TEMPLATE_EMAIL_VERIFICATION)

    @property
    def GEMINI_API_KEY(self) -> str:
        """Gets the Gemini API key, defaulting to GOOGLE_API_KEY if not set."""
        return self._GEMINI_API_KEY or self.GOOGLE_API_KEY
    
    @GEMINI_API_KEY.setter
    def GEMINI_API_KEY(self, value: str):
        self._GEMINI_API_KEY = value


# Create a single, importable settings instance
settings = Settings() 
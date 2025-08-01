#!/usr/bin/env python3
"""
Hierarchical Web Directory Image Question Generator using Gemini

This script processes a hierarchical web directory structure:
- Main URL contains multiple subjects
- Each subject contains multiple topics 
- Each topic contains multiple images

It downloads images, extracts text using Gemini, generates questions, and stores them in MongoDB.

Usage:
    python simple_drive_downloader.py --url "https://codept.in/public/sb_may/"

Requirements:
    pip install beautifulsoup4 requests pymongo google-generativeai python-dotenv
"""

import os
import argparse
import re
import requests
import json
import io
import tempfile
import logging
import time
from urllib.parse import urlparse, parse_qs, unquote
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import asyncio

try:
    from bs4 import BeautifulSoup
    from google import genai
    from pymongo import MongoClient
    from dotenv import load_dotenv
except ImportError:
    print("Error: Required libraries not installed.")
    print("Please run: pip install beautifulsoup4 requests pymongo google-generativeai python-dotenv")
    exit(1)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Back to INFO level for normal operation
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('question_generator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HierarchicalWebDirectoryProcessor:
    def __init__(self):
        self.session = requests.Session()
        
        # Initialize Gemini
        self.gemini_api_key = os.getenv('GOOGLE_API_KEY')
        if not self.gemini_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        self.gemini_client = genai.Client(api_key=self.gemini_api_key)
        
        # Initialize MongoDB
        self.mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
        self.mongo_client = MongoClient(self.mongo_uri)
        self.db = self.mongo_client['x_cbse']
        
        logger.info("Gemini and MongoDB initialized successfully")
    
    def _format_collection_name(self, subject_name: str, topic_name: str) -> str:
        """Format collection name as subject_name_topic_name with safe characters."""
        # Remove special characters and normalize spaces
        safe_subject = re.sub(r'[^\w\s-]', '', subject_name).strip()
        safe_topic = re.sub(r'[^\w\s-]', '', topic_name).strip()
        
        # Replace spaces with underscores and convert to lowercase
        safe_subject = re.sub(r'\s+', '_', safe_subject).lower()
        safe_topic = re.sub(r'\s+', '_', safe_topic).lower()
        
        collection_name = f"{safe_subject}_{safe_topic}"
        logger.debug(f"Formatted collection name: '{subject_name}' + '{topic_name}' -> '{collection_name}'")
        return collection_name
    
    def get_image_files_from_directory(self, directory_url: str) -> List[Dict[str, str]]:
        """
        Get list of image files from a web directory listing.
        """
        try:
            print(f"Getting image file list from: {directory_url}")
            
            response = self.session.get(directory_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for image file links
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
            image_files = []
            
            # Find all links to image files
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                link_text = link.get_text(strip=True)
                
                # Skip parent directory links
                if (href in ['../', '../', '/'] or 
                    'Parent Directory' in link_text or 
                    'UpParent Directory' in link_text or
                    link_text.startswith('Up') or
                    href.endswith('../')):
                    continue
                
                # Check if this is an image file by looking for img with alt="[IMG]" or file extension
                img_tag = link.find('img')
                is_image = (img_tag and img_tag.get('alt') == '[IMG]') or any(href.lower().endswith(ext) for ext in image_extensions)
                
                if is_image:
                    # Construct full URL
                    if href.startswith('http'):
                        full_url = href
                    else:
                        # Extract base domain from directory_url
                        parsed_url = urlparse(directory_url)
                        base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
                        
                        # href already contains the full path from root, so just append to base domain
                        if href.startswith('/'):
                            full_url = f"{base_domain}{href}"
                        else:
                            # If href is relative, append to directory_url
                            base_url = directory_url.rstrip('/')
                            full_url = f"{base_url}/{href}"
                    
                    # Extract filename from link text (more reliable in this format)
                    filename = link_text.strip() if link_text.strip() else href.split('/')[-1]
                    
                    image_files.append({
                        'name': filename,
                        'url': full_url,
                        'relative_path': href
                    })
                    logger.debug(f"Found image: {filename}")
            
            logger.info(f"Found {len(image_files)} image files in directory")
            return image_files
            
        except Exception as e:
            logger.error(f"Error getting image files from directory: {e}")
            return []
    
    def get_subjects_from_main_url(self, main_url: str) -> List[Dict[str, str]]:
        """
        Get list of subject directories from the main URL.
        """
        try:
            logger.info(f"Getting subjects from main URL: {main_url}")
            
            response = self.session.get(main_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            subjects = []
            
            logger.debug(f"Found {len(soup.find_all('a', href=True))} total links on page")
            
            # Look for directory links (indicated by image with alt="Directory")
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                link_text = link.get_text(strip=True)
                
                # Skip parent directory links
                if (href in ['../', '../', '/'] or 
                    'Parent Directory' in link_text or 
                    'UpParent Directory' in link_text or
                    link_text.startswith('Up') or
                    href.endswith('../')):
                    continue
                
                # Check if this is a subject directory by looking for img with alt="Directory"
                img_tag = link.find('img')
                img_alt = img_tag.get('alt') if img_tag else None
                
                logger.debug(f"Processing link: href='{href}', text='{link_text}', img_alt='{img_alt}'")
                
                if img_tag and img_tag.get('alt') == 'Directory' and href:
                    # Extract subject name from the link text
                    subject_name = link_text.strip()
                    if not subject_name and '/' in href:
                        subject_name = unquote(href.strip('/').split('/')[-1])
                    
                    # Construct full subject URL
                    if href.startswith('http'):
                        subject_url = href
                    else:
                        parsed_url = urlparse(main_url)
                        base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
                        if href.startswith('/'):
                            subject_url = f"{base_domain}{href}"
                        else:
                            base_url = main_url.rstrip('/')
                            subject_url = f"{base_url}/{href}"
                    
                    subjects.append({
                        'name': subject_name,
                        'url': subject_url,
                        'relative_path': href
                    })
                    logger.info(f"Found subject: {subject_name}")
            
            logger.info(f"Found {len(subjects)} subjects")
            return subjects
            
        except Exception as e:
            logger.error(f"Error getting subjects from main URL: {e}")
            return []
    
    def get_topics_from_subject_url(self, subject_url: str, subject_name: str) -> List[Dict[str, str]]:
        """
        Get list of topic directories from a subject URL.
        """
        try:
            logger.info(f"Getting topics from subject: {subject_name}")
            
            response = self.session.get(subject_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            topics = []
            
            # Look for directory links and image files
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                link_text = link.get_text(strip=True)
                
                # Skip parent directory links
                if (href in ['../', '../', '/'] or 
                    'Parent Directory' in link_text or 
                    'UpParent Directory' in link_text or
                    link_text.startswith('Up') or
                    href.endswith('../')):
                    continue
                
                # Check if this is a topic directory by looking for img with alt="Directory"
                img_tag = link.find('img')
                if img_tag and img_tag.get('alt') == 'Directory' and href:
                    # Extract topic name from the link text
                    topic_name = link_text.strip()
                    if not topic_name and '/' in href:
                        topic_name = unquote(href.strip('/').split('/')[-1])
                    
                    # Construct full topic URL
                    if href.startswith('http'):
                        topic_url = href
                    else:
                        parsed_url = urlparse(subject_url)
                        base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
                        if href.startswith('/'):
                            topic_url = f"{base_domain}{href}"
                        else:
                            base_url = subject_url.rstrip('/')
                            topic_url = f"{base_url}/{href}"
                    
                    topics.append({
                        'name': topic_name,
                        'url': topic_url,
                        'relative_path': href,
                        'subject_name': subject_name
                    })
                    logger.info(f"Found topic: {topic_name}")
            
            logger.info(f"Found {len(topics)} topics in {subject_name}")
            return topics
            
        except Exception as e:
            logger.error(f"Error getting topics from subject {subject_name}: {e}")
            return []
    
    def download_image_to_bytes(self, image_url: str, filename: str) -> bytes:
        """
        Download an image from URL and return its bytes.
        """
        try:
            logger.debug(f"Downloading image: {filename}")
            
            response = self.session.get(image_url, stream=True)
            response.raise_for_status()
            
            image_bytes = response.content
            logger.debug(f"Downloaded {len(image_bytes)} bytes")
            return image_bytes
            
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            return b''
    
    
    
    async def extract_text_from_image_bytes(self, image_bytes: bytes, file_name: str) -> str:
        """
        Extract text from image bytes using Gemini Vision API with pure IO.
        """
        try:
            if not image_bytes:
                logger.warning(f"No image bytes to process for {file_name}")
                return ""
            
            # Create BytesIO object for Gemini upload
            import io
            image_buffer = io.BytesIO(image_bytes)
            
            # Determine mime type from file extension
            file_ext = Path(file_name).suffix.lower()
            mime_type = 'image/jpeg'  # default
            if file_ext in ['.png']:
                mime_type = 'image/png'
            elif file_ext in ['.gif']:
                mime_type = 'image/gif'
            elif file_ext in ['.webp']:
                mime_type = 'image/webp'
            
            # Upload image bytes directly to Gemini using client
            uploaded_file = self.gemini_client.files.upload(
                file=image_buffer,
                config=dict(mime_type=mime_type, display_name=file_name)
            )
            
            # Create prompt for text extraction
            text_extraction_prompt = """
            Extract ALL text content from this image. 
            If this appears to be an educational document, question paper, or study material:
            - Extract all questions exactly as they appear
            - Include all answer choices/options
            - Preserve formatting and numbering
            - Include any instructions or headers
            - Extract all explanations or solutions if present
            
            Return the extracted text in a clear, readable format.
            """
            
            # Generate text extraction using Gemini 2.5 Pro
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[uploaded_file, text_extraction_prompt],
            )
            
            extracted_text = response.text
            
            # Clean up - delete uploaded file and close buffer
            try:
                self.gemini_client.files.delete(name=uploaded_file.name)
            except Exception:
                pass
            
            image_buffer.close()
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"Error extracting text from {file_name}: {e}")
            return ""
    
    async def generate_questions_from_text(
        self, 
        extracted_text: str, 
        subject: str, 
        topic: str, 
        subtopic: str
    ) -> List[Dict[str, Any]]:
        """
        Generate questions from extracted text using Gemini.
        """
        try:
            if not extracted_text.strip():
                logger.warning("No text extracted, skipping question generation")
                return []
            
            # Get current timestamp for all questions
            current_timestamp = datetime.now().isoformat()
            
            # Create prompt for question generation
            prompt = self._create_question_generation_prompt(subject, topic, subtopic, current_timestamp, extracted_text)
            
            # Generate content using Gemini 2.5 Pro
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[prompt],
            )
            raw_response_content = response.text
            
            # Clean and format the JSON response using Gemini
            cleaned_json = await self._clean_json_with_gemini(raw_response_content)
            
            # Parse and process the response
            questions = self._parse_response(cleaned_json, subject, topic, subtopic, current_timestamp)
            
            return questions
            
        except Exception as e:
            logger.error(f"Error generating questions from text: {e}")
            return []
    
    def _create_question_generation_prompt(self, subject: str, topic: str, subtopic: str, current_timestamp: str, extracted_text: str) -> str:
        """Create the prompt for question generation from extracted text."""
        return f"""
Analyze the provided text content extracted from an image and EXTRACT questions from its content.
For EACH question identified in the text, generate a JSON object.
Determine the question_type from: "MCQ", "VERY_SHORT_ANSWER", "SHORT_ANSWER", "LONG_ANSWER", "CASE_STUDY", "TRUEFALSE".
The output should be a JSON array containing these objects.
Write questions and options in LaTeX if needed. Do NOT include an "_id" field in your response.

TEXT CONTENT TO ANALYZE:
{extracted_text}

IMPORTANT: Make sure to extract all the questions from the text and classify them appropriately:
- MCQ: Has distinct options (A, B, C, D)
- VERY_SHORT_ANSWER: Requires 1-3 words (definitions, terms)
- SHORT_ANSWER: Requires 1-3 sentences (brief explanations)
- LONG_ANSWER: Requires detailed explanations (multiple paragraphs)
- CASE_STUDY: Scenario-based application questions
- TRUEFALSE: True/False questions

**General Instructions for ALL questions:**
- "questionset" MUST be "scanned".
- "created_at" MUST use the provided timestamp: {current_timestamp}.
- "question_image" and "explaination_image" MUST be empty strings "" for now.
- `subject`, `topic`, and `subtopic` will be provided as context variables; use their values.
- `level`: Determine the difficulty level for this question [1-3] (1 Easy, 2 Medium, 3 Hard).
- `marks`: Extract the marks allocated for this question, if visible. If not visible, suggest a reasonable default (e.g., 1 for simple MCQs, 2-5 for descriptive, based on complexity).
- `grading_criteria`: Must not be empty.
- `model_answer`: Must not be empty.

If the question is primarily multiple-choice (has distinct options), use this format:
{{
  "question": "Extract the main question text from the document for this specific question (in LaTeX if needed)",
  "option1": "You must extract the first multiple-choice option for this question (in LaTeX if needed)",
  "option2": "You must extract the second multiple-choice option for this question (in LaTeX if needed)",
  "option3": "You must extract the third multiple-choice option for this question (in LaTeX if needed)",
  "option4": "You must extract the fourth multiple-choice option for this question (in LaTeX if needed)",
  "correctanswer": "Identify and provide the correct answer label (e.g., 'option1', 'option2', 'option3', or 'option4'). This field MUST be filled.",
  "explaination": "Provide a detailed explanation for why the correct answer is correct and, if applicable, why the other options are incorrect.  or if present in the document, provide the explanation from the document. This field MUST be filled.",
  "grading_criteria": "Typically for MCQs: 'Full marks if the correct option is selected, zero otherwise.' or similar based on the question's nature. This field MUST be filled.",
  "question_image": "",
  "question_type": "MCQ",
  "explaination_image": "",
  "subject": "{subject}",
  "topic": "{topic}",
  "subtopic": "{subtopic}",
  "level": "Determine the difficulty level for this question [1-3] 1 Easy , 2 Medium, 3 Hard",
  "questionset": "scanned",
  "marks": "Extract the marks allocated for this question, if visible. If not, suggest a default like 1.",
  "created_at": "{current_timestamp}",
  "ignore": true
}}

If the question requires a VERY SHORT ANSWER (1-3 words), use this format:
{{
  "question": "Extract the main question text from the document for this specific question (in LaTeX if needed)",
  "model_answer": "Expected very short answer (1-3 words or brief phrase)",
  "grading_criteria": "Full marks for exact/equivalent term, partial marks for close answers, zero for incorrect",
  "explaination": "Brief explanation about the expected answer and key concept",
  "question_image": "",
  "explaination_image": "",
  "question_type": "VERY_SHORT_ANSWER",
  "subject": "{subject}",
  "topic": "{topic}",
  "subtopic": "{subtopic}",
  "level": "Determine the difficulty level for this question [1-3] 1 Easy , 2 Medium, 3 Hard",
  "questionset": "scanned",
  "marks": "Extract the marks allocated for this question, if visible. If not, suggest default like 1.",
  "created_at": "{current_timestamp}",
  "ignore": true
}}

If the question requires a SHORT ANSWER (1-3 sentences), use this format:
{{
  "question": "Extract the main question text from the document for this specific question (in LaTeX if needed)",
  "model_answer": "Brief but complete answer (1-3 sentences covering key points)",
  "grading_criteria": "Key points breakdown: main concept (X marks), supporting detail (Y marks), clarity (Z marks)",
  "explaination": "Brief explanation about the question and key concepts tested",
  "question_image": "",
  "explaination_image": "",
  "question_type": "SHORT_ANSWER",
  "subject": "{subject}",
  "topic": "{topic}",
  "subtopic": "{subtopic}",
  "level": "Determine the difficulty level for this question [1-3] 1 Easy , 2 Medium, 3 Hard",
  "questionset": "scanned",
  "marks": "Extract the marks allocated for this question, if visible. If not, suggest default like 3.",
  "created_at": "{current_timestamp}",
  "ignore": true
}}

If the question requires a LONG ANSWER (detailed explanation), use this format:
{{
  "question": "Extract the main question text from the document for this specific question (in LaTeX if needed)",
  "model_answer": "Comprehensive detailed answer with multiple key points and examples",
  "grading_criteria": "Detailed breakdown: concept understanding (X marks), examples/evidence (Y marks), analysis/evaluation (Z marks), structure/clarity (W marks)",
  "explaination": "Explanation of the depth and scope expected in the answer",
  "question_image": "",
  "explaination_image": "",
  "question_type": "LONG_ANSWER",
  "subject": "{subject}",
  "topic": "{topic}",
  "subtopic": "{subtopic}",
  "level": "Determine the difficulty level for this question [1-3] 1 Easy , 2 Medium, 3 Hard",
  "questionset": "scanned",
  "marks": "Extract the marks allocated for this question, if visible. If not, suggest default like 5.",
  "created_at": "{current_timestamp}",
  "ignore": true
}}

If the question is a CASE STUDY (scenario-based application), use this format:
{{
  "question": "Extract the scenario and question text from the document (in LaTeX if needed)",
  "model_answer": "Comprehensive case analysis with problem identification, theoretical application, and practical solutions",
  "grading_criteria": "Case analysis (X marks), theoretical application (Y marks), practical solutions (Z marks), justification (W marks)",
  "explaination": "Overview of the case scenario and key learning objectives",
  "question_image": "",
  "explaination_image": "",
  "question_type": "CASE_STUDY",
  "subject": "{subject}",
  "topic": "{topic}",
  "subtopic": "{subtopic}",
  "level": "Determine the difficulty level for this question [1-3] 1 Easy , 2 Medium, 3 Hard",
  "questionset": "scanned",
  "marks": "Extract the marks allocated for this question, if visible. If not, suggest default like 10.",
  "created_at": "{current_timestamp}",
  "ignore": true
}}

If the question is a TRUEFALSE question, use this format:
{{
  "question": "Extract the question text from the document (in LaTeX if needed)",
  "correctanswer": "True or False",
  "explaination": "Brief explanation about the question and key concepts tested",
  "question_image": "",
  "explaination_image": "",
  "question_type": "TRUEFALSE",
  "subject": "{subject}",
  "topic": "{topic}",
  "subtopic": "{subtopic}",
  "level": "Determine the difficulty level for this question [1-3] 1 Easy , 2 Medium, 3 Hard",
  "questionset": "scanned",
  "marks": "Extract the marks allocated for this question, if visible. If not, suggest default like 1.",
  "created_at": "{current_timestamp}",
  "ignore": true
}}

Return a JSON array where each element is a JSON object formatted as described above.
If no questions are found, return an empty array [].
Ensure the output is a single valid JSON array string.
RETURN ONLY JSON NO ADDITIONAL TEXT OR COMMENTS MAKE SURE YOU GET ALL THE QUESTIONS AT ONCe
IMPORTANT: Make sure the FINAL JSON STRUCTURE is PARSABLE with json.loads()
IMPORTANT: Make sure the final JSON has only utf-8 encoded characters.
IMPORTANT: Make sure to Escape all backslashes and other charecters which could cause errors in JSON parsing.
IMPORTANT: FOR MCQ Do not include the options in the question keep them in the options field.
IMPORTANT: FOR all text-based questions (VERY_SHORT_ANSWER, SHORT_ANSWER, LONG_ANSWER, CASE_STUDY) model_answer and grading_criteria MUST NOT BE EMPTY.
"""
    
    async def _clean_json_with_gemini(self, raw_json_text: str) -> str:
        """Clean and format JSON response using Gemini to ensure it's parsable."""
        try:
            cleaning_prompt = f"""
You are a JSON formatter and validator. Your task is to take the provided text and:

1. Extract ONLY the JSON array from the text (ignore any surrounding text, explanations, or markdown)
2. Fix any JSON formatting issues (missing commas, quotes, brackets, etc.)
3. Properly escape all special characters (backslashes, quotes, etc.) that could cause JSON parsing errors
4. Ensure all strings are properly quoted
5. Remove any invalid characters or syntax errors
6. Return ONLY the clean, valid JSON array - no additional text, explanations, or markdown formatting

Input text to clean:
{raw_json_text}

Return only the cleaned JSON array:
"""
            
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[cleaning_prompt],
            )
            
            cleaned_json = response.text.strip()
            
            if cleaned_json.startswith("```json"):
                cleaned_json = cleaned_json[7:]
            if cleaned_json.startswith("```"):
                cleaned_json = cleaned_json[3:]
            if cleaned_json.endswith("```"):
                cleaned_json = cleaned_json[:-3]
            
            cleaned_json = cleaned_json.strip()
            logger.debug(f"CLEANED JSON LENGTH: {len(cleaned_json)}")
            
            return cleaned_json
            
        except Exception as e:
            logger.error(f"Error cleaning JSON with Gemini: {str(e)}")
            return raw_json_text
    
    def _parse_response(
        self, 
        response_text: str, 
        subject: str, 
        topic: str, 
        subtopic: str, 
        current_timestamp: str
    ) -> List[Dict[str, Any]]:
        """Parse the Gemini response and process the questions."""
        try:
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            logger.debug(f"RAW RESPONSE: {response_text[:500]}...")

            list_of_generated_json = json.loads(response_text)
            processed_questions = []
            
            if isinstance(list_of_generated_json, list):
                for item_json in list_of_generated_json:
                    # Common fields enforced for all types
                    item_json["subject"] = subject
                    item_json["topic"] = topic
                    item_json["subtopic"] = subtopic
                    item_json["questionset"] = "scanned"
                    item_json["created_at"] = current_timestamp  # Always empty as per requirement
                    
                    question_type = str(item_json.get("question_type", "")).upper()
                    
                    if question_type in ["VERY_SHORT_ANSWER", "SHORT_ANSWER", "LONG_ANSWER", "CASE_STUDY"]:
                        item_json["question_type"] = question_type
                        
                        if "model_answer" not in item_json:
                            item_json["model_answer"] = ""
                        if "grading_criteria" not in item_json:
                            item_json["grading_criteria"] = ""
                        
                        mcq_keys = ["option1", "option2", "option3", "option4", "correctanswer"]
                        for key in mcq_keys:
                            item_json.pop(key, None)
                            
                        if question_type == "VERY_SHORT_ANSWER" and item_json.get("marks", "") == "":
                            item_json["marks"] = "1"
                        elif question_type == "SHORT_ANSWER" and item_json.get("marks", "") == "":
                            item_json["marks"] = "3"
                        elif question_type == "LONG_ANSWER" and item_json.get("marks", "") == "":
                            item_json["marks"] = "5"
                        elif question_type == "CASE_STUDY" and item_json.get("marks", "") == "":
                            item_json["marks"] = "10"
                    else:
                        if question_type not in ["MCQ", "SINGLE_SELECT_MCQ"]:
                            if any(opt_key in item_json for opt_key in ["option1", "option2", "option3", "option4"]):
                                item_json["question_type"] = "MCQ"
                            else:
                                item_json["question_type"] = "MCQ"
                        if question_type == "TRUEFALSE" and "correctanswer" not in item_json:
                            item_json["correctanswer"] = ""
                            
                        if "correctanswer" not in item_json:
                            item_json["correctanswer"] = ""
                        
                        text_based_keys = ["model_answer", "grading_criteria"]
                        for key in text_based_keys:
                            item_json.pop(key, None)
                    
                    processed_questions.append(item_json)
                
            return processed_questions
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing response: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            return []
    
    def insert_questions_to_db(self, questions: List[Dict[str, Any]], subject_name: str, topic_name: str) -> int:
        """Insert questions into MongoDB in the appropriate collection and return count of successful insertions."""
        try:
            if not questions:
                return 0
            
            # Get the collection for this subject and topic
            collection_name = self._format_collection_name(subject_name, topic_name)
            collection = self.db[collection_name]
            
            logger.info(f"Inserting {len(questions)} questions into collection: {collection_name}")
            
            successful_count = 0
            for question in questions:
                try:
                    question.pop("_id", None)
                    result = collection.insert_one(question)
                    if result.inserted_id:
                        successful_count += 1
                        logger.debug(f"Inserted question: {question.get('question', 'Unknown')[:50]}...")
                    else:
                        logger.warning(f"Failed to insert question: {question.get('question', 'Unknown')}")
                        
                except Exception as e:
                    logger.error(f"Error inserting question: {e}")
                    continue
            
            logger.info(f"Successfully inserted {successful_count}/{len(questions)} questions into {collection_name}")
            return successful_count
            
        except Exception as e:
            logger.error(f"Error inserting questions to database: {e}")
            return 0
    
    async def process_hierarchical_structure(self, main_url: str, resume_from: Dict[str, str] = None):
        """
        Main method to process the complete hierarchical structure:
        1. Get all subjects from main URL
        2. For each subject, get all topics
        3. For each topic, get all images and process them
        """
        try:
            logger.info(f"Starting hierarchical processing from: {main_url}")
            start_time = time.time()
            
            # Initialize progress tracking
            total_questions_generated = 0
            total_images_processed = 0
            total_errors = 0
            
            # Resume handling
            resume_subject = resume_from.get('subject') if resume_from else None
            resume_topic = resume_from.get('topic') if resume_from else None
            
            # Get all subjects
            subjects = self.get_subjects_from_main_url(main_url)
            if not subjects:
                logger.error("No subjects found in main URL")
                return
            
            logger.info(f"Found {len(subjects)} subjects to process")
            
            for subject_idx, subject in enumerate(subjects, 1):
                subject_name = subject['name']
                subject_url = subject['url']
                
                # Resume logic: skip subjects before resume point
                if resume_subject and subject_name != resume_subject:
                    logger.info(f"Skipping subject {subject_name} (resuming from {resume_subject})")
                    continue
                
                logger.info(f"\n[{subject_idx}/{len(subjects)}] Processing subject: {subject_name}")
                
                try:
                    # Get all topics for this subject
                    topics = self.get_topics_from_subject_url(subject_url, subject_name)
                    
                    if not topics:
                        logger.warning(f"No topics found for subject: {subject_name}")
                        continue
                    
                    logger.info(f"Found {len(topics)} topics in {subject_name}")
                    
                    for topic_idx, topic in enumerate(topics, 1):
                        topic_name = topic['name']
                        topic_url = topic['url']
                        
                        # Resume logic: skip topics before resume point
                        if resume_topic and resume_subject == subject_name and topic_name != resume_topic:
                            logger.info(f"Skipping topic {topic_name} (resuming from {resume_topic})")
                            continue
                        
                        logger.info(f"\n[{topic_idx}/{len(topics)}] Processing topic: {topic_name}")
                        
                        try:
                            # Process this topic's images
                            topic_stats = await self.process_topic_images(
                                topic_url, 
                                subject_name, 
                                topic_name
                            )
                            
                            # Update overall statistics
                            total_questions_generated += topic_stats['questions']
                            total_images_processed += topic_stats['images']
                            total_errors += topic_stats['errors']
                            
                            logger.info(f"Topic {topic_name} completed: "
                                      f"{topic_stats['questions']} questions, "
                                      f"{topic_stats['images']} images")
                            
                            # Clear resume topic after successful processing
                            if resume_topic and resume_subject == subject_name and topic_name == resume_topic:
                                resume_topic = None
                        
                        except Exception as e:
                            logger.error(f"Failed to process topic {topic_name} in {subject_name}: {e}")
                            total_errors += 1
                            
                            # Log the exact failure point for resume
                            logger.error(f"To resume from this point, use: --resume-subject '{subject_name}' --resume-topic '{topic_name}'")
                            continue
                    
                    # Clear resume subject after successful processing
                    if resume_subject == subject_name:
                        resume_subject = None
                        
                except Exception as e:
                    logger.error(f"Failed to process subject {subject_name}: {e}")
                    total_errors += 1
                    
                    # Log the exact failure point for resume
                    logger.error(f"To resume from this point, use: --resume-subject '{subject_name}'")
                    continue
            
            # Final statistics
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"\nHierarchical processing complete!")
            logger.info(f"Total time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            logger.info(f"Total statistics:")
            logger.info(f"Subjects processed: {len(subjects)}")
            logger.info(f"Questions generated: {total_questions_generated}")
            logger.info(f"Images processed: {total_images_processed}")
            logger.info(f"Errors encountered: {total_errors}")
            
            if total_images_processed > 0:
                logger.info(f"Avg questions per image: {total_questions_generated/total_images_processed:.2f}")
            
        except Exception as e:
            logger.error(f"Fatal error in hierarchical processing: {e}")
        finally:
            if hasattr(self, 'mongo_client'):
                self.mongo_client.close()
    
    async def process_topic_images(self, topic_url: str, subject_name: str, topic_name: str) -> Dict[str, int]:
        """
        Process all images in a specific topic directory.
        Returns statistics dictionary with counts.
        """
        try:
            stats = {'questions': 0, 'images': 0, 'errors': 0}
            
            # Get list of image files from topic directory
            image_files = self.get_image_files_from_directory(topic_url)
            
            if not image_files:
                logger.warning(f"No image files found in topic: {topic_name}")
                return stats
            
            logger.info(f"Found {len(image_files)} images in {topic_name}")
            
            # Process each image file individually
            for img_idx, file_info in enumerate(image_files, 1):
                file_name = file_info['name']
                image_url = file_info['url']
                
                logger.info(f"[{img_idx}/{len(image_files)}] Processing: {file_name}")
                
                try:
                    # Download individual file to bytes
                    file_bytes = self.download_image_to_bytes(image_url, file_name)
                    
                    if not file_bytes:
                        logger.warning(f"Failed to download {file_name}")
                        stats['errors'] += 1
                        continue
                    
                    # Extract text from image bytes
                    extracted_text = await self.extract_text_from_image_bytes(file_bytes, file_name)
                    
                    if extracted_text:
                        logger.info(f"Extracted {len(extracted_text)} characters of text")
                        
                        # Generate questions from extracted text
                        questions = await self.generate_questions_from_text(
                            extracted_text,
                            subject=subject_name.lower().replace(' ', '_'),
                            topic=topic_name,
                            subtopic=""
                        )
                        
                        if questions:
                            # Add source image information to each question
                            for question in questions:
                                question['source_image_url'] = image_url
                                question['source_image_view_url'] = image_url
                                question['source_image_name'] = file_name
                                question['source_folder_name'] = topic_name
                                question['source_subject_name'] = subject_name
                                question['source_topic_url'] = topic_url
                            
                            # Insert questions immediately
                            inserted_count = self.insert_questions_to_db(questions, subject_name, topic_name)
                            stats['questions'] += inserted_count
                            logger.info(f"Generated and stored {inserted_count} questions from {file_name}")
                        else:
                            logger.warning(f"No questions generated from {file_name}")
                            stats['errors'] += 1
                    else:
                        logger.warning(f"No text extracted from {file_name}")
                        stats['errors'] += 1
                    
                    stats['images'] += 1
                    
                    # Clear file bytes from memory
                    del file_bytes
                    
                except Exception as e:
                    logger.error(f"Error processing {file_name}: {e}")
                    stats['errors'] += 1
                    continue
            
            return stats
            
        except Exception as e:
            logger.error(f"Error processing topic {topic_name}: {e}")
            return {'questions': 0, 'images': 0, 'errors': 1}
    
    async def process_directory_images(self, directory_url: str):
        """
        Main method to process images from a web directory one file at a time:
        1. Get list of image files from directory
        2. For each image file, download bytes and process individually
        3. Store questions immediately after each file
        """
        try:
            logger.info(f"Processing directory: {directory_url}")
            
            # Extract subject and topic from URL path
            subject = "social_science"  # Default subject
            topic = "NATIONALISM IN INDIA"  # Default topic
            try:
                # Try to extract subject and topic from URL path
                path_parts = unquote(directory_url).split('/')
                
                # Find subject (looks for x_subject_name pattern)
                for part in path_parts:
                    if part.startswith('x_') and part not in ['', 'public', 'sb_may']:
                        subject = part[2:]  # Remove 'x_' prefix
                        break
                
                # Find topic (last meaningful part of path)
                for part in reversed(path_parts):
                    if part and part not in ['public', 'sb_may'] and not part.startswith('x_'):
                        topic = part.replace('%20', ' ')
                        break
            except:
                pass
            
            logger.info(f"Extracted - Subject: '{subject}', Topic: '{topic}'")
            
            # Get list of image files from directory
            image_files = self.get_image_files_from_directory(directory_url)
            
            if not image_files:
                logger.warning("No image files found in directory")
                return
            
            total_questions_generated = 0
            
            # Process each file individually using IO bytes
            for i, file_info in enumerate(image_files, 1):
                file_name = file_info['name']
                image_url = file_info['url']
                
                logger.info(f"\n[{i}/{len(image_files)}] Processing: {file_name}")
                
                # Download individual file to bytes
                file_bytes = self.download_image_to_bytes(image_url, file_name)
                
                if not file_bytes:
                    logger.warning(f"Failed to download {file_name}")
                    continue
                
                # Extract text from image bytes
                extracted_text = await self.extract_text_from_image_bytes(file_bytes, file_name)
                
                if extracted_text:
                    logger.info(f"Extracted {len(extracted_text)} characters of text")
                    
                    # Generate questions from extracted text
                    questions = await self.generate_questions_from_text(
                        extracted_text,
                        subject="social_science",  # Based on URL path
                        topic=topic,
                        subtopic=""
                    )
                    
                    if questions:
                        # Add source image information to each question before insertion
                        for question in questions:
                            question['source_image_url'] = image_url           # Direct download URL
                            question['source_image_view_url'] = image_url      # Same as download URL
                            question['source_image_name'] = file_name
                            question['source_folder_name'] = topic
                            question['source_directory_url'] = directory_url   # Source directory
                        
                        # Insert questions immediately
                        inserted_count = self.insert_questions_to_db(questions, subject, topic)
                        total_questions_generated += inserted_count
                        logger.info(f"Generated and stored {inserted_count} questions from {file_name}")
                    else:
                        logger.warning(f"No questions generated from {file_name}")
                else:
                    logger.warning(f"No text extracted from {file_name}")
                
                # Clear file bytes from memory
                del file_bytes
            
            logger.info(f"\nProcessing complete!")
            logger.info(f"Total questions generated and stored: {total_questions_generated}")
            logger.info(f"Files processed: {len(image_files)}")
            logger.info(f"Topic: {topic}")
            
        except Exception as e:
            logger.error(f"Error processing directory images: {e}")
        finally:
            if hasattr(self, 'mongo_client'):
                self.mongo_client.close()

async def main():
    parser = argparse.ArgumentParser(
        description="Generate questions from images in hierarchical web directory structure using Gemini AI"
    )
    parser.add_argument(
        '--url',
        required=True,
        help='Main URL containing subject directories (e.g., https://codept.in/public/sb_may/)'
    )
    parser.add_argument(
        '--resume-subject',
        help='Subject name to resume processing from (for error recovery)'
    )
    parser.add_argument(
        '--resume-topic',
        help='Topic name to resume processing from (requires --resume-subject)'
    )
    parser.add_argument(
        '--single-directory',
        action='store_true',
        help='Process a single directory of images instead of hierarchical structure'
    )
    
    args = parser.parse_args()
    
    try:
        processor = HierarchicalWebDirectoryProcessor()
        
        # Validate resume arguments
        if args.resume_topic and not args.resume_subject:
            logger.error("--resume-topic requires --resume-subject to be specified")
            return
        
        resume_from = None
        if args.resume_subject:
            resume_from = {'subject': args.resume_subject}
            if args.resume_topic:
                resume_from['topic'] = args.resume_topic
            logger.info(f"Resuming from: {resume_from}")
        
        if args.single_directory:
            logger.info(f"Processing single directory: {args.url}")
            await processor.process_directory_images(args.url)
        else:
            logger.info(f"Processing hierarchical structure: {args.url}")
            await processor.process_hierarchical_structure(args.url, resume_from)
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print("\nMake sure you have set the following environment variables:")
        print("- GOOGLE_API_KEY: Your Google Gemini API key")
        print("- MONGODB_URI: Your MongoDB connection string (optional, defaults to localhost)")
    except Exception as e:
        logger.error(f"Fatal error: {e}")

if __name__ == '__main__':
    asyncio.run(main()) 
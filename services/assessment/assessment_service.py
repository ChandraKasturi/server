import json
import dateutil.parser
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional, Any
import uuid

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings
from repositories.mongo_repository import HistoryRepository, QuestionRepository
from repositories.postgres_text_repository import PostgresTextRepository
from repositories.pdf_repository import PDFRepository
from services.langchain.langchain_service import LangchainService
from models.pdf_models import ProcessingStatus, QuestionType
from langchain.embeddings import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector

class AssessmentService:
    """Service for handling assessment-related operations."""
    
    def __init__(self):
        """Initialize assessment service."""
        self.history_repo = HistoryRepository()
        self.question_repo = QuestionRepository()
        self.langchain_service = LangchainService()
        self.pdf_repository = PDFRepository()
        self.postgres_text_repository = PostgresTextRepository()
        self.llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY)
        self.ug_llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY, model="gpt-4o")
        '''self.llm = ChatGoogleGenerativeAI(google_api_key=settings.GOOGLE_API_KEY, model="gemini-2.0-flash")
        self.ug_llm = ChatGoogleGenerativeAI(google_api_key=settings.GOOGLE_API_KEY, model="gemini-2.0-flash")'''
    
    def check_assessment_content(self, subject: str, topic: str = None, topics: List[str] = None, subtopic: Optional[str] = None, 
                            level: int = 1, num_questions: int = 5, 
                            session_id: Optional[str] = None, student_id: str = None,
                            question_types: List[str] = None) -> Tuple[Dict, int]:
        """Generate assessment questions based on the provided parameters.
        
        Args:
            subject: Subject for the questions
            topic: Single topic within the subject (legacy parameter, use topics instead)
            topics: List of topics within the subject
            subtopic: Optional subtopic within the topic
            level: Difficulty level (1-5)
            num_questions: Number of questions to generate
            session_id: JWT token from X-Auth-Session to use as conversation ID
            student_id: ID of the student
            question_types: List of question types to generate (e.g., ["MCQ", "DESCRIPTIVE"])
                           If None or empty, defaults to ["MCQ"]
            
        Returns:
            Tuple of (result_data, status_code)
        """
        try:
            # Handle both single topic and topics list
            topic_list = []
            if topics:
                topic_list = topics
            elif topic:
                topic_list = [topic]
                
            if not topic_list:
                return {"message": "At least one topic must be provided"}, 400
                
            # Default to MCQ if no question types are provided
            if not question_types:
                question_types = ["MCQ"]
            
            # Convert all question types to uppercase for consistency
            question_types = [qt.upper() for qt in question_types]
            
            # Validate question types
            valid_types = ["MCQ", "DESCRIPTIVE", "FILL_BLANKS", "TRUEFALSE"]
            question_types = [qt for qt in question_types if qt in valid_types]
            
            # Default to MCQ if all provided types were invalid
            if not question_types:
                question_types = ["MCQ"]
            
            # Find matching questions in the database
            db_questions = []
            
            # Distribute questions evenly among topics
            questions_per_topic = self._distribute_questions_among_topics(topic_list, num_questions)
            
            # Get questions for each topic
            for topic_name, topic_question_count in questions_per_topic.items():
                # Find matching questions for this topic
                base_query = {
                    "subject": subject,
                    "topic": topic_name,
                }
                
                if subtopic:
                    base_query["subtopic"] = subtopic
                    
                if level:
                    base_query["level"] = str(level)
                
                # If we have only one question type, fetch all questions of that type
                if len(question_types) == 1:
                    query = base_query.copy()
                    query["question_type"] = question_types[0]
                    
                    topic_questions = self.question_repo.find_questions(
                        query=query,
                        limit=topic_question_count
                    )
                    db_questions.extend(topic_questions)
                else:
                    # For multiple types, distribute questions evenly among types
                    import random
                    
                    # Shuffle the types to ensure a random order
                    types_to_use = question_types.copy()
                    random.shuffle(types_to_use)
                    
                    # Calculate questions per type (at least 1 per type if possible)
                    questions_per_type = {}
                    remaining = topic_question_count
                    
                    # Initial distribution - at least 1 per type if possible
                    for q_type in types_to_use:
                        if remaining > 0:
                            questions_per_type[q_type] = 1
                            remaining -= 1
                        else:
                            questions_per_type[q_type] = 0
                    
                    # Distribute remaining questions randomly
                    while remaining > 0:
                        q_type = random.choice(types_to_use)
                        questions_per_type[q_type] += 1
                        remaining -= 1
                    
                    # Fetch questions for each type according to distribution
                    for q_type, count in questions_per_type.items():
                        if count > 0:
                            query = base_query.copy()
                            query["question_type"] = q_type
                            
                            type_questions = self.question_repo.find_questions(
                                query=query,
                                limit=count
                            )
                            
                            db_questions.extend(type_questions)
            
            # If not enough questions, generate some using LLM
            if len(db_questions) < num_questions:
                # Number of questions to generate
                num_to_generate = num_questions - len(db_questions)
                
                # Distribute remaining questions to generate among topics
                remaining_questions_per_topic = self._distribute_questions_among_topics(
                    topic_list, 
                    num_to_generate
                )
                
                # Generate questions for each topic
                for topic_name, topic_question_count in remaining_questions_per_topic.items():
                    if topic_question_count > 0:
                        subtopic_to_use = subtopic or topic_name
                        generated_questions = self._generate_questions(
                            subject, topic_name, subtopic_to_use, level, topic_question_count, question_types
                        )
                        db_questions.extend(generated_questions)
            
            # Create assessment object
            assessment = {
                "questions": db_questions,
                "student_id": student_id,
                "timestamp": datetime.utcnow(),
                "session_id": session_id,
                "question_types": question_types,
                "subject": subject,
                "topics": topic_list,  # Store all topics
                "level": level
            }
            
            # Store assessment
            assessment_id = self.history_repo.add_assessment(student_id, assessment)
            
            # Add this interaction to chat history if session_id is provided
            if session_id:
                self.langchain_service.add_to_chat_history(
                    student_id, 
                    session_id, 
                    f"Generated assessment with ID: {assessment_id}"
                )
            
            # Return assessment data
            return {
                "message": "Assessment generated successfully",
                "assessment_id": assessment_id,
                "questions": db_questions
            }, 200
            
        except Exception as e:
            return {"message": f"Error processing assessment request: {str(e)}"}, 500
    
    def _distribute_questions_among_topics(self, topics: List[str], num_questions: int) -> Dict[str, int]:
        """Distribute questions evenly among multiple topics.
        
        Args:
            topics: List of topics
            num_questions: Total number of questions to distribute
            
        Returns:
            Dictionary mapping topics to number of questions
        """
        import random
        
        # Create a copy of topics to avoid modifying the original
        topics_to_use = topics.copy()
        random.shuffle(topics_to_use)
        
        # Initial distribution - assign at least 1 question per topic if possible
        questions_per_topic = {}
        remaining = num_questions
        
        for topic in topics_to_use:
            if remaining > 0:
                questions_per_topic[topic] = 1
                remaining -= 1
            else:
                questions_per_topic[topic] = 0
        
        # Distribute remaining questions randomly
        while remaining > 0:
            topic = random.choice(topics_to_use)
            questions_per_topic[topic] += 1
            remaining -= 1
            
        return questions_per_topic
    
    def _generate_questions(self, subject: str, topic: str, subtopic: str, level: int, 
                           num_questions: int, question_types: List[str] = None) -> List[Dict]:
        """Generate questions using AI.
        
        Args:
            subject: Subject of the questions
            topic: Topic of the questions
            subtopic: Subtopic of the questions
            level: Difficulty level of the questions
            num_questions: Number of questions to generate
            question_types: List of question types to generate (e.g., ["MCQ", "DESCRIPTIVE"])
            
        Returns:
            List of generated questions
        """
        # Default to MCQ if no question types are provided
        if not question_types:
            question_types = ["MCQ"]
            
        # Convert all question types to uppercase for consistency
        question_types = [qt.upper() for qt in question_types]
        
        # Validate question types
        valid_types = ["MCQ", "DESCRIPTIVE", "FILL_BLANKS", "TRUEFALSE"]
        question_types = [qt for qt in question_types if qt in valid_types]
        
        # Default to MCQ if all provided types were invalid
        if not question_types:
            question_types = ["MCQ"]
        
        # Generate questions for all the requested types
        all_questions = []
        
        # If we have only one question type, generate all questions of that type
        if len(question_types) == 1:
            all_questions = self._generate_questions_of_type(
                subject, topic, subtopic, level, num_questions, question_types[0]
            )
        else:
            # Distribute questions among the types
            # We'll use a simple random approach to distribute the questions
            import random
            
            # Shuffle the types to ensure a random distribution
            types_to_use = question_types.copy()
            random.shuffle(types_to_use)
            
            questions_per_type = {}
            remaining = num_questions
            
            # Initial distribution - at least 1 per type if possible
            for q_type in types_to_use:
                if remaining > 0:
                    questions_per_type[q_type] = 1
                    remaining -= 1
                else:
                    questions_per_type[q_type] = 0
            
            # Distribute remaining questions randomly
            while remaining > 0:
                q_type = random.choice(types_to_use)
                questions_per_type[q_type] += 1
                remaining -= 1
            
            # Generate questions for each type
            for q_type, count in questions_per_type.items():
                if count > 0:
                    type_questions = self._generate_questions_of_type(
                        subject, topic, subtopic, level, count, q_type
                    )
                    all_questions.extend(type_questions)
                    
        return all_questions
    
    def _generate_questions_of_type(self, subject: str, topic: str, subtopic: str, 
                                   level: int, num_questions: int, question_type: str) -> List[Dict]:
        """Generate questions of a specific type.
        
        Args:
            subject: Subject of the questions
            topic: Topic of the questions
            subtopic: Subtopic of the questions
            level: Difficulty level of the questions
            num_questions: Number of questions to generate
            question_type: Type of questions to generate (MCQ, DESCRIPTIVE, FILL_BLANKS, or TRUEFALSE)
            
        Returns:
            List of generated questions
        """
        # Create prompt based on question type
        if question_type == "MCQ":
            prompt_template = """
            You are an expert educator. Generate {num_questions} multiple-choice questions about {subject}, specifically on the 
            topic of {topic} and subtopic {subtopic}. Make these questions suitable for difficulty 
            level {level} (where 1 is easiest and 5 is hardest).
            
            For each question:
            1. Create a challenging but fair question
            2. Provide four options (A, B, C, D)
            3. Indicate the correct answer
            4. Include a brief explanation of why the answer is correct
            
            Format your response as a JSON array of objects with the following structure:
            [
              {{
                "question": "The question text",
                "option1": "Option A",
                "option2": "Option B",
                "option3": "Option C",
                "option4": "Option D",
                "correctanswer": "option1, option2, option3, or option4",
                "explaination": "Explanation of the correct answer",
                "question_type": "MCQ"
              }},
              // more questions...
            ]
            
            Make sure the questions test understanding of key concepts, not just trivial details.
            """
        elif question_type == "DESCRIPTIVE":
            prompt_template = """
            You are an expert educator. Generate {num_questions} descriptive questions about {subject}, specifically on the 
            topic of {topic} and subtopic {subtopic}. Make these questions suitable for difficulty 
            level {level} (where 1 is easiest and 5 is hardest).
            
            For each question:
            1. Create a thought-provoking question that requires explanation or analysis
            2. Provide a model answer that would receive full marks
            3. Include grading criteria or key points that should be included in a good answer
            
            Format your response as a JSON array of objects with the following structure:
            [
              {{
                "question": "The question text",
                "model_answer": "A comprehensive model answer",
                "grading_criteria": "Key points that should be included",
                "question_type": "DESCRIPTIVE",
                "explaination": "Brief explanation about the question and its importance"
              }},
              // more questions...
            ]
            
            Make sure the questions test deep understanding and critical thinking.
            """
        elif question_type == "FILL_BLANKS":
            prompt_template = """
            You are an expert educator. Generate {num_questions} fill-in-the-blank questions about {subject}, specifically on the 
            topic of {topic} and subtopic {subtopic}. Make these questions suitable for difficulty 
            level {level} (where 1 is easiest and 5 is hardest).
            
            For each question:
            1. Create a sentence or paragraph with key terms removed and replaced with blanks
            2. Provide the correct answers for each blank
            3. Include a brief explanation for why these answers are correct
            
            Format your response as a JSON array of objects with the following structure:
            [
              {{
                "question": "The sentence with blanks indicated by _____",
                "answers": ["Answer for blank 1", "Answer for blank 2", ...],
                "explaination": "Explanation of the correct answers",
                "question_type": "FILL_BLANKS"
              }},
              // more questions...
            ]
            
            Make sure the blanks focus on important terminology or concepts.
            """
        elif question_type == "TRUEFALSE":
            prompt_template = """
            You are an expert educator. Generate {num_questions} true/false questions about {subject}, specifically on the 
            topic of {topic} and subtopic {subtopic}. Make these questions suitable for difficulty 
            level {level} (where 1 is easiest and 5 is hardest).
            
            For each question:
            1. Create a challenging but clear statement that is either true or false
            2. Indicate whether the statement is true or false
            3. Include a brief explanation of why the statement is true or false
            
            Format your response as a JSON array of objects with the following structure:
            [
              {{
                "question": "The statement to evaluate as true or false",
                "correct_answer": "true or false (lowercase)",
                "explaination": "Explanation of why the statement is true or false",
                "question_type": "TRUEFALSE"
              }},
              // more questions...
            ]
            
            Make sure the questions test understanding of key concepts, not just trivial details.
            """
        else:
            # Default to MCQ if the type is not recognized
            question_type = "MCQ"
            prompt_template = """
            You are an expert educator. Generate {num_questions} multiple-choice questions about {subject}, specifically on the 
            topic of {topic} and subtopic {subtopic}. Make these questions suitable for difficulty 
            level {level} (where 1 is easiest and 5 is hardest).
            
            For each question:
            1. Create a challenging but fair question
            2. Provide four options (A, B, C, D)
            3. Indicate the correct answer
            4. Include a brief explanation of why the answer is correct
            
            Format your response as a JSON array of objects with the following structure:
            [
              {{
                "question": "The question text",
                "option1": "Option A",
                "option2": "Option B",
                "option3": "Option C",
                "option4": "Option D",
                "correctanswer": "option1, option2, option3, or option4",
                "explaination": "Explanation of the correct answer",
                "question_type": "MCQ"
              }},
              // more questions...
            ]
            
            Make sure the questions test understanding of key concepts, not just trivial details.
            """
        
        generate_prompt = PromptTemplate.from_template(prompt_template)
        generate_chain = generate_prompt | self.ug_llm | StrOutputParser()
        
        questions_json = generate_chain.invoke({
            "num_questions": num_questions,
            "subject": subject,
            "topic": topic,
            "subtopic": subtopic,
            "level": level
        })
        
        try:
            # Parse generated questions
            questions = json.loads(questions_json.replace("```json", "").replace("```", ""))
            
            # Ensure proper structure and add metadata
            for q in questions:
                q["subject"] = subject
                q["topic"] = topic
                q["subtopic"] = subtopic
                q["level"] = str(level)
                q["questionset"] = "generated"
                q["marks"] = "1"  # Default mark
                q["created_at"] = datetime.utcnow().isoformat()
                
                # Set required fields based on question type
                q_type = q.get("question_type", question_type).upper()
                
                if q_type == "MCQ":
                    required_fields = ["question", "option1", "option2", "option3", "option4", 
                                      "correctanswer", "explaination"]
                    for field in required_fields:
                        if field not in q:
                            q[field] = ""
                
                elif q_type == "DESCRIPTIVE":
                    required_fields = ["question", "model_answer", "grading_criteria", "explaination"]
                    for field in required_fields:
                        if field not in q:
                            q[field] = ""
                
                elif q_type == "FILL_BLANKS":
                    if "question" not in q:
                        q["question"] = ""
                    if "explaination" not in q:
                        q["explaination"] = ""
                    if "answers" not in q:
                        q["answers"] = []
                
                elif q_type == "TRUEFALSE":
                    if "question" not in q:
                        q["question"] = ""
                    if "correct_answer" not in q:
                        q["correct_answer"] = "true"
                    if "explaination" not in q:
                        q["explaination"] = ""
                    
                    # Ensure the correct_answer is lowercase (true or false)
                    if isinstance(q.get("correct_answer"), str):
                        q["correct_answer"] = q["correct_answer"].lower()
                
                # Ensure question_type is included
                if "question_type" not in q:
                    q["question_type"] = question_type
                
                # Insert question into question bank
                try:
                    # Create a copy of the question for insertion to avoid modifying the original
                    question_copy = q.copy()
                    
                    # Insert into question bank
                    success = self.question_repo.insert_question(question_copy)
                    
                    if success:
                        # If insertion was successful, the _id field was added to question_copy
                        # Add it to original question
                        if '_id' in question_copy:
                            q['id'] = str(question_copy['_id'])
                    else:
                        print(f"Failed to insert question into question bank: {q['question']}")
                except Exception as e:
                    print(f"Error inserting question into question bank: {str(e)}")
            
            return questions
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing generated questions: {str(e)}")
            # If JSON parsing fails, try to extract JSON using basic pattern matching
            try:
                # Find JSON array in the text
                start_idx = questions_json.find('[')
                end_idx = questions_json.rfind(']') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    extracted_json = questions_json[start_idx:end_idx]
                    questions = json.loads(extracted_json)
                    
                    # Process questions as above
                    for q in questions:
                        q["subject"] = subject
                        q["topic"] = topic
                        q["subtopic"] = subtopic
                        q["level"] = str(level)
                        q["questionset"] = "generated"
                        q["marks"] = "1"  # Default mark
                        q["created_at"] = datetime.utcnow().isoformat()
                        
                        # Ensure question_type is included
                        if "question_type" not in q:
                            q["question_type"] = question_type
                            
                        # Insert question into question bank
                        try:
                            # Create a copy of the question for insertion to avoid modifying the original
                            question_copy = q.copy()
                            
                            # Insert into question bank
                            success = self.question_repo.insert_question(question_copy)
                            
                            if success:
                                # If insertion was successful, the _id field was added to question_copy
                                # Add it to original question
                                if '_id' in question_copy:
                                    q['id'] = str(question_copy['_id'])
                        except Exception as e:
                            print(f"Error inserting question into question bank: {str(e)}")
                    
                    return questions
            except:
                pass
            
            # Return empty list if all parsing attempts fail
            return []
    
    def submit_assessment(self, assessment_id: str, student_answers: List[Dict], student_id: str) -> Tuple[Dict, int]:
        """Submit and grade an assessment.
        
        Args:
            assessment_id: ID of the assessment to grade
            student_answers: List of questions with student answers
            student_id: ID of the student
            
        Returns:
            Tuple of (result_data, status_code)
        """
        try:
            # Get the assessment
            print(f"Getting assessment with ID: {assessment_id}")
            assessment = self.history_repo.get_assessment_by_id(student_id, assessment_id)
            print(f"Assessment: {assessment}")
            if not assessment:
                return {"Message": f"Assessment with ID {assessment_id} not found"}, 404
                
            # Get questions from the assessment
            assessment_questions = assessment.get("questions", [])
            print(f"Assessment questions: {assessment_questions}")
            
            # Create a dictionary of questions by ID for faster lookup
            questions_by_id = {}
            for q in assessment_questions:
                # Handle both string IDs and ObjectIds
                q_id = str(q.get("id", "")) or str(q.get("_id", ""))
                if q_id:
                    questions_by_id[q_id] = q
            
            # Process each answer
            results = []
            correct_count = 0
            total_questions = len(student_answers)
            
            for answer_data in student_answers:
                question_id = answer_data.questionid
                student_answer = answer_data.studentanswer
                
                # Skip questions without answers
                if not student_answer:
                    continue
                
                # Find the question in the assessment
                question = questions_by_id.get(question_id)
                
                if not question:
                    # Question not found in this assessment
                    result = {
                        "questionid": question_id,
                        "is_correct": False,
                        "feedback": f"Question not found in assessment {assessment_id}"
                    }
                    results.append(result)
                    continue
                
                # Store student's answer in the question object
                question["student_answer"] = student_answer
                
                # Check answer based on question type
                question_type = question.get("question_type", "MCQ")
                is_correct = False
                feedback = ""
                
                if question_type == "MCQ":
                    # For multiple-choice questions
                    correct_option = question.get("correct_option")
                    if correct_option and student_answer.upper() == correct_option.upper():
                        is_correct = True
                        feedback = question.get("explanation", "Correct answer!")
                    else:
                        feedback = question.get("explanation", "Incorrect answer.")
                
                elif question_type == "DESCRIPTIVE":
                    # For descriptive questions, use AI to evaluate
                    model_answer = question.get("model_answer", "")
                    grading_criteria = question.get("grading_criteria", "")
                    
                    # Use GPT-4 to evaluate
                    prompt = PromptTemplate.from_template(
                        "You are an expert grader. Evaluate the student's answer against the model answer "
                        "and grading criteria.\n\n"
                        "Question: {question}\n\n"
                        "Model Answer: {model_answer}\n\n"
                        "Grading Criteria: {grading_criteria}\n\n"
                        "Student Answer: {student_answer}\n\n"
                        "Evaluate the answer and provide a score from 0 to 100 percent, and feedback. "
                        "Format your response as a JSON object with the following structure:\n"
                        "{{\"score\": percentage_score, \"feedback\": \"detailed feedback\"}}"
                    )
                    
                    grade_chain = prompt | self.ug_llm | StrOutputParser()
                    
                    evaluation_result = grade_chain.invoke({
                        "question": question.get("question", ""),
                        "model_answer": model_answer,
                        "grading_criteria": grading_criteria,
                        "student_answer": student_answer
                    })
                    
                    try:
                        # Parse the evaluation
                        print(f"Evaluation result: {evaluation_result}")
                        evaluation_result = evaluation_result.replace("```json", "").replace("```", "")
                        evaluation = json.loads(evaluation_result)
                        score = evaluation.get("score", 0)
                        feedback = evaluation.get("feedback", "")
                        
                        # Store the score in the question object
                        question["score"] = score
                        
                        # Consider correct if score is above 70%
                        is_correct = score >= 70
                    except json.JSONDecodeError:
                        # Fallback if JSON parsing fails
                        is_correct = False
                        feedback = "Error evaluating answer. Please try again."
                
                elif question_type == "FILL_BLANKS":
                    # For fill-in-the-blanks questions
                    correct_answers = question.get("answers", [])
                    
                    # Split student answer by comma if multiple blanks
                    student_answers_list = [ans.strip() for ans in student_answer.split(',')]
                    
                    # Check if the number of answers matches
                    if len(student_answers_list) != len(correct_answers):
                        is_correct = False
                        feedback = "Number of answers provided does not match the number of blanks."
                    else:
                        # Check each answer
                        all_correct = True
                        for i, (student_ans, correct_ans) in enumerate(zip(student_answers_list, correct_answers)):
                            if student_ans.lower() != correct_ans.lower():
                                all_correct = False
                                break
                        
                        is_correct = all_correct
                        if is_correct:
                            feedback = question.get("explanation", "All answers are correct!")
                        else:
                            feedback = question.get("explanation", "Some answers are incorrect.")
                
                elif question_type == "TRUEFALSE":
                    # For true/false questions
                    correct_answer = question.get("correct_answer", "").lower()
                    student_answer_normalized = student_answer.lower().strip()
                    
                    # Handle various forms of true/false answers
                    is_true = student_answer_normalized in ["true", "t", "yes", "y", "1"]
                    is_false = student_answer_normalized in ["false", "f", "no", "n", "0"]
                    
                    if (is_true and correct_answer == "true") or (is_false and correct_answer == "false"):
                        is_correct = True
                        feedback = question.get("explanation", "Correct answer!")
                    else:
                        is_correct = False
                        feedback = question.get("explanation", "Incorrect answer.")
                
                else:
                    # Default handling for unknown question types
                    is_correct = False
                    feedback = "Unknown question type."
                
                # Store evaluation result in the question object
                question["is_correct"] = is_correct
                question["feedback"] = feedback
                
                # Add result for this question
                result = {
                    "questionid": question_id,
                    "is_correct": is_correct,
                    "feedback": feedback
                }
                results.append(result)
                
                if is_correct:
                    correct_count += 1
            
            # Calculate score
            score_percentage = (correct_count / total_questions) * 100 if total_questions > 0 else 0
            
            # Create result object
            assessment_result = {
                "assessment_id": assessment_id,
                "student_id": student_id,
                "submission_time": datetime.utcnow(),
                "results": results,
                "correct_count": correct_count,
                "total_questions": total_questions,
                "score_percentage": score_percentage
            }
            
            # Update assessment with results and update questions with student answers
            print(f"Assessment questions before update: {assessment_questions}")
            self.history_repo.update_assessment(
                student_id,
                assessment_id,
                {
                    "last_submission": assessment_result,
                    "submission_count": assessment.get("submission_count", 0) + 1,
                    "last_submission_time": datetime.utcnow(),
                    "questions": assessment_questions  # Update questions with student answers
                }
            )
            
            return assessment_result, 200
            
        except Exception as e:
            return {"Message": f"Error submitting assessment: {str(e)}"}, 500
    
    def get_assessments(self, student_id: str, time_str: Optional[str] = None, subject: Optional[str] = None, topic: Optional[str] = None) -> Tuple[List[Dict], int]:
        """Get a student's assessments.
        
        Args:
            student_id: ID of the student
            time_str: Optional time string to filter assessments
            subject: Optional subject to filter assessments
            topic: Optional topic to filter assessments
            
        Returns:
            Tuple of (assessments, status_code)
        """
        try:
            # Parse time if provided
            from_date = None
            if time_str:
                try:
                    from_date = dateutil.parser.parse(time_str)
                except Exception:
                    # Default to one week ago if parsing fails
                    from_date = datetime.utcnow() - timedelta(weeks=1)
            else:
                # Default to one week ago
                from_date = datetime.utcnow() - timedelta(weeks=1)
            
            # Get assessments
            assessments = self.history_repo.get_assessments(student_id, from_date, subject, topic)
            
            # Ensure each assessment has a 'topics' field
            for assessment in assessments:
                if 'topics' not in assessment:
                    if 'topic' in assessment:
                        assessment['topics'] = [assessment['topic']]
                    else:
                        # Try to extract topics from the questions
                        topics = set()
                        for question in assessment.get('questions', []):
                            if 'topic' in question:
                                topics.add(question['topic'])
                        
                        if topics:
                            assessment['topics'] = list(topics)
                        else:
                            assessment['topics'] = []
            
            return assessments, 200
        except Exception as e:
            return [], 500
    
    def get_assessment_by_id(self, student_id: str, assessment_id: str) -> Tuple[Dict, int]:
        """Get a specific assessment by ID.
        
        Args:
            student_id: ID of the student
            assessment_id: ID of the assessment
            
        Returns:
            Tuple of (assessment_data, status_code)
        """
        try:
            assessment = self.history_repo.get_assessment_by_id(student_id, assessment_id)
            
            if not assessment:
                return {"Message": "Assessment not found"}, 404
            
            # Ensure 'topics' field exists in the assessment
            if 'topics' not in assessment:
                # If assessment has a single 'topic' field, convert it to list format
                if 'topic' in assessment:
                    assessment['topics'] = [assessment['topic']]
                else:
                    # Check if any topic info is in the questions
                    topics = set()
                    for question in assessment.get('questions', []):
                        if 'topic' in question:
                            topics.add(question['topic'])
                    
                    if topics:
                        assessment['topics'] = list(topics)
                    else:
                        assessment['topics'] = []
                
            return assessment, 200
        except Exception:
            return {"Message": "Error retrieving assessment"}, 500
    
    def get_history(self, student_id: str, time_str: Optional[str] = None) -> Tuple[List[Dict], int]:
        """Get a student's assessment history.
        
        Args:
            student_id: ID of the student
            time_str: Optional time string to filter history
            
        Returns:
            Tuple of (history_items, status_code)
        """
        try:
            # Parse time if provided
            from_date = None
            if time_str:
                try:
                    from_date = dateutil.parser.parse(time_str)
                except Exception:
                    # Default to one week ago if parsing fails
                    from_date = datetime.utcnow() - timedelta(weeks=1)
            else:
                # Default to one week ago
                from_date = datetime.utcnow() - timedelta(weeks=1)
            
            # Get history
            history = self.history_repo.get_history(student_id, from_date)
            
            return history, 200
        except Exception:
            return [], 500
    
    def generate_assessment_from_pdf(self, pdf_id: str, student_id: str, 
                                     question_types: List[str] = None, 
                                     num_questions: int = 10) -> Tuple[Dict, int]:
        """Generate assessment questions from a PDF document.
        
        Args:
            pdf_id: ID of the PDF document
            student_id: ID of the student
            question_types: List of question types to generate (e.g., ["MCQ", "DESCRIPTIVE"])
                           If None or empty, defaults to ["MCQ"]
            num_questions: Number of questions to generate
            
        Returns:
            Tuple of (assessment_data, status_code)
        """
        try:
            # Default to MCQ if no question types are provided
            if not question_types:
                question_types = ["MCQ"]
                
            # Convert all question types to uppercase for consistency
            question_types = [qt.upper() for qt in question_types]
            
            # Validate question types
            valid_types = ["MCQ", "DESCRIPTIVE", "FILL_BLANKS", "TRUEFALSE"]
            question_types = [qt for qt in question_types if qt in valid_types]
            
            # Default to MCQ if all provided types were invalid
            if not question_types:
                question_types = ["MCQ"]
            
            # Get PDF document
            pdf_document = self.pdf_repository.get_pdf_document(pdf_id)
            
            if not pdf_document:
                return {"Message": "PDF not found"}, 404
            
            # Check if PDF belongs to the student
            if pdf_document.user_id != student_id:
                return {"Message": "You don't have permission to access this PDF"}, 403
            
            # Check if PDF has been processed
            if pdf_document.processing_status != ProcessingStatus.COMPLETED:
                return {"Message": "PDF has not been fully processed yet"}, 400
            
            # Get PDF text from PostgreSQL
            pdf_text = self.postgres_text_repository.get_pdf_text(student_id, pdf_id)
            
            if not pdf_text or not pdf_text.get('content'):
                return {"Message": "PDF text not found or is empty"}, 404
            
            # Limit the number of questions to a reasonable range
            num_questions = max(1, min(num_questions, 20))
            
            # Generate questions using LLM
            questions = self._generate_questions_from_pdf(
                pdf_text['content'],
                pdf_document.title,
                question_types,
                num_questions,
                pdf_id,
                student_id
            )
            
            if not questions:
                return {"Message": "Failed to generate questions"}, 500
            
            # Add metadata to questions
            for q in questions:
                q["pdf_id"] = pdf_id
                q["generated_at"] = datetime.utcnow().isoformat()
                q["id"] = str(uuid.uuid4())
            
            # Create assessment object
            assessment = {
                "id": str(uuid.uuid4()),
                "pdf_id": pdf_id,
                "title": f"Assessment for {pdf_document.title}",
                "description": f"Generated assessment based on {pdf_document.title}",
                "questions": questions,
                "student_id": student_id,
                "created_at": datetime.utcnow(),
                "question_count": len(questions),
                "question_types": question_types
            }
            
            # Store assessment in MongoDB
            assessment_id = self.history_repo.add_assessment(student_id, assessment)
            
            # Return assessment data
            return {
                "Message": "Assessment generated successfully",
                "assessment_id": assessment_id,
                "assessment": assessment
            }, 200
            
        except Exception as e:
            return {"Message": f"Error generating assessment: {str(e)}"}, 500
    
    def _generate_questions_from_pdf(self, content: str, pdf_title: str, 
                                    question_types: List[str], num_questions: int, pdf_id: str = None, student_id: str = None) -> List[Dict]:
        """Generate questions from PDF content.
        
        Args:
            content: Text content of the PDF
            pdf_title: Title of the PDF
            question_types: List of question types to generate
            num_questions: Number of questions to generate
            pdf_id: ID of the PDF document (for image retrieval)
            student_id: ID of the student (for image retrieval)
            
        Returns:
            List of generated questions
        """
        # Truncate content if too large
        max_tokens = 12000  # Limit to avoid context length issues
        content = content[:max_tokens] if len(content) > max_tokens else content
        
        # Generate questions for all the requested types
        all_questions = []
        
        # If we have only one question type, generate all questions of that type
        if len(question_types) == 1:
            all_questions = self._generate_questions_of_type_from_pdf(
                content, pdf_title, question_types[0], num_questions, pdf_id, student_id
            )
        else:
            # Distribute questions among the types
            # We'll use a simple random approach to distribute the questions
            import random
            
            # Shuffle the types to ensure a random distribution
            types_to_use = question_types.copy()
            random.shuffle(types_to_use)
            
            questions_per_type = {}
            remaining = num_questions
            
            # Initial distribution - at least 1 per type if possible
            for q_type in types_to_use:
                if remaining > 0:
                    questions_per_type[q_type] = 1
                    remaining -= 1
                else:
                    questions_per_type[q_type] = 0
            
            # Distribute remaining questions randomly
            while remaining > 0:
                q_type = random.choice(types_to_use)
                questions_per_type[q_type] += 1
                remaining -= 1
            
            # Generate questions for each type
            for q_type, count in questions_per_type.items():
                if count > 0:
                    type_questions = self._generate_questions_of_type_from_pdf(
                        content, pdf_title, q_type, count, pdf_id, student_id
                    )
                    all_questions.extend(type_questions)
                
        # If we have a PDF ID and student ID, try to find relevant images for each question
        if pdf_id and student_id and all_questions:
            # Add relevant images to each question
            all_questions = self._add_images_to_questions(all_questions, pdf_id, student_id)
            
        return all_questions
            
    def _generate_questions_of_type_from_pdf(self, content: str, pdf_title: str, 
                                           question_type: str, num_questions: int, 
                                           pdf_id: str = None, student_id: str = None) -> List[Dict]:
        """Generate questions of a specific type from PDF content.
        
        Args:
            content: Text content of the PDF
            pdf_title: Title of the PDF
            question_type: Type of questions to generate
            num_questions: Number of questions to generate
            pdf_id: ID of the PDF document (for image retrieval)
            student_id: ID of the student (for image retrieval)
            
        Returns:
            List of generated questions
        """
        # Create prompt based on question type
        if question_type == "MCQ":
            prompt_template = """
            You are an expert educator. Generate {num_questions} multiple-choice questions based on the following text from the document titled "{pdf_title}".
            
            Text content:
            {content}
            
            For each question:
            1. Create a challenging but fair question
            2. Provide four options (A, B, C, D)
            3. Indicate the correct answer
            4. Include a brief explanation of why the answer is correct
            
            
            Format your response as a JSON array of objects with the following structure:
            [
              {{
                "question": "The question text",
                "option1": "Option A",
                "option2": "Option B",
                "option3": "Option C",
                "option4": "Option D",
                "correct_option": "The letter of the correct option (A, B, C, or D)",
                "explanation": "Explanation of the correct answer",
                "question_type": "MCQ"
              }},
              // more questions...
            ]
            
            Make sure the questions test understanding of key concepts from the text, not just trivial details.
            """
        elif question_type == "DESCRIPTIVE":
            prompt_template = """
            You are an expert educator. Generate {num_questions} descriptive questions based on the following text from the document titled "{pdf_title}".
            
            Text content:
            {content}
            
            For each question:
            1. Create a thought-provoking question that requires explanation or analysis
            2. Provide a model answer that would receive full marks
            3. Include grading criteria or key points that should be included in a good answer
            
            Format your response as a JSON array of objects with the following structure:
            [
              {{
                "question": "The question text",
                "model_answer": "A comprehensive model answer",
                "grading_criteria": "Key points that should be included",
                "question_type": "DESCRIPTIVE"
              }},
              // more questions...
            ]
            
            Make sure the questions test deep understanding and critical thinking about the content.
            """
        elif question_type == "FILL_BLANKS":
            prompt_template = """
            You are an expert educator. Generate {num_questions} fill-in-the-blank questions based on the following text from the document titled "{pdf_title}".
            
            Text content:
            {content}
            
            For each question:
            1. Create a sentence or paragraph with key terms removed and replaced with blanks
            2. Provide the correct answers for each blank
            3. Include a brief explanation for why these answers are correct
            
            Format your response as a JSON array of objects with the following structure:
            [
              {{
                "question": "The sentence with blanks indicated by _____",
                "answers": ["Answer for blank 1", "Answer for blank 2", ...],
                "explanation": "Explanation of the correct answers",
                "question_type": "FILL_BLANKS"
              }},
              // more questions...
            ]
            
            Make sure the blanks focus on important terminology or concepts from the text.
            """
        elif question_type == "TRUEFALSE":
            prompt_template = """
            You are an expert educator. Generate {num_questions} true/false questions based on the following text from the document titled "{pdf_title}".
            
            Text content:
            {content}
            
            For each question:
            1. Create a challenging but clear statement that is either true or false based on the text
            2. Indicate whether the statement is true or false
            3. Include a brief explanation of why the statement is true or false
            
            Format your response as a JSON array of objects with the following structure:
            [
              {{
                "question": "The statement to evaluate as true or false",
                "correct_answer": "true or false (lowercase)",
                "explanation": "Explanation of why the statement is true or false",
                "question_type": "TRUEFALSE"
              }},
              // more questions...
            ]
            
            Make sure the statements are substantive and test important concepts from the text.
            """
        else:
            # Default to MCQ if the type is not recognized
            question_type = "MCQ"
            prompt_template = """
            You are an expert educator. Generate {num_questions} multiple-choice questions based on the following text from the document titled "{pdf_title}".
            
            Text content:
            {content}
            
            For each question:
            1. Create a challenging but fair question
            2. Provide four options (A, B, C, D)
            3. Indicate the correct answer
            4. Include a brief explanation of why the answer is correct
            
            
            Format your response as a JSON array of objects with the following structure:
            [
              {{
                "question": "The question text",
                "option1": "Option A",
                "option2": "Option B",
                "option3": "Option C",
                "option4": "Option D",
                "correct_option": "The letter of the correct option (A, B, C, or D)",
                "explanation": "Explanation of the correct answer",
                "question_type": "MCQ"
              }},
              // more questions...
            ]
            
            Make sure the questions test understanding of key concepts from the text, not just trivial details.
            """
        
        prompt = PromptTemplate.from_template(prompt_template)
        
        # Use the more powerful model for question generation
        chain = prompt | self.ug_llm | StrOutputParser()
        
        questions_json = chain.invoke({
            "num_questions": num_questions,
            "pdf_title": pdf_title,
            "content": content
        })
        
        try:
            # Parse the generated questions
            questions = json.loads(questions_json.replace("```json", "").replace("```", ""))
            
            # Ensure all questions have the correct question_type
            for q in questions:
                if "question_type" not in q:
                    q["question_type"] = question_type
                    
                # Handle specific question type validation
                if question_type == "TRUEFALSE" and "correct_answer" in q:
                    # Ensure the correct_answer is lowercase (true or false)
                    if isinstance(q["correct_answer"], str):
                        q["correct_answer"] = q["correct_answer"].lower()
                
            return questions
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON using basic pattern matching
            try:
                # Find JSON array in the text
                start_idx = questions_json.find('[')
                end_idx = questions_json.rfind(']') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    extracted_json = questions_json[start_idx:end_idx]
                    questions = json.loads(extracted_json)
                    
                    # Ensure all questions have the correct question_type
                    for q in questions:
                        if "question_type" not in q:
                            q["question_type"] = question_type
                            
                        # Handle specific question type validation
                        if question_type == "TRUEFALSE" and "correct_answer" in q:
                            # Ensure the correct_answer is lowercase (true or false)
                            if isinstance(q["correct_answer"], str):
                                q["correct_answer"] = q["correct_answer"].lower()
                                
                    return questions
            except:
                pass
            
            # Return empty list if all parsing attempts fail
            return []
    
    def _add_images_to_questions(self, questions: List[Dict], pdf_id: str, student_id: str) -> List[Dict]:
        """Add relevant images to assessment questions using RAG on image captions.
        
        Args:
            questions: List of generated questions
            pdf_id: ID of the PDF document
            student_id: ID of the student
            
        Returns:
            List of questions with image URLs added
        """
        try:
            # Format student ID for database connection
            safe_student_id = student_id.replace('-', '_')
            
            # Create connection string for student's database
            base_connection = settings.POSTGRES_CONNECTION_STRING
            if "://" in base_connection and "@" in base_connection:
                prefix = base_connection[:base_connection.rindex('@') + 1]
                suffix = base_connection[base_connection.rindex('@') + 1:]
                
                if '/' in suffix:
                    host_port = suffix[:suffix.index('/')]
                    connection_string = f"{prefix}{host_port}/student_{safe_student_id}"
                else:
                    connection_string = f"{base_connection}/student_{safe_student_id}"
            else:
                connection_string = base_connection
            
            # Initialize embeddings and vector store
            embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
            '''embeddings = GoogleGenerativeAIEmbeddings(google_api_key=settings.GOOGLE_API_KEY,model="models/gemini-embedding-exp-03-07")'''
            collection_name = f"pdf_{pdf_id}_images"
            
            # Connect to PGVector with image captions collection
            try:
                vector_store = PGVector(
                    embeddings=embeddings,
                    collection_name=collection_name,
                    connection=connection_string,
                    use_jsonb=True
                )
                
                # Process each question to find a relevant image
                for question in questions:
                    # Extract question text to use for RAG
                    question_text = question.get("question", "")
                    
                    # If MCQ, also consider the options in the query
                    if question.get("question_type") == "MCQ" and "options" in question:
                        options_text = " ".join(question["options"])
                        query_text = f"{question_text} {options_text}"
                    else:
                        query_text = question_text
                    
                    # Perform similarity search to find relevant image
                    try:
                        print(f"Searching for images for question: {question_text}")
                        print(f"Query text: {query_text}")
                        results = vector_store.similarity_search_with_score(
                            query_text, 
                            k=1  # Get the most relevant image
                        )
                        
                        if results and len(results) > 0:
                            # Extract image URL from the metadata
                            doc, score = results[0]
                            if doc.metadata and "image_url" in doc.metadata:
                                # Add image URL and caption to question
                                question["image_url"] = doc.metadata["image_url"]
                                question["image_caption"] = doc.page_content
                                question["has_image"] = True
                            
                    except Exception as e:
                        print(f"Error searching for images for question: {str(e)}")
                        
            except Exception as e:
                print(f"Error connecting to image vectors: {str(e)}")
                        
        except Exception as e:
            print(f"Error adding images to questions: {str(e)}")
            
        return questions 
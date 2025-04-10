import json
import dateutil.parser
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional, Any
import uuid

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI

from config import settings
from repositories.mongo_repository import HistoryRepository, QuestionRepository
from repositories.postgres_text_repository import PostgresTextRepository
from repositories.pdf_repository import PDFRepository
from services.langchain.langchain_service import LangchainService
from models.pdf_models import ProcessingStatus, QuestionType

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
    
    def check_assessment_content(self, content: str, session_id: str, student_id: str) -> Tuple[Dict, int]:
        """Check if input text is requesting an assessment and process accordingly.
        
        Args:
            content: Input text from the user
            session_id: JWT token from X-Auth-Session to use as conversation ID
            student_id: ID of the student
            
        Returns:
            Tuple of (result_data, status_code)
        """
        try:
            # Check if content contains the word "Assessment"
            check_prompt = PromptTemplate.from_template(
                "Answer Only AS YES OR NO IN CAPTIALS Answer whether the following statement contains the "
                "word case insensitive 'Assessment' : \n Statement:{statement}"
            )
            check_chain = check_prompt | self.llm | StrOutputParser()
            assessment_check = check_chain.invoke({"statement": content})
            
            if assessment_check != "YES":
                # Not an assessment request
                # Try to answer using LangChain
                answer, status_code = self.langchain_service.answer_question(
                    content, 
                    student_id,
                    session_id
                )
                
                if status_code == 200:
                    return {"answer": answer}, 200
                else:
                    return {"Message": "Not an assessment request"}, 400
            
            # Parse what subjects/topics to assess on
            json_structure = [i for i in self.question_repo.get_all_topics_subtopics()]
            
            get_final_json_prompt = PromptTemplate.from_template(
                "Given The Question Produce a correct JSON containing a key called questions which "
                "contains an array of json which in turn has subject,topic,subtopic,level,NumberOfQuestions "
                "for each subject from the Question Keep the Default value for NumberOfQuestions as 5 and "
                "Default Level as 1 if not found in the Question and the omit and remove fields and remove "
                "the keys too from json do not keep them as empty or null if not found in the question "
                "remove the fields and keys completely\n Question: {question}"
            )
            get_final_json_chain = get_final_json_prompt | self.llm | StrOutputParser()
            json_with_questions = get_final_json_chain.invoke({"question": content})
            
            # Parse the JSON
            try:
                assessment_data = json.loads(json_with_questions)
                questions_list = assessment_data.get("questions", [])
            except (json.JSONDecodeError, TypeError):
                # Fallback to empty list if JSON is invalid
                questions_list = []
            
            if not questions_list:
                return {"Message": "Failed to parse assessment request"}, 400
            
            # Generate questions for each subject/topic
            final_questions = []
            
            for question_spec in questions_list:
                subject = question_spec.get("subject")
                topic = question_spec.get("topic")
                subtopic = question_spec.get("subtopic", topic)
                level = question_spec.get("level", 1)
                num_questions = question_spec.get("NumberOfQuestions", 5)
                
                # Find matching questions in the database
                query = {
                    "subject": subject,
                    "topic": topic
                }
                
                if subtopic:
                    query["subtopic"] = subtopic
                    
                if level:
                    query["level"] = str(level)
                
                # Get questions from the database
                db_questions = self.question_repo.find_questions(
                    query=query,
                    limit=num_questions
                )
                
                # If not enough questions, generate some using LLM
                if len(db_questions) < num_questions:
                    # Number of questions to generate
                    num_to_generate = num_questions - len(db_questions)
                    
                    # Generate questions
                    generated_questions = self._generate_questions(
                        subject, topic, subtopic, level, num_to_generate
                    )
                    
                    # Add generated questions
                    db_questions.extend(generated_questions)
                
                # Add to final list
                final_questions.extend(db_questions)
            
            # Create assessment object
            assessment = {
                "questions": final_questions,
                "student_id": student_id,
                "timestamp": datetime.utcnow(),
                "original_query": content,
                "session_id": session_id
            }
            
            # Store assessment
            assessment_id = self.history_repo.add_assessment(student_id, assessment)
            
            # Add this interaction to chat history
            self.langchain_service.add_to_chat_history(
                student_id, 
                session_id, 
                f"Generated assessment with ID: {assessment_id}"
            )
            
            # Return assessment data
            return {
                "Message": "Assessment generated successfully",
                "assessment_id": assessment_id,
                "questions": final_questions
            }, 200
            
        except Exception as e:
            return {"Message": f"Error processing assessment request: {str(e)}"}, 500
    
    def _generate_questions(self, subject: str, topic: str, subtopic: str, level: int, 
                           num_questions: int) -> List[Dict]:
        """Generate questions using AI.
        
        Args:
            subject: Subject of the questions
            topic: Topic of the questions
            subtopic: Subtopic of the questions
            level: Difficulty level of the questions
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        # Construct prompt
        generate_prompt = PromptTemplate.from_template(
            "Generate {num_questions} multiple-choice questions about {subject}, specifically on the "
            "topic of {topic} and subtopic {subtopic}. Make these questions suitable for difficulty "
            "level {level} (where 1 is easiest and 5 is hardest).\n\n"
            "For each question, provide:\n"
            "1. The question text\n"
            "2. Four options (option1, option2, option3, option4)\n"
            "3. The correct answer (as option1, option2, option3, or option4)\n"
            "4. A brief explanation of the answer\n\n"
            "Format your response as a JSON array of objects, where each object has the fields: "
            "question, option1, option2, option3, option4, correctanswer, explaination.\n\n"
            "Make sure the questions are factually accurate and educational."
        )
        
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
            questions = json.loads(questions_json)
            
            # Ensure proper structure and add metadata
            for q in questions:
                q["subject"] = subject
                q["topic"] = topic
                q["subtopic"] = subtopic
                q["level"] = str(level)
                q["questionset"] = "generated"
                q["marks"] = "1"  # Default mark
                
                # Ensure all required fields exist
                required_fields = ["question", "option1", "option2", "option3", "option4", 
                                  "correctanswer", "explaination"]
                for field in required_fields:
                    if field not in q:
                        q[field] = ""
            
            return questions
        except (json.JSONDecodeError, TypeError):
            # Return empty list if generation fails
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
                
                else:
                    # Default handling for unknown question types
                    is_correct = False
                    feedback = "Unknown question type."
                
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
            
            # Update assessment with results (optional)
            self.history_repo.update_assessment(
                student_id,
                assessment_id,
                {
                    "last_submission": assessment_result,
                    "submission_count": assessment.get("submission_count", 0) + 1,
                    "last_submission_time": datetime.utcnow()
                }
            )
            
            return assessment_result, 200
            
        except Exception as e:
            return {"Message": f"Error submitting assessment: {str(e)}"}, 500
    
    def get_assessments(self, student_id: str, time_str: Optional[str] = None) -> Tuple[List[Dict], int]:
        """Get a student's assessments.
        
        Args:
            student_id: ID of the student
            time_str: Optional time string to filter assessments
            
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
            assessments = self.history_repo.get_assessments(student_id, from_date)
            
            return assessments, 200
        except Exception:
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
                                     question_type: str = "MIXED", 
                                     num_questions: int = 10) -> Tuple[Dict, int]:
        """Generate assessment questions from a PDF document.
        
        Args:
            pdf_id: ID of the PDF document
            student_id: ID of the student
            question_type: Type of questions to generate (MCQ, DESCRIPTIVE, FILL_BLANKS, or MIXED)
            num_questions: Number of questions to generate
            
        Returns:
            Tuple of (assessment_data, status_code)
        """
        try:
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
            
            # Validate question type
            question_type = question_type.upper()
            valid_types = ["MCQ", "DESCRIPTIVE", "FILL_BLANKS", "MIXED"]
            if question_type not in valid_types:
                question_type = "MIXED"
            
            # Limit the number of questions to a reasonable range
            num_questions = max(1, min(num_questions, 20))
            
            # Generate questions using LLM
            questions = self._generate_questions_from_pdf(
                pdf_text['content'],
                pdf_document.title,
                question_type,
                num_questions
            )
            
            if not questions:
                return {"Message": "Failed to generate questions"}, 500
            
            # Add metadata to questions
            for q in questions:
                q["pdf_id"] = pdf_id
                q["question_type"] = q.get("question_type", question_type)
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
                "question_type": question_type
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
                                    question_type: str, num_questions: int) -> List[Dict]:
        """Generate questions from PDF content.
        
        Args:
            content: Text content of the PDF
            pdf_title: Title of the PDF
            question_type: Type of questions to generate
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        # Truncate content if too large
        max_tokens = 12000  # Limit to avoid context length issues
        content = content[:max_tokens] if len(content) > max_tokens else content
        
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
                "options": ["Option A", "Option B", "Option C", "Option D"],
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
        else:  # MIXED type
            prompt_template = """
            You are an expert educator. Generate {num_questions} questions of mixed types based on the following text from the document titled "{pdf_title}".
            
            Text content:
            {content}
            
            Create a mix of:
            - Multiple-choice questions (MCQ)
            - Descriptive questions (DESCRIPTIVE)
            - Fill-in-the-blank questions (FILL_BLANKS)
            
            For multiple-choice questions, include:
            - The question text
            - Four options (A, B, C, D)
            - The correct answer
            - An explanation
            
            For descriptive questions, include:
            - The question text
            - A model answer
            - Grading criteria
            
            For fill-in-the-blank questions, include:
            - A sentence with blanks (marked as _____)
            - The correct answers for each blank
            - An explanation
            
            Format your response as a JSON array of objects with different structures based on type:
            [
              {{
                "question": "MCQ question text",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_option": "A/B/C/D",
                "explanation": "Explanation",
                "question_type": "MCQ"
              }},
              {{
                "question": "Descriptive question text",
                "model_answer": "Model answer",
                "grading_criteria": "Criteria",
                "question_type": "DESCRIPTIVE"
              }},
              {{
                "question": "Fill in: _____",
                "answers": ["Answer"],
                "explanation": "Explanation",
                "question_type": "FILL_BLANKS"
              }},
              // more questions...
            ]
            
            Aim for roughly equal distribution of question types.
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
            questions = json.loads(questions_json)
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
                    return questions
            except:
                pass
            
            # Return empty list if all parsing attempts fail
            return [] 
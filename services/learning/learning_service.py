from typing import List, Dict, Any, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_postgres.vectorstores import PGVector
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime

from config import settings
from repositories.pdf_repository import PDFRepository
from repositories.mongo_repository import HistoryRepository

class LearningService:
    """Service for subject-specific learning using RAG."""
    
    # Subject-specific collection names in PGVector
    SUBJECT_COLLECTIONS = {
        "science": "science",
        "social_science": "social_science",
        "mathematics": "mathematics", 
        "english": "english",
        "hindi": "hindi"
    }
    
    # Subject-specific system prompts
    SUBJECT_PROMPTS = {
        "science": "You are a science education assistant. Focus on providing clear, accurate explanations of scientific concepts, principles, and phenomena. Use precise scientific terminology and explain complex ideas in an accessible way. When possible, relate scientific concepts to real-world applications and everyday experiences.",
        
        "social_science": "You are a social science education assistant. Provide balanced, nuanced perspectives on historical events, geographical concepts, civics, and economic principles. Present multiple viewpoints when relevant and emphasize critical thinking about social issues. Avoid politically biased language and present information objectively.",
        
        "mathematics": "You are a mathematics education assistant. Present mathematical concepts with precision and clarity. Walk through problem-solving steps methodically, explaining each step's reasoning. Use mathematical notation appropriately, and when possible, provide visual representations or analogies to aid understanding of abstract concepts.",
        
        "english": "You are an English language and literature education assistant. Provide clear explanations of grammar rules, literary devices, and writing techniques. When analyzing texts, focus on themes, character development, narrative structure, and authorial intent. Offer constructive guidance for improving writing and language skills.",
        
        "hindi": "You are a Hindi language education assistant. Focus on proper grammar, vocabulary, and language usage. Explain linguistic concepts clearly and provide examples that aid understanding. When discussing Hindi literature, highlight cultural contexts, literary traditions, and notable works. Support students in developing their Hindi language skills with clear, practical guidance."
    }
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the learning service.
        
        Args:
            openai_api_key: API key for OpenAI. If None, uses the one from settings.
        """
        self.api_key = openai_api_key or settings.OPENAI_API_KEY
        self.embeddings = OpenAIEmbeddings(api_key=self.api_key)
        self.llm = ChatOpenAI(api_key=self.api_key)
        self.ug_llm = ChatOpenAI(api_key=self.api_key, model="gpt-4o")
        self.pdf_repository = PDFRepository()
        self.history_repository = HistoryRepository()
    
    def _setup_chat_history(self, student_id: str, session_id: str, subject: str) -> MongoDBChatMessageHistory:
        """Set up a MongoDB-based chat history for a student with subject-specific collection.
        
        Args:
            student_id: ID of the student
            session_id: JWT token from X-Auth-Session 
            subject: Subject for this chat history
            
        Returns:
            MongoDBChatMessageHistory instance
        """
        # Use subject-specific collection name
        collection_name = f"{settings.MONGO_DATABASE_HISTORY}_{subject}"
        
        return MongoDBChatMessageHistory(
            connection_string=settings.MONGO_URI,
            database_name=student_id,
            collection_name=collection_name,
            session_id=session_id,
            history_size=10
        )
    
    def _add_to_chat_history(self, student_id: str, session_id: str, subject: str,
                           message: str, is_ai: bool = True) -> None:
        """Add a message to the subject-specific chat history.
        
        Args:
            student_id: ID of the student
            session_id: JWT token from X-Auth-Session
            subject: Subject for this chat history
            message: Message content to add
            is_ai: Whether the message is from the AI (True) or user (False)
        """
        history = self._setup_chat_history(student_id, session_id, subject)
        
        if is_ai:
            history.add_message(AIMessage(content=message))
        else:
            history.add_message(HumanMessage(content=message))
        
        # Additionally, store in sahasra_history for persistence and retrieval
        history_data = {
            "subject": subject,
            "message": message,
            "is_ai": is_ai,
            "time": datetime.utcnow(),
            "session_id": session_id
        }
        
        # Add to sahasra_history
        self.history_repository.add_history_item(student_id, history_data)
    
    def _get_student_specific_connection_string(self, student_id: str) -> str:
        """Get a student-specific connection string for PGVector.
        
        Args:
            student_id: ID of the student
            
        Returns:
            Connection string for the student's database
        """
        base_connection = settings.POSTGRES_CONNECTION_STRING
        
        # Parse the base connection string to insert student_id
        if "://" in base_connection and "@" in base_connection:
            prefix = base_connection[:base_connection.rindex('@') + 1]
            suffix = base_connection[base_connection.rindex('@') + 1:]
            
            # If suffix contains a slash, it has host:port/dbname
            if '/' in suffix:
                host_port = suffix[:suffix.index('/')]
                # Replace the database name with student-specific one
                student_db = f"student_{student_id}"
                
                # Reconstruct the connection string
                connection_string = f"{prefix}{host_port}/{student_db}"
            else:
                # If no database name in original connection, just append it
                connection_string = f"{base_connection}/student_{student_id}"
        else:
            # Fallback: just use base connection
            connection_string = base_connection
            
        return connection_string
    
    def _get_subject_pdf_content(self, student_id: str, subject: str, question: str) -> List[Dict]:
        """Retrieve content from user's PDFs related to the specified subject.
        
        Args:
            student_id: ID of the student
            subject: Subject to filter PDFs by
            question: Question to retrieve relevant content for
            
        Returns:
            List of documents with relevant content
        """
        pdf_docs = []
        
        try:
            # Get all PDFs for this student that have been processed successfully
            user_pdfs = self.pdf_repository.get_user_pdf_documents(student_id)
            completed_pdfs = [pdf for pdf in user_pdfs if pdf.processing_status == "completed"]
            
            # Filter PDFs by subject if specified
            subject_pdfs = []
            for pdf in completed_pdfs:
                pdf_subject = pdf.metadata.get("subject", "").lower() if pdf.metadata else ""
                
                # Match PDFs with this subject or with no subject specified
                if subject.lower() in pdf_subject or not pdf_subject:
                    subject_pdfs.append(pdf)
            
            if subject_pdfs:
                # Create student-specific connection string
                connection_string = self._get_student_specific_connection_string(student_id)
                
                # For each processed PDF, try to retrieve relevant content
                for pdf in subject_pdfs[:5]:  # Limit to 5 PDFs for performance
                    try:
                        # Create a collection name specific to this PDF
                        collection_name = f"pdf_{pdf.id}"
                        
                        # Initialize PGVector with the student-specific connection
                        pdf_vector_store = PGVector(
                            embeddings=self.embeddings,
                            collection_name=collection_name,
                            connection=connection_string,
                            use_jsonb=True
                        )
                        
                        # Create a retriever from the vector store
                        pdf_retriever = pdf_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2})
                        
                        # Get relevant documents from this PDF
                        pdf_results = pdf_retriever.get_relevant_documents(question)
                        
                        # Add source information to each document
                        for doc in pdf_results:
                            if not hasattr(doc, "metadata"):
                                doc.metadata = {}
                            doc.metadata["source"] = f"From your document: {pdf.title}"
                        
                        # Add to the list of retrieved documents
                        pdf_docs.extend(pdf_results)
                    except Exception as e:
                        print(f"Error retrieving from PDF {pdf.id}: {str(e)}")
                        continue
        except Exception as e:
            print(f"Error accessing student PDFs: {str(e)}")
        
        return pdf_docs
    
    def learn_subject(self, 
                    subject: str, 
                    question: str, 
                    student_id: str, 
                    session_id: str = None,
                    include_pdfs: bool = True) -> Tuple[str, int]:
        """Learn about a specific subject.
        
        Args:
            subject: Subject to learn about (science, social_science, mathematics, english, hindi)
            question: Question about the subject
            student_id: ID of the student
            session_id: JWT token from X-Auth-Session (optional)
            include_pdfs: Whether to include user's PDFs in the answer
            
        Returns:
            Tuple of (answer, status_code)
        """
        try:
            # Validate subject
            if subject not in self.SUBJECT_COLLECTIONS:
                return f"Invalid subject: {subject}. Valid subjects are: {', '.join(self.SUBJECT_COLLECTIONS.keys())}", 400
            
            # Add the user question to the subject-specific chat history
            if session_id:
                self._add_to_chat_history(student_id, session_id, subject, question, is_ai=False)
            else:
                # Even without session_id, store in sahasra_history
                history_data = {
                    "subject": subject,
                    "message": question,
                    "is_ai": False,
                    "time": datetime.utcnow()
                }
                self.history_repository.add_history_item(student_id, history_data)
            
            # STEP 1: Get relevant documents from the subject knowledge base
            subject_collection = self.SUBJECT_COLLECTIONS[subject]
            
            # Initialize PGVector with the subject collection
            subject_vector_store = PGVector(
                embeddings=self.embeddings,
                collection_name=subject_collection,
                connection=settings.PGVECTOR_CONNECTION_STRING,
                use_jsonb=True
            )
            
            # Create a retriever from the vector store
            subject_retriever = subject_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
            
            # Get relevant documents from subject collection
            subject_docs = subject_retriever.get_relevant_documents(question)
            
            # Extract content from subject knowledge documents
            subject_context = [doc.page_content for doc in subject_docs]
            
            # STEP 2: Get relevant documents from user's PDFs if requested
            pdf_docs = []
            if include_pdfs:
                pdf_docs = self._get_subject_pdf_content(student_id, subject, question)
            
            # Extract content from PDF documents with source info
            pdf_context = [f"{doc.metadata.get('source', '')}: {doc.page_content}" for doc in pdf_docs]
            
            # STEP 3: Combine both contexts
            all_context_parts = []
            
            # Add subject knowledge context if available
            if subject_context:
                all_context_parts.append(f"{subject.capitalize()} Knowledge:")
                all_context_parts.extend(subject_context)
            
            # Add personal PDF context if available
            if pdf_context:
                all_context_parts.append("\nFrom Your Documents:")
                all_context_parts.extend(pdf_context)
            
            # Join all context parts
            context = "\n".join(all_context_parts) if all_context_parts else ""
            
            # STEP 4: Get chat history for this subject
            history = None
            if session_id:
                history = self._setup_chat_history(student_id, session_id, subject)
            
            # STEP 5: Create prompt with subject-specific system message
            system_prompt = self.SUBJECT_PROMPTS.get(subject, "You are an educational assistant.")
            
            prompt_messages = [
                ("system", f"{system_prompt}\n\n"
                          "Use the provided context to give accurate answers. "
                          "If your answer includes information from the student's own documents, clearly indicate this. "
                          "If you're unsure or the answer is not in the context, be honest about it.\n\n"
                          "Context: {context}")
            ]
            
            # Add chat history if available
            if history and history.messages:
                prompt_messages.append(MessagesPlaceholder(variable_name="history"))
            
            # Add the current question
            prompt_messages.append(("human", "{question}"))
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages(prompt_messages)
            
            # STEP 6: Create chain and generate answer
            chain = prompt | self.ug_llm | StrOutputParser()
            
            # Prepare input for the chain
            chain_input = {
                "context": context,
                "question": question
            }
            
            # Add history if available
            if history and history.messages:
                chain_input["history"] = history.messages
            
            # Run chain to get answer
            answer = chain.invoke(chain_input)
            
            # STEP 7: Add the AI's answer to the chat history
            if session_id:
                self._add_to_chat_history(student_id, session_id, subject, answer, is_ai=True)
            else:
                # Even without session_id, store in sahasra_history
                history_data = {
                    "subject": subject,
                    "message": answer,
                    "is_ai": True,
                    "time": datetime.utcnow()
                }
                self.history_repository.add_history_item(student_id, history_data)
            
            return answer, 200
            
        except Exception as e:
            error_message = f"Error learning about {subject}: {str(e)}"
            print(error_message)
            return error_message, 500
    
    # Subject-specific convenience methods
    
    def learn_science(self, question: str, student_id: str, session_id: str = None, include_pdfs: bool = True) -> Tuple[str, int]:
        """Learn about science.
        
        Args:
            question: Question about science
            student_id: ID of the student
            session_id: JWT token from X-Auth-Session (optional)
            include_pdfs: Whether to include user's PDFs in the answer
            
        Returns:
            Tuple of (answer, status_code)
        """
        return self.learn_subject("science", question, student_id, session_id, include_pdfs)
    
    def learn_social_science(self, question: str, student_id: str, session_id: str = None, include_pdfs: bool = True) -> Tuple[str, int]:
        """Learn about social science.
        
        Args:
            question: Question about social science
            student_id: ID of the student
            session_id: JWT token from X-Auth-Session (optional)
            include_pdfs: Whether to include user's PDFs in the answer
            
        Returns:
            Tuple of (answer, status_code)
        """
        return self.learn_subject("social_science", question, student_id, session_id, include_pdfs)
    
    def learn_mathematics(self, question: str, student_id: str, session_id: str = None, include_pdfs: bool = True) -> Tuple[str, int]:
        """Learn about mathematics.
        
        Args:
            question: Question about mathematics
            student_id: ID of the student
            session_id: JWT token from X-Auth-Session (optional)
            include_pdfs: Whether to include user's PDFs in the answer
            
        Returns:
            Tuple of (answer, status_code)
        """
        return self.learn_subject("mathematics", question, student_id, session_id, include_pdfs)
    
    def learn_english(self, question: str, student_id: str, session_id: str = None, include_pdfs: bool = True) -> Tuple[str, int]:
        """Learn about English.
        
        Args:
            question: Question about English
            student_id: ID of the student
            session_id: JWT token from X-Auth-Session (optional)
            include_pdfs: Whether to include user's PDFs in the answer
            
        Returns:
            Tuple of (answer, status_code)
        """
        return self.learn_subject("english", question, student_id, session_id, include_pdfs)
    
    def learn_hindi(self, question: str, student_id: str, session_id: str = None, include_pdfs: bool = True) -> Tuple[str, int]:
        """Learn about Hindi.
        
        Args:
            question: Question about Hindi
            student_id: ID of the student
            session_id: JWT token from X-Auth-Session (optional)
            include_pdfs: Whether to include user's PDFs in the answer
            
        Returns:
            Tuple of (answer, status_code)
        """
        return self.learn_subject("hindi", question, student_id, session_id, include_pdfs) 
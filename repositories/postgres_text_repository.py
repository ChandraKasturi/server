import psycopg2
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime

from config import settings
import json


class PostgresTextRepository:
    """Repository for storing extracted text in PostgreSQL.
    
    This class provides functionality to store and retrieve extracted text
    from PDFs in student-specific PostgreSQL databases.
    """
    
    def __init__(self, base_connection_string: Optional[str] = None):
        """Initialize the PostgreSQL text repository.
        
        Args:
            base_connection_string: Base connection string for PostgreSQL.
                If None, uses the one from settings.
        """
        self.base_connection_string = base_connection_string or settings.PGVECTOR_CONNECTION_STRING
        # Parse the base connection string to extract components
        self._parse_connection_string()
        
    def _parse_connection_string(self):
        """Parse the connection string to extract components."""
        conn_str = self.base_connection_string
        if "postgresql+psycopg://" in conn_str:
            # Extract the part after the scheme
            conn_details = conn_str.split("postgresql+psycopg://")[1]
            user_pass, host_port_db = conn_details.split("@")
            
            if ":" in user_pass:
                self.user, self.password = user_pass.split(":")
            else:
                self.user, self.password = user_pass, ""
                
            host_port, self.base_db = host_port_db.split("/")
            
            if ":" in host_port:
                self.host, self.port = host_port.split(":")
            else:
                self.host, self.port = host_port, "5432"
        else:
            # Default values if parsing fails
            self.host = "localhost"
            self.port = "5432"
            self.user = "postgres"
            self.password = "postgres"
            self.base_db = "postgres"
    
    def _get_connection_params(self, student_id: str) -> Dict[str, str]:
        """Get connection parameters for a specific student database.
        
        Args:
            student_id: ID of the student
            
        Returns:
            Dictionary of connection parameters
        """
        return {
            'host': self.host,
            'port': self.port,
            'user': self.user,
            'password': self.password,
            'dbname': f"student_{student_id.replace('-', '_')}"
        }
    
    def _get_connection_string(self, student_id: str) -> str:
        """Get a connection string for a specific student database.
        
        Args:
            student_id: ID of the student
            
        Returns:
            Connection string for the student database
        """
        safe_student_id = student_id.replace('-', '_')
        base_parts = self.base_connection_string.split('/')
        # Replace the database name in the connection string
        base_parts[-1] = f"student_{safe_student_id}"
        return '/'.join(base_parts)
    
    def _create_student_database(self, student_id: str) -> bool:
        """Create a database for a specific student if it doesn't exist.
        
        Args:
            student_id: ID of the student
            
        Returns:
            True if database was created or already exists
        """
        # Connect to the default database to create a new one
        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=self.base_db
        )
        conn.autocommit = True  # Required for CREATE DATABASE
        
        try:
            with conn.cursor() as cur:
                safe_student_id = student_id.replace('-', '_')
                db_name = f"student_{safe_student_id}"
                
                # Check if database exists
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
                exists = cur.fetchone()
                
                if not exists:
                    # Create the database
                    cur.execute(f"CREATE DATABASE {db_name}")
                    
                    # Connect to the new database to create tables
                    conn.close()
                    conn = psycopg2.connect(**self._get_connection_params(student_id))
                    conn.autocommit = True
                    
                    with conn.cursor() as schema_cur:
                        # Create the pdf_texts table
                        schema_cur.execute("""
                            CREATE TABLE IF NOT EXISTS pdf_texts (
                                id VARCHAR(36) PRIMARY KEY,
                                pdf_id VARCHAR(36) UNIQUE NOT NULL,
                                title TEXT,
                                content TEXT NOT NULL,
                                page_count INTEGER,
                                word_count INTEGER,
                                created_at TIMESTAMP DEFAULT NOW(),
                                metadata JSONB
                            )
                        """)
                        
                        # Create the pdf_chunks table for storing chunked content
                        schema_cur.execute("""
                            CREATE TABLE IF NOT EXISTS pdf_chunks (
                                id VARCHAR(36) PRIMARY KEY,
                                pdf_id VARCHAR(36) NOT NULL,
                                chunk_index INTEGER NOT NULL,
                                page_number INTEGER,
                                content TEXT NOT NULL,
                                word_count INTEGER,
                                created_at TIMESTAMP DEFAULT NOW(),
                                FOREIGN KEY (pdf_id) REFERENCES pdf_texts(pdf_id) ON DELETE CASCADE
                            )
                        """)
                        
                        # Create the necessary indices
                        schema_cur.execute("CREATE INDEX IF NOT EXISTS idx_pdf_texts_pdf_id ON pdf_texts(pdf_id)")
                        schema_cur.execute("CREATE INDEX IF NOT EXISTS idx_pdf_chunks_pdf_id ON pdf_chunks(pdf_id)")
                        schema_cur.execute("CREATE INDEX IF NOT EXISTS idx_pdf_chunks_chunk_index ON pdf_chunks(chunk_index)")
                
                return True
                
        except Exception as e:
            print(f"Error creating student database: {str(e)}")
            return False
        finally:
            conn.close()
    
    def store_pdf_text(self, 
                       student_id: str, 
                       pdf_id: str, 
                       title: str,
                       content: str,
                       page_count: int,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store extracted text from a PDF for a specific student.
        
        Args:
            student_id: ID of the student
            pdf_id: ID of the PDF document
            title: Title of the PDF
            content: Extracted text content
            page_count: Number of pages in the PDF
            metadata: Additional metadata for the PDF
            
        Returns:
            ID of the stored text entry
        """
        # Ensure student database exists
        self._create_student_database(student_id)
        
        # Connect to the student database
        conn = psycopg2.connect(**self._get_connection_params(student_id))
        
        try:
            with conn.cursor() as cur:
                # Generate a unique ID for the text entry
                text_id = str(uuid.uuid4())
                
                # Count words (simple approximation)
                word_count = len(content.split())
                #check if content has null characters

                
                # Store the PDF text
                cur.execute("""
                    INSERT INTO pdf_texts (id, pdf_id, title, content, page_count, word_count, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    text_id,
                    pdf_id,
                    title,
                    content,
                    page_count,
                    word_count,
                    json.dumps(metadata) if metadata else None,
                    datetime.utcnow()
                ))
                
                conn.commit()
                return text_id
                
        except Exception as e:
            conn.rollback()
            print(f"Error storing PDF text: {str(e)}")
            raise
        finally:
            conn.close()
    
    def store_pdf_chunks(self, 
                         student_id: str, 
                         pdf_id: str, 
                         chunks: List[Dict[str, Any]]) -> int:
        """Store PDF text chunks for a specific student.
        
        Args:
            student_id: ID of the student
            pdf_id: ID of the PDF document
            chunks: List of chunk dictionaries with keys:
                    - chunk_index: Index of the chunk
                    - page_number: Page number of the chunk
                    - content: Text content of the chunk
            
        Returns:
            Number of chunks stored
        """
        # Ensure student database exists
        self._create_student_database(student_id)
        
        # Connect to the student database
        conn = psycopg2.connect(**self._get_connection_params(student_id))
        
        try:
            with conn.cursor() as cur:
                # Insert all chunks
                values = []
                for chunk in chunks:
                    chunk_id = str(uuid.uuid4())
                    word_count = len(chunk['content'].split())
                    
                    values.append((
                        chunk_id,
                        pdf_id,
                        chunk['chunk_index'],
                        chunk.get('page_number', 0),
                        chunk['content'],
                        word_count,
                        datetime.utcnow()
                    ))
                
                # Use executemany for efficient insertion
                cur.executemany("""
                    INSERT INTO pdf_chunks 
                    (id, pdf_id, chunk_index, page_number, content, word_count, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, values)
                
                conn.commit()
                return len(chunks)
                
        except Exception as e:
            conn.rollback()
            print(f"Error storing PDF chunks: {str(e)}")
            raise
        finally:
            conn.close()
    
    def get_pdf_text(self, student_id: str, pdf_id: str) -> Optional[Dict[str, Any]]:
        """Get the full text of a PDF document.
        
        Args:
            student_id: ID of the student
            pdf_id: ID of the PDF document
            
        Returns:
            Dictionary with PDF text data or None if not found
        """
        # Connect to the student database
        try:
            conn = psycopg2.connect(**self._get_connection_params(student_id))
            
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, pdf_id, title, content, page_count, word_count, metadata, created_at
                    FROM pdf_texts
                    WHERE pdf_id = %s
                """, (pdf_id,))
                
                result = cur.fetchone()
                
                if result:
                    return {
                        'id': result[0],
                        'pdf_id': result[1],
                        'title': result[2],
                        'content': result[3],
                        'page_count': result[4],
                        'word_count': result[5],
                        'metadata': result[6],
                        'created_at': result[7]
                    }
                return None
                
        except Exception as e:
            print(f"Error getting PDF text: {str(e)}")
            return None
        finally:
            conn.close()
    
    def get_pdf_chunks(self, student_id: str, pdf_id: str) -> List[Dict[str, Any]]:
        """Get chunks of a PDF document.
        
        Args:
            student_id: ID of the student
            pdf_id: ID of the PDF document
            
        Returns:
            List of dictionaries with PDF chunk data
        """
        # Connect to the student database
        try:
            conn = psycopg2.connect(**self._get_connection_params(student_id))
            
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, pdf_id, chunk_index, page_number, content, word_count, created_at
                    FROM pdf_chunks
                    WHERE pdf_id = %s
                    ORDER BY chunk_index ASC
                """, (pdf_id,))
                
                results = cur.fetchall()
                
                chunks = []
                for row in results:
                    chunks.append({
                        'id': row[0],
                        'pdf_id': row[1],
                        'chunk_index': row[2],
                        'page_number': row[3],
                        'content': row[4],
                        'word_count': row[5],
                        'created_at': row[6]
                    })
                
                return chunks
                
        except Exception as e:
            print(f"Error getting PDF chunks: {str(e)}")
            return []
        finally:
            conn.close()
    
    def search_pdf_text(self, student_id: str, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for PDF texts matching a query.
        
        Args:
            student_id: ID of the student
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries with PDF text data
        """
        # Connect to the student database
        try:
            conn = psycopg2.connect(**self._get_connection_params(student_id))
            
            with conn.cursor() as cur:
                # Basic text search using ILIKE
                cur.execute("""
                    SELECT id, pdf_id, title, substring(content, 1, 300) as content_preview, 
                           page_count, word_count, metadata, created_at
                    FROM pdf_texts
                    WHERE content ILIKE %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (f"%{query}%", limit))
                
                results = cur.fetchall()
                
                texts = []
                for row in results:
                    texts.append({
                        'id': row[0],
                        'pdf_id': row[1],
                        'title': row[2],
                        'content_preview': row[3],
                        'page_count': row[4],
                        'word_count': row[5],
                        'metadata': row[6],
                        'created_at': row[7]
                    })
                
                return texts
                
        except Exception as e:
            print(f"Error searching PDF texts: {str(e)}")
            return []
        finally:
            conn.close()
            
    def delete_pdf_text(self, student_id: str, pdf_id: str) -> bool:
        """Delete a PDF text and its chunks.
        
        Args:
            student_id: ID of the student
            pdf_id: ID of the PDF document
            
        Returns:
            True if deletion was successful
        """
        # Connect to the student database
        try:
            conn = psycopg2.connect(**self._get_connection_params(student_id))
            conn.autocommit = False
            
            try:
                with conn.cursor() as cur:
                    # Delete the PDF text (will cascade to chunks)
                    cur.execute("DELETE FROM pdf_texts WHERE pdf_id = %s", (pdf_id,))
                    rows_deleted = cur.rowcount
                    
                conn.commit()
                return rows_deleted > 0
                
            except Exception as e:
                conn.rollback()
                print(f"Error deleting PDF text: {str(e)}")
                return False
                
        except Exception as e:
            print(f"Error connecting to database: {str(e)}")
            return False
        finally:
            conn.close() 
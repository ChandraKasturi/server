import uuid
from typing import List, Optional, Any, Dict
import psycopg2
from langchain.embeddings import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import PGVector
from langchain_core.documents import Document

from config import settings

class PgVectorRepository:
    """Repository for PostgreSQL vector operations.
    
    This class provides a wrapper around direct PostgreSQL operations
    using psycopg for vector storage and retrieval.
    """
    
    def __init__(self, table: str, openai_api_key: Optional[str] = None):
        """Initialize PostgreSQL vector store.
        
        Args:
            table: The table name to store vectors in
            openai_api_key: API key for OpenAI embeddings. If None, uses the one from settings.
        """
        self.table = table
        self.api_key = openai_api_key or settings.OPENAI_API_KEY
        self.embeddings = OpenAIEmbeddings(api_key=self.api_key)
        '''self.embeddings = GoogleGenerativeAIEmbeddings(google_api_key=settings.GOOGLE_API_KEY)'''
        
        # Parse connection string to extract connection parameters
        conn_str = settings.PGVECTOR_CONNECTION_STRING
        if "postgresql+psycopg://" in conn_str:
            # For simplicity, extract the part after the scheme
            conn_details = conn_str.split("postgresql+psycopg://")[1]
            user_pass, host_port_db = conn_details.split("@")
            
            if ":" in user_pass:
                user, password = user_pass.split(":")
            else:
                user, password = user_pass, ""
                
            host_port, db = host_port_db.split("/")
            
            if ":" in host_port:
                host, port = host_port.split(":")
            else:
                host, port = host_port, "5432"
                
            self.connection_params = {
                'host': host,
                'port': port,
                'user': user,
                'password': password,
                'dbname': db
            }
        else:
            # Fallback for direct connection params
            self.connection_params = {
                'host': 'localhost',
                'port': '5432',
                'user': 'myuser',
                'password': 'mypassword',
                'dbname': 'cbse_x'
            }
        
        self.conn = psycopg2.connect(**self.connection_params)
        
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of document IDs
        """
        doc_ids = [str(uuid.uuid4()) for _ in documents]
        
        with self.conn.cursor() as cur:
            for i, doc in enumerate(documents):
                # Generate embedding for the document
                embedding_vector = self.embeddings.embed_documents([doc.page_content])[0]
                
                # Insert document with its embedding
                cur.execute(
                    f"INSERT INTO {self.table} (id, content, embedding) VALUES (%s, %s, %s)",
                    (doc_ids[i], doc.page_content, embedding_vector)
                )
                
            self.conn.commit()
            
        return doc_ids
    
    def search(self, query_text: str, match_count: int = 5, filter_json: str = '{}') -> List[Dict]:
        """Search for similar documents.
        
        Args:
            query_text: Text to search for
            match_count: Maximum number of results to return
            filter_json: JSON string with additional filter criteria
            
        Returns:
            List of matching documents
        """
        # Generate embedding for the query
        query_vector = self.embeddings.embed_query(query_text)
        
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM match_{self.table}(%s::vector, %s::int, %s::jsonb);",
                (query_vector, match_count, filter_json)
            )
            return cur.fetchall()
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

class LangchainVectorRepository:
    """Repository for vector operations using LangChain's PGVector.
    
    This class provides a wrapper around LangChain's PGVector implementation
    for vector storage and retrieval.
    """
    
    def __init__(self, collection_name: str, openai_api_key: Optional[str] = None):
        """Initialize LangChain PGVector store.
        
        Args:
            collection_name: The collection name in PGVector
            openai_api_key: API key for OpenAI embeddings. If None, uses the one from settings.
        """
        self.collection_name = collection_name
        self.api_key = openai_api_key or settings.OPENAI_API_KEY
        self.embeddings = OpenAIEmbeddings(api_key=self.api_key)
        
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=collection_name,
            connection_string=settings.PGVECTOR_CONNECTION_STRING,
            use_jsonb=True,
        )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of document IDs
        """
        # Generate UUIDs for each document
        ids = [str(uuid.uuid4()) for _ in documents]
        
        # Add documents to vector store with the generated IDs
        self.vector_store.add_documents(documents, ids=ids)
        
        return ids
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents using similarity search.
        
        Args:
            query: Text to search for
            k: Number of documents to return
            
        Returns:
            List of similar documents
        """
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple[Document, float]]:
        """Search for similar documents with relevance scores.
        
        Args:
            query: Text to search for
            k: Number of documents to return
            
        Returns:
            List of tuples containing document and relevance score
        """
        return self.vector_store.similarity_search_with_score(query, k=k) 
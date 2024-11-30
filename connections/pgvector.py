import psycopg
from langchain.embeddings import OpenAIEmbeddings
import uuid
class PostgresVectorStore:
    def __init__(self, connection_params, table, key):
        self.conn = psycopg.connect(**connection_params)
        self.table = table
        self.embeddings = OpenAIEmbeddings(api_key=key)  # Initialize your embedding model here

    def add_documents(self, documents):
        with self.conn.cursor() as cur:
            for doc in documents:
                embedding_vector = self.embeddings.embed_documents([doc.page_content])[0]
                cur.execute(f"INSERT INTO {self.table} (id,content, embedding) VALUES (%s, %s, %s)",
                            (uuid.uuid4(),doc.page_content, embedding_vector))
            self.conn.commit()

    def search(self, query_vector,match_count=5, filter_json='{}'):
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT * FROM match_{self.table}(%s::vector, %s::int, %s::jsonb);",
                        (query_vector, match_count, filter_json))
            return cur.fetchall()
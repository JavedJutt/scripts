from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from tqdm.auto import tqdm
import math
import os
from dotenv import load_dotenv
import uuid


load_dotenv()


class PineconeManager:
    def __init__(self):
        self.api_key = os.getenv('PINECONE_API_KEY')
        self.index_name = os.getenv('PINECONE_INDEX_NAME')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')

        # Settings class handles validation, so no need for explicit checks here

        self.pc = Pinecone(api_key=self.api_key)
        
        # Initialize OpenAI only if key is available
        if self.openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
            except Exception as e:
                print(f"Warning: Failed to initialize OpenAI client: {e}")
                self.openai_client = None
        else:
            self.openai_client = None
            
        self.index = None # This will be set when an index is created or retrieved

    def create_index(self, dimension: int = 1536, metric: str = 'cosine', replace_existing: bool = False):
        existing_indexes = self.list_indexes()
        if self.index_name in existing_indexes:
            if replace_existing:
                print(f"Index '{self.index_name}' already exists. Deleting and recreating.")
                self.delete_index()
            else:
                print(f"Index '{self.index_name}' already exists. Skipping creation.")
                self.index = self.pc.Index(self.index_name)
                return

        print(f"Creating index '{self.index_name}' with dimension {dimension} and metric '{metric}'.")
        self.pc.create_index(name=self.index_name, dimension=dimension, metric=metric, spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ))
        self.index = self.pc.Index(self.index_name)
        print(f"Index '{self.index_name}' created successfully.")



    def upsert_supplier(self, supplier: dict, namespace: str = "dmc") -> str:
        """
        Upsert a new supplier using the same structure as process_exhibitors().
        supplier: dict with keys:
        name, country, address, phone, email, website, description
        Returns the generated vector_id (UUID-based).
        """
        # Build text (same as CSV pipeline)
        text = f"Company Name: {supplier.get('name','')} \n "
        if supplier.get('country'):
            text += f"Country: {supplier['country']} \n "
        if supplier.get('address'):
            text += f"Address: {supplier['address']} \n "
        if supplier.get('phone'):
            text += f"Phone: {supplier['phone']} \n "
        if supplier.get('email'):
            text += f"Email: {supplier['email']} \n "
        if supplier.get('listing_url'):
            text += f"Listing_url: {supplier['listing_url']} \n "
        if supplier.get('description'):
            text += f"Description: {supplier['description']} \n "

        # Metadata
        metadata = {
            "name": supplier.get("name",""),
            "country": supplier.get("country",""),
            "address": supplier.get("address",""),
            "phone": supplier.get("phone",""),
            "email": supplier.get("email",""),
            "listing_url": supplier.get("listing_url",""),
            "text": text
        }

        # UUID4 (random) → safe for Pinecone ids; no hyphens via .hex
        vector_id = f"{uuid.uuid4().hex}"
        print(vector_id)

        embeddings = self.create_embeddings([text])
        if not embeddings:
            print("❌ Failed to create embedding for supplier")
            return ""

        vector = {"id": vector_id, "values": embeddings[0], "metadata": metadata}
        self.upsert_data([vector], namespace=namespace)
        print(f"✅ Supplier '{supplier.get('name')}' upserted into namespace '{namespace}' as {vector_id}")
        return vector_id

    def delete_index(self):
        if self.index_name in self.pc.list_indexes():
            self.pc.delete_index(self.index_name)
            print(f"Index '{self.index_name}' deleted successfully.")
            self.index = None
        else:
            print(f"Index '{self.index_name}' does not exist.")

    def list_indexes(self):
        indexes = self.pc.list_indexes()
        return [idx['name'] for idx in indexes]

    def get_or_create_index(self):
        if self.index is None:
            existing_indexes = self.list_indexes()
            if self.index_name in existing_indexes:
                print(f"Index '{self.index_name}' found. Connecting to existing index.")
                self.index = self.pc.Index(self.index_name)
            else:
                print(f"Index '{self.index_name}' does not exist. Calling create_index.")
                self.create_index()
                self.index = self.pc.Index(self.index_name)
                print(f"Index '{self.index_name}' is now set after creation.")
        return self.index

    def create_embeddings(self, texts: list[str], model: str = "text-embedding-ada-002") -> list[list[float]]:
        if not texts:
            print("No texts provided for embedding. Returning empty list.")
            return []
        try:
            response = self.openai_client.embeddings.create(
                input=texts,
                model=model
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return []

    def upsert_data(self, vectors: list[tuple[str, list[float], dict]], namespace: str = "exhibitors", batch_size: int = 100):
        if not vectors:
            print("No vectors to upsert.")
            return

        print(f"Attempting to upsert {len(vectors)} vectors to index '{self.index_name}' in namespace '{namespace}' with batch size {batch_size}...")
        try:
            index = self.get_or_create_index()
            
            # Calculate number of batches
            num_batches = math.ceil(len(vectors) / batch_size)

            for i in tqdm(range(num_batches), desc="Upserting vectors"): # Add tqdm progress bar
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(vectors))
                batch = vectors[start_idx:end_idx]
                
                index.upsert(
                    vectors=batch,
                    namespace=namespace
                )
            print(f"Successfully upserted {len(vectors)} vectors to index '{self.index_name}'.")
        except Exception as e:
            print(f"Error upserting data: {e}")

    def query_index(self, query_embedding: list[float], namespace:str="supplier", top_k: int = 20, include_values: bool = False, include_metadata: bool = True, filter: dict = None ):
        try:
            if self.index is None:
                self.index = self.get_or_create_index()
            return self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_values=include_values,
                include_metadata=include_metadata,
                namespace=namespace,
                filter=filter
            )
        except Exception as e:
            print(f"Error querying index: {e}")
            return None

    def describe_index_stats(self):
        try:
            index = self.get_or_create_index()
            return index.describe_index_stats()
        except Exception as e:
            print(f"Error describing index stats: {e}")
            return None

from src.helper import load_pdf_file, text_splitter, download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os
from pinecone import ServerlessSpec, Pinecone

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')


extracted_data = load_pdf_file("data/")
text_chunks = text_splitter(extracted_data)
embeddings = download_hugging_face_embeddings()

index_name = "medicalchatbot"

pc = Pinecone(api_key=PINECONE_API_KEY)
if 'medicalchatbot' not in pc.list_indexes().names():
    pc.create_index(
        name='medicalchatbot',
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )


index_name="medicalchatbot"

docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

class RAGDatabase:
    def __init__(self, models_instance, db_path="./chroma_db"):
        self.models = models_instance
        self.chroma_client = chromadb.PersistentClient(path=db_path, settings=Settings(allow_reset=True))
        
        try:
            self.chroma_client.delete_collection(name="readme_rag")
        except:
            pass
        self.collection = self.chroma_client.create_collection(name="readme_rag")

    def list_collections(self):
        return self.chroma_client.list_collections()
    
    def reset_database(self):
        return self.chroma_client.reset()

    def process_and_store(self, documents):
        chunk_args = {
            "chunk_size" : 1500,
            "chunk_overlap" : 200
        }
        splitter_dictionary = {
            "C" : RecursiveCharacterTextSplitter.from_language(
                language=Language.C,
                **chunk_args
            ),
            "CPP" : RecursiveCharacterTextSplitter.from_language(
                language=Language.CPP,
                **chunk_args
            ),
            "DEFAULT" : 
            RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                **chunk_args
            ),
            "JAVA" : RecursiveCharacterTextSplitter.from_language(
                language=Language.JAVA,
                **chunk_args
            ),
            "MARKDOWN" : RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN,
                **chunk_args
            ),
            "PYTHON" : RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                **chunk_args
            ),
        }

        chunk_id = 0
        for doc in documents:
            splitter = splitter_dictionary.get(doc["language"], "DEFAULT")
            chunks = splitter.split_text(doc['content'])
            print(f"Processing {doc['path']} -> {len(chunks)} chunks")

            for chunk in chunks:

                embedding = self.models.get_gemini_embedding(chunk)
                
                self.collection.add(
                    ids=[f"chunk_{chunk_id}"],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{"file_path": doc['path']}]
                )
                chunk_id += 1
                
        print(f"\nSuccessfully stored {chunk_id} chunks in the database.")
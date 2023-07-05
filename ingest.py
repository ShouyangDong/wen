"""Save the dataset into embedding datavector."""
import os
import pickle
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def save_database(embedding_cache_path, file_name):
    # Check if embedding cache path exists
    if not os.path.exists(embedding_cache_path):
        with open(file_name, "r", encoding="utf-8") as f:
            content = f.read()

        print("Encode the corpus. This might take a while")
        corpus_embeddings = model.encode(
            content, show_progress_bar=True, convert_to_tensor=True
        )

        print("Store file on disc")
        with open(embedding_cache_path, "wb") as fOut:
            pickle.dump(
                {"sentences": content, "embeddings": corpus_embeddings}, fOut
            )
    else:
        raise RuntimeError("File exist")


if __name__ == "__main__":
    embedding_cache_path = "./Cambricon-documentation.pkl"
    file_name = "new_docs_segments.txt"
    save_database(embedding_cache_path, file_name)

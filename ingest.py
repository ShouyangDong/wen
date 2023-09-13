"""Save the dataset into embedding datavector."""
import os
import pickle

from util import get_embedding_from_api


def save_database(embedding_cache_path, file_name):
    # Check if embedding cache path exists
    if not os.path.exists(embedding_cache_path):
        with open(file_name, "r", encoding="utf-8") as f:
            content = f.read()

        print("Encode the corpus. This might take a while")
        corpus_embeddings = get_embedding_from_api(content)

        print("Store file on disc")
        with open(embedding_cache_path, "wb") as fOut:
            pickle.dump({"sentences": content, "embeddings": corpus_embeddings}, fOut)
    else:
        raise RuntimeError("File exist")


if __name__ == "__main__":
    embedding_cache_path = "./documentation.pkl"
    file_name = "new_docs_segments.txt"
    save_database(embedding_cache_path, file_name)
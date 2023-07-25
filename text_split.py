"""Split the text by given chunk size."""
from numpy import array, average


TEXT_EMBEDDING_CHUNK_SIZE = 2048
# Split a text into smaller chunks of size n, preferably ending at the end of a sentence
def chunks(text, size, tokenizer):
    """Yield successive n-sized chunks from text."""
    tokens = tokenizer.encode(text)
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * size), len(tokens))
        while j > i + int(0.5 * size):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if (
                chunk.endswith(".")
                or chunk.endswith("\n")
                or chunk.endswith("。")
                or chunk.endswith("！")
                or chunk.endswith("？")
            ):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * size):
            j = min(i + size, len(tokens))
        yield tokens[i:j]
        i = j


def create_embeddings_for_text(text, tokenizer):
    """Return a list of tuples (text_chunk, embedding) and an average embedding for a text."""
    token_chunks = list(chunks(text, TEXT_EMBEDDING_CHUNK_SIZE, tokenizer))
    text_chunks = [tokenizer.decode(chunk) for chunk in token_chunks]
    return text_chunks


def get_col_average_from_list_of_lists(list_of_lists):
    """Return the average of each column in a list of lists."""
    if len(list_of_lists) == 1:
        return list_of_lists[0]

    list_of_lists_array = array(list_of_lists)
    average_embedding = average(list_of_lists_array, axis=0)
    return average_embedding.tolist()


def get_embeddings(text_array, tokenizer):
    """Get the embedding according to the responding tokenizer."""
    return tokenizer.encode(text_array)


# Split a text into smaller chunks of size n, preferably ending at the end of a sentence
def text_chunks(text, n):
    """Yield successive n-sized chunks from text."""
    i = 0
    while i < len(text):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(text))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = text[i:j]
            if (
                chunk.endswith(".")
                or chunk.endswith("\n")
                or chunk.endswith("。")
                or chunk.endswith("！")
                or chunk.endswith("？")
            ):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(text))

        yield text[i:j]
        i = j


def create_chunk_for_text(text, chunk_size=2000):
    """Return a list text_chunk."""
    return list(text_chunks(text, TEXT_EMBEDDING_CHUNK_SIZE))

import json

import numpy as np
import requests


def get_embedding_from_api(
    word, model="chinese-alpaca-plus-13B-clean-qa-cambricon-epoch-20"
):
    """Get the corresponding embedding of word."""
    url = "http://10.100.207.69:8001/v1/create_embeddings"
    headers = {"Content-Type": "application/json"}
    data = json.dumps({"model": model, "input": word})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        embedding = np.array(response.json()["data"][0]["embedding"])
        return embedding
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

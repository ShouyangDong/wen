from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = "static"

from tqdm import tqdm
import json
import openai
from datetime import datetime
import torch

from elasticsearch import Elasticsearch
import os
import urllib3
import numpy as np
from numpy.linalg import norm
import requests
from scipy.spatial.distance import cosine
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# only do once
# 连接到 Elasticsearch 服务器
es = Elasticsearch(
    ["https://10.100.158.12:9200"],
    verify_certs=False,
    http_auth=("elastic", "*qdSlBZ7AmkaHhyf0VLN"),
)

# 设置OpenAI API
openai.api_key = "YOUR_API_KEY"
openai.api_base = "http://10.100.207.69:8001/v1"



def get_embedding_from_api(word, model="chinese-alpaca-plus-13B-clean-qa-cambricon-epoch-20"):
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


def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def query_question(USER_QUESTION):
    model = "chinese-alpaca-plus-13B-clean-qa-cambricon-epoch-20"
    
    # search function
    def strings_ranked_by_relatedness(query, corpus_list, top_n=100):
        """Returns a list of strings and relatednesses, sorted from most related to least."""
        query_embedding = get_embedding_from_api(query)
        # Calculate cosine similarity
        cosine_similarities = []
        
        for corpus in corpus_list:
            corpus_embedding = get_embedding_from_api(corpus, model)
            cosine_similarities.append(cosine_similarity(query_embedding, corpus_embedding))
        # Sort articles by cosine similarity
        score = np.sort(cosine_similarities)
        index = np.argsort(cosine_similarities)
        return score[:top_n], index[:top_n]

    def strings_ranked_by_keyword(query, text_embeddings=None, insert=False):
        if insert:
            for doc_content in text_embeddings:
                es.index(
                    index="doc",
                    body={"content": doc_content, "publish_date": datetime.now()},
                )

        # 搜索包含关键字 "elasticsearch" 的文档
        res = es.search(index="doc", body={"query": {"match": {"content": query}}})
        formatted_top_results = set()
        for hit in res["hits"]["hits"]:
            formatted_top_results.add(hit["_source"]["content"])
        formatted_top_results = list(formatted_top_results)
        return formatted_top_results

    # examples(vector search is not used)
    # related_corpus_section = []
    # indexs, relatednesses = strings_ranked_by_relatedness(USER_QUESTION, corpus_embeddings, top_n=10)
    # for index, relatedness in zip(indexs, relatednesses):
    #     related_corpus_section.append(corpus_sentences[index])

    query_prompt = f"""
        请提取给定的问题中的关键字， 返回一个关键词列表。
        注意关键字必须为问题中出现的词语。

        问题：{USER_QUESTION}

        关键字列表格式为: [关键字1， 关键字2]
    """
    query_completion = openai.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": query_prompt}]
    )
    query_response = query_completion.choices[0].message.content.strip()
    query_response = query_response.replace("[", "")
    query_response = query_response.replace("]", "")
    query_response = query_response.split(",")

    related_corpus_section = []
    for QUESTION in list(query_response):
        related_corpus_section.extend(strings_ranked_by_keyword(QUESTION))

    completion = openai.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": USER_QUESTION}]
    )
    final_response = completion.choices[0].message.content.strip()
    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(5, len(related_corpus_section))

    from text_split import create_chunk_for_text

    related_corpus_sentences = []
    for sentence in related_corpus_section:
        # Here the max_seq_length of llama model is defined as 2048
        related_corpus_sentences.extend(create_chunk_for_text(sentence))
    if len(related_corpus_sentences) == 0:
        return final_response

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    top_results = strings_ranked_by_relatedness(final_response, related_corpus_sentences, top_n=top_k)

    # If the highest score < 0.6, just print the hypothetical ideal answer
    if top_results[0][0] >= 0.6:
        try:
            selected_data = related_corpus_sentences[top_results[1][0]]
            prompt = f"""
            请你参考：
            {selected_data.strip()}
            （请忽略与问题无关的部分）
            来回答问题：
            {USER_QUESTION}
            """
            completion = openai.ChatCompletion.create(
                model=model, messages=[{"role": "user", "content": prompt}],
            )
            final_response = completion.choices[0].message.content.strip()
        except Exception as e:
            return None
    return final_response


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    question = request.args.get("msg")
    chat_answer = query_question(question)
    return chat_answer


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

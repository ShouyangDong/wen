from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'
import random                                                                                                                                                                                              
from tqdm import tqdm
import json
import openai
import pickle
from sentence_transformers import SentenceTransformer, util
import torch
from datetime import datetime
from elasticsearch import Elasticsearch
import os
 
# only do once
# 连接到 Elasticsearch 服务器
es = Elasticsearch(["https://10.100.158.12:9200"], verify_certs=False, http_auth=('elastic', '*qdSlBZ7AmkaHhyf0VLN'))

embedder = SentenceTransformer('all-MiniLM-L6-v2')
# 设置OpenAI API
openai.api_key = "YOUR_API_KEY"
openai.api_base = "http://10.100.209.14:8001/v1"
             
def query_question(USER_QUESTION):
    model = "chinese-alpaca-plus-13B-docs-human-filtered-api-epoch-25"
    # search function
    def strings_ranked_by_relatedness(
        query,
        corpus_embeddings,
        top_n= 100
    ):
        """Returns a list of strings and relatednesses, sorted from most related to least."""
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_n)
        return top_results[1][:top_n], top_results[0][:top_n]
    
    def strings_ranked_by_keyword(query, text_embeddings=None, insert=False):
        if insert:
            for doc_content in text_embeddings:
                es.index(index='doc', body={
                    'content': doc_content,
                    'publish_date': datetime.now(),
                })
        
        # 搜索包含关键字 "elasticsearch" 的文档
        res = es.search(index='doc', body={'query': {'match': {'content': query}}})
        print("Got %d Hits:" % res['hits']['total']['value'])
        formatted_top_results = set()
        for hit in res['hits']['hits']:
            formatted_top_results.add(hit['_source']['content'])
        formatted_top_results = list(formatted_top_results)
        return formatted_top_results

    # examples(vector search is not used)
    # related_corpus_section = []
    # indexs, relatednesses = strings_ranked_by_relatedness(USER_QUESTION, corpus_embeddings, top_n=10)
    # for index, relatedness in zip(indexs, relatednesses):
    #     related_corpus_section.append(corpus_sentences[index])

    query_prompt = f'''
        请提取给定的问题中的关键字， 返回一个关键词列表。
        注意关键字必须为问题中出现的词语。

        问题：{USER_QUESTION}

        关键字列表格式为: [关键字1， 关键字2]
    '''
    query_completion = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": query_prompt}]
    )
    query_response = query_completion.choices[0].message.content
    query_response = query_response.replace("[", "")
    query_response = query_response.replace("]", "")
    query_response = query_response.split(",")
    
    related_corpus_section = []
    for QUESTION in list(query_response):
        related_corpus_section.extend(strings_ranked_by_keyword(QUESTION))
    
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": USER_QUESTION}]
    )
    final_response = completion.choices[0].message.content
    # print(related_corpus_section)
    # # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(5, len(related_corpus_section))
    query_embedding = embedder.encode(final_response, convert_to_tensor=True)

    from text_split import create_embeddings_for_text

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    related_corpus_sentences = []
    for sentence in related_corpus_section:
        related_corpus_sentences.extend(create_embeddings_for_text(sentence, tokenizer))
    if len(related_corpus_sentences) == 0:
        return final_response
    related_corpus_embeddings = embedder.encode(related_corpus_sentences, convert_to_tensor=True)
    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, related_corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    
    # If the highest score < 0.6, just print the hypothetical ideal answer
    if top_results[0][0] >= 0.6:
        try: 
            selected_data = related_corpus_sentences[top_results[1][0]]
            prompt = f'''
            请你参考：
            {selected_data.strip()}
            （请忽略与问题无关的部分）
            来回答问题：
            {USER_QUESTION}
            '''
            completion = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            final_response = completion.choices[0].message.content
        except Exception as e:
            return None
    return final_response

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    question = request.args.get('msg')
    chat_answer = query_question(question)
    return chat_answer


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
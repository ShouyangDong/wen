from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = "static"
import time
import json
import os
from datetime import datetime
import argparse
import numpy as np
import requests
import urllib3
import glob
import markdown
import markdown.extensions.fenced_code
import markdown.extensions.codehilite
from elasticsearch import Elasticsearch
from numpy.linalg import norm
from scipy.spatial.distance import cosine
from tqdm import tqdm
from util import get_embedding_from_api
from pygments.formatters import HtmlFormatter
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import torch_mlu
torch.set_grad_enabled(False)
from transformers.generation.utils import GenerationConfig
from fastchat.model import load_model, get_conversation_template, add_model_args
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import warnings
warnings.filterwarnings("ignore")
import pickle
import re
# only do once
# connect Elasticsearch server
torch.set_grad_enabled(False)
es = Elasticsearch(
    ["https://10.100.158.12:9200"],
    verify_certs=False,
    http_auth=("elastic", "*qdSlBZ7AmkaHhyf0VLN"),
)

with open("./../data/sentence_embedding.pkl", "rb") as fIn:
    cache_data = pickle.load(fIn)
    corpus_chunk = cache_data['corpus_embeddings']

def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)


def load_baichuan_model(model_path):
    torch_mlu.core.mlu_model.set_memory_strategy(True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    max_memory_mapping = {0: "40GB"}
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, max_memory=max_memory_mapping)
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    return model, tokenizer

def get_baichuan_api_response(model, tokenizer, prompt, max_new_tokens=4096):
    messages = []
    messages.append({"role": "user", "content": prompt})
    response = model.chat(tokenizer, messages)
    return response

def get_reponse(msg, model, tokenizer, model_path):
    conv = get_conversation_template(model_path)
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer([prompt]).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).mlu(),
        do_sample=True,
        temperature=0.7,
        max_new_tokens=2048,
    )
    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(input_ids[0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    return outputs

baichuan_model, baichuan_tokenizer = load_baichuan_model("./Baichuan2-13B-Chat/")
parser = argparse.ArgumentParser()
add_model_args(parser)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--message", type=str, default="Hello! Who are you?")
args = parser.parse_args()
document_model, document_tokenizer = load_model(
    "/data1/AE/llm/models/cambricon-chatbot/",
    args.device,
    args.num_gpus,
    args.max_gpu_memory,
    args.load_8bit,
    args.cpu_offloading,
    debug=args.debug,
)


def rerank_search(USER_QUESTION, answer=None):
    # search function
    def strings_ranked_by_relatedness(query, corpus_list, document_model, document_tokenizer, top_n=100):
        """Returns a list of strings and relatednesses, sorted from most related to least."""
        query_embedding = get_embedding_from_api(document_model, document_tokenizer, query)
        
        # Calculate cosine similarity
        cosine_similarities = []
        corpus_embedding = []
        for corpus in corpus_list:
            corpus_embedding.extend(corpus_chunk[corpus]["embedding"])
        for embedding in corpus_embedding:
            cosine_similarities.append(cosine_similarity(query_embedding, embedding))
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

        res = es.search(index="doc", body={"query": {"match": {"content": query}}})
        formatted_top_results = set()
        for hit in res["hits"]["hits"]:
            formatted_top_results.add(hit["_source"]["content"])
        formatted_top_results = list(formatted_top_results)
        return formatted_top_results
    print("[INFO]***************query: ", USER_QUESTION)
    
    query_prompt = f"""
    请提取以下内容中的关键字，严格按照格式输出，注意函数名或术语必须作为一个整体。
    内容：
    {USER_QUESTION}
    关键字：
    关键字列表格式为: [关键字1， 关键字2]
    """
    
    key_words = get_baichuan_api_response(baichuan_model, baichuan_tokenizer, query_prompt)
    if "[" in key_words:
        key_words = re.findall(r'\[(.*?)\]', key_words)
    else:
        key_words = [key_words]
    key_words = key_words[0].replace("“", "")
    key_words = key_words.replace("”", "")
    key_words = key_words.replace("，", ",")
    key_words = key_words.replace(" ", "")
    key_words = key_words.replace('"', '')
    key_words = key_words.split(",")

    related_corpus_section = []
    for QUESTION in list(key_words):
        if QUESTION.lower() not in ["return", "example", "parameters", "function"]:
            related_corpus_section.extend(strings_ranked_by_keyword(QUESTION))
    if answer is None:
        final_response = get_reponse(USER_QUESTION, document_model, document_tokenizer, "/data1/AE/llm/models/cambricon-chatbot/")
    else:
        final_response = answer
    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(5, len(related_corpus_section))

 
    if len(related_corpus_section) == 0:
        return final_response

    related_corpus_sentences = []
    for sentence in related_corpus_section:
        # Here the max_seq_length of llama model is defined as 2048
        related_corpus_sentences.extend(corpus_chunk[sentence]["corpus"])
    
    # We use cosine-similarity and find the highest top_k scores
    top_results = strings_ranked_by_relatedness(
        final_response, related_corpus_section, document_model, document_tokenizer, top_n=top_k
    )

    selected_data = related_corpus_sentences[top_results[1][0]]    
    return selected_data.strip()

@torch.no_grad()
def long_think(selected_data, USER_QUESTION):
    prompt = f"""
    请你根据以下文本：
    {selected_data}

    来回答问题：
    {USER_QUESTION}
    （请忽略与问题无关的部分） 
    """
    final_response = get_baichuan_api_response(baichuan_model, baichuan_tokenizer, prompt)
    return final_response

def iterative_answer(question):
    answer = None
    context = rerank_search(question, answer)
    n = 3
    for i in range(n):
        answer = long_think(question, context)
        print("-------------------\n")
        print(answer)
        context = rerank_search(question, answer)


if __name__ == "__main__":
    response_content = []
    in_file = "./../data/test_qa_cambricon_bang_c_API.json"
    content = json.load(open(in_file, "r"))
    for ques_dict in tqdm(content):
        qa_map = {}
        query = list(ques_dict.keys())[0]
        answer = list(ques_dict.values())[0]
        final_response = iterative_answer(query)
        print("============================================\n")

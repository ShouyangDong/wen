
from util import get_embedding_from_api
from pygments.formatters import HtmlFormatter
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch_mlu
import numpy as np
from numpy.linalg import norm
 
import torch.nn.functional as F

from fastchat.model import load_model, add_model_args
import argparse
from transformers.generation.utils import GenerationConfig


parser = argparse.ArgumentParser()
add_model_args(parser)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--message", type=str, default="Hello! Who are you?")
args = parser.parse_args()

document_model, document_tokenizer = load_model(
    "/chatbot/",
    args.device,
    args.num_gpus,
    args.max_gpu_memory,
    args.load_8bit,
    args.cpu_offloading,
    debug=args.debug,
)

def load_baichun_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).to('mlu')
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    return model, tokenizer


query = "Hi, I'm Minwoo Park from seoul, korea."
from sklearn.metrics.pairwise import cosine_similarity

corpus_list = ["Hi, I'm Minwoo Park from seoul, korea.",
               "Which athletes won the gold medal in curling at the 2022 Winter Olympics? Hi, I'm Minwoo Park from seoul, korea. ",
              "Hi, I'm Minwoo Park from seoul, korea. Which athletes won the gold medal in curling at the 2022 Winter Olympics?"]
import time

def encode(document_model, document_tokenizer, query):
    input_ids = document_tokenizer.encode(query, return_tensors="pt").to("mlu")
    model_output = document_model(input_ids, output_hidden_states=True)
    data = model_output.hidden_states[-1][0]
    embedding = torch.mean(data, dim=0)
    return embedding


def batch_encode(model, tokenizer, corpus_list):
    encoding = document_tokenizer.batch_encode_plus(
        corpus_list, padding=True, return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to("mlu")
    attention_mask = encoding["attention_mask"].to("mlu")
    model_output = document_model(
        input_ids, attention_mask, output_hidden_states=True
    )
    data = model_output.hidden_states[-1]
    mask = attention_mask.unsqueeze(-1).expand(data.size()).float()
    masked_embeddings = data * mask
    sum_embeddings = torch.sum(masked_embeddings, dim=1)
    seq_length = torch.sum(mask, dim=1)
    embedding = sum_embeddings / seq_length
    normalized_embeddings = F.normalize(embedding, p=2, dim=1)
    return normalized_embeddings.tolist()

print(document_model)
t1 = time.time()
query_embedding = encode(document_model, document_tokenizer, query)
print("=======================")
corpus_embedding = batch_encode(document_model, document_tokenizer, corpus_list)
t2= time.time()
print("+++++++++++++++++++++++")
print(t2-t1)
# compute cosine similarity
cosine_similarities = np.dot(query_embedding, np.transpose(corpus_embedding))/(norm(query_embedding)*norm(corpus_embedding))
print(cosine_similarities)

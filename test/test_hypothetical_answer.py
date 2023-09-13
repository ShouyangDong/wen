from util import get_embedding_from_api
from pygments.formatters import HtmlFormatter
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch_mlu
import argparse
from fastchat.model import load_model, get_conversation_template, add_model_args
import glob
import json
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

response_content = []
for in_file in glob.glob("./../data/test_qa*.json"):
    content = json.load(open(in_file, "r"))
    ques_dict = content[0]
    query = list(ques_dict.keys())[0]
    answer = list(ques_dict.values())[0]

    final_response = get_reponse(query, document_model, document_tokenizer, "/chatbot/")
    print("=======llama=============\n")
    print(final_response)
import re
import markdown
with open("./cntoolkit_3.5.2_cambricon_bang_c_4.5.1.md", "r") as f:
    markdown_string = f.read()

html_string = markdown.markdown(markdown_string)
from bs4 import BeautifulSoup
new_text = "".join(BeautifulSoup(html_string).findAll(string=True))

# 定义匹配代码块的正则表达式模式
pattern = r"```[a-zA-Z]*\n([\s\S]*?)\n```"

# 使用正则表达式搜索匹配的代码块
matches = re.findall(pattern, new_text)


# Import necessary libraries
import os
import sys
import time
import subprocess
import openai
import json
from tqdm import tqdm
# Set OpenAI API key
openai.api_key = "YOUR_API_KEY"
openai.api_base = "http://10.100.207.69:8001/v1"

# Set starting prompt and history for OpenAI chatbot
# Modify it according to your use case (this is just an example)
starting_prompt = dict(
    {
        "role": "system",
        "content": "我将会给你一个BANGC Kernel函数，请描述该BANGC Kernel函数的功能以及对应的接口参数。",
    }
)
history = [
    starting_prompt,
]


# Define function to add docstring to Python functions
def addDocstring(matches):
    currentTime = time.time()
    content = []
    # Loop through all functions in the Python file
    for node in tqdm(matches):
        if time.time() - currentTime < 20:
            # Sleep for remaining time
            time.sleep(20 - (time.time() - currentTime) + 1)


        # Send the function code to ChatGPT API for generating docstring (offcourse use GPT4 API if you hace access to it)
        response = openai.ChatCompletion.create(
            model="vicuna-33b-v1.3",
            temperature=0.2,
            messages=[
                *history,
                {"role": "user", "content": node},
            ],
        )

        currentTime = time.time()

        # Extract the generated docstring from the OpenAI response
        docstring = response.choices[0].message.content
        # Add the function code and generated docstring to history
        history.append({"role": "user", "content": node})
        history.append(
            {
                "role": "assistant",
                "content": docstring,
            }
        )


        content.append(
            {
                "id": f"identity_{len(content)}",
                "conversations": [
                    {"from": "human", "value": docstring},
                    {"from": "gpt", "value": node},
                ]
            }
        )

    with open('./code.json', 'w', encoding='utf8') as json_file:
        json.dump(content, json_file, ensure_ascii=False, indent=2)



addDocstring(matches)
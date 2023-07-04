import json
import sys
import os
import time
import argparse
from typing import List

import openai
import openai.error as openai_error

openai.api_key = "Enter your OpenAI API Key Here"

CHAT_MODEL = "gpt-3.5-turbo"

def gen_prompt(text):
    prompt = f'''
        请根据文本生成尽可能多的问答，每个问答都要独立成篇，并且确保每个问题的答案都能够在文本中找到。此外，请不要在问答中使用指代原文的代词。
        例如：
        文本：
        在法国，巴黎是一个重要的城市，被誉为“光之城”。它是法国最大的城市之一，也是法国政治、商业、文化和旅游的中心。巴黎有许多著名的景点，例如埃菲尔铁塔、卢浮宫、圣母院、凯旋门等。此外，巴黎还以其美食、时装和艺>术而闻名于世界。
        问答
        Q: 巴黎被称为什么？
        A: 巴黎被称为“光之城”。

        Q: 巴黎是什么？
        A: 巴黎是法国最大的城市之一，也是法国政治、商业、文化和旅游的中心。

        Q: 巴黎有哪些著名景点？
        A: 巴黎有许多著名的景点，例如埃菲尔铁塔、卢浮宫、圣母院、凯旋门等。

        Q: 巴黎以什么而闻名于世界？
        A: 巴黎以其美食、时装和艺术而闻名于世界。

        ...

        请根据文本{text}, 开始生成QA问答。
    '''
    return prompt


# Initialize the ArgumentParser object
parser = argparse.ArgumentParser(description='Generate Output using ChatGPT')
def openai_gpt(prompt: str, verbose: bool = False, max_attempts: int = 3) -> str:
    """
    This function sends a prompt to the OpenAI GPT API and returns the response.
    It tries the creation several times (max_attempts) in case of exception.
    If the model is text-davinci-003, it uses the Completion API, otherwise it uses the ChatCompletion API.

    Args:
        prompt (str): Prompt to send to the API.
        config (_type_): Configuration object.
        verbose (bool, optional): If True, print the prompt and response. Defaults to False.
        max_attempts (int, optional): Number of attempts to make in case of exception. Defaults to 3.

    Returns:
        str: The response from the API.
    """
    # send the prompt to gpt and return the response
    # try the creation several times in case of exception
    for attempt in range(1, max_attempts + 1):
        try:
            messages = gen_prompt(text)
            response = openai.ChatCompletion.create(
                model=CHAT_MODEL,
                messages=messages,
                max_tokens=3072,
                temperature=0.5,
                top_p=1.0,
                stop=["\n20", "20.", "20."]
            )
            choices = [choice["message"]["content"] for choice in response["choices"]]

            if verbose:
                print("*" * 20)
                print(f"Model: {CHAT_MODEL}")
                print(f"Prompt: {prompt}")
                print(f"Chat Response: {response['choices'][0]['message']['content']}")

            return choices[0]
        except openai_error.OpenAIError as e:
            if attempt < max_attempts:
                print(f"Error on attempt {attempt}: {e}. Retrying...")
                time.sleep(2)  # Wait for 2 seconds before retrying
            else:
                print(f"Error on attempt {attempt}: {e}. All attempts failed.")
                # we will return None if all attempts failed because raising an exception will stop the program and we will lose all the data we have collected so far
                return None
            
def chat_completions(dataset, num_samples, output_file, verbose):
    """ Generate completions using the Chat Model

    Args:
        dataset (str): Filename of the dataset to read
        num_samples (int): How many instructions in the dataset to process
        output_file (str): output filename
        verbose (bool, optional): If True, print the prompt and response. Defaults to False.
    """
    # generate the prompt for the row
    rows = json.load(dataset)
    num_samples = min(num_samples, len(rows))
    
    with open(output_file,"w") as outfile:
        outfile.write("[\n")
        for i in range(num_samples):
            print (f"Working on {i+1} of {num_samples}")
            prompt = "instruction: '" + rows[i]['instruction'] + "'\ninput: '" + rows[i]['input'] + "'"
            rows[i]['output'] = openai_gpt(prompt, verbose)
            if rows == None:
                break
            data = json.dumps(rows[i], indent=4)
            outfile.write(f"{data}")
            if i < num_samples-1:
                outfile.write(",\n")
        outfile.write("\n]")
        
def main():
    global parser
    
    # Define the arguments
    parser.add_argument('--dataset', type=str, default="alpaca_data_cleaned.json", help='Alpaca Dataset name')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples')
    parser.add_argument('--output_file', type=str, default="chat_dataset.json", help="output filename")
    parser.add_argument('--verbose', type=bool, default=True, help="Verbose output")
    
    # Parse the arguments
    args = parser.parse_args()
    chat_completions(args.dataset, args.num_samples, args.output_file, args.verbose)

if __name__ == "__main__":
    main()
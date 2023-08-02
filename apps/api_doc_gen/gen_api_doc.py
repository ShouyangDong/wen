# Import necessary libraries
import os
import sys
import time
import subprocess
import openai
from redbaron import RedBaron

# Set OpenAI API key
openai.api_key = "YOUR_API_KEY"
openai.api_base = "http://10.100.207.69:8001/v1"

# Set starting prompt and history for OpenAI chatbot
# Modify it according to your use case (this is just an example)
starting_prompt = dict(
    {
        "role": "system",
        "content": "I will send you a code of Python function. You need to analyse the code and the docstring to generate accurate documentation.",
    }
)
history = [
    starting_prompt,
]


# Define function to add docstring to Python functions
def addDocstring(filePath):
    """
    Generates API documentation using OpenAI API

    Args:
        filePath (str): Path to the Python file

    Returns:
        None
    """
    currentTime = time.time()

    # Open the Python file using RedBaron library
    with open(filePath, "r", encoding="utf-8") as file:
        code = RedBaron(file.read())

    output_doc = []
    # Loop through all functions in the Python file
    for node in code.find_all("def"):
        # To avoid OpenAI rate limit (only free trial accounts have rate limit, comment the code below if you have a paid account)
        # Free trial accounts have a hard cap of 1 request every 20 seconds
        if time.time() - currentTime < 20:
            # Sleep for remaining time
            time.sleep(20 - (time.time() - currentTime) + 1)

        # Extract the function code
        function_code = node.dumps()

        # Send the function code to ChatGPT API for generating docstring (offcourse use GPT4 API if you hace access to it)
        response = openai.ChatCompletion.create(
            model="vicuna-33b-v1.3/",
            temperature=0.2,
            messages=[
                *history,
                {"role": "user", "content": function_code},
            ],
        )

        currentTime = time.time()

        # Extract the generated docstring from the OpenAI response
        docstring = response.choices[0].message.content

        # Remove the quotes from the generated docstring if present
        if docstring.startswith('"""') or docstring.startswith("'''"):
            docstring = docstring[3:-3]
        if docstring.startswith('"'):
            docstring = docstring[1:-1]

        # Add the function code and generated docstring to history
        history.append({"role": "user", "content": function_code})
        history.append(
            {
                "role": "assistant",
                "content": docstring,
            }
        )

        output_doc.append(docstring)

    # Write the modified Python file back to disk
    with open(filePath +".txt", "w", encoding="utf-8") as file:
        file.write(output_doc)



# Run the function if this script is called directly
if __name__ == "__main__":
    filePath = sys.argv[1]
    addDocstring(filePath)
import argparse
import os

import openai
import requests

parameters: dict
# set the OpenAI API
openai.api_key = "YOUR_API_KEY"
openai.api_base = "http://10.100.207.69:8001/v1"


def code_review(parameters: dict):
    for filename in parameters["files"]:
        # open text file in read mode
        text_file = open(filename, "r")

        # read whole file to a string
        content = text_file.read()
        try:
            response = openai.ChatCompletion.create(
                model=parameters["model"],
                messages=[
                    {
                        "role": "user",
                        "content": (f"{parameters['prompt']}:\n```{content}```"),
                    }
                ],
                temperature=parameters["temperature"],
            )

            print(
                f"ChatGPT's review about `{filename}` file:\n {response['choices'][0]['message']['content']}"
            )
        except Exception as ex:
            message = (
                f"🚨 Fail code review process for file **{filename}**.\n\n`{str(ex)}`"
            )
            print(message)

        # close file
        text_file.close()


def make_prompt(dev_lang: str) -> str:
    review_prompt = f"Review this {dev_lang} code for potential bugs or Code Smells and suggest improvements. Generate your response in markdown format"

    return review_prompt


def make_resume_for_pull_request(pr) -> str:
    comment = f"""
    Starting review process for this pull request send by **{pr.user.name}**
    **Commits** in this pull request: {pr.commits}

    **Additions**: {pr.additions}
    **Changed** files: {pr.changed_files}
    **Deletions**: {pr.deletions}
    """

    comment = comment.format(length="multi-line", ordinal="second")

    return comment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n", "--files", nargs="+", default=[], help="files to be viewed"
    )
    parser.add_argument("--dev-lang", help="Development language used for this request")
    parser.add_argument(
        "--openai-engine",
        default="chinese-alpaca-plus-13B-clean-qa-cambricon-epoch-20",
        help="GPT-3.5 model to use. Options: text-davinci-003, text-davinci-002, text-babbage-001, text-curie-001, text-ada-001",
    )
    parser.add_argument(
        "--openai-temperature",
        default=0.0,
        help="Sampling temperature to use. Higher values means the model will take more risks. Recommended: 0.5",
    )
    parser.add_argument(
        "--openai-max-tokens",
        default=4096,
        help="The maximum number of tokens to generate in the completion.",
    )

    args = parser.parse_args()

    review_parameters = {
        "files": args.files,
        "prompt": make_prompt(dev_lang=args.dev_lang),
        "temperature": float(args.openai_temperature),
        "max_tokens": int(args.openai_max_tokens),
        "model": args.openai_engine,
    }

    code_review(parameters=review_parameters)

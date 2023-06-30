from tqdm import tqdm
import json
import openai

# Set OpenAI key
openai.api_key = "YOUR_API_KEY"
openai.api_base = "http://10.100.209.14:8001/v1"

def get_eval_prompt(query, result, answer):
    template = f"""您是一名教师，正在为一张测验打分。
    您会得到一个问题、学生的回答和正确答案，并被要求将学生的回答评分为“正确”或“错误”。

    样例格式为：
    问题： 此处为问题
    学生的回答：此处为学生的回答
    正确答案： 此处为正确答案
    评分： 此处为正确或者错误

    仅基于学生答案的事实准确性来评分。忽略学生答案和正确答案之间标点和措辞上的差异。如果学生答案包含比正确答案更多的信息，只要不包含任何相互冲突的陈述即可。开始！

    问题： {query}
    学生的回答： {result}
    正确答案：{answer}
    评分："""
    return template

def check_qa_result(query, result, answer):
    qa_prompt = get_eval_prompt(query, result, answer)
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": qa_prompt}]
    )
    response = completion.choices[0].message.content.strip()
    return response


if __name__ == "__main__":
    database = json.load(open("./qa_pair.json", "r"))
    correct_num = 0
    quize_result = []
    for data in tqdm(database):
        result = check_qa_result(data["question"], data["result"], data["answer"])
        if "正确" in result:
            data["评分"] = "True"
            correct_num += 1
        else:
            data["评分"] = "False"
        quize_result.append(data)

    print("[INFO]*********************correct rate: ", correct_num/len(database))
    with open('qa_pair_result.json', 'w', encoding='utf8') as json_file:
        json.dump(quize_result, json_file, ensure_ascii=False, indent=2)
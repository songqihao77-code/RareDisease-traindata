from openai import OpenAI

client = OpenAI(
    api_key="sk-darvsndztjrjyqdfqeahtttgabcvittiqhebrbraklrlyodh",
    base_url="https://api.siliconflow.cn/v1"
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个有用的助手"},
        {"role": "user", "content": "只回复：ok"}
    ]
)

print(response.choices[0].message.content)
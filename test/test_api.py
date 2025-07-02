
# from openai import OpenAI

# client = OpenAI(api_key="sk-cg-r1w8a8-5e9f673e115a77224b8ed56a8c699f6bbec22a4", base_url="http://223.2.249.70:7019/v1")

# response = client.chat.completions.create(
#     model="r1w8a8",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": "Hello"},
#     ],
#     stream=False
# )

# print(response.choices[0].message.content)


from openai import OpenAI

client = OpenAI(api_key="cg-ai_brain-InternVL3-8B-c56726f1022c122f2e095fc90bbb484VDS", base_url="http://223.2.249.70:7025/v1")

response = client.chat.completions.create(
    model="InternVL3-8B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)

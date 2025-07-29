# 我将根据用户要求修改代码：
# 1. 将 `imageUrls` 指定的图片下载本地后再送入模型处理。
# 2. 每张图都生成一个描述。
# 3. 在输入工单中添加字段：processActionCode="EventTypeRecognition"，processResultCode="", processResultData=""
# 4. 输出时保持 JSON 工单结构，成功则 processResultCode=0，失败为1，processResultData 为富文本。
# 5. 支持 WebSocket 流式传输模拟：每一步都打印一部分，最后统一输出完整结果。

import os
import json
import requests
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# 初始化模型
llm = ChatOpenAI(
    model="r1w8a8",
    temperature=0,
    openai_api_key=os.environ["W8A8_API_KEY"],
    openai_api_base="http://223.2.249.70:7019/v1",
)

def ask_deepseek(prompt: str) -> str:
    response = llm.invoke(prompt)
    return response.content.strip()

vlm = ChatOpenAI(
    model="InternVL3-8B",
    temperature=0,
    openai_api_key=os.environ["INTERNVL_API_KEY"],
    openai_api_base="http://223.2.249.70:7025/v1",
)


# # 下载图片并生成描述
# def download_and_describe_images(image_urls):
#     descriptions = []
#     for idx, url in enumerate(image_urls):
#         try:
#             # 下载图片
#             response = requests.get(url, timeout=10)
#             file_path = f"temp_image_{idx}.jpg"
#             with open(file_path, "wb") as f:
#                 f.write(response.content)
            
#             # 构造 prompt 描述图片
#             prompt = f"请你详细描述以下图片的内容：这是图片文件 `{file_path}` 的内容。"
#             caption = ask_deepseek(prompt)
#             descriptions.append(caption)

#             print(f"[WebSocket]: 图片{idx+1}描述完成：{caption}")
#         except Exception as e:
#             error_msg = f"图片下载或描述失败: {str(e)}"
#             descriptions.append(error_msg)
#             print(f"[WebSocket]: {error_msg}")
#     return descriptions


import base64

def encode_image_to_base64(file_path):
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    return encoded

def download_and_describe_images(image_urls):
    descriptions = []
    for idx, url in enumerate(image_urls):
        try:
            # Step 1: 下载图片
            response = requests.get(url, timeout=10)
            file_path = f"temp_image_{idx}.jpg"
            with open(file_path, "wb") as f:
                f.write(response.content)
            
            # Step 2: 转为 base64 格式
            image_base64 = encode_image_to_base64(file_path)

            # Step 3: 构造 multimodal 输入（ChatOpenAI格式，支持vision）
            prompt = [
                {"type": "text", "text": "请你描述这张图片的内容。"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
            response = vlm.invoke(prompt)
            caption = response.content.strip()
            descriptions.append(caption)

            print(f"[WebSocket]: 图片{idx+1}描述完成：{caption}")
        except Exception as e:
            error_msg = f"图片下载或描述失败: {str(e)}"
            descriptions.append(error_msg)
            print(f"[WebSocket]: {error_msg}")
    return descriptions


def check_scene_consistency(scene_type, text_description, image_descriptions):
    prompt = f"""你是一个城市治理专家，请判断以下事件是否与场景类型一致。\n
场景类型: {scene_type}
工单文字描述: {text_description}
图片描述: {'；'.join(image_descriptions)}

请你只回答“是”或“否”，并简要解释理由。"""
    response = ask_deepseek(prompt)
    print(f"[WebSocket]: 一致性分析结果：{response}")
    return "是" in response[:5], response

def build_html_hint(scene_type, image_descriptions, reason):
    return f"""
    <p>
      <b>提示：</b>经系统分析，<span style="color:red;">照片内容</span>与事件描述中的
      <span style="color:blue;">“{scene_type}”</span>场景不一致。<br/>
      <b>图片显示：</b><span style="color:red;">{'；'.join(image_descriptions)}</span><br/>
      <b>模型分析：</b>{reason}<br/>
      <b>建议：</b>请核对上报图片是否正确，或修改“sceneType”为更准确的分类。
    </p>
    """

def process_event(event_json: dict):
    # 初始化结构
    event_json["processActionCode"] = "EventTypeRecognition"
    event_json["processResultCode"] = ""
    event_json["processResultData"] = ""

    print("[WebSocket]: 开始处理工单图像内容...")

    image_urls = event_json.get("imageUrls", [])
    image_descriptions = download_and_describe_images(image_urls)
    event_json["imageDescription"] = " ".join(image_descriptions)

    print("[WebSocket]: 图像内容分析完成，正在进行一致性判断...")

    is_consistent, reason = check_scene_consistency(
        event_json.get("sceneType", ""),
        event_json.get("description", ""),
        image_descriptions
    )

    if is_consistent:
        event_json["consistencyCheck"] = "一致"
        event_json["processResultCode"] = 0
        print("[WebSocket]: 场景一致，生成结果中...")
    else:
        event_json["consistencyCheck"] = "不一致"
        event_json["processResultCode"] = 1
        event_json["processResultData"] = build_html_hint(
            event_json["sceneType"], image_descriptions, reason
        )
        print("[WebSocket]: 场景不一致，生成富文本提示...")

    # 模拟所有内容推送完毕
    print("[WebSocket]: 所有信息生成完毕，推送完整工单：")
    print(json.dumps({
        "think": reason,
        "final_json": event_json
    }, ensure_ascii=False, indent=2))

    return {
        "think": reason,
        "final_json": event_json
    }

# 测试入口
if __name__ == "__main__":
    input_event = {
        "eventId": "SX-2025-06-19-00001",
        "sceneType": "泥头车",
        "reportLocation": "广东省深圳市福田区梅林街道林海山庄",
        "gridCode": "440304008004003",
        "reportTime": "2025-06-20 06:48:20",
        "reporter": "张三",
        "reporterSubject": "城市管家",
        "imageUrls": [
            "https://city001-bt-test.obs.cn-south-1.myhuaweicloud.com/2025/07/07/811702d688c348f8b0be009cb9b2805d.jpg",
            "https://city001-bt-test.obs.cn-south-1.myhuaweicloud.com/2025/07/07/7c7e65a9840b45688289c2b0be4da81a.jpg"
        ],
        "longitude": 121.473,
        "latitude": 31.230,
        "description": "滨河路。东往西滨河新洲立交处，该车辆超高超载。未密闭，沿途撒落。"
    }

    _ = process_event(input_event)

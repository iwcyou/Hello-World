import os
import json
import requests
from datetime import datetime
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # 读取 .env 文件

# 初始化语言模型和视觉模型
llm = ChatOpenAI(
    model="r1w8a8",
    temperature=0,
    openai_api_key=os.environ["W8A8_API_KEY"],
    openai_api_base="http://223.2.249.70:7019/v1",
)

vlm = ChatOpenAI(
    model="InternVL3-8B",
    temperature=0,
    openai_api_key=os.environ["INTERNVL_API_KEY"],
    openai_api_base="http://223.2.249.70:7025/v1",
)

# 文本模型调用
def ask_deepseek(prompt: str) -> str:
    response = llm.invoke(prompt)
    return response.content.strip()

import base64

# 下载图片并使用 VLM 生成描述（Base64 模式）
def get_image_description(image_urls):
    descriptions = []
    for idx, url in enumerate(image_urls):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                raise Exception(f"图片下载失败: HTTP {response.status_code}")

            # 编码为 Base64
            image_bytes = response.content
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{image_base64}"

            # 构建视觉模型输入
            human_msg = HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "请你用一句话描述这张图片内容。"}
            ])
            response = vlm.invoke([human_msg])
            descriptions.append(response.content.strip())

        except Exception as e:
            descriptions.append(f"图片处理失败: {str(e)}")

    return " ".join(descriptions)


# 判断文字和图片是否匹配
def check_scene_consistency(scene_type, text_description, image_description):
    prompt = f"""你是一个城市治理专家，请判断以下事件是否与场景类型一致。\n
场景类型: {scene_type}
工单文字描述: {text_description}
图片描述: {image_description}

请你只回答“是”或“否”，并简要解释理由。"""
    response = ask_deepseek(prompt)
    pure_response = response.split("</think>")[-1].strip()
    is_consistent = "是" in pure_response[:5]
    return is_consistent, response

# 富文本提示
def build_html_hint(scene_type, image_description, reason):
    return f"""
    <p>
      <b>提示：</b>经系统分析，<span style="color:red;">照片内容</span>与事件描述中的
      <span style="color:blue;">“{scene_type}”</span>场景不一致。<br/>
      <b>图片显示：</b><span style="color:red;">{image_description}</span><br/>
      <b>模型分析：</b>{reason}<br/>
      <b>建议：</b>请核对上报图片是否正确，或修改“sceneType”为更准确的分类。
    </p>
    """

# 主处理函数
def process_event(event_json: dict):
    # 记录开始时间
    start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    event_json["processStartAt"] = start_time

    data = event_json.get("processDataParams", {})
    has_error = False  # 错误标志

    # Step 1: 图像 caption
    image_description = get_image_description(data.get("imageUrls", []))

    # 判断是否存在失败描述
    if any(desc.startswith("图片处理失败") for desc in image_description.split(" ")):
        has_error = True

    # Step 2: 一致性判断
    scene_type = data.get("sceneType", "")
    description = data.get("description", "")
    try:
        is_consistent, reason = check_scene_consistency(scene_type, description, image_description)
        if not reason:
            raise ValueError("模型响应为空")
    except Exception as e:
        is_consistent = False
        reason = f"一致性判断失败: {str(e)}"
        has_error = True

    # Step 3: 构造 HTML 提示内容
    html_result = f"""
    <p>
      <b>提示：</b>经系统分析，照片内容与事件描述中的
      <span style="color:blue;">“{scene_type}”</span>场景
      <span style="color:{'green' if is_consistent else 'red'};">{'一致' if is_consistent else '不一致'}</span>。<br/>
      <b>图片显示：</b><span style="color:red;">{image_description}</span><br/>
      <b>模型分析：</b>{reason}<br/>
    </p>
    """

    # Step 4: 构建新的 processResultData 结构
    event_json["processResultData"] = {
        "htmlHint": html_result.strip(),
        "imageDescription": image_description,
        "consistencyCheck": "一致" if is_consistent else "不一致"
    }

    # Step 5: 处理结果码
    event_json["processResultCode"] = "1" if has_error else "0"

    # Step 6: 记录结束时间
    end_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    event_json["processStartEnd"] = end_time

    return event_json


# 测试入口
if __name__ == "__main__":
    input_event = {
        "eventId": "SX-2025-06-19-00001",
        "processResultCode": "",
        "processActionCode": "EventTypeRecognition",
        "processResultData": "",
        "processStartAt": "",
        "processStartEnd": "",
        "processDataParams": {
            "districtName": "福田",
            "sourceEntity": "城市管家",
            "sceneType": "泥头车",
            "reportLocation": "广东省深圳市福田区梅林街道林海山庄",
            "gridCode": "440304008004003",
            "reportTime": "2025-06-20 06:48:20",
            "reporter": "张三",
            "reportEntity": "城市管家",
            "longitude": 121.473,
            "latitude": 31.230,
            "imageUrls": [
                "https://city001-bt-test.obs.cn-south-1.myhuaweicloud.com/2025/07/07/811702d688c348f8b0be009cb9b2805d.jpg",
                "https://city001-bt-test.obs.cn-south-1.myhuaweicloud.com/2025/07/07/7c7e65a9840b45688289c2b0be4da81a.jpg"
            ],
            "description": "滨河路。东往西滨河新洲立交处，该车辆超高超载。未密闭，沿途撒落。"
        }
    }

    result = process_event(input_event)

    if isinstance(result, dict):
        print("✅ 返回增强后的 JSON 工单：")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("⚠️ 未返回 JSON 格式，请检查代码逻辑。")

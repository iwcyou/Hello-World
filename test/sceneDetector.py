import os
import json
from langchain_openai import ChatOpenAI

#Load environment variables
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


# ✅ 1. 初始化你自定义的 DeepSeek-R1 模型
llm = ChatOpenAI(
    model="r1w8a8",
    temperature=0,
    openai_api_key=os.environ["W8A8_API_KEY"],
    openai_api_base="http://223.2.249.70:7019/v1",
)

# ✅ 2. 调用 DeepSeek-R1 接口
def ask_deepseek(prompt: str) -> str:
    response = llm.invoke(prompt)
    return response.content.strip()


# ✅ 3. 图片描述（caption）
def get_image_description(image_urls):
    descriptions = []
    for url in image_urls:
        prompt = f"请你描述以下图片的内容：{url}"
        try:
            response = ask_deepseek(prompt)
            descriptions.append(response)
        except Exception as e:
            descriptions.append(f"图片处理失败: {str(e)}")
    return " ".join(descriptions)


# ✅ 4. 判断 sceneType 与内容是否一致
def check_scene_consistency(scene_type, text_description, image_description):
    prompt = f"""你是一个城市治理专家，请判断以下事件是否与场景类型一致。\n
场景类型: {scene_type}
工单文字描述: {text_description}
图片描述: {image_description}

请你只回答“是”或“否”，并简要解释理由。"""
    response = ask_deepseek(prompt)
    is_consistent = "是" in response[:5]
    return is_consistent, response


# ✅ 5. 富文本提示生成
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


# ✅ 6. 主处理函数
def process_event(event_json: dict):
    # Step 1: 图像 caption
    image_description = get_image_description(event_json.get("imageUrls", []))
    event_json["imageDescription"] = image_description

    # Step 2: 一致性判断
    scene_type = event_json.get("sceneType", "")
    description = event_json.get("description", "")
    is_consistent, reason = check_scene_consistency(scene_type, description, image_description)

    # Step 3: 分支输出
    if is_consistent:
        event_json["consistencyCheck"] = "一致"
        return event_json
    else:
        return build_html_hint(scene_type, image_description, reason)


# ✅ 7. 测试入口
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

    result = process_event(input_event)

    # ✅ 打印处理结果
    if isinstance(result, dict):
        print("✅ 一致，返回增强后的 JSON 工单：")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("❌ 不一致，返回富文本 HTML 提示：")
        print(result)

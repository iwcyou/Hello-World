import os
import json
from typing import TypedDict
from urllib import response
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

# ========== 配置大模型（DeepSeek-R1） ==========
llm = ChatOpenAI(
    model="r1w8a8",
    temperature=0,
    openai_api_key=os.environ["W8A8_API_KEY"],
    openai_api_base="http://223.2.249.70:7019/v1"
)

# ========== 定义状态结构 ==========
class State(TypedDict):
    tickets: list[dict]  # 原始工单数组
    newTickets: list[dict]  # 新生成的处理后工单数组


# ========== 法规内容检索 ==========
def get_law_context(index_path: str, query: str, top_k=5):
    db = FAISS.load_local(
        folder_path=index_path,
        embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh"),
        allow_dangerous_deserialization=True
    )
    docs = db.similarity_search(query, k=top_k)
    return "\n".join([doc.page_content for doc in docs])

# ========== 使用大模型进行责任判断 ==========
def ask_deepseek(event_data, law_context):
    prompt = f"""你是一个城市治理专家，专门处理泥头车遗洒责任问题。请根据以下工单信息和法规内容判断责任应归属谁，并给出理由。

【工单信息】
事件描述: {event_data["description"]}
图片描述: {event_data["imageDescription"]}
上报位置: {event_data["reportLocation"]}
上报人所属单位: {event_data["sourceEntity"]}

【法规内容】
{law_context}

请你返回格式如下（用中文）：
1. 责任主体：
2. 决策理由（包括参考的法规）：
"""
    res = llm.invoke(prompt)
    return res.content

# ========== 工单信息验证 ==========
def validate_event_info(state: State):
    if not state["tickets"]:
        raise ValueError("❌ 没有待处理的工单")

    # 只验证第一个工单
    ticket = state["tickets"][0]
    data = ticket.get("processDataParams", {})
    missing = []

    for field in ["description", "reportLocation", "sourceEntity"]:
        if not data.get(field):
            missing.append(field)

    # 尝试获取 imageDescription
    image_desc = ticket.get("context", {}).get("imageDescription")
    if image_desc:
        ticket["imageDescription"] = image_desc
    else:
        missing.append("imageDescription")

    if missing:
        raise ValueError(f"❌ 工单缺失必要字段：{', '.join(missing)}")

    # 替换 tickets 中第一个工单
    state["tickets"][0] = ticket
    return state


# ========== 泳道选择节点 ==========
def swimlane_selector(state: State):
    mapping = {
        "运输企业": "transport_company_node",
        "工地企业": "construction_site_node",
        "街道办": "subdistrict_office_node",
        "城市管家": "city_housekeeper_node",
        "市政服务第三方企业": "municipal_services_node"
    }

    if not state["tickets"]:
        return END

    first_ticket = state["tickets"][0]
    result_data = first_ticket.get("processResultData", {})
    target = result_data.get("target", None)

    if target and target.strip() in mapping:
        return mapping[target.strip()]

    source = first_ticket.get("processDataParams", {}).get("sourceEntity", "").strip()
    return mapping.get(source, END)


def extract_target_and_reason(text: str) -> tuple[str, str]:
    target, reason = "", ""

    lines = text.strip().splitlines()
    current_section = None
    reason_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("1.") and "责任主体" in line:
            parts = line.split("：", 1)
            if len(parts) == 2:
                target = parts[1].strip()
            current_section = "target"
        elif line.startswith("2.") and "决策理由" in line:
            current_section = "reason"
        elif current_section == "reason":
            reason_lines.append(line)

    reason = "\n".join(reason_lines).strip()
    return target, reason


# ========== 主体节点函数生成器 ==========
def make_responsibility_node(index_path):
    def node_fn(state: State):
        new_results = []

        for ticket in state["tickets"]:
            query = ticket["processDataParams"]["description"] + "\n" + ticket["imageDescription"]
            law_context = get_law_context(index_path, query)

            event_data = ticket["processDataParams"].copy()
            event_data["imageDescription"] = ticket["imageDescription"]
            response = ask_deepseek(event_data, law_context)
            pure_response = response.split("</think>")[-1].strip()

            target, reason = extract_target_and_reason(pure_response)

            # 构造新工单，不修改原始工单结构
            result_ticket = {
                "eventId": ticket.get("eventId", ""),
                "processActionCode": ticket.get("processActionCode", ""),
                "processResultData": {
                    "target": target,
                    "content": reason
                }
            }

            state["newTickets"].append(result_ticket)

        return state

    return node_fn


# ========== 城市管家节点 ==========
def city_housekeeper_node(state: State):
    new_results = []

    ticket = state["tickets"][0]
    original_data = ticket["processDataParams"]
    image_desc = ticket["imageDescription"]

    # 合并查询内容
    base_query = original_data["description"] + "\n" + image_desc

    # ========== 功能1：判断是否需要生成执法队工单 ==========
    target = ticket.get("processResultData", {}).get("target", "").strip()
    if not target:
        law_context = get_law_context(
            "./test/faiss_law_index/city_housekeeper_index",
            base_query + "\n案件报送执法队，协助执法队取证"
        )
        event_data = original_data.copy()
        event_data["imageDescription"] = image_desc

        response = ask_deepseek(event_data, law_context)
        pure_response = response.split("</think>")[-1].strip()
        _, reason = extract_target_and_reason(pure_response)

        law_ticket = {
            "eventId": ticket.get("eventId", ""),
            "processActionCode": ticket.get("processActionCode", ""),
            "processResultData": {
                "target": "执法队",
                "content": reason,
                "action": "PUSH_API"
            }
        }
        state["newTickets"].append(law_ticket)

        # ========== 功能2：生成城市管家工单 ==========
        law_context_caretaker = get_law_context(
            "./test/faiss_law_index/city_housekeeper_index",
            base_query + "\n留存基础证据.初步安全防护准备（放警示锥等）.协助交警临时管制（若污染较大）"
        )
        event_data2 = original_data.copy()
        event_data2["imageDescription"] = image_desc

        response2 = ask_deepseek(event_data2, law_context_caretaker)
        pure_response2 = response2.split("</think>")[-1].strip()
        _, reason2 = extract_target_and_reason(pure_response2)

        caretaker_ticket = {
            "eventId": ticket.get("eventId", ""),
            "processActionCode": ticket.get("processActionCode", ""),
            "processResultData": {
                "target": "城市管家",
                "content": reason2,
                "action": "PUSH_API"
            }
        }
        state["newTickets"].append(caretaker_ticket)

        # ========== 功能3：生成市政服务第三方企业工单 ==========
        law_context_municipal = get_law_context(
            "./test/faiss_law_index/city_housekeeper_index",
            base_query
        )
        response3 = ask_deepseek(event_data2, law_context_municipal)
        pure_response3 = response3.split("</think>")[-1].strip()
        _, reason3 = extract_target_and_reason(pure_response3)

        municipal_ticket = {
            "eventId": ticket.get("eventId", ""),
            "processActionCode": ticket.get("processActionCode", ""),
            "processResultData": {
                "target": "市政服务第三方企业",
                "content": reason3,
                "action": "PUSH_API",
                "timeLimit": 60
            }
        }
        state["newTickets"].append(municipal_ticket)

    return state


# ========== 构建 LangGraph 图 ==========
def build_graph():
    graph = StateGraph(State)

    # 注册节点
    graph.add_node("validate_event_info", validate_event_info)
    graph.add_node("transport_company_node", make_responsibility_node("./faiss_law_index/transport_company_index"))
    graph.add_node("construction_site_node", make_responsibility_node("./faiss_law_index/construction_site_index"))
    graph.add_node("subdistrict_office_node", make_responsibility_node("./faiss_law_index/subdistrict_office_index"))
    graph.add_node("city_housekeeper_node", city_housekeeper_node)
    graph.add_node("municipal_services_node", make_responsibility_node("./faiss_law_index/municipal_services_index"))

    # 设置起点
    graph.set_entry_point("validate_event_info")

    # 条件跳转
    graph.add_conditional_edges("validate_event_info", swimlane_selector)

    # 终点
    graph.add_edge("transport_company_node", END)
    graph.add_edge("construction_site_node", END)
    graph.add_edge("subdistrict_office_node", END)
    graph.add_edge("city_housekeeper_node", END)
    graph.add_edge("municipal_services_node", END)

    return graph.compile()

# ========== 主入口 ==========
if __name__ == "__main__":
    # 原始工单
    raw_event = {
        "eventId": "SX-2025-06-19-00001",
        "processActionCode": "EventHandlerLookup",
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
            "latitude": 31.23,
            "description": "滨河路。东往西滨河新洲立交处，该车辆超高超载。未密闭，沿途撒落。"
        },
        "context": {
            "imageDescription": "[811702d688c3: 这张图片显示了一辆载满土方的卡车行驶在深圳市福田区的道路上，路牌指示了前往南山、香蜜湖路、新洲路等方向。] [7c7e65a9840b: 这张图片显示了一辆载满土方的卡车行驶在深圳市福田区的道路上，背景是高楼大厦。]"
        }
    }

    # 构建输入 State
    initial_state = {
        "tickets": [raw_event],
        "newTickets": []
    }

    app = build_graph()
    result_state = app.invoke(initial_state)

    # ✅ 打印新生成的处理工单
    print("\n✅ 处理结果：")
    print(json.dumps(result_state["newTickets"], ensure_ascii=False, indent=2))



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
    ticket: dict  # 原始工单
    newTicket: dict  # 新生成的处理后工单


# ========== 法规内容检索 ==========
def get_law_context(index_path: str, query: str, top_k=10):
    db = FAISS.load_local(
        folder_path=index_path,
        embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5"),
        allow_dangerous_deserialization=True
    )
    docs = db.similarity_search(query, k=top_k)
    return "\n".join([doc.page_content for doc in docs])

# ========== 法规内容检索（带引用） ==========
def get_law_context_with_citations(index_path: str, query: str, top_k=10):
    db = FAISS.load_local(
        folder_path=index_path,
        embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5"),
        allow_dangerous_deserialization=True
    )
    docs = db.similarity_search(query, k=top_k)
    
    # 提取文档内容和来源文件名
    content = "\n".join([doc.page_content for doc in docs])
    citations = list(set([doc.metadata.get("source", "未知文档") for doc in docs if doc.metadata.get("source")]))
    
    return content, citations

# ========== 使用大模型进行责任判断 ==========
def ask_deepseek(event_data, law_context, target_entity=None):
    if target_entity:
        prompt = f"""你是一个城市治理专家，专门处理泥头车遗洒责任问题。请根据以下工单信息和法规内容，针对{target_entity}制定具体的处理方案。

【工单信息】
事件描述: {event_data["description"]}
图片描述: {event_data["imageDescription"]}
上报位置: {event_data["reportLocation"]}
上报人所属单位: {event_data["sourceEntity"]}

【法规内容】
{law_context}

请针对{target_entity}详细制定处理方案：

请你返回格式如下（用中文）：
1. 责任主体：{target_entity}
2. 决策理由（包括参考的法规）：
3. 应执行的具体操作：
   - 立即措施：（列出{target_entity}需要立即执行的紧急处理措施）
   - 后续跟进：（列出{target_entity}的后续监管和整改措施）
   - 责任追究：（列出{target_entity}相关的法律责任和处罚措施）
   - 预防措施：（列出{target_entity}防范类似事件再次发生的措施）
"""
    else:
        prompt = f"""你是一个城市治理专家，专门处理泥头车遗洒责任问题。请根据以下工单信息和法规内容判断责任应归属谁，并给出理由。

【工单信息】
事件描述: {event_data["description"]}
图片描述: {event_data["imageDescription"]}
上报位置: {event_data["reportLocation"]}
上报人所属单位: {event_data["sourceEntity"]}

【法规内容】
{law_context}

请你分析责任主体并详细说明该主体应该执行的具体操作和措施。

请你返回格式如下（用中文）：
1. 责任主体：
2. 决策理由（包括参考的法规）：
3. 应执行的具体操作：
   - 立即措施：（列出紧急处理措施）
   - 后续跟进：（列出后续监管和整改措施）
   - 责任追究：（列出相关法律责任和处罚措施）
   - 预防措施：（列出防范类似事件再次发生的措施）
"""
    res = llm.invoke(prompt)
    return res.content

# ========== 工单信息验证 ==========
def validate_event_info(state: State):
    if not state["ticket"]:
        raise ValueError("❌ 没有待处理的工单")

    ticket = state["ticket"]
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
        # 信息缺失，返回错误工单
        error_ticket = {
            "eventId": ticket.get("eventId", ""),
            "processResultCode": 1,
            "processActionCode": ticket.get("processActionCode", ""),
            "processResultData": {
                "error": f"工单缺失必要字段：{', '.join(missing)}"
            }
        }
        state["newTicket"] = error_ticket
        return state

    # 更新 ticket
    state["ticket"] = ticket
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

    if not state["ticket"]:
        return END

    ticket = state["ticket"]
    
    # 首先检查 dispatchTarget 字段
    dispatch_target = ticket.get("context", {}).get("dispatchTarget", "")
    if dispatch_target and dispatch_target.strip():
        return mapping.get(dispatch_target.strip(), "END")
    
    # 如果 dispatchTarget 为空，检查 processResultData 中的 target
    result_data = ticket.get("processResultData", {})
    target = result_data.get("target", None)
    if target and target.strip() in mapping:
        return mapping[target.strip()]

    # 最后使用 sourceEntity 字段
    source = ticket.get("processDataParams", {}).get("sourceEntity", "").strip()
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


# ========== 运输企业节点 ==========
def transport_company_node(state: State):
    ticket = state["ticket"]
    original_data = ticket["processDataParams"]
    image_desc = ticket["imageDescription"]

    base_query = original_data["description"] + "\n" + image_desc
    query = base_query + "\n留存基础证据；初步安全防护准备（放警示锥等）；协助交警临时管制（若污染较大）"

    # 获取法规内容
    law_context = get_law_context(
        "./test/faiss_law_index",
        query
    )

    # 组装 event_data 用于提问
    event_data = original_data.copy()
    event_data["imageDescription"] = image_desc

    # 大模型生成内容
    response = ask_deepseek(event_data, law_context)
    pure_response = response.split("</think>")[-1].strip()
    _, reason = extract_target_and_reason(pure_response)

    # 构造城市管家工单
    caretaker_ticket = {
        "eventId": ticket.get("eventId", ""),
        "processActionCode": ticket.get("processActionCode", ""),
        "processResultData": {
            "target": "城市管家",
            "content": reason,
            "action": "PUSH_API"
        }
    }

    state["newTicket"] = caretaker_ticket
    return state


# ========== 工地企业节点 ==========
def construction_site_node(state: State):
    ticket = state["ticket"]
    original_data = ticket["processDataParams"]
    image_desc = ticket["imageDescription"]

    base_query = original_data["description"] + "\n" + image_desc
    query = base_query + "\n提醒肇事人员及时整改；批评教育（观看污染治理记录片+法规学习）；留存基础证据"

    # 获取法规内容
    law_context = get_law_context(
        "./test/faiss_law_index",
        query
    )

    # 组装 event_data 用于提问
    event_data = original_data.copy()
    event_data["imageDescription"] = image_desc

    # 大模型生成内容
    response = ask_deepseek(event_data, law_context)
    pure_response = response.split("</think>")[-1].strip()
    _, reason = extract_target_and_reason(pure_response)

    # 构造街道办工单
    subdistrict_ticket = {
        "eventId": ticket.get("eventId", ""),
        "processActionCode": ticket.get("processActionCode", ""),
        "processResultData": {
            "target": "街道办",
            "content": reason,
            "action": "PUSH_API"
        }
    }

    state["newTicket"] = subdistrict_ticket
    return state


# ========== 街道办节点 ==========
def subdistrict_office_node(state: State):
    ticket = state["ticket"]
    original_data = ticket["processDataParams"]
    image_desc = ticket["imageDescription"]

    # 合并查询内容
    base_query = original_data["description"] + "\n" + image_desc

    actions = []

    # ========== 生成工地企业工单 ==========
    law_context, citations = get_law_context_with_citations(
        "./test/faiss_law_index",
        base_query + "\n提醒肇事人员及时整改；批评教育（观看污染治理记录片+法规学习）；街道办应该留存基础证据。"
    )
    event_data = original_data.copy()
    event_data["imageDescription"] = image_desc

    response = ask_deepseek(event_data, law_context, target_entity="工地企业")
    pure_response = response.split("</think>")[-1].strip()
    think_match = response.split("</think>")[0] if "</think>" in response else ""
    _, reason = extract_target_and_reason(pure_response)

    actions.append({
        "target": "工地企业",
        "content": reason,
        "action": "PUSH_SMS",
        "timeLimit": 120,
        "think": think_match,
        "citations": citations
    })

    # 使用大模型生成总结
    summary = generate_summary(actions)

    # 构造合并后的工单
    merged_ticket = {
        "eventId": ticket.get("eventId", ""),
        "processResultCode": 0,
        "processActionCode": ticket.get("processActionCode", ""),
        "processResultData": {
            "actions": actions,
            "summary": summary
        }
    }

    state["newTicket"] = merged_ticket
    return state


# ========== 市政服务第三方企业节点 ==========
def municipal_services_node(state: State):
    ticket = state["ticket"]
    original_data = ticket["processDataParams"]
    image_desc = ticket["imageDescription"]

    base_query = original_data["description"] + "\n" + image_desc

    # ========== 功能1：生成城市管家工单 ==========
    law_context_caretaker = get_law_context(
        "./test/faiss_law_index",
        base_query + "\n留存基础证据.初步安全防护准备（放警示锥等）.协助交警临时管制（若污染较大）"
    )

    event_data1 = original_data.copy()
    event_data1["imageDescription"] = image_desc

    response1 = ask_deepseek(event_data1, law_context_caretaker)
    pure_response1 = response1.split("</think>")[-1].strip()
    _, reason1 = extract_target_and_reason(pure_response1)

    # 只返回第一个工单（城市管家工单）
    caretaker_ticket = {
        "eventId": ticket.get("eventId", ""),
        "processActionCode": ticket.get("processActionCode", ""),
        "processResultData": {
            "target": "城市管家",
            "content": reason1,
            "action": "PUSH_API"
        }
    }
    state["newTicket"] = caretaker_ticket
    return state


# ========== 使用大模型生成总结 ==========
def generate_summary(actions):
    # 构建actions信息
    actions_info = []
    for action in actions:
        target = action.get("target", "")
        content = action.get("content", "")
        time_limit = action.get("timeLimit", "")
        time_info = f"，{time_limit}分钟内完成" if time_limit else ""
        actions_info.append(f"- {target}：{content[:100]}...{time_info}")
    
    actions_text = "\n".join(actions_info)
    
    prompt = f"""你是一个城市治理专家，请对以下处置方案进行总结。

【处置方案】
{actions_text}

请用HTML格式返回总结，要求：
1. 突出处置单位和主要任务
2. 使用颜色标记重要信息
3. 保持简洁明了
4. 格式示例：经大模型分析，找到<span style="color:blue;">X个</span>该事件的处置单位，分别是<span style="color:red;">单位A</span>负责<span style="color:green;">任务描述</span>，<span style="color:red;">单位B</span>负责<span style="color:green;">任务描述</span>，请确认是否进行分拨处理

请直接返回HTML格式的总结内容："""
    
    response = llm.invoke(prompt)
    response_content = response.content  # 先获取content属性
    pure_response = response_content.split("</think>")[-1].strip() if "</think>" in response_content else response_content.strip()
    
    return pure_response


# ========== 城市管家节点 ==========
def city_housekeeper_node(state: State):
    ticket = state["ticket"]
    original_data = ticket["processDataParams"]
    image_desc = ticket["imageDescription"]

    # 合并查询内容
    base_query = original_data["description"] + "\n" + image_desc

    actions = []

    # ========== 功能1：生成执法队工单 ==========
    law_context, citations1 = get_law_context_with_citations(
        "./test/faiss_law_index",
        base_query + "\n案件报送执法队，协助执法队取证。"
    )
    event_data = original_data.copy()
    event_data["imageDescription"] = image_desc

    response = ask_deepseek(event_data, law_context, target_entity="执法队")
    pure_response = response.split("</think>")[-1].strip()
    think_match = response.split("</think>")[0] if "</think>" in response else ""
    _, reason = extract_target_and_reason(pure_response)

    actions.append({
        "target": "执法队",
        "content": reason,
        "action": "PUSH_API",
        "think": think_match,
        "citations": citations1
    })

    # ========== 功能2：生成城市管家工单 ==========
    law_context_caretaker, citations2 = get_law_context_with_citations(
        "./test/faiss_law_index",
        base_query + "\n留存基础证据；初步安全防护准备（放警示锥等）；协助交警临时管制（若污染较大）。" + "\n协调组织相应环卫企业清洁。"
    )

    response2 = ask_deepseek(event_data, law_context_caretaker, target_entity="城市管家")
    pure_response2 = response2.split("</think>")[-1].strip()
    think_match2 = response2.split("</think>")[0] if "</think>" in response2 else ""
    _, reason2 = extract_target_and_reason(pure_response2)

    actions.append({
        "target": "城市管家",
        "content": reason2,
        "action": "PUSH_API",
        "think": think_match2,
        "citations": citations2
    })

    # 使用大模型生成总结
    summary = generate_summary(actions)

    # 构造合并后的工单
    merged_ticket = {
        "eventId": ticket.get("eventId", ""),
        "processResultCode": 0,
        "processActionCode": ticket.get("processActionCode", ""),
        "processResultData": {
            "actions": actions,
            "summary": summary
        }
    }

    state["newTicket"] = merged_ticket
    return state


# ========== 构建 LangGraph 图 ==========
def build_graph():
    graph = StateGraph(State)

    # 注册节点
    graph.add_node("validate_event_info", validate_event_info)
    graph.add_node("transport_company_node", transport_company_node)
    graph.add_node("construction_site_node", construction_site_node)
    graph.add_node("subdistrict_office_node", subdistrict_office_node)
    graph.add_node("city_housekeeper_node", city_housekeeper_node)
    graph.add_node("municipal_services_node", municipal_services_node)

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
def handler_finder(raw_event):
    from datetime import datetime
    
    # 记录开始时间
    start_time = datetime.now()

    # 构建输入 State
    initial_state = {
        "ticket": raw_event,
        "newTicket": {}
    }

    app = build_graph()
    result_state = app.invoke(initial_state)

    # 记录结束时间
    end_time = datetime.now()

    # 为生成的工单添加时间戳
    if result_state["newTicket"]:
        result_state["newTicket"]["processStartAt"] = start_time.isoformat()
        result_state["newTicket"]["processStartEnd"] = end_time.isoformat()

    # ✅ 打印新生成的处理工单
    print("\n✅ 处理结果：")
    print(json.dumps(result_state["newTicket"], ensure_ascii=False, indent=2))


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
        "reportEntity": "城市管家巡查员",
        "longitude": 121.473,
        "latitude": 31.23,
        "description": "滨河路。东往西滨河新洲立交处，该车辆超高超载。未密闭，沿途撒落。"
    },
    "context": {
        "imageDescription": "[811702d688c3: 这张图片显示了一辆载满土方的卡车行驶在深圳市福田区的道路上，路牌指示了前往南山、香蜜湖路、新洲路等方向。] [7c7e65a9840b: 这张图片显示了一辆载满土方的卡车行驶在深圳市福田区的道路上，背景是高楼大厦。]"
    }
}
    handler_finder(raw_event)


#Load environment variables
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from langchain_core.messages import HumanMessage
from langchain.chat_models import ChatOpenAI  # 替换为 DeepSeek-R1 的兼容模型
import json
import datetime

# 1. 定义状态结构
class EventState(TypedDict):
    raw_input: str
    filled_input: str
    json_data: Optional[dict]
    missing_fields: list
    assigned_entity: Optional[str]
    dispatch_status: Optional[str]

# 2. 初始化模型（可替换为 DeepSeek-R1）
llm = ChatOpenAI(
    model="r1w8a8",
    temperature=0,
    openai_api_key=os.environ["W8A8_API_KEY"],
    openai_api_base="http://223.2.249.70:7019/v1",
)

# 3. 节点定义

def receive_report(state: EventState) -> EventState:
    state['filled_input'] = state['raw_input']
    return state

def check_and_complete_info(state: EventState) -> EventState:
    prompt = f"""
你是一位城市治理智能助理，现在收到一条描述，请判断是否包含以下信息：时间、地点、事件情况。
请以如下格式返回：
{{
  "time": true/false,
  "location": true/false,
  "incident": true/false,
  "missing_fields": ["time", "location"]
}}
描述如下：
{state['filled_input']}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    result = json.loads(response.content)
    state['missing_fields'] = result.get("missing_fields", [])
    return state

def ask_for_more(state: EventState) -> EventState:
    missing = ", ".join(state['missing_fields'])
    print(f"⚠️ 信息缺失：{missing}，请补充如下内容。")
    user_reply = input("请输入补充描述：")
    state['filled_input'] += " " + user_reply
    return state

def extract_json(state: EventState) -> EventState:
    prompt = f"""
请从下列事件描述中提取如下 JSON 信息：
{{
  "time": "",
  "location": "",
  "description": "",
  "reporter": "",
  "images": []
}}
描述如下：
{state['filled_input']}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    state['json_data'] = json.loads(response.content)
    return state

def assign_responsibility(state: EventState) -> EventState:
    prompt = f"""
你是一位城市治理责任判定专家，请判断事件责任应归属于哪一类：运输企业、工地企业或环卫公司。
返回格式：{{"responsibility": "运输企业", "reason": "..."}}
描述如下：
{state['filled_input']}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    result = json.loads(response.content)
    state['assigned_entity'] = result['responsibility']
    return state

def dispatch_task(state: EventState) -> EventState:
    now = datetime.datetime.now()
    deadline = now + datetime.timedelta(hours=2)
    dispatch_record = {
        "event_id": now.strftime("%Y%m%d%H%M%S"),
        "time": state['json_data'].get("time", str(now)),
        "location": state['json_data'].get("location", "未知"),
        "description": state['json_data'].get("description", ""),
        "dispatched_to": state['assigned_entity'],
        "dispatch_deadline": deadline.strftime("%Y-%m-%d %H:%M"),
        "status": "已派遣"
    }
    state['dispatch_status'] = json.dumps(dispatch_record, ensure_ascii=False, indent=2)
    print("✅ 派遣成功，记录如下：")
    print(state['dispatch_status'])
    return state

# 4. 构建LangGraph
workflow = StateGraph(EventState)

workflow.add_node("receive_report", receive_report)
workflow.add_node("check_and_complete_info", check_and_complete_info)
workflow.add_node("ask_for_more", ask_for_more)
workflow.add_node("extract_json", extract_json)
workflow.add_node("assign_responsibility", assign_responsibility)
workflow.add_node("dispatch_task", dispatch_task)

workflow.set_entry_point("receive_report")

def is_info_complete(state: EventState) -> str:
    return "extract_json" if not state['missing_fields'] else "ask_for_more"

workflow.add_conditional_edges("check_and_complete_info", is_info_complete)
workflow.add_edge("receive_report", "check_and_complete_info")
workflow.add_edge("ask_for_more", "check_and_complete_info")
workflow.add_edge("extract_json", "assign_responsibility")
workflow.add_edge("assign_responsibility", "dispatch_task")
workflow.add_edge("dispatch_task", END)

app = workflow.compile()

# 5. 示例运行
if __name__ == "__main__":
    user_input = input("请输入泥头车遗撒事件描述：")
    initial_state = {
        "raw_input": user_input,
        "filled_input": "",
        "json_data": {},
        "missing_fields": [],
        "assigned_entity": None,
        "dispatch_status": None
    }
    app.invoke(initial_state)

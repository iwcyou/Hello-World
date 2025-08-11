import os
import json
import requests
import re
from datetime import datetime
from typing import Any, Dict, TypedDict
from urllib import response
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file


# ========== 配置大模型（DeepSeek-R1） ==========
llm = ChatOpenAI(
    model="r1w8a8",
    temperature=0,
    openai_api_key=os.environ["W8A8_API_KEY"],
    openai_api_base="http://223.2.249.70:7019/v1"
)

vlm = ChatOpenAI(
    model="InternVL3-8B",
    temperature=0,
    openai_api_key=os.environ["INTERNVL_API_KEY"],
    openai_api_base="http://223.2.249.70:7025/v1",
)


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

            # 获取图片文件名或编号
            filename = os.path.basename(url).split("?")[0][:12]  # 提取图片 ID 前缀
            descriptions.append(f"[{filename}: {response.content.strip()}]")

        except Exception as e:
            descriptions.append(f"[image_{idx+1}: 图片处理失败: {str(e)}]")

    return " ".join(descriptions)


# ==========================
# 状态数据结构
# ==========================
class State(TypedDict):
    ticket: dict      # 单个原始工单
    newTicket: dict   # 单个最终输出工单


# ==========================
# 节点1 - 校验事件信息
# ==========================
def validate_event_info(state: State) -> State:
    ticket = state["ticket"]

    # 检查基本必须字段
    process_data = ticket.get("processDataParams", {})
    context = ticket.get("context", {})
    result_info = ticket.get("resultInfo", [])

    # 必须字段检查
    required_fields = [
        process_data.get("description"),
        context.get("imageDescription"),
        context.get("dispatchTarget")
    ]

    # 检查 resultInfo 是否存在且为有效数组
    if not result_info or not isinstance(result_info, list) or len(result_info) == 0:
        required_fields.append(None)  # 标记为缺失
    else:
        # 检查第一个结果是否有 handleResult
        first_result = result_info[0]
        if not first_result.get("handleResult"):
            required_fields.append(None)  # 标记为缺失

    # 如果有缺失，直接返回默认结果
    if any(f is None or f == "" for f in required_fields):
        state["newTicket"] = {
            "eventId": ticket.get("eventId", ""),
            "processResultCode": "1"
        }
        return state

    return state


# ==========================
# 节点2 - swimlane选择器
# ==========================
def swimlane_selector(state: State) -> str:
    target = state["ticket"].get("context", {}).get("dispatchTarget", "")
    mapping = {
        "运输企业": "transport_company_node",
        "工地企业": "construction_site_node",
        "街道办": "subdistrict_office_node",
        "城市管家": "city_housekeeper_node",
        "市政服务第三方企业": "municipal_services_node"
    }
    return mapping.get(target, "END")


# ========== 运输企业节点 ==========
def transport_company_node(state: State) -> State:
    # 运输企业，不做处理
    return state


# ========== 工地企业节点 ==========
def construction_site_node(state: State) -> State:
    ticket = state["ticket"]

    # 时间记录
    start_time = datetime.now().isoformat()

    # 获取处理结果信息
    result_info = ticket.get("resultInfo", [])
    if not result_info or not isinstance(result_info, list):
        # 如果没有结果信息，返回未完成
        new_ticket = {
            "eventId": ticket.get("eventId", ""),
            "processResultCode": 0,
            "processActionCode": "CheckHandleResult",
            "processResultData": {
                "completed": False,
                "reason": "缺少处理结果信息",
                "think": "未找到有效的处理结果信息，无法判断完成状态。",
                "handleResultImageDescription": "",
                "actions": ticket.get("context", {}).get("actions", [{"target": "工地企业"}])
            },
            "processStartAt": start_time,
            "processStartEnd": datetime.now().isoformat()
        }
        state["newTicket"] = new_ticket
        return state

    # 获取第一个处理结果（通常只有一个）
    first_result = result_info[0]
    handle_result = first_result.get("handleResult", "")
    image_urls = first_result.get("handleResultImageUrls", [])

    # 生成图片描述
    image_description = ""
    if image_urls:
        image_description = get_image_description(image_urls)

    # 使用大模型进行决策
    event_desc = ticket.get("processDataParams", {}).get("description", "")
    original_image_desc = ticket.get("context", {}).get("imageDescription", "")
    
    # 获取原始分拨要求
    original_actions = ticket.get("context", {}).get("actions", [])
    action_content = ""
    for action in original_actions:
        if action.get("target") == "工地企业" and action.get("content"):
            action_content = action.get("content")
            break
    
    prompt = f"""你是一个城市治理专家，请根据以下信息判断工地企业的处理结果是否完成：

【原始事件描述】
{event_desc}

【原始现场图片描述】
{original_image_desc}

【分拨节点要求工地企业应做的事情】
{action_content}

【处理结果描述】
{handle_result}

【处理后图片描述】
{image_description}

请你分析：
1. 处理结果是否充分解决了原始问题
2. 是否完成了分拨节点要求的具体工作内容
3. 图片证据是否支持处理结果的真实性
4. 是否需要进一步处理

请返回JSON格式：
{{
    "completed": true/false,
    "reason": "详细的判断理由和分析过程"
}}
"""

    try:
        response = llm.invoke(prompt)
        response_content = response.content
        
        print(f"🔍 大模型原始响应: {response_content}")  # 调试信息
        
        # 提取思考过程（如果存在<think>标签）
        think_match = re.search(r'<think>(.*?)</think>', response_content, re.DOTALL)
        think = think_match.group(1).strip() if think_match else ""
        
        # 从response中移除<think>标签后的内容进行JSON解析
        content_after_think = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()
        
        print(f"🔍 移除think标签后: {content_after_think}")  # 调试信息
        
        # 尝试解析JSON响应
        json_match = re.search(r'\{.*\}', content_after_think, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            print(f"🔍 提取的JSON字符串: {json_str}")  # 调试信息
            result_json = json.loads(json_str)
            completed = result_json.get("completed", False)
            reason = result_json.get("reason", "大模型判断结果")
            # 如果没有提取到think，使用reason作为think
            if not think:
                think = reason
        else:
            print("⚠️ 未找到有效的JSON格式")
            # 如果无法解析JSON，使用备用逻辑
            completed = "完成" in response_content or "充分" in response_content or "解决" in response_content
            reason = "根据大模型分析，" + ("该工单已经完成处理" if completed else "该工单处理不充分或未完成")
            think = think if think else content_after_think[:200] + "..."  # 截取前200字符
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误: {e}")
        # JSON解析失败的备用逻辑
        completed = "完成" in handle_result or "清洗" in handle_result or "处理" in handle_result
        reason = f"JSON解析失败，使用备用判断：" + ("该工单已经完成处理" if completed else "该工单处理不充分或未完成")
        think = f"JSON解析异常: {str(e)}"
    except Exception as e:
        print(f"❌ 大模型调用异常: {type(e).__name__}: {e}")
        # 大模型调用失败时的备用逻辑
        completed = "完成" in handle_result or "清洗" in handle_result or "处理" in handle_result
        reason = f"大模型调用失败({str(e)})，使用备用判断：" + ("该工单已经完成处理" if completed else "该工单处理不充分或未完成")
        think = f"由于大模型调用异常，根据处理结果描述'{handle_result}'进行简单判断。"

    # 动作列表
    if completed:
        actions = [{"target": "住建局", "action": "PUSH_API"}]
    else:
        # 如果未完成，复制原始工单中的actions字段
        original_actions = ticket.get("context", {}).get("actions", [])
        actions = original_actions if original_actions else [{"target": "工地企业"}]

    # 输出结果
    new_ticket = {
        "eventId": ticket.get("eventId", ""),
        "processResultCode": 0,
        "processActionCode": "CheckHandleResult",
        "processResultData": {
            "completed": completed,
            "reason": reason,
            "think": think,
            "handleResultImageDescription": image_description,
            "actions": actions
        },
        "processStartAt": start_time,
        "processStartEnd": datetime.now().isoformat()
    }

    state["newTicket"] = new_ticket
    return state


# ========== 街道办节点 ==========
def subdistrict_office_node(state: State) -> State:
    ticket = state["ticket"]

    # 时间记录
    start_time = datetime.now().isoformat()

    # 检查是否存在checklistInfo，判断是否进入复核阶段
    checklist_info = ticket.get("checklistInfo", {})
    if checklist_info:
        # 进入复核阶段
        handle_items = checklist_info.get("handleItems", [])
        
        # 检查复查结果
        review_result = None
        for item in handle_items:
            if item.get("item") == "复查结果":
                review_result = item.get("result")
                break
        
        if review_result == "已完成":
            # 复查完成，结案
            new_ticket = {
                "eventId": ticket.get("eventId", ""),
                "processResultCode": 0,
                "processActionCode": "CheckHandleResult",
                "processResultData": {
                    "completed": True,
                    "closed": True,
                    "reason": "已结案，正负清单保存在····点击按钮继续生成治理报告···"
                },
                "processStartAt": start_time,
                "processStartEnd": datetime.now().isoformat()
            }
        else:
            # 复查未完成，再次分拨
            original_actions = ticket.get("context", {}).get("actions", [])
            new_ticket = {
                "eventId": ticket.get("eventId", ""),
                "processResultCode": 0,
                "processActionCode": "CheckHandleResult",
                "processResultData": {
                    "completed": False,
                    "actions": original_actions
                },
                "processStartAt": start_time,
                "processStartEnd": datetime.now().isoformat()
            }
    else:
        # 没有checklistInfo，按原逻辑处理（如果需要的话）
        new_ticket = {
            "eventId": ticket.get("eventId", ""),
            "processResultCode": 0,
            "processActionCode": "CheckHandleResult",
            "processResultData": {
                "completed": False,
                "reason": "街道办节点暂未实现超时反馈任务",
            },
            "processStartAt": start_time,
            "processStartEnd": datetime.now().isoformat()
        }

    state["newTicket"] = new_ticket
    return state


# ========== 城市管家节点 ==========
def city_housekeeper_node(state: State) -> State:
    ticket = state["ticket"]

    # 时间记录
    start_time = datetime.now().isoformat()

    # 获取处理结果信息
    result_info = ticket.get("resultInfo", [])
    if not result_info or not isinstance(result_info, list):
        # 如果没有结果信息，返回未完成
        new_ticket = {
            "eventId": ticket.get("eventId", ""),
            "processResultCode": 0,
            "processActionCode": "CheckHandleResult",
            "processResultData": {
                "completed": False,
                "reason": "缺少处理结果信息",
                "think": "未找到有效的处理结果信息，无法判断完成状态。",
                "handleResultImageDescription": "",
                "actions": ticket.get("context", {}).get("actions", [{"target": "城市管家"}])
            },
            "processStartAt": start_time,
            "processStartEnd": datetime.now().isoformat()
        }
        state["newTicket"] = new_ticket
        return state

    # 合并所有处理结果
    all_results = []
    all_image_urls = []
    for result in result_info:
        handle_result = result.get("handleResult", "")
        entity = result.get("handleEntity", "")
        if handle_result:
            all_results.append(f"[{entity}: {handle_result}]")
        
        image_urls = result.get("handleResultImageUrls", [])
        all_image_urls.extend(image_urls)

    combined_results = " ".join(all_results)

    # 生成图片描述
    image_description = ""
    if all_image_urls:
        image_description = get_image_description(all_image_urls)

    # 使用大模型进行决策
    event_desc = ticket.get("processDataParams", {}).get("description", "")
    original_image_desc = ticket.get("context", {}).get("imageDescription", "")
    
    # 获取原始分拨要求
    original_actions = ticket.get("context", {}).get("actions", [])
    action_content = ""
    for action in original_actions:
        if action.get("target") == "城市管家" and action.get("content"):
            action_content = action.get("content")
            break
    
    prompt = f"""你是一个城市治理专家，请根据以下信息判断城市管家的处理结果是否完成：

【原始事件描述】
{event_desc}

【原始现场图片描述】
{original_image_desc}

【分拨节点要求城市管家应做的事情】
{action_content}

【所有处理结果描述】
{combined_results}

【处理后图片描述】
{image_description}

请你分析：
1. 城市管家是否完成了现场安全防护（如设置警示锥等）
2. 市政服务企业是否完成了清理工作（渣土清运、路面清洗）
3. 是否完成了分拨节点要求的具体工作内容
4. 所有处理措施是否充分解决了原始问题
5. 图片证据是否支持处理结果的真实性

请返回JSON格式：
{{
    "completed": true/false,
    "reason": "详细的判断理由和分析过程"
}}
"""

    try:
        response = llm.invoke(prompt)
        response_content = response.content
        
        print(f"🔍 大模型原始响应: {response_content}")  # 调试信息
        
        # 提取思考过程（如果存在<think>标签）
        think_match = re.search(r'<think>(.*?)</think>', response_content, re.DOTALL)
        think = think_match.group(1).strip() if think_match else ""
        
        # 从response中移除<think>标签后的内容进行JSON解析
        content_after_think = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()
        
        print(f"🔍 移除think标签后: {content_after_think}")  # 调试信息
        
        # 尝试解析JSON响应
        json_match = re.search(r'\{.*\}', content_after_think, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            print(f"🔍 提取的JSON字符串: {json_str}")  # 调试信息
            result_json = json.loads(json_str)
            completed = result_json.get("completed", False)
            reason = result_json.get("reason", "大模型判断结果")
            # 如果没有提取到think，使用reason作为think
            if not think:
                think = reason
        else:
            print("⚠️ 未找到有效的JSON格式")
            # 如果无法解析JSON，使用备用逻辑
            completed = "完成" in response_content or "充分" in response_content or "解决" in response_content
            reason = "根据大模型分析，" + ("该工单已经完成处理" if completed else "该工单处理不充分或未完成")
            think = think if think else content_after_think[:200] + "..."  # 截取前200字符
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误: {e}")
        # JSON解析失败的备用逻辑
        completed = "完成" in combined_results or "清洗" in combined_results or "警示锥" in combined_results
        reason = f"JSON解析失败，使用备用判断：" + ("该工单已经完成处理" if completed else "该工单处理不充分或未完成")
        think = f"JSON解析异常: {str(e)}"
    except Exception as e:
        print(f"❌ 大模型调用异常: {type(e).__name__}: {e}")
        # 大模型调用失败时的备用逻辑
        completed = "完成" in combined_results or "清洗" in combined_results or "警示锥" in combined_results
        reason = f"大模型调用失败({str(e)})，使用备用判断：" + ("该工单已经完成处理" if completed else "该工单处理不充分或未完成")
        think = f"由于大模型调用异常，根据处理结果描述进行简单判断。"

    # 动作列表
    if completed:
        actions = [{"target": "街道办", "action": "PUSH_API"}]
    else:
        # 如果未完成，复制原始工单中的actions字段
        original_actions = ticket.get("context", {}).get("actions", [])
        actions = original_actions if original_actions else [{"target": "城市管家"}]

    # 输出结果
    new_ticket = {
        "eventId": ticket.get("eventId", ""),
        "processResultCode": 0,
        "processActionCode": "CheckHandleResult",
        "processResultData": {
            "completed": completed,
            "reason": reason,
            "think": think,
            "handleResultImageDescription": image_description,
            "actions": actions
        },
        "processStartAt": start_time,
        "processStartEnd": datetime.now().isoformat()
    }

    state["newTicket"] = new_ticket
    return state


# ========== 市政服务第三方企业节点 ==========
def municipal_services_node(state: State) -> State:
    # 市政服务第三方企业，不做处理
    return state


# ==========================
# 构建 LangGraph 图
# ==========================
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


# ==========================
# 主流程入口
# ==========================
def main_handler(input_ticket: Dict[str, Any]) -> Dict[str, Any]:
    # 构建输入 State
    initial_state: State = {
        "ticket": input_ticket,
        "newTicket": {}
    }

    # 创建并执行图
    app = build_graph()
    result_state = app.invoke(initial_state)

    # 返回处理结果
    return result_state["newTicket"]


# ==========================
# 测试
# ==========================
if __name__ == "__main__":
    sample_ticket = {
	"eventId": "SX-2025-06-19-00001",
	"eventType": "泥头车遗撒",
	"processActionCode": "CheckHandleResult",
	"processDataParams": {
        "districtName": "福田",
        "sourceEntity": "城市管家",
        "sceneType": "泥头车",
        "regulatedEntity": "某某运输有限公司",
        "regulatedEntityType": "运输企业",
        "reportLocation": "广东省深圳市福田区梅林街道林海山庄",
        "gridCode": "440304008004003",
        "reportTime": "2025-06-20 06:48:20",
        "reporter": "张三",
        "reportEntity": "城市管家",
        "longitude": 121.473,
        "latitude": 31.230,
        "description": "滨河路。东往西滨河新洲立交处，该车辆超高超载。未密闭，沿途撒落。"
	    },
    "context": {
        "imageDescription": "[811702d688c3: 这张图片显示了一辆载满土方的卡车行驶在深圳市福田区的道路上，路牌指示了前往南山、香蜜湖路、新洲路等方向。] [7c7e65a9840b: 这张图片显示了一辆载满土方的卡车行驶在深圳市福田区的道路上，背景是高楼大厦。]",
        "dispatchTarget": "街道办",
		"actions": [
            {
            "target": "城市管家",
            "content": "- **工地企业**的责任依据：\n- 根据《深圳市建筑废弃物管理办法》第二十二条第一款，工地企业需在出入口设置冲洗设施并配备专人检查车辆装载情况，严禁超高超载。工单中车辆存在超高超载且未密闭的情况，表明工地企业未履行检查义务，导致违规车辆离场。\n- 依据《深圳经济特区市容和环境卫生管理条例》第四十三条第一款第三项，工地企业应确保车辆出场前冲洗清理，禁止车轮、车厢外挂泥。事件中车辆未密闭导致遗洒，说明工地企业未落实车辆清洁管理。\n- 区住建局的监管职责明确要求工地企业禁止车体不洁、车厢外挂泥、超载等车辆出场（行政监管主体部分），而工地企业未有效执行。\n- **运输企业及驾驶员**的责任依据：\n- 根据《深圳市建筑废弃物管理办法》第二十七条，运输车辆需按规定行驶，不得超高超载。工单明确车辆存在超高超载行为，直接违反此条款。\n- 驾驶员未遵守《处置/整改主体》中“车辆操作规范”要求（驾驶全密闭式泥头车、装载量不超核定标准），导致未密闭和遗洒问题。\n- 运输企业未履行动态监管责任（如GPS监控、车辆密封性检查），违反《自查主体》中“车辆安全规范”和“运输流程合规性”要求。\n**综合判定**：工地企业作为源头管理主体，未严格审核车辆装载情况并允许违规车辆出场，负主要责任；运输企业及驾驶员因直接实施超载、未密闭等违规行为，负连带责任。"
            }
        ]
		},
	"checklistInfo": {
	    "handleItems": [
            { "item": "是否及时响应","result": "是"},
            { "item": "是否及时处置","result": "是"},
            { "item": "证据是否完整规范","result": "是"},
            { "item": "是否配合跟进整改","result": "是"},
            { "item": "协作效率是否高效","result": "是"},
            { "item": "复查结果","result": "已完成"}
            ]
    }
    }

    result = main_handler(sample_ticket)
    print(json.dumps(result, ensure_ascii=False, indent=2))

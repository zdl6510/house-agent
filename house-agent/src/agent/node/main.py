from langchain_core.messages import SystemMessage, filter_messages, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.agent.common.context import ContextSchema
from src.agent.common.llm import model
from src.agent.state.main import State, NeedReserveOutput


# 节点：查询持久化信息
def get_store_info(state: State, runtime: Runtime[ContextSchema], config: RunnableConfig, * , store: BaseStore):
    # 搜索用户信息
    # 优先从 runtime.context 获取，如果为 None 则从 config.configurable 中获取
    ctx = runtime.context
    if ctx is None:
        ctx = config.get("configurable", {})
    user_id = str(ctx.get("user_id") or "anonymous")
    namespace = (user_id, "preferences")
    prefs_result = store.search(namespace)
    if prefs_result and prefs_result[0]:
        return {
            "user_preferences": prefs_result[0].value
        }
    else:
        return {
            "user_preferences": {}
        }

class UserMessage(BaseModel):
    type: Literal["recommend_house", "listing_detail", "reserve_house", "get_info", "others"] = Field(
        description="根据用户问题描述判断问题类型：推荐房源、查看指定房源详情、预定房源、获取信息、其它内容"
    )

# 节点：识别用户意图
def identify_question(state: State):
    # state["messages"] # 用户问题  -》 LLM  -> 结构化输出（type） : 推荐、预定、我的、其它
    messages = state.get("messages", [])
    if not messages:
        return {"user_intent": "others"}
    if str(messages[-1].content).strip().startswith("[推荐房源]"):
        return {"user_intent": "recommend_house"}
    if str(messages[-1].content).strip().startswith("[房源详情查询]"):
        return {"user_intent": "listing_detail"}
    try:
        user_intent = model.with_structured_output(UserMessage).invoke(
            [SystemMessage(content="你是一个根据描述提取信息的提取专家。请从用户的描述中提取想要咨询的相关信息。"
                        "严谨根据语义推断信息，但是不能猜测或者编造信息。"), messages[-1]]
        )
    except Exception:
        # The router remains useful during a transient LLM failure.
        content = str(messages[-1].content).lower()
        if any(word in content for word in ("推荐", "找房", "租房", "房源")):
            intent = "recommend_house"
        elif any(word in content for word in ("预定", "预订", "预约")):
            intent = "reserve_house"
        elif any(word in content for word in ("我的", "偏好", "订单", "预算")):
            intent = "get_info"
        else:
            intent = "others"
        return {"user_intent": intent}
    return {
        "user_intent": user_intent.type  # 条件边使用
    }

# 节点：中断询问是否需要帮助预定房源
def need_reserve(state: State) -> NeedReserveOutput:
    prompt = f"已经为您推荐合适的房源，是否需要帮您预订房源？\n"
    prompt += "如果不需要,请输入'**不需要**'。\n"
    prompt += "如果需要,请输入'**需要**'。\n(注意输入其它值无效)\n"
    answer = interrupt(prompt)
    return {"reserve": answer}  # 条件边获取到后，是否执行预定子图

# 节点：返回用户偏好信息
def get_user_preferences(state: State):

    # 获取最新历史偏好信息（参考答案）
    prefs = state.get("user_preferences", {})
    # 筛选用户消息（获取到用户问题）
    user_messages = filter_messages(state["messages"], include_types="human")
    reserved_info = prefs.get("reserved_info", [])
    if reserved_info:
        # 有预定过的信息
        reserved_str = "\n"
        for i, item in enumerate(reserved_info, 1):
            reserved_str += f"{i}. 预定工单ID: {item.get('order_id')}，" \
                            f"房源标题：{item.get('title')}，" \
                            f"预定电话：{item.get('phone_number')}\n"
    else:
        # 没有预定
        reserved_str = "无"

    if not user_messages:
        return {"messages": [AIMessage(content="暂时没有可查询的问题，请告诉我您想了解的偏好或预订信息。")]}
    try:
        result = model.invoke(
            [SystemMessage(content="""你是一个乐于助人的助手，可以根据用户偏好信息进行回复。
如果有的偏好数据为空，不要猜测或编造数据。
不要直接回复偏好数据是什么，要结合问题进行生动回复。
如果问题与用户偏好数据无关，直接回复即可。""") ,
             HumanMessage(content="用户的历史偏好信息如下\n"
                          f"1. 最低预算：{prefs.get('budget_min')}\n"
                          f"2. 最高预算：{prefs.get('budget_max')}\n"
                          f"3. 已预定过的信息：{reserved_str}"),
             user_messages[-1]]
        )
    except Exception:
        result = AIMessage(content=(
            f"您的历史预算为 {prefs.get('budget_min') or '未设置'} - "
            f"{prefs.get('budget_max') or '未设置'} 元/月，历史预订：{reserved_str}。"
        ))
    return {
        "messages": [result]
    }




from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import MessagesState

from src.agent.common.llm import model


def extend_node(state: MessagesState):
    try:
        response = model.invoke(
            [SystemMessage(content="你是一个专业的租房助手。优先回答租房相关问题；信息不足时明确说明，不编造房源信息。")]
            + state.get("messages", [])
        )
    except Exception:
        response = AIMessage(content="智能服务暂时繁忙。您仍可使用首页真实房源筛选，或稍后再次发送问题。")
    return {"messages": [response]}

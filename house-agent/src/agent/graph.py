from typing import Literal

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from src.agent.extend import extend_graph
from src.agent.recommend import recommended_graph
from src.agent.reserve import reserve_graph
from src.agent.common.context import ContextSchema
from src.agent.node.main import get_store_info, identify_question, get_user_preferences, need_reserve
from src.agent.state.main import State, NeedReserveOutput

builder = StateGraph(State, context_schema=ContextSchema)

builder.add_node(get_store_info)
builder.add_node(identify_question)
builder.add_node("recommended_graph", recommended_graph)
builder.add_node("reserve_graph", reserve_graph)
builder.add_node("extend_graph", extend_graph)
builder.add_node(get_user_preferences)
builder.add_node(need_reserve)

builder.add_edge(START, "get_store_info")
builder.add_edge("get_store_info", "identify_question")

# 智能路由
def router_message(state: State) -> Literal["recommended_graph", "reserve_graph", "extend_graph", "get_user_preferences"]:
    user_intent = state["user_intent"]
    if user_intent == "recommend_house":
        return "recommended_graph"
    elif user_intent == "reserve_house":
        return "reserve_graph"
    elif user_intent == "get_info":
        return "get_user_preferences"
    else:
        return "extend_graph"

builder.add_conditional_edges(
    "identify_question",
    router_message,
    ["recommended_graph", "reserve_graph", "extend_graph", "get_user_preferences"]
)

# 路由1：推荐子图：根据用户中断信息决定后续是否继续预定
builder.add_edge("recommended_graph", "need_reserve")
def should_reserve(state: NeedReserveOutput):
    reserve = state["reserve"]
    if reserve == "需要":
        return "reserve_graph"
    else:
        return END
builder.add_conditional_edges(
    "need_reserve",
    should_reserve,
    ["reserve_graph", END]
)

# 路由2：预定子图
builder.add_edge("reserve_graph", END)

# 路由3：查询我的
builder.add_edge("get_user_preferences", END)

# 路由4：其它
builder.add_edge("extend_graph", END)
graph = builder.compile()
print(graph.get_graph(xray=True).draw_mermaid())



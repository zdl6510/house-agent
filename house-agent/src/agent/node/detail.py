"""Database-backed listing detail response for the rental assistant."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.catalog import lookup_listing_detail
from src.agent.common.database import DatabaseUnavailable
from src.agent.state.main import State


DETAIL_PREFIX = "[房源详情查询]"


def _request_from_state(state: State) -> tuple[str, str]:
    messages = state.get("messages", [])
    message = next((item for item in reversed(messages) if isinstance(item, HumanMessage)), None)
    content = str(message.content if message else "")
    payload = content.removeprefix(DETAIL_PREFIX).strip()
    try:
        parsed = json.loads(payload)
        return str(parsed.get("id") or ""), str(parsed.get("title") or "")
    except (json.JSONDecodeError, AttributeError):
        return "", payload


def query_listing_detail(state: State):
    """Query the selected listing directly and return a user-friendly detail card."""

    listing_id, title = _request_from_state(state)
    try:
        listing = lookup_listing_detail(listing_id, title)
    except DatabaseUnavailable as exc:
        return {"messages": [AIMessage(content=f"暂时无法连接房源数据库：{exc}")]}
    except Exception:
        return {"messages": [AIMessage(content="查询房源详情时遇到问题，请稍后重试。")]}

    if not listing:
        return {"messages": [AIMessage(content="数据库中没有找到这套房源，它可能已下架或信息已更新。请重新搜索房源。")]} 

    location = " · ".join(filter(None, (listing["city"], listing["region"], listing["address"]))) or "待确认"
    price = f"¥{listing['price']:,.0f}/月" if listing["price"] is not None else "价格面议"
    tags = "、".join(listing["tags"]) or "暂无"
    content = (
        f"### {listing['title']}\n\n"
        f"- 房源编号：{listing['id'] or '暂无'}\n"
        f"- 租金：{price}\n"
        f"- 位置：{location}\n"
        f"- 户型：{listing['layout'] or '待确认'}\n"
        f"- 朝向：{listing['orientation'] or '待确认'}\n"
        f"- 房源状态：{listing['status'] or '以实时咨询为准'}\n"
        f"- 房源特色：{tags}\n\n"
        f"**房源介绍**\n\n{listing['intro'] or '数据库暂未提供更多介绍。'}"
    )
    return {"messages": [AIMessage(content=content)]}

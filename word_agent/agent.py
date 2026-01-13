# Copyright (c) Microsoft. All rights reserved.
"""Word generation agent for Werewolf (谁是卧底) game.

This agent uses OpenAI-compatible API (e.g., SiliconFlow) to generate word pairs.
"""

import os
import asyncio
from dataclasses import dataclass
from dotenv import load_dotenv

from agent_framework.openai import OpenAIChatClient

load_dotenv()


@dataclass
class WordPair:
    """A pair of words for the game - one civilian word and one spy word."""
    civilian_word: str
    spy_word: str
    category: str


def setup_word_agent():
    """Setup the OpenAI-based word generation agent (supports SiliconFlow, OpenAI, etc.)."""
    client = OpenAIChatClient(
        model_id=os.getenv("OPENAI_CHAT_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1"),
    )

    agent = client.create_agent(
        name="WordAgent",
        instructions="""
        您是"谁是卧底"游戏的词语生成助手。您需要生成一对相似的词语：
        - 一个平民词（大多数人看到）
        - 一个卧底词（卧底看到）

        要求：
        1. 两个词语必须相似但不完全相同
        2. 卧底词与平民词应该有一些共同特征，容易混淆
        3. 词语应该简单易懂，适合游戏
        4. 输出格式严格按照以下JSON格式，不要包含其他内容：

        {"civilian_word": "平民词", "spy_word": "卧底词", "category": "类别"}

        示例：
        {"civilian_word": "苹果", "spy_word": "梨", "category": "水果"}
        {"civilian_word": "狗", "spy_word": "猫", "category": "动物"}
        {"civilian_word": "汽车", "spy_word": "火车", "category": "交通工具"}
        """
    )

    return agent


async def generate_word_pair(agent, category: str = "随机") -> WordPair:
    """Generate a word pair for the game."""
    prompt = f"""
    生成一个"{category}"类别的卧底游戏词语对。

    请严格按照JSON格式输出，不要包含其他内容。
    """

    response = await agent.run(prompt)
    result = response.text.strip()

    # Parse the JSON response
    import json
    try:
        # Try to find JSON in the response
        start = result.find('{')
        end = result.rfind('}') + 1
        if start != -1 and end != -1:
            json_str = result[start:end]
            data = json.loads(json_str)
            return WordPair(
                civilian_word=data["civilian_word"],
                spy_word=data["spy_word"],
                category=data.get("category", category)
            )
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    # Fallback: try to parse manually
    # Try common Chinese word pairs as fallback
    fallback_pairs = [
        ("苹果", "梨", "水果"),
        ("香蕉", "芭蕉", "水果"),
        ("狗", "猫", "动物"),
        ("狗", "狼", "动物"),
        ("汽车", "火车", "交通工具"),
        ("飞机", "直升机", "交通工具"),
        ("桌子", "椅子", "家具"),
        ("床", "沙发", "家具"),
        ("手机", "电话", "电子产品"),
        ("电脑", "平板", "电子产品"),
    ]

    import random
    civilian, spy, cat = random.choice(fallback_pairs)
    return WordPair(civilian_word=civilian, spy_word=spy, category=cat)

# Copyright (c) Microsoft. All rights reserved.
"""Game logic agent for Werewolf (谁是卧底) game.

This agent handles game logic, AI player speech generation, and AI voting.
Supports OpenAI-compatible API (e.g., SiliconFlow, OpenAI).
"""

import os
import asyncio
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from dotenv import load_dotenv

from agent_framework.openai import OpenAIChatClient

load_dotenv()


@dataclass
class Player:
    """Represents a player in the game."""
    name: str
    role: str  # "civilian" or "spy"
    word: str
    is_human: bool
    is_alive: bool = True
    speech: str = ""  # Current round's speech
    vote: str = ""  # Current round's vote

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "role": self.role,
            "is_human": self.is_human,
            "is_alive": self.is_alive,
            "speech": self.speech,
            "vote": self.vote
        }


@dataclass
class GameState:
    """Represents the current state of the game."""
    category: str
    civilian_word: str
    spy_word: str
    players: List[Player] = field(default_factory=list)
    current_round: int = 1
    current_player_index: int = 0
    phase: str = "waiting"  # waiting, speaking, voting, result
    eliminated_player: Optional[Player] = None
    winner: Optional[str] = None
    vote_results: Dict[str, int] = field(default_factory=dict)
    history: List[dict] = field(default_factory=list)  # Game history

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "civilian_word": self.civilian_word,
            "spy_word": self.spy_word,
            "players": [p.to_dict() for p in self.players],
            "current_round": self.current_round,
            "current_player_index": self.current_player_index,
            "phase": self.phase,
            "eliminated_player": self.eliminated_player.to_dict() if self.eliminated_player else None,
            "winner": self.winner,
            "vote_results": self.vote_results,
            "history": self.history
        }

    def get_human_player(self) -> Optional[Player]:
        """Get the human player."""
        for p in self.players:
            if p.is_human:
                return p
        return None

    def get_alive_players(self) -> List[Player]:
        """Get all alive players."""
        return [p for p in self.players if p.is_alive]

    def get_current_player(self) -> Optional[Player]:
        """Get the current player whose turn it is."""
        alive = self.get_alive_players()
        if self.current_player_index < len(alive):
            return alive[self.current_player_index]
        return None


def setup_ai_speech_agent():
    """Setup the OpenAI-based AI speech generation agent."""
    client = OpenAIChatClient(
        model_id=os.getenv("OPENAI_CHAT_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1"),
    )

    agent = client.create_agent(
        name="AISpeechAgent",
        instructions="""
        您是"谁是卧底"游戏中的AI玩家。您需要根据您的身份和词语，生成一句描述。

        规则：
        1. 如果您是平民：围绕您的平民词进行描述，但不要过于明显
        2. 如果您是卧底：试图模仿平民词的描述特征，混淆视听
        3. 描述要简短，一句话即可
        4. 不要直接说出词语本身
        5. 语言要自然，像正常人说话一样

        请只输出您的描述文字，不要输出其他内容。
        """
    )

    return agent


def setup_ai_voting_agent():
    """Setup the OpenAI-based AI voting agent."""
    client = OpenAIChatClient(
        model_id=os.getenv("OPENAI_CHAT_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1"),
    )

    agent = client.create_agent(
        name="AIVotingAgent",
        instructions="""
        您是"谁是卧底"游戏中的AI玩家。现在需要投票指认卧底。

        您会收到：
        - 所有玩家的发言
        - 游戏轮次

        请分析所有发言，判断谁最可疑，然后投票给您认为的卧底。

        请只输出您要投票的玩家名字，不要输出其他内容。
        """
    )

    return agent


def create_game(
    category: str,
    civilian_word: str,
    spy_word: str,
    num_ai_players: int = 3
) -> GameState:
    """Create a new game with players. Exactly one spy. Human player is always first (Speaker 0)."""
    players = []

    # 先给人类玩家随机分配身份 (25% 概率是卧底)
    is_human_spy = random.random() < 0.25
    human_role = "spy" if is_human_spy else "civilian"
    human_word = spy_word if is_human_spy else civilian_word

    # 人类玩家始终放在第一个位置（Speaker 0）
    players.append(Player(
        name="你",
        role=human_role,
        word=human_word,
        is_human=True
    ))

    # Add AI players
    ai_names = ["小红", "小明", "张三", "李四", "王五", "赵六", "钱七", "孙八"]
    random.shuffle(ai_names)

    for i in range(num_ai_players):
        players.append(Player(
            name=ai_names[i],
            role="civilian",  # 先都设为平民
            word=civilian_word,
            is_human=False
        ))

    # 如果人类是平民，随机选一个AI当卧底
    if not is_human_spy:
        # 排除人类玩家，AI玩家索引从1开始
        spy_index = random.randint(1, num_ai_players)
        players[spy_index].role = "spy"
        players[spy_index].word = spy_word

    # Debug: 打印所有玩家信息
    print("\n" + "="*50)
    print("🎮 谁是卧底 - 游戏调试信息")
    print("="*50)
    print(f"📚 类别: {category}")
    print(f"👥 平民词: {civilian_word}")
    print(f"🕵️ 卧底词: {spy_word}")
    print("-"*50)
    print("👤 玩家列表:")
    for i, p in enumerate(players, 1):
        role_name = "🕵️ 卧底" if p.role == "spy" else "👥 平民"
        print(f"  {i}. {p.name} [{role_name}] - 词: {p.word} {'(人类)' if p.is_human else '(AI)'}")
    print("="*50 + "\n")

    return GameState(
        category=category,
        civilian_word=civilian_word,
        spy_word=spy_word,
        players=players,
        phase="speaking"
    )


async def generate_ai_speech(agent, player: Player, game_state: GameState) -> str:
    """Generate speech for an AI player."""
    alive_players = game_state.get_alive_players()
    other_speeches = []
    for p in alive_players:
        if p.name != player.name and p.speech:
            other_speeches.append(f"{p.name}: {p.speech}")

    prompt = f"""
    您是玩家「{player.name}」。
    您的身份是：{"卧底" if player.role == "spy" else "平民"}

    你的词语是：{player.word}

    其他玩家的发言：
    {chr(10).join(other_speeches) if other_speeches else "暂无其他玩家发言"}

    请根据您的身份和当前情况，生成一句描述。
    """

    response = await agent.run(prompt)
    return response.text.strip()


async def generate_ai_vote(agent, player: Player, game_state: GameState) -> str:
    """Generate vote for an AI player."""
    alive_players = game_state.get_alive_players()

    prompt = f"""
    您是玩家「{player.name}」，身份是{"卧底" if player.role == "spy" else "平民"}。

    当前存活玩家：{[p.name for p in alive_players]}

    本轮所有发言：
    """

    for p in alive_players:
        if p.speech:
            prompt += f"  {p.name}: {p.speech}\n"

    prompt += f"""
    游戏进行到第{game_state.current_round}轮。

    请分析所有发言，判断谁最可疑，然后投票给您认为的卧底。
    请只输出玩家名字，不要输出其他内容。
    """

    response = await agent.run(prompt)
    return response.text.strip()


def next_player(game_state: GameState) -> None:
    """Move to the next player."""
    alive = game_state.get_alive_players()

    # Find next alive player
    next_index = game_state.current_player_index + 1
    if next_index >= len(alive):
        # All players have spoken, move to voting
        game_state.phase = "voting"
    else:
        game_state.current_player_index = next_index


def process_vote(game_state: GameState) -> None:
    """Process voting and eliminate a player."""
    alive = game_state.get_alive_players()

    # Count votes
    vote_count = {}
    for player in alive:
        if player.vote:
            vote_count[player.vote] = vote_count.get(player.vote, 0) + 1

    game_state.vote_results = vote_count

    if not vote_count:
        # No votes, eliminate a random player
        eliminated = random.choice(alive)
    else:
        # Find player with most votes
        max_votes = max(vote_count.values())
        candidates = [name for name, count in vote_count.items() if count == max_votes]

        if len(candidates) == 1:
            # Single player with most votes
            eliminated_name = candidates[0]
        else:
            # Tie, pick random
            eliminated_name = random.choice(candidates)

        eliminated = next((p for p in alive if p.name == eliminated_name), alive[0])

    # Eliminate the player
    eliminated.is_alive = False
    game_state.eliminated_player = eliminated

    # Record to history
    game_state.history.append({
        "round": game_state.current_round,
        "eliminated": eliminated.name,
        "role": eliminated.role,
        "votes": vote_count,
        "vote_details": {p.name: p.vote for p in alive}  # 记录谁投了谁
    })

    # Check win conditions
    alive_after = game_state.get_alive_players()
    alive_spies = [p for p in alive_after if p.role == "spy"]
    alive_civilians = [p for p in alive_after if p.role == "civilian"]

    # 卧底被投出 → 平民胜利
    if eliminated.role == "spy":
        game_state.winner = "civilians"
        game_state.phase = "result"
        return

    # 卧底存活，检查是否 1卧底+1平民
    if len(alive_spies) == 1 and len(alive_civilians) == 1:
        game_state.winner = "spy"
        game_state.phase = "result"
        return

    # 继续下一轮
    game_state.current_round += 1
    game_state.current_player_index = 0
    game_state.phase = "speaking"

    # Clear speeches
    for p in alive_after:
        p.speech = ""
        p.vote = ""


def reset_game(game_state: GameState) -> None:
    """Reset the game state for a new game."""
    for p in game_state.players:
        p.is_alive = True
        p.speech = ""
        p.vote = ""
    game_state.current_round = 1
    game_state.current_player_index = 0
    game_state.phase = "speaking"
    game_state.eliminated_player = None
    game_state.winner = None
    game_state.vote_results = {}

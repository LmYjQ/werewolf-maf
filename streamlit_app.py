# Copyright (c) Microsoft. All rights reserved.
"""Streamlit Web UI for Werewolf (谁是卧底) game.

This module provides the web interface for playing the Werewolf game.
"""

import streamlit as st
import asyncio
import random
from dataclasses import asdict
from typing import List, Optional
import os
from dotenv import load_dotenv

# Voice service for TTS
from voice_service import VoiceService, VoiceConfig, get_voice_service

# 加载 .env 文件
load_dotenv()

# 验证环境变量
def check_env_config():
    """检查环境变量配置状态（静默检查，不打印到命令行）"""
    issues = []

    provider = os.getenv("LLM_PROVIDER", "openai")
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        model_id = os.getenv("OPENAI_CHAT_MODEL_ID", "")
        base_url = os.getenv("OPENAI_BASE_URL", "")

        if not api_key:
            issues.append("❌ OPENAI_API_KEY 未设置")
        if not model_id:
            issues.append("⚠️ OPENAI_CHAT_MODEL_ID 未设置，使用默认值")
        if not base_url:
            issues.append("⚠️ OPENAI_BASE_URL 未设置，使用默认值")
    else:
        model_id = os.getenv("OLLAMA_CHAT_MODEL_ID", "")
        if not model_id:
            issues.append("⚠️ OLLAMA_CHAT_MODEL_ID 未设置，使用默认值")

    return issues

# 环境变量检查（不打印到命令行）
env_issues = check_env_config()

# 根据环境变量决定默认使用哪个客户端
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

from word_agent import generate_word_pair
from game_agent import (
    create_game,
    GameState,
    Player,
    generate_ai_speech,
    generate_ai_vote,
    next_player,
    process_vote,
    reset_game,
)

# 根据选择导入不同的客户端
if DEFAULT_PROVIDER == "openai":
    from agent_framework.openai import OpenAIChatClient
    ClientClass = OpenAIChatClient
else:
    from agent_framework.ollama import OllamaChatClient
    ClientClass = OllamaChatClient

# Page config
st.set_page_config(
    page_title="谁是卧底 - AI对战版",
    page_icon="🎭",
    layout="wide"
)

# CSS styles
st.markdown("""
<style>
    .game-title {
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        color: #FF6B6B;
        margin-bottom: 20px;
    }
    .word-display {
        text-align: center;
        font-size: 36px;
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 20px 0;
    }
    .player-card {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        background: #f0f2f6;
    }
    .speech-box {
        padding: 15px;
        border-radius: 10px;
        background: #e8f4fd;
        border-left: 5px solid #2196F3;
        margin: 10px 0;
    }
    .vote-button {
        width: 100%;
        margin: 5px 0;
    }
    .result-box {
        text-align: center;
        padding: 30px;
        border-radius: 20px;
        font-size: 24px;
        margin: 20px 0;
    }
    .spy-win {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .civilian-win {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "game_state" not in st.session_state:
        st.session_state.game_state = None
    if "word_pair" not in st.session_state:
        st.session_state.word_pair = None
    if "page" not in st.session_state:
        st.session_state.page = "setup"
    if "speech_agent" not in st.session_state:
        st.session_state.speech_agent = None
    if "vote_agent" not in st.session_state:
        st.session_state.vote_agent = None
    if "llm_config" not in st.session_state:
        st.session_state.llm_config = None
    if "user_speech" not in st.session_state:
        st.session_state.user_speech = ""
    if "user_vote" not in st.session_state:
        st.session_state.user_vote = ""
    if "voice_mode" not in st.session_state:
        st.session_state.voice_mode = False
    if "voice_service" not in st.session_state:
        st.session_state.voice_service = None
    if "voice_cache" not in st.session_state:
        st.session_state.voice_cache = {}  # Cache for generated audio


def setup_page():
    """Render the game setup page."""
    st.markdown('<div class="game-title">🎭 谁是卧底 🎭</div>', unsafe_allow_html=True)

    # 显示环境变量配置状态（仅在设置页面）
    if env_issues:
        with st.expander("⚙️ 环境变量配置", expanded=True):
            for issue in env_issues:
                st.write(issue)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.info("🤖 这是一个AI人机对战版本，你将与AI玩家进行游戏。")

        # 显示环境变量配置状态
        with st.expander("📋 环境变量配置状态", expanded=True):
            provider = os.getenv("LLM_PROVIDER", "openai")
            if provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY", "")
                model_id = os.getenv("OPENAI_CHAT_MODEL_ID", "")
                base_url = os.getenv("OPENAI_BASE_URL", "")

                if api_key:
                    masked = api_key[:4] + "****" + api_key[-4:]
                    st.success(f"✅ OPENAI_API_KEY: {masked}")
                else:
                    st.error("❌ OPENAI_API_KEY 未设置")

                if model_id:
                    st.success(f"✅ OPENAI_CHAT_MODEL_ID: {model_id}")
                else:
                    st.warning("⚠️ 使用默认值: Qwen/Qwen2.5-7B-Instruct")

                if base_url:
                    st.success(f"✅ OPENAI_BASE_URL: {base_url}")
                else:
                    st.warning("⚠️ 使用默认值: https://api.siliconflow.cn/v1")
            else:
                model_id = os.getenv("OLLAMA_CHAT_MODEL_ID", "")
                if model_id:
                    st.success(f"✅ OLLAMA_CHAT_MODEL_ID: {model_id}")
                else:
                    st.warning("⚠️ 使用默认值: qwen3:8b")

        st.markdown("### 游戏设置")

        # 语音模式开关
        voice_mode = st.toggle(
            "🎙️ 语音模式",
            value=False,
            help="开启后，AI玩家的发言将转换为语音播放。需要连接VibeVoice服务。"
        )

        if voice_mode:
            col_voice1, col_voice2 = st.columns(2)
            with col_voice1:
                voice_server_url = st.text_input(
                    "TTS 服务地址",
                    value=os.getenv("TTS_SERVER_URL", "https://development-1717-xllvcwtu-8090.550w.link"),
                    help=f"本地服务: {os.getenv("TTS_SERVER_URL", "https://development-1717-xllvcwtu-8090.550w.link")}"
                )
            with col_voice2:
                if st.button("🔗 测试连接", use_container_width=True):
                    voice_config = VoiceConfig(server_url=voice_server_url)
                    voice_service = get_voice_service(voice_config)
                    with st.spinner("正在测试连接..."):
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        success, message = loop.run_until_complete(
                            voice_service.test_connection()
                        )
                        if success:
                            # Refresh voices from server after successful connection
                            loop.run_until_complete(voice_service.refresh_voices())
                            st.success(f"✅ {message}")

                            # 测试 TTS 服务，生成并播放测试语音
                            with st.spinner("正在测试 TTS 服务..."):
                                test_text = "文字转语音服务正常运行"
                                # VibeVoice 需要 speaker: text 格式
                                voice_preset = voice_service.config.voice_presets.get("你", "WHMale")
                                formatted_text = f"{voice_preset}: {test_text}"
                                audio_data, tts_error = loop.run_until_complete(
                                    voice_service.text_to_speech(formatted_text, voice_key="你", use_full=True)
                                )
                                if tts_error:
                                    st.warning(f"⚠️ 连接成功但 TTS 测试失败: {tts_error}")
                                elif audio_data:
                                    st.audio(audio_data, format="audio/wav")
                                    st.info("🔊 TTS 服务测试通过，音频已播放")

                            with st.expander("已加载的声音配置"):
                                st.json(voice_service.config.voice_presets)
                        else:
                            st.error(f"❌ {message}")
                            with st.expander("查看解决方案"):
                                st.markdown("""
                                **启动本地 TTS 服务：**

                                ```bash
                                # 安装依赖
                                pip install vibevoice fastapi uvicorn aiohttp

                                # 启动服务
                                python tts_server.py --model-path microsoft/VibeVoice-1.5b
                                ```

                            **或使用本地模型路径：**

                            ```bash
                            python tts_server.py --model-path ./your-local-model
                            ```
                            """)

        st.markdown("---")

        # LLM 服务商选择
        provider = st.selectbox(
            "选择 LLM 服务商",
            ["Ollama (本地)", "OpenAI/SiliconFlow (在线)"],
            index=0 if DEFAULT_PROVIDER == "ollama" else 1,
            key="provider_select"
        )
        provider_type = "ollama" if "Ollama" in provider else "openai"

        # 根据选择显示不同的模型ID
        if provider_type == "ollama":
            model_id = st.text_input("Ollama 模型 ID", value=os.getenv("OLLAMA_CHAT_MODEL_ID", "qwen3:8b"))
        else:
            model_id = st.text_input("模型 ID", value=os.getenv("OPENAI_CHAT_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"))
            api_key = st.text_input("API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
            base_url = st.text_input("Base URL", value=os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1"))

        st.markdown("---")

        category = st.selectbox(
            "选择词语类别",
            ["水果", "动物", "食物", "交通工具", "电子产品", "日常用品", "职业", "自然"]
        )

        num_ai = st.slider("AI玩家数量", 2, 5, 3)

        if st.button("🎮 开始游戏", type="primary", use_container_width=True):
            # 收集配置
            config = {
                "provider": provider_type,
                "model_id": model_id,
                "voice_mode": voice_mode,
            }
            if provider_type == "openai":
                config["api_key"] = api_key
                config["base_url"] = base_url
            if voice_mode:
                config["voice_server_url"] = voice_server_url

            start_game(category, num_ai_players=num_ai, llm_config=config)

        st.markdown("---")
        st.markdown("### 游戏规则")
        st.markdown("""
        1. 每个人看到一个词语，大部分人的词语相同（平民），有1人的词语相似但不同（卧底）
        2. 轮流用一句话描述自己的词语
        3. 根据发言判断谁是卧底
        4. 投票指认卧底，被指认最多的玩家出局
        5. 如果卧底被投出，平民获胜；如果卧底存活到最后，卧底获胜
        """)


def create_client(llm_config: dict):
    """根据配置创建 LLM 客户端"""
    provider = llm_config.get("provider", "ollama")
    model_id = llm_config.get("model_id", "qwen3:8b")

    if provider == "openai":
        return ClientClass(
            model_id=model_id,
            api_key=llm_config.get("api_key"),
            base_url=llm_config.get("base_url"),
        )
    else:
        return ClientClass(model_id=model_id)


def setup_agents_with_config(llm_config: dict):
    """根据配置创建 agents"""
    client = create_client(llm_config)

    # 词语生成 agent
    word_agent = client.create_agent(
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
        """
    )

    # AI发言 agent
    speech_agent = client.create_agent(
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

    # AI投票 agent
    vote_agent = client.create_agent(
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

    return word_agent, speech_agent, vote_agent


def start_game(category: str, num_ai_players: int, llm_config: dict):
    """Start a new game."""
    # 保存配置到 session state
    st.session_state.llm_config = llm_config
    st.session_state.voice_mode = llm_config.get("voice_mode", False)

    # Initialize agents with config
    word_agent, speech_agent, vote_agent = setup_agents_with_config(llm_config)
    st.session_state.speech_agent = speech_agent
    st.session_state.vote_agent = vote_agent

    # Initialize voice service if voice mode is enabled
    if st.session_state.voice_mode:
        voice_server_url = llm_config.get("voice_server_url", "http://localhost:3000")
        voice_config = VoiceConfig(server_url=voice_server_url)
        st.session_state.voice_service = get_voice_service(voice_config)
        st.session_state.voice_cache = {}

    with st.spinner("正在生成词语..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        word_pair = loop.run_until_complete(generate_word_pair(word_agent, category))
        loop.close()

    st.session_state.word_pair = word_pair

    # Create game
    game_state = create_game(
        category=word_pair.category,
        civilian_word=word_pair.civilian_word,
        spy_word=word_pair.spy_word,
        num_ai_players=num_ai_players
    )

    st.session_state.game_state = game_state
    st.session_state.page = "game"
    st.session_state.user_speech = ""
    st.session_state.user_vote = ""

    st.rerun()


def game_page():
    """Render the main game page."""
    game_state = st.session_state.game_state
    word_pair = st.session_state.word_pair

    st.markdown(f"### 第 {game_state.current_round} 轮 - {get_phase_text(game_state.phase)}")

    # Show player's word (不显示身份，保护游戏体验)
    human = game_state.get_human_player()
    if human:
        st.markdown(f"""
        <div class="word-display">
            你的词：{human.word}
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Show all players' speeches
        st.markdown("### 📝 玩家发言")
        alive_players = game_state.get_alive_players()
        voice_service = st.session_state.get("voice_service") if st.session_state.get("voice_mode") else None

        for player in alive_players:
            if player.speech:
                st.markdown(f"""
                <div class="speech-box">
                    <strong>{player.name}</strong>: {player.speech}
                </div>
                """, unsafe_allow_html=True)

                # Show audio player if voice mode is enabled
                if voice_service and player.speech:
                    try:
                        speaker_index = game_state.players.index(player) + 1
                        voice_key = voice_service.get_voice_for_player(player.name)
                        cache_key = f"{speaker_index}:{player.speech}"

                        # Get or generate audio
                        if cache_key not in st.session_state.voice_cache:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            audio_data, error = loop.run_until_complete(
                                voice_service.text_to_speech(player.speech, voice_key, speaker_index=speaker_index)
                            )
                            loop.close()
                            if error:
                                st.session_state.voice_cache[cache_key] = None
                                audio_data = None
                            else:
                                st.session_state.voice_cache[cache_key] = audio_data
                        else:
                            audio_data = st.session_state.voice_cache[cache_key]

                        # Display audio player
                        if audio_data and len(audio_data) > 0:
                            st.audio(audio_data, format="audio/wav")
                    except Exception:
                        st.caption("🔇 语音不可用")

        # Show eliminated players (不显示词，保护游戏体验)
        eliminated = [p for p in game_state.players if not p.is_alive]
        if eliminated:
            st.markdown("---")
            st.markdown("### ❌ 已出局玩家")
            for p in eliminated:
                role_icon = "🔴" if p.role == "spy" else "🟢"
                st.write(f"{role_icon} {p.name}")

    with col2:
        # Current player's turn or voting
        if game_state.phase == "speaking":
            render_speaking_phase(game_state, alive_players)
        elif game_state.phase == "voting":
            render_voting_phase(game_state, alive_players)
        elif game_state.phase == "result":
            render_result_page(game_state)


def get_phase_text(phase: str) -> str:
    """Get Chinese text for game phase."""
    phase_map = {
        "waiting": "等待中",
        "speaking": "发言阶段",
        "voting": "投票阶段",
        "result": "游戏结束"
    }
    return phase_map.get(phase, phase)


def render_speaking_phase(game_state: GameState, alive_players: List[Player]):
    """Render the speaking phase."""
    current_player = game_state.get_current_player()

    if current_player is None:
        return

    st.markdown("### 🎤 发言")

    if current_player.is_human:
        # Human player's turn
        st.info(f"现在是 **{current_player.name}** 的发言（你）")

        speech = st.text_area(
            "请用一句话描述你的词语：",
            value=st.session_state.user_speech,
            height=100,
            key="speech_input"
        )

        if st.button("提交发言", type="primary", use_container_width=True):
            if speech.strip():
                current_player.speech = speech.strip()
                st.session_state.user_speech = ""
                next_player(game_state)
                st.rerun()
            else:
                st.error("请输入发言内容！")
    else:
        # AI player's turn
        with st.spinner(f"AI玩家 {current_player.name} 正在思考..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            speech = loop.run_until_complete(
                generate_ai_speech(st.session_state.speech_agent, current_player, game_state)
            )
            loop.close()

            current_player.speech = speech

            # Display speech with optional audio
            st.success(f"**{current_player.name}**: {speech}")

            # Play voice if voice mode is enabled
            if st.session_state.get("voice_mode", False):
                voice_service = st.session_state.get("voice_service")
                if voice_service:
                    # Human player is Speaker 1, AI players are 2, 3, 4...
                    # Players list: [human, ai1, ai2, ai3, ...]
                    # So speaker_index = player_index + 1
                    speaker_index = game_state.players.index(current_player) + 1
                    # Get voice preset for this player
                    voice_key = voice_service.get_voice_for_player(current_player.name)
                    # Generate or retrieve cached audio
                    # VibeVoice requires format: "Speaker X: text"
                    cache_key = f"{speaker_index}:{speech}"

                    audio_data = None  # Initialize

                    if cache_key not in st.session_state.voice_cache:
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            audio_data, error = loop.run_until_complete(
                                voice_service.text_to_speech(speech, voice_key, speaker_index=speaker_index)
                            )
                            loop.close()
                            if error:
                                st.session_state.voice_cache[cache_key] = None
                                audio_data = None
                                if "CORS" in error or "Connection" in error:
                                    st.warning(f"🔇 语音服务暂时不可用: {error}")
                            else:
                                st.session_state.voice_cache[cache_key] = audio_data
                        except Exception as e:
                            st.session_state.voice_cache[cache_key] = None
                            st.warning(f"🔇 语音合成失败: {e}")
                            audio_data = None
                    else:
                        audio_data = st.session_state.voice_cache[cache_key]

                    # Display audio player if audio was generated
                    if audio_data and len(audio_data) > 0:
                        st.audio(audio_data, format="audio/wav")
                    else:
                        # Show fallback message
                        st.caption("🔇 语音未生成")

            # Auto advance after delay
            time = st.empty()
            for i in range(3, 0, -1):
                time.info(f"{i}秒后自动进入下一位玩家...")
                import time as tm
                tm.sleep(1)
            time.empty()

            next_player(game_state)
            st.rerun()


def render_voting_phase(game_state: GameState, alive_players: List[Player]):
    """Render the voting phase."""
    st.markdown("### 🗳️ 投票")

    human = game_state.get_human_player()

    if human:
        st.info(f"请选择你认为的卧底 ({human.name})")

        # Filter out self
        vote_options = [p.name for p in alive_players if p.name != human.name]

        vote = st.radio(
            "选择你要投票的玩家：",
            vote_options,
            key="vote_radio"
        )

        if st.button("提交投票", type="primary", use_container_width=True):
            human.vote = vote
            st.session_state.user_vote = vote

            # Process all AI votes
            process_ai_votes(game_state)
            st.rerun()

    # Show other players' votes (already cast)
    votes_shown = [p.vote for p in alive_players if p.vote]
    if votes_shown:
        st.write("其他玩家已投票完成")


def process_ai_votes(game_state: GameState):
    """Process AI player votes."""
    alive = game_state.get_alive_players()

    for player in alive:
        if not player.is_human and not player.vote:
            with st.spinner(f"AI玩家 {player.name} 正在投票..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                vote = loop.run_until_complete(
                    generate_ai_vote(st.session_state.vote_agent, player, game_state)
                )
                loop.close()

                # Validate vote target exists
                valid_targets = [p.name for p in alive if p.name != player.name]
                if vote not in valid_targets:
                    vote = random.choice(valid_targets)

                player.vote = vote

    # Process the voting result
    process_vote(game_state)


def render_result_page(game_state: GameState):
    """Render the game result page."""
    st.markdown("### 🏆 游戏结果")

    human = game_state.get_human_player()

    # Determine result
    if game_state.winner == "civilians":
        civilians_won = (human and human.role == "civilian") or (human is None)
        won = civilians_won

        if won:
            st.markdown('<div class="result-box civilian-win">🎉 平民获胜！ 🎉</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box spy-win">😈 卧底获胜！ 😈</div>', unsafe_allow_html=True)
    elif game_state.winner == "spy":
        won = human and human.role == "spy"

        if won:
            st.markdown('<div class="result-box spy-win">🎭 你是卧底，卧底获胜！ 🎭</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box civilian-win">😢 卧底获胜，平民惜败</div>', unsafe_allow_html=True)

    # Show the answer
    st.markdown("### 📋 揭晓答案")
    col1, col2 = st.columns(2)

    with col1:
        civilian_players = [p for p in game_state.players if p.role == "civilian"]
        st.markdown(f"""
        <div style="padding: 20px; background: #4facfe; border-radius: 10px; color: white;">
            <h3 style="margin: 0;">平民词</h3>
            <p style="font-size: 32px; margin: 10px 0;">{game_state.civilian_word}</p>
            <p>平民玩家：{', '.join([p.name for p in civilian_players])}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        spy_players = [p for p in game_state.players if p.role == "spy"]
        st.markdown(f"""
        <div style="padding: 20px; background: #f5576c; border-radius: 10px; color: white;">
            <h3 style="margin: 0;">卧底词</h3>
            <p style="font-size: 32px; margin: 10px 0;">{game_state.spy_word}</p>
            <p>卧底玩家：{', '.join([p.name for p in spy_players])}</p>
        </div>
        """, unsafe_allow_html=True)

    # Show vote results
    if game_state.eliminated_player:
        st.markdown(f"""
        ### 📊 投票结果
        本轮被投出的玩家：**{game_state.eliminated_player.name}**
        """)

        if game_state.vote_results:
            st.write("得票情况:")
            for name, count in game_state.vote_results.items():
                st.write(f"  {name}: {count} 票")

        # 显示谁投了谁
        if game_state.history:
            last_record = game_state.history[-1]
            if 'vote_details' in last_record:
                st.write("投票详情:")
                vote_details = last_record['vote_details']
                for voter, target in vote_details.items():
                    if target:
                        st.write(f"  {voter} → {target}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 再来一局", type="primary", use_container_width=True):
            reset_game(game_state)
            st.session_state.page = "setup"
            st.rerun()

    with col2:
        if st.button("📊 查看游戏历史", use_container_width=True):
            st.session_state.page = "history"
            st.rerun()


def history_page():
    """Render the game history page."""
    game_state = st.session_state.game_state

    st.markdown("### 📜 游戏历史")

    if game_state.history:
        for i, record in enumerate(game_state.history, 1):
            with st.expander(f"第 {record['round']} 轮"):
                st.write(f"**出局玩家**: {record['eliminated']} (身份: {record['role']})")
                st.write(f"**投票数**: {record['votes']}")
                if 'vote_details' in record:
                    st.write("**投票详情**:")
                    for voter, target in record['vote_details'].items():
                        if target:
                            st.write(f"  {voter} → {target}")
    else:
        st.info("暂无游戏记录")

    if st.button("返回游戏", type="primary"):
        st.session_state.page = "game"
        st.rerun()


def main():
    """Main application entry point."""
    init_session_state()

    # Navigation
    if st.session_state.page == "setup":
        setup_page()
    elif st.session_state.page == "game":
        game_page()
    elif st.session_state.page == "history":
        history_page()


if __name__ == "__main__":
    main()

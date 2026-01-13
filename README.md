# 谁是卧底 - AI对战版

基于 Agent Framework 工作流框架的 AI 人机对战"谁是卧底"游戏应用。

## 游戏介绍

"谁是卧底"是一款多人语言推理游戏。在本版本中，你将与 AI 玩家进行对战，体验完整的游戏乐趣。

## 游戏规则

1. **分发词语**：游戏开始时，每个人看到一个词语。大部分人看到的是相同的词（平民），但有 1-2 人看到的是相似但不同的词（卧底）
2. **轮流发言**：玩家轮流用一句话描述自己的词语，不能太明显也不能太隐晦
3. **投票环节**：根据发言内容判断谁是卧底，所有玩家投票指认
4. **出局判定**：被指认最多的玩家出局
5. **胜负判定**：
   - 如果卧底被投出，平民获胜
   - 如果卧底存活到最后（只剩 1 人），卧底获胜

## 项目结构

```
werewolf_game/
├── README.md                     # 本文档
├── streamlit_app.py              # Streamlit Web 界面
├── workflow.py                   # 工作流定义
├── tts_server.py                 # VibeVoice TTS 服务 (FastAPI)
├── .env                          # 环境变量配置
├── requirements.txt              # 依赖列表
├── word_agent/                   # 词语生成代理
│   ├── __init__.py
│   └── agent.py
├── game_agent/                   # 游戏逻辑代理
│   ├── __init__.py
│   └── agent.py
└── VibeVoice/                    # VibeVoice TTS 库 (子模块)
```

## 核心模块

### 1. 词语生成代理 (word_agent)

负责生成游戏所需的词语对：
- **输入**：主题类别（动物/食物/水果/日常用品等）
- **输出**：平民词、卧底词（相似但有区分度）
- **技术**：使用 OpenAI 兼容 API（如 SiliconFlow）生成词语对

### 2. 游戏逻辑代理 (game_agent)

负责游戏核心逻辑：
- **玩家管理**：人类玩家 + AI 玩家
- **角色分配**：随机分配卧底/平民（人类 25% 概率是卧底）
- **AI 发言生成**：LLM 根据玩家身份生成描述
- **AI 投票策略**：根据发言内容智能投票

### 3. 工作流 (workflow)

使用 Agent Framework 工作流框架协调游戏流程，包括词语生成和 AI 行为触发。

### 4. TTS 服务 (tts_server.py)

基于 VibeVoice 的文本转语音服务：
- **框架**：FastAPI
- **模型**：VibeVoice 1.5B
- **功能**：流式/非流式 TTS 合成
- **端口**：默认 8001

## 技术栈

- **Agent Framework**：工作流编排
- **LLM (SiliconFlow/Qwen)**：AI 推理
- **Streamlit**：Web 界面
- **FastAPI**：TTS 服务
- **VibeVoice**：语音合成

## 环境配置

确保已配置 `.env` 文件：

```bash
# LLM 配置 (OpenAI 兼容 API)
OPENAI_API_KEY="your-api-key"
OPENAI_CHAT_MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
OPENAI_BASE_URL="https://api.siliconflow.cn/v1"

# TTS 服务配置
TTS_SERVER_URL="http://localhost:8001"
```

## 依赖安装

### 1. 安装项目依赖

```bash
pip install -r requirements.txt
```

### 2. 安装 VibeVoice TTS（用于语音合成）

```bash
export HF_ENDPOINT=https://hf-mirror.com
git clone https://ghfast.top/https://github.com/vibevoice-community/VibeVoice.git
cd VibeVoice
pip install -e .
```

## 运行游戏

### 1. 启动 TTS 服务（可选，用于语音播放）

```bash
cd werewolf_game
python tts_server.py --model-mode non-streaming --host 0.0.0.0 --port 8001
```

### 2. 启动游戏界面

```bash
cd werewolf_game
streamlit run streamlit_app.py
```

## 游戏界面

### 首页
- 选择词语类别
- 设置 AI 玩家数量
- 开始游戏

### 游戏页
- 显示你的词语和身份
- 查看所有玩家发言
- 输入你的发言
- 投票指认卧底

### 结果页
- 显示胜负结果
- 揭晓答案
- 查看投票统计

## 界面预览

```
🎭 谁是卧底 🎭
┌─────────────────────────────┐
│  你的词：苹果  👤 平民      │
├─────────────────────────────┤
│  📝 玩家发言                 │
│  ─────────────────          │
│  小红: 这是一种水果         │
│  小明: 红色的很甜           │
│  ...                        │
├─────────────────────────────┤
│  🎤 发言                    │
│  [请输入你的描述...]         │
│  [提交发言]                 │
└─────────────────────────────┘
```

## 扩展功能

后续可扩展的功能：
- 不同难度级别
- 自定义词语库
- 多人在线对战
- 游戏记录和统计
- 多种游戏模式

## 参考

本项目参考了同目录下的 Podcast Application 实现模式，使用 Agent Framework 工作流框架进行开发。

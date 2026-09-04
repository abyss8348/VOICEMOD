# FlowVoice 🎙️

A minimal, real-time AI voice agent built with **Python**, **LiveKit Agents**, and **Rime TTS**.

FlowVoice establishes a bidirectional real-time audio pipeline over WebRTC, processing speech inputs, generating responses via LLM, and synthesizing natural speech with high-fidelity Rime TTS voices.

---

## 🏗️ Architecture & Pipeline

- **Transport / WebRTC**: [LiveKit](https://livekit.io) Real-time Agents framework (`livekit-agents` v1.7.1)
- **Text-to-Speech (TTS)**: [Rime](https://rime.ai) (`livekit-plugins-rime`)
  - Supports `coda` (default), `mistv2`, and `mistv3` models
  - Configurable speakers (e.g. `astra`, `amber`, `marsh`, `bayou`)
- **Speech-to-Text (STT)**: OpenAI Whisper (`livekit-plugins-openai`)
- **LLM Reasoning**: OpenAI GPT-4o-mini (`livekit-plugins-openai`)
- **Voice Activity Detection (VAD)**: Silero VAD (`livekit-plugins-silero`)

---

## 📋 Prerequisites

- **Python 3.10+** (Tested on Python 3.14 / macOS ARM64 & Linux)
- A **LiveKit Cloud** account or self-hosted LiveKit instance
- A **Rime API Key** (from [rime.ai](https://rime.ai))
- An **OpenAI API Key** (from [platform.openai.com](https://platform.openai.com))

---

## 🚀 Quickstart Guide

### 1. Set Up Virtual Environment

```bash
# Navigate to the project root
cd "voice bot"

# Activate the existing virtual environment or create a new one
source .venv/bin/activate
# or: python3 -m venv .venv && source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your API credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```ini
# LiveKit Cloud or Server Credentials
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# Rime TTS
RIME_API_KEY=your_rime_api_key
RIME_MODEL=coda
RIME_SPEAKER=astra

# OpenAI (STT & LLM)
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini

# Logging
LOG_LEVEL=INFO
```

---

## 🏃 How to Run the Agent

### Development Mode (Recommended for testing with LiveKit Cloud / Playground)

Run the agent with hot-reloading:

```bash
python -m backend.agent dev
```

### Connecting via LiveKit Agents Playground

1. Go to [agents-playground.livekit.io](https://agents-playground.livekit.io) or your LiveKit Cloud dashboard.
2. Connect using your LiveKit project credentials.
3. Once you join the room, FlowVoice will automatically connect, greet you through Rime TTS, and engage in real-time voice conversation!

### Production Mode

```bash
python -m backend.agent start
```

### Console Mode (Local audio testing without WebRTC frontend)

```bash
python -m backend.agent console
```

---

## 🧪 Running Tests

To run the automated configuration and integration tests:

```bash
source .venv/bin/activate
python -m unittest discover -s tests
```

---

## 📁 Project Structure

```
.
├── .env.example              # Environment variables template
├── .gitignore                # Secret & cache exclusions
├── README.md                 # Project documentation & run guide
├── requirements.txt          # Python dependencies
├── backend/
│   ├── __init__.py           # Package marker
│   ├── agent.py              # Main voice agent worker & Rime pipeline
│   ├── config.py             # Environment configuration & validation
│   └── logger.py             # Formatted structured logging
└── tests/
    ├── __init__.py           # Test package marker
    └── test_agent_config.py  # Unit and integration tests
```

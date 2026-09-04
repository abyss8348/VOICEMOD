"""Unit and integration tests for FlowVoice configuration, Rime TTS, and Agent setup."""

import asyncio
import os
import unittest
from unittest.mock import patch

from backend.config import Settings
from backend.logger import setup_logger
from livekit.plugins import rime, openai, silero
from livekit.agents.voice import Agent, AgentSession
from livekit.agents import WorkerOptions
from backend.agent import entrypoint, prewarm


class TestFlowVoiceConfig(unittest.TestCase):
    """Test configuration loading and credential validation."""

    def test_settings_validation_missing_all(self):
        """Ensure missing variables are accurately detected."""
        empty_settings = Settings(
            livekit_url="",
            livekit_api_key="",
            livekit_api_secret="",
            rime_api_key="",
            openai_api_key="",
        )
        missing = empty_settings.validate(require_livekit=True)
        self.assertIn("LIVEKIT_URL", missing)
        self.assertIn("LIVEKIT_API_KEY", missing)
        self.assertIn("LIVEKIT_API_SECRET", missing)
        self.assertIn("RIME_API_KEY", missing)
        self.assertIn("OPENAI_API_KEY", missing)

    def test_settings_validation_valid(self):
        """Ensure valid settings pass validation."""
        valid_settings = Settings(
            livekit_url="wss://test.livekit.cloud",
            livekit_api_key="test_key",
            livekit_api_secret="test_secret",
            rime_api_key="rime_secret_key",
            openai_api_key="openai_secret_key",
        )
        missing = valid_settings.validate(require_livekit=True)
        self.assertEqual(len(missing), 0)


class TestRimeTTSIntegration(unittest.TestCase):
    """Test Rime TTS plugin instantiation and options."""

    def test_rime_tts_initialization(self):
        """Verify Rime TTS initializes with custom model and speaker options."""
        tts_instance = rime.TTS(
            model="coda",
            speaker="astra",
            api_key="dummy_key",
        )
        self.assertEqual(tts_instance._opts.model, "coda")
        self.assertEqual(tts_instance._opts.speaker, "astra")
        self.assertEqual(tts_instance._api_key, "dummy_key")

    def test_rime_tts_requires_api_key(self):
        """Verify Rime TTS raises ValueError if no API key is set."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                rime.TTS(api_key=None)


class TestAgentPipelineStructure(unittest.IsolatedAsyncioTestCase):
    """Test Voice Agent and AgentSession pipeline construction."""

    async def test_agent_session_creation(self):
        """Verify AgentSession can be constructed with Rime TTS and OpenAI STT/LLM."""
        tts_plugin = rime.TTS(api_key="test_key", model="coda", speaker="astra")
        stt_plugin = openai.STT(api_key="test_key")
        llm_plugin = openai.LLM(api_key="test_key", model="gpt-4o-mini")

        session = AgentSession(
            stt=stt_plugin,
            llm=llm_plugin,
            tts=tts_plugin,
        )

        agent = Agent(
            instructions="You are FlowVoice, a helpful voice assistant."
        )

        self.assertIsNotNone(session)
        self.assertIsNotNone(agent)
        self.assertEqual(session.tts, tts_plugin)
        self.assertEqual(session.stt, stt_plugin)
        self.assertEqual(session.llm, llm_plugin)

    async def test_whisper_agent_session_creation(self):
        """Verify AgentSession can be constructed with Whisper STT and Ollama LLM."""
        from backend.agent import WhisperSTT
        from livekit.agents import stt

        vad_plugin = silero.VAD.load()
        whisper_plugin = WhisperSTT(model="tiny.en")
        stream_stt = stt.StreamAdapter(stt=whisper_plugin, vad=vad_plugin)
        llm_plugin = openai.LLM.with_ollama(model="llama3.2:3b", base_url="http://localhost:11434/v1")
        tts_plugin = rime.TTS(api_key="test_key", model="coda", speaker="astra")

        session = AgentSession(
            stt=stream_stt,
            llm=llm_plugin,
            tts=tts_plugin,
            vad=vad_plugin,
        )

        self.assertIsNotNone(session)
        self.assertEqual(session.tts, tts_plugin)
        self.assertEqual(session.stt, stream_stt)
        self.assertEqual(session.llm, llm_plugin)

    async def test_agent_session_with_userdata_regression(self):
        """Regression test ensuring AgentSession starts up cleanly with userdata without ValueError."""
        from backend.agent import WhisperSTT
        from backend.generation import GenerationManager
        from livekit.agents import stt

        vad_plugin = silero.VAD.load()
        whisper_plugin = WhisperSTT(model="tiny.en")
        stream_stt = stt.StreamAdapter(stt=whisper_plugin, vad=vad_plugin)
        llm_plugin = openai.LLM.with_ollama(model="llama3.2:3b", base_url="http://localhost:11434/v1")
        tts_plugin = rime.TTS(api_key="test_key", model="coda", speaker="astra")

        gen_manager = GenerationManager()
        session = AgentSession(
            stt=stream_stt,
            llm=llm_plugin,
            tts=tts_plugin,
            vad=vad_plugin,
            userdata={"gen_manager": gen_manager},
        )
        gen_manager.set_session(session)

        # Ensure session.userdata does NOT raise ValueError and correctly exposes gen_manager
        self.assertIsNotNone(session.userdata)
        self.assertIn("gen_manager", session.userdata)
        self.assertIs(session.userdata["gen_manager"], gen_manager)

    def test_worker_options_configuration(self):
        """Verify WorkerOptions binds entrypoint and prewarm functions properly."""
        opts = WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
        self.assertEqual(opts.entrypoint_fnc, entrypoint)
        self.assertEqual(opts.prewarm_fnc, prewarm)


if __name__ == "__main__":
    unittest.main()

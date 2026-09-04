"""FlowVoice - Minimal Realtime Voice Agent using LiveKit Agents, Local Whisper STT, Ollama LLM & Rime TTS."""

import asyncio
import io
import sys
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    AgentServer,
    cli,
    llm,
    stt,
    utils,
)

from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, rime, silero
from faster_whisper import WhisperModel

from .config import settings
from .logger import logger
from .generation import GenerationManager, GenerationTaskResult


class WhisperSTT(stt.STT):
    """Local Speech-to-Text using faster-whisper on CPU/MPS without external APIs or keys."""

    def __init__(
        self,
        *,
        model: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self._model_name = model
        self._language = language
        self._model = WhisperModel(model, device=device, compute_type=compute_type)

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "faster-whisper"

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: str | None = None,
        conn_options: stt.APIConnectOptions | None = None,
    ) -> stt.SpeechEvent:
        combined = rtc.combine_audio_frames(buffer)
        wav_bytes = combined.to_wav_bytes()
        lang = language or self._language

        def _transcribe():
            segments, info = self._model.transcribe(
                io.BytesIO(wav_bytes),
                language=lang if lang != "auto" else None,
                beam_size=1,
            )
            return " ".join([s.text for s in segments]).strip(), info.language

        text, detected_lang = await asyncio.to_thread(_transcribe)

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    text=text,
                    language=detected_lang or lang or "en",
                )
            ],
        )


def prewarm(proc: JobProcess) -> None:
    """Prewarm computationally intensive models during worker process initialization."""
    try:
        logger.info("Prewarming Silero VAD model...")
        proc.userdata["vad"] = silero.VAD.load()
        logger.info("Silero VAD model successfully prewarmed.")
    except Exception as e:
        logger.warning(f"Silero VAD prewarm skipped or failed: {e}")

    try:
        logger.info("Prewarming local Whisper STT model (base.en)...")
        proc.userdata["whisper"] = WhisperSTT(model="base.en")
        logger.info("Local Whisper STT model successfully prewarmed.")
    except Exception as e:
        logger.warning(f"Whisper STT prewarm skipped or failed: {e}")


server = AgentServer(
    setup_fnc=prewarm,
)


@server.rtc_session(agent_name="flowvoice")
async def entrypoint(ctx: JobContext) -> None:
    """Main job entrypoint called when a room dispatch is assigned to this worker."""
    logger.info(f"Agent assigned to room: {ctx.room.name}")

    missing_vars = settings.validate(require_livekit=False)
    if missing_vars:
        logger.warning(
            f"Missing recommended environment variables: {', '.join(missing_vars)}. "
            "Ensure they are set in your .env file."
        )

    logger.info("Connecting to LiveKit room (AUDIO_ONLY auto-subscription)...")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    logger.info("Waiting for participant to join the room...")
    participant = await ctx.wait_for_participant()
    logger.info(
        f"Participant connected: identity='{participant.identity}', name='{participant.name}'"
    )

    # Initialize Rime Text-to-Speech as the primary TTS provider
    logger.info(
        f"Initializing Rime TTS with model='{settings.rime_model}', speaker='{settings.rime_speaker}'"
    )
    tts_plugin = rime.TTS(
        model=settings.rime_model,
        speaker=settings.rime_speaker,
        api_key=settings.rime_api_key or None,
    )

    # Initialize local Ollama LLM
    logger.info("Initializing Ollama LLM (model='llama3.2:3b')...")
    llm_plugin = openai.LLM.with_ollama(
        model="llama3.2:3b",
        base_url="http://localhost:11434/v1",
    )

    # Use prewarmed VAD if available, or initialize a new instance
    vad_plugin = ctx.proc.userdata.get("vad")
    if vad_plugin is None:
        logger.info("Loading fresh Silero VAD instance...")
        vad_plugin = silero.VAD.load()

    # Use prewarmed Whisper STT if available, or initialize a new instance
    whisper_stt = ctx.proc.userdata.get("whisper")
    if whisper_stt is None:
        logger.info("Loading fresh Whisper STT instance (model='base.en')...")
        whisper_stt = WhisperSTT(model="base.en")

    # Adapt non-streaming Whisper STT with streaming VAD
    logger.info("Configuring StreamAdapter with local Whisper STT and Silero VAD...")
    stt_plugin = stt.StreamAdapter(stt=whisper_stt, vad=vad_plugin)

    # Initialize Safe Generation & Interruption Manager
    gen_manager = GenerationManager()

    # Create the Voice Agent Session with userdata pre-initialized
    logger.info("Creating AgentSession...")
    session = AgentSession(
        stt=stt_plugin,
        llm=llm_plugin,
        tts=tts_plugin,
        vad=vad_plugin,
        userdata={"gen_manager": gen_manager},
    )
    gen_manager.set_session(session)
    initial_gen = gen_manager.next_generation(reason="session_start")

    # Define slow demo task for testing interruption and stale result prevention
    @llm.function_tool(description="Execute a slow background computation or data retrieval.")
    async def slow_task(delay_seconds: int = 3) -> str:
        """Execute a deliberately slow task that can be safely interrupted and cancelled."""
        task_gen = gen_manager.current_generation_id
        logger.info(f"GEN {task_gen} SLOW TASK STARTED (delay={delay_seconds}s)")

        async def _work():
            await asyncio.sleep(delay_seconds)
            return f"Completed slow task after {delay_seconds}s (GEN {task_gen})"

        is_valid, result = await gen_manager.run_task("slow_task", _work(), gen_id=task_gen)
        if not is_valid:
            logger.info(f"GEN {task_gen} RESULT ARRIVED -> DISCARDED")
            return f"Task from GEN {task_gen} was cancelled due to user interruption."

        return result

    # Register event handlers for lifecycle, user interruption, and transcription
    @session.on("user_state_changed")
    def on_user_state_changed(ev):
        logger.debug(f"[User State]: {ev.old_state} -> {ev.new_state}")
        if ev.new_state == "speaking":
            # If agent is currently speaking or processing, handle interruption
            if session.agent_state == "speaking" or getattr(session, "current_speech", None) is not None:
                gen_manager.handle_user_interruption()

    @session.on("user_input_transcribed")
    def on_user_transcript(ev):
        if ev.is_final:
            logger.info(f"[User Final Transcript]: {ev.transcript}")
            # Ensure generation is updated for the new user input turn
            if not gen_manager.is_current(gen_manager.current_generation_id):
                gen_manager.next_generation(reason="user_transcript")
        else:
            logger.debug(f"[User Interim Transcript]: {ev.transcript}")

    @session.on("agent_state_changed")
    def on_agent_state_changed(ev):
        logger.info(f"[Agent State]: {ev.old_state} -> {ev.new_state}")
        if ev.new_state == "speaking":
            logger.info(f"RIME STREAMING (GEN {gen_manager.current_generation_id})")
        elif ev.old_state == "speaking" and ev.new_state != "speaking":
            logger.info(f"RIME STOP (GEN {gen_manager.current_generation_id})")

    @session.on("error")
    def on_error(ev):
        logger.error(f"[Agent Pipeline Error from {ev.source}]: {ev.error}")

    agent = Agent(
        instructions=(
            "You are FlowVoice, a friendly, concise, and helpful real-time AI voice assistant. "
            "Your responses are converted to speech using Rime TTS. "
            "Keep your answers natural, conversational, and direct. "
            "Do not output markdown formatting, bullet points, or emojis, as they will be spoken aloud. "
            "You have a demo tool named 'slow_task' for demonstrating slow background operations."
        ),
        tools=[slow_task],
    )

    logger.info("Starting AgentSession in room...")
    await session.start(agent=agent, room=ctx.room)

    greeting_text = (
        "Hello! I am FlowVoice, your realtime voice assistant. "
        "How can I assist you today?"
    )
    logger.info(f"Sending initial greeting: '{greeting_text}'")
    gen_manager.safe_say(greeting_text, gen_id=initial_gen)


if __name__ == "__main__":
    cli.run_app(server)

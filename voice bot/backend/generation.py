"""Safe Interruption and Generation Tracking for FlowVoice."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Set, Dict, List, Tuple
from livekit.agents.voice import AgentSession

from .logger import logger


@dataclass(frozen=True)
class GenerationTaskResult:
    """Represents the output of an asynchronous task tagged with its originating generation."""

    generation_id: int
    task_name: str
    result: Any
    is_valid: bool


class GenerationManager:
    """Manages monotonically increasing generation IDs, task cancellation,

    and stale result filtering to ensure safe user interruption.
    """

    def __init__(self, session: Optional[AgentSession] = None) -> None:
        self._current_generation_id: int = 0
        self._session: Optional[AgentSession] = session
        self._active_tasks: Dict[int, Set[asyncio.Task]] = {}
        self._spoken_history: List[Tuple[int, str]] = []

    @property
    def current_generation_id(self) -> int:
        """Return the currently active generation ID."""
        return self._current_generation_id

    @property
    def spoken_history(self) -> List[Tuple[int, str]]:
        """Return history of successfully spoken utterances tagged with generation ID."""
        return list(self._spoken_history)

    def is_current(self, gen_id: int) -> bool:
        """Check if the provided generation ID matches the currently active generation."""
        return gen_id == self._current_generation_id

    def set_session(self, session: AgentSession) -> None:
        """Attach or update the LiveKit AgentSession reference."""
        self._session = session

    def next_generation(self, reason: str = "new_turn") -> int:
        """Advance to a new generation ID, invalidating all preceding generations and cancelling their tasks."""
        old_gen = self._current_generation_id
        self._current_generation_id += 1
        new_gen = self._current_generation_id

        if old_gen > 0:
            logger.info(f"GEN {old_gen} INVALIDATED ({reason})")
            self._cancel_generation_tasks(old_gen)

        logger.info(f"GEN {new_gen} START")
        return new_gen

    def handle_user_interruption(self) -> int:
        """Handle user interruption: stop active speech, invalidate the current generation,

        cancel in-flight work, and start a fresh generation immediately.
        """
        logger.info("USER INTERRUPTED")
        if self._session is not None:
            try:
                # Cancel current agent speech via LiveKit AgentSession interruption API
                if getattr(self._session, "agent_state", None) == "speaking" or getattr(
                    self._session, "current_speech", None
                ) is not None:
                    logger.info("RIME STOP")
                    self._session.interrupt(force=True)
            except Exception as e:
                logger.debug(f"Session interruption call error: {e}")

        return self.next_generation(reason="interruption")

    def _cancel_generation_tasks(self, gen_id: int) -> None:
        """Cancel all pending asyncio tasks registered under the specified generation ID."""
        tasks = self._active_tasks.pop(gen_id, set())
        for task in tasks:
            if not task.done():
                task.cancel()

    def register_task(self, gen_id: int, task: asyncio.Task) -> None:
        """Register an active background task to be tracked and cancelled if the generation is interrupted."""
        if gen_id not in self._active_tasks:
            self._active_tasks[gen_id] = set()
        self._active_tasks[gen_id].add(task)
        task.add_done_callback(
            lambda t: self._active_tasks.get(gen_id, set()).discard(t)
            if gen_id in self._active_tasks
            else None
        )

    async def run_task(
        self,
        task_name: str,
        coro: Awaitable[Any],
        gen_id: Optional[int] = None,
    ) -> Tuple[bool, Any]:
        """Execute an asynchronous coroutine bound to a specific generation ID.

        Returns (is_valid, result). If the generation was invalidated or interrupted,
        the task is cancelled or discarded, returning (False, None).
        """
        target_gen = gen_id if gen_id is not None else self._current_generation_id

        if not self.is_current(target_gen):
            logger.info(f"GEN {target_gen} TASK '{task_name}' NOT STARTED -> STALE")
            return False, None

        # Wrap in an isolated child task for clean cancellation
        child_task = asyncio.create_task(coro, name=f"gen_{target_gen}_{task_name}")
        self.register_task(target_gen, child_task)

        try:
            res = await child_task
            if self.is_current(target_gen):
                logger.info(f"GEN {target_gen} RESULT -> ACCEPTED")
                return True, res
            else:
                logger.info(f"GEN {target_gen} RESULT ARRIVED -> DISCARDED")
                return False, None
        except asyncio.CancelledError:
            logger.info(f"GEN {target_gen} TASK '{task_name}' CANCELLED")
            return False, None
        except Exception as e:
            if not self.is_current(target_gen):
                logger.info(
                    f"GEN {target_gen} ERROR ARRIVED AFTER INVALIDATION -> DISCARDED ({e})"
                )
                return False, None
            raise

    def safe_say(self, text: str, gen_id: Optional[int] = None) -> bool:
        """Send speech to Rime TTS via AgentSession ONLY if the target generation is currently active."""
        target_gen = gen_id if gen_id is not None else self._current_generation_id

        if not self.is_current(target_gen):
            logger.warning(
                f"GEN {target_gen} SPEECH SUPPRESSED (STALE) -> '{text}'"
            )
            return False

        logger.info(f"RIME STREAMING (GEN {target_gen}): '{text}'")
        self._spoken_history.append((target_gen, text))
        if self._session is not None:
            try:
                self._session.say(text)
            except RuntimeError as e:
                logger.warning(f"Speech suppressed (session not active): {e}")
                return False
        return True

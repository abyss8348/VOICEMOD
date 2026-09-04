"""Unit and integration tests for Safe Interruption and Stale Result Prevention."""

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from backend.generation import GenerationManager, GenerationTaskResult


class TestSafeInterruptionAndStalePrevention(unittest.IsolatedAsyncioTestCase):
    """Test suite verifying generation tracking, task cancellation, and stale result filtering."""

    async def test_normal_completion(self):
        """Verify normal task completion accepts results and permits speech in current generation."""
        mock_session = MagicMock()
        mock_session.say = MagicMock()
        mock_session.agent_state = "idle"
        mock_session.current_speech = None

        gm = GenerationManager(session=mock_session)
        gen1 = gm.next_generation(reason="turn_1")
        self.assertEqual(gm.current_generation_id, 1)

        async def sample_work():
            await asyncio.sleep(0.01)
            return "Sample task completed successfully"

        is_valid, result = await gm.run_task("sample_work", sample_work(), gen_id=gen1)
        self.assertTrue(is_valid)
        self.assertEqual(result, "Sample task completed successfully")

        # Result is valid and spoken
        spoken = gm.safe_say(result, gen_id=gen1)
        self.assertTrue(spoken)
        mock_session.say.assert_called_once_with("Sample task completed successfully")
        self.assertIn((gen1, "Sample task completed successfully"), gm.spoken_history)

    async def test_interruption_during_slow_task(self):
        """Verify that user interruption cancels active slow tasks and invalidates generation."""
        mock_session = MagicMock()
        mock_session.interrupt = MagicMock()
        mock_session.agent_state = "speaking"
        mock_session.current_speech = MagicMock()

        gm = GenerationManager(session=mock_session)
        gen1 = gm.next_generation(reason="turn_1")

        task_started = asyncio.Event()
        task_cancelled = asyncio.Event()

        async def slow_work():
            try:
                task_started.set()
                await asyncio.sleep(0.5)
                return "Slow task finished"
            except asyncio.CancelledError:
                task_cancelled.set()
                raise

        # Launch slow task in background
        bg_task = asyncio.create_task(
            gm.run_task("slow_work", slow_work(), gen_id=gen1)
        )
        await task_started.wait()

        # User interrupts while task is in flight
        gen2 = gm.handle_user_interruption()
        self.assertEqual(gen2, 2)
        mock_session.interrupt.assert_called_once_with(force=True)

        is_valid, result = await bg_task
        self.assertFalse(is_valid)
        self.assertIsNone(result)
        self.assertTrue(task_cancelled.is_set())

    async def test_stale_result_after_interruption(self):
        """Verify that any result arriving from an older generation is completely discarded."""
        mock_session = MagicMock()
        gm = GenerationManager(session=mock_session)
        gen1 = gm.next_generation(reason="turn_1")

        async def delayed_work():
            await asyncio.sleep(0.05)
            return "Delayed result from Gen 1"

        bg_task = asyncio.create_task(
            gm.run_task("delayed_work", delayed_work(), gen_id=gen1)
        )

        # Interrupt before task finishes
        await asyncio.sleep(0.01)
        gen2 = gm.handle_user_interruption()
        self.assertEqual(gen2, 2)

        is_valid, result = await bg_task
        self.assertFalse(is_valid)
        self.assertIsNone(result)
        self.assertFalse(gm.is_current(gen1))

    async def test_two_consecutive_interruptions(self):
        """Verify handling of rapid consecutive interruptions across multiple generations."""
        mock_session = MagicMock()
        gm = GenerationManager(session=mock_session)

        # Gen 1
        gen1 = gm.next_generation(reason="turn_1")
        t1 = asyncio.create_task(
            gm.run_task("t1", asyncio.sleep(0.1, result="res1"), gen_id=gen1)
        )

        # First interruption -> Gen 2
        await asyncio.sleep(0.01)
        gen2 = gm.handle_user_interruption()
        self.assertEqual(gen2, 2)
        t2 = asyncio.create_task(
            gm.run_task("t2", asyncio.sleep(0.1, result="res2"), gen_id=gen2)
        )

        # Second interruption -> Gen 3
        await asyncio.sleep(0.01)
        gen3 = gm.handle_user_interruption()
        self.assertEqual(gen3, 3)
        t3 = asyncio.create_task(
            gm.run_task("t3", asyncio.sleep(0.01, result="res3"), gen_id=gen3)
        )

        v1, r1 = await t1
        v2, r2 = await t2
        v3, r3 = await t3

        # Gen 1 and Gen 2 must be discarded
        self.assertFalse(v1)
        self.assertIsNone(r1)
        self.assertFalse(v2)
        self.assertIsNone(r2)

        # Gen 3 must succeed
        self.assertTrue(v3)
        self.assertEqual(r3, "res3")

    async def test_cancellation_race(self):
        """Verify concurrent task execution and cancellation under high concurrency without errors."""
        mock_session = MagicMock()
        gm = GenerationManager(session=mock_session)
        gen1 = gm.next_generation(reason="concurrent_turn")

        async def worker(idx: int):
            await asyncio.sleep(0.1 + 0.01 * idx)
            return f"worker_{idx}"

        tasks = [
            asyncio.create_task(gm.run_task(f"w_{i}", worker(i), gen_id=gen1))
            for i in range(10)
        ]

        # Trigger interruption while all tasks are still in-flight
        await asyncio.sleep(0.02)
        gen2 = gm.handle_user_interruption()
        self.assertEqual(gen2, 2)

        results = await asyncio.gather(*tasks)
        for is_valid, res in results:
            # All tasks from gen1 must be cancelled/discarded
            self.assertFalse(is_valid)
            self.assertIsNone(res)

    async def test_stale_result_never_spoken(self):
        """Verify that stale results from previous generations cannot be spoken via safe_say."""
        mock_session = MagicMock()
        mock_session.say = MagicMock()
        gm = GenerationManager(session=mock_session)

        gen1 = gm.next_generation(reason="turn_1")
        stale_text = "Stale answer from interrupted generation"

        # Advance generation to simulate interruption
        gen2 = gm.handle_user_interruption()

        # Attempt to speak stale text tagged with Gen 1
        spoken_stale = gm.safe_say(stale_text, gen_id=gen1)
        self.assertFalse(spoken_stale)
        mock_session.say.assert_not_called()

        # Current generation text can be spoken
        fresh_text = "Fresh answer for new user turn"
        spoken_fresh = gm.safe_say(fresh_text, gen_id=gen2)
        self.assertTrue(spoken_fresh)
        mock_session.say.assert_called_once_with(fresh_text)


if __name__ == "__main__":
    unittest.main()

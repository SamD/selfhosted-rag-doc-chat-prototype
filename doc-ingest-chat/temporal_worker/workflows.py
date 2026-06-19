#!/usr/bin/env python3
"""
Temporal Workflow for WhisperX transcription.
Coordinates the transcribe_media Activity with proper retry policy and timeouts.
"""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from models.transcription_input import TranscriptionInput, TranscriptionResult
    from temporal_worker.activities import transcribe_media

@workflow.defn
class TranscribeWorkflow:
    @workflow.run
    async def run(self, input: TranscriptionInput) -> TranscriptionResult:
        return await workflow.execute_activity(
            transcribe_media,
            input,
            schedule_to_close_timeout=timedelta(minutes=90),
            start_to_close_timeout=timedelta(minutes=90),
            retry_policy=RetryPolicy(
                maximum_attempts=3,
                initial_interval=timedelta(seconds=5),
                backoff_coefficient=2.0,
                maximum_interval=timedelta(seconds=60),
            ),
        )
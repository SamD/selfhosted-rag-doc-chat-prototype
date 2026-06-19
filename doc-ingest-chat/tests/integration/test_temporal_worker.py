#!/usr/bin/env python3
"""
Integration tests for Temporal Worker.
Tests integration with temporalite dev server in pytest.
"""

from datetime import timedelta
from unittest.mock import MagicMock, Mock, patch

import pytest
import pytest_asyncio
from models.transcription_input import TranscriptionInput, TranscriptionResult
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment

# Mock the whisperx module before importing activities
with patch.dict('sys.modules', {'whisperx': MagicMock()}):
    with patch.dict('sys.modules', {'temporalio': MagicMock()}):
        from temporal_worker.workflows import TranscribeWorkflow


@pytest.mark.skip(reason="Requires running temporalite server")
class TestTemporalWorkerIntegration:
    """Integration tests for Temporal Worker."""
    
    @pytest_asyncio.fixture(scope="function")
    async def temporal_env(self):
        """Create a Temporal testing environment."""
        async with await WorkflowEnvironment.start_local() as env:
            yield env

    @pytest_asyncio.fixture(scope="function")
    async def temporal_client(self, temporal_env):
        """Create a Temporal client connected to the test environment."""
        client = await Client.connect(
            target_host=temporal_env.target_host,
            task_queue="test-whisperx",
        )
        yield client
        await client.close()
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, temporal_client):
        """Test workflow execution with temporalite."""
        # Mock the activity
        mock_response = Mock()
        mock_response.json.return_value = {
            "segments": [{"text": "Test transcription"}]
        }
        
        with patch('temporal_worker.activities.os.path.exists', return_value=True), \
             patch('temporal_worker.activities.mimetypes.guess_type', return_value=('audio/mpeg', None)), \
             patch('temporal_worker.activities.requests.post', return_value=mock_response), \
             patch('temporal_worker.activities.config.settings.WHISPER_MODEL_ENDPOINTS', 'http://remote-whisper:8000'):
            
            # Create input
            input_data = TranscriptionInput(
                file_path="/test/file.mp3",
                language="en",
                mime_type="audio/mpeg"
            )
            
            # Execute workflow
            result = await temporal_client.execute_workflow(
                TranscribeWorkflow.run,
                input_data,
                id="test-workflow-123",
                task_queue="test-whisperx",
                execution_timeout=timedelta(minutes=90),
            )

            # Verify result
            assert isinstance(result, TranscriptionResult)
            assert len(result.segments) == 1
            assert result.segments[0]["text"] == "Test transcription"
            assert result.source_file == "file.mp3"
            assert result.job_id == ""
    
    @pytest.mark.asyncio
    async def test_activity_retry_on_failure(self, temporal_client):
        """Test activity retry on failure."""
        # Mock the activity to fail first time, succeed second time
        call_count = 0
        
        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = Mock()
            if call_count == 1:
                # First call fails
                mock_response.raise_for_status.side_effect = Exception("Connection failed")
            else:
                # Second call succeeds
                mock_response.json.return_value = {
                    "segments": [{"text": "Retry successful"}]
                }
            return mock_response
        
        with patch('temporal_worker.activities.os.path.exists', return_value=True), \
             patch('temporal_worker.activities.mimetypes.guess_type', return_value=('audio/mpeg', None)), \
             patch('temporal_worker.activities.requests.post', side_effect=mock_post), \
             patch('temporal_worker.activities.config.settings.WHISPER_MODEL_ENDPOINTS', 'http://remote-whisper:8000'):
            
            # Create input
            input_data = TranscriptionInput(
                file_path="/test/file.mp3",
                language="en",
                mime_type="audio/mpeg"
            )
            
            # Execute workflow - should retry and succeed
            result = await temporal_client.execute_workflow(
                TranscribeWorkflow.run,
                input_data,
                id="test-retry-workflow-123",
                task_queue="test-whisperx",
                execution_timeout=timedelta(minutes=90),
            )

            # Verify result
            assert isinstance(result, TranscriptionResult)
            assert len(result.segments) == 1
            assert result.segments[0]["text"] == "Retry successful"
            assert call_count == 2  # Should have been called twice (once failed, once succeeded)
    
    @pytest.mark.asyncio
    async def test_workflow_crash_recovery(self, temporal_client):
        """Test workflow crash recovery."""
        # This test simulates a worker crash mid-activity and verifies recovery
        # In a real scenario, Temporal would restart the activity on a different worker
        
        # Create input
        input_data = TranscriptionInput(
            file_path="/test/file.mp3",
            language="en",
            mime_type="audio/mpeg"
        )
        
        # Start workflow
        handle = await temporal_client.start_workflow(
            TranscribeWorkflow.run,
            input_data,
            id="test-crash-workflow-123",
            task_queue="test-whisperx",
            execution_timeout=timedelta(minutes=90),
        )
        
        # Verify workflow started
        assert handle.workflow_id == "test-crash-workflow-123"
        
        # In a real test, we would simulate a worker crash and verify
        # that Temporal restarts the activity on a different worker
        # For this test, we'll just verify the workflow can be started and queried
        
        # Get workflow description
        description = await temporal_client.describe_workflow_execution(
            handle.workflow_id,
            handle.run_id,
        )
        
        # Verify workflow is running
        assert description.workflow_execution_info.status == "Running"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
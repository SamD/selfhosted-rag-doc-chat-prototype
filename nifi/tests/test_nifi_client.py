"""
Unit tests for NifiClient.
"""
import sys
import types
import unittest
from unittest.mock import MagicMock

for k in list(sys.modules.keys()):
    if k.startswith("nipyapi"):
        del sys.modules[k]

nifiapi_mock = types.ModuleType("nipyapi")
nifiapi_mock.__path__ = ["/tmp/nipyapi"]
nifiapi_mock.__file__ = "/tmp/nipyapi/__init__.py"

config_mod = types.ModuleType("nipyapi.config")
config_mod.__file__ = "/tmp/nipyapi/config.py"
utils_mod = types.ModuleType("nipyapi.utils")
utils_mod.__file__ = "/tmp/nipyapi/utils.py"
canvas_mod = types.ModuleType("nipyapi.canvas")
canvas_mod.__file__ = "/tmp/nipyapi/canvas.py"
nifi_mod = types.ModuleType("nipyapi.nifi")
nifi_mod.__file__ = "/tmp/nipyapi/nifi.py"
versioning_mod = types.ModuleType("nipyapi.versioning")
versioning_mod.__file__ = "/tmp/nipyapi/versioning.py"

nifiapi_mock.config = config_mod
nifiapi_mock.utils = utils_mod
nifiapi_mock.canvas = canvas_mod
nifiapi_mock.nifi = nifi_mod
nifiapi_mock.versioning = versioning_mod

sys.modules["nipyapi"] = nifiapi_mock
sys.modules["nipyapi.config"] = config_mod
sys.modules["nipyapi.utils"] = utils_mod
sys.modules["nipyapi.canvas"] = canvas_mod
sys.modules["nipyapi.nifi"] = nifi_mod
sys.modules["nipyapi.versioning"] = versioning_mod

from nifi.nifi_client import NifiClient  # noqa: E402

requests_mock = MagicMock()
sys.modules["nifi.nifi_client"].requests = requests_mock


def _configure_mocks():
    """Set up nipyapi mocks for a test."""
    config_mod.nifi_config = MagicMock()
    config_mod.registry_config = MagicMock()
    utils_mod.set_endpoint = MagicMock()
    canvas_mod.get_root_pg_id = MagicMock(return_value="root-pg-id")
    canvas_mod.list_all_process_groups = MagicMock(return_value=[])
    canvas_mod.create_process_group = MagicMock()
    canvas_mod.schedule_process_group = MagicMock()
    canvas_mod.get_processor_type = MagicMock()
    canvas_mod.create_processor = MagicMock()
    canvas_mod.get_processor = MagicMock()
    canvas_mod.create_connection = MagicMock()
    canvas_mod.list_all_processors = MagicMock(return_value=[])
    canvas_mod.delete_process_group = MagicMock()


class TestNifiClient(unittest.TestCase):
    """Test cases for NifiClient."""

    def setUp(self):
        """Set up test fixtures."""
        _configure_mocks()

        token_response = MagicMock()
        token_response.text = "fake-token"
        token_response.raise_for_status = MagicMock()

        self.mock_post_response = MagicMock()
        self.mock_post_response.status_code = 201
        self.mock_post_response.raise_for_status = MagicMock()
        self.mock_post_response.json.return_value = {"component": {"id": "default-id"}}
        self.mock_post_response.text = "ok"

        self.mock_get_response = MagicMock()
        self.mock_get_response.status_code = 200
        self.mock_get_response.raise_for_status = MagicMock()

        requests_mock.post.side_effect = None
        requests_mock.post.return_value = self.mock_post_response
        requests_mock.get.return_value = self.mock_get_response

        self._token_response = token_response

    def test_init_configures_nifi(self):
        """Test that __init__ configures NiFi connection."""
        client = NifiClient(
            base_url="https://nifi.example.com:8443/nifi-api",
            username="admin",
            password="secret",
            ssl_verify=False,
        )

        self.assertEqual(client.base_url, "https://nifi.example.com:8443/nifi-api")
        self.assertEqual(client.username, "admin")
        self.assertEqual(client.password, "secret")
        self.assertFalse(client.ssl_verify)

        self.assertEqual(
            config_mod.nifi_config.host,
            "https://nifi.example.com:8443/nifi-api"
        )
        self.assertFalse(config_mod.nifi_config.verify_ssl)
        self.assertEqual(config_mod.nifi_config.username, "admin")
        self.assertEqual(config_mod.nifi_config.password, "secret")

        utils_mod.set_endpoint.assert_called_once_with(
            "https://nifi.example.com:8443/nifi-api",
            ssl=True,
            login=True,
        )

    def test_init_configures_registry(self):
        """Test that __init__ configures NiFi Registry if URL provided."""
        client = NifiClient(
            base_url="https://nifi.example.com:8443/nifi-api",
            username="admin",
            password="secret",
            ssl_verify=False,
            registry_url="http://registry.example.com:18080/nifi-registry-api",
        )

        self.assertEqual(
            client.registry_url,
            "http://registry.example.com:18080/nifi-registry-api"
        )

        self.assertEqual(
            config_mod.registry_config.host,
            "http://registry.example.com:18080/nifi-registry-api"
        )
        self.assertFalse(config_mod.registry_config.verify_ssl)

    def test_wait_for_nifi_success(self):
        """Test wait_for_nifi returns True when NiFi is available."""
        client = NifiClient(
            base_url="https://nifi.example.com:8443/nifi-api",
            username="admin",
            password="secret",
        )

        canvas_mod.get_root_pg_id.return_value = "root-pg-id"

        result = client.wait_for_nifi(max_retries=3)

        self.assertTrue(result)
        canvas_mod.get_root_pg_id.assert_called()

    def test_wait_for_nifi_retry_success(self):
        """Test wait_for_nifi retries and eventually succeeds."""
        client = NifiClient(
            base_url="https://nifi.example.com:8443/nifi-api",
            username="admin",
            password="secret",
        )

        canvas_mod.get_root_pg_id.side_effect = [
            Exception("Connection refused"),
            Exception("Connection refused"),
            "root-pg-id",
        ]

        result = client.wait_for_nifi(max_retries=3)

        self.assertTrue(result)
        self.assertEqual(canvas_mod.get_root_pg_id.call_count, 3)

    def test_wait_for_nifi_all_retries_fail(self):
        """Test wait_for_nifi returns False after all retries fail."""
        client = NifiClient(
            base_url="https://nifi.example.com:8443/nifi-api",
            username="admin",
            password="secret",
        )

        canvas_mod.get_root_pg_id.side_effect = Exception("Connection refused")

        result = client.wait_for_nifi(max_retries=3)

        self.assertFalse(result)
        self.assertEqual(canvas_mod.get_root_pg_id.call_count, 3)

    def test_create_process_group(self):
        """Test create_process_group creates a new process group."""
        client = NifiClient(
            base_url="https://nifi.example.com:8443/nifi-api",
            username="admin",
            password="secret",
        )

        mock_pg = MagicMock()
        mock_pg.id = "new-pg-id"
        canvas_mod.create_process_group.return_value = mock_pg

        pg_id = client.create_process_group("Test Process Group")

        self.assertEqual(pg_id, "new-pg-id")
        canvas_mod.create_process_group.assert_called_once()

    def test_check_flow_exists_found(self):
        """Test check_flow_exists returns ID when process group exists."""
        client = NifiClient(
            base_url="https://nifi.example.com:8443/nifi-api",
            username="admin",
            password="secret",
        )

        mock_pg = MagicMock()
        mock_pg.id = "existing-pg-id"
        mock_pg.component.name = "Test Process Group"
        canvas_mod.list_all_process_groups.return_value = [mock_pg]

        pg_id = client.check_flow_exists("Test Process Group")

        self.assertEqual(pg_id, "existing-pg-id")

    def test_check_flow_exists_not_found(self):
        """Test check_flow_exists returns None when process group doesn't exist."""
        client = NifiClient(
            base_url="https://nifi.example.com:8443/nifi-api",
            username="admin",
            password="secret",
        )

        canvas_mod.list_all_process_groups.return_value = []

        pg_id = client.check_flow_exists("Nonexistent Process Group")

        self.assertIsNone(pg_id)

    def test_create_consumer_processor(self):
        """Test create_consumer_processor creates a RedisQueueConsumer."""
        self.mock_post_response.json.return_value = {"component": {"id": "consumer-processor-id"}}

        client = NifiClient(
            base_url="https://nifi.example.com:8443/nifi-api",
            username="admin",
            password="secret",
        )

        processor_id = client.create_consumer_processor(
            process_group_id="pg-id",
            queue_name="test_queue_input",
            redis_host="localhost",
            redis_port=6379,
            redis_db=0,
        )

        self.assertEqual(processor_id, "consumer-processor-id")

    def test_create_producer_processor(self):
        """Test create_producer_processor creates a RedisQueueProducer."""
        self.mock_post_response.json.return_value = {"component": {"id": "producer-processor-id"}}

        client = NifiClient(
            base_url="https://nifi.example.com:8443/nifi-api",
            username="admin",
            password="secret",
        )

        processor_id = client.create_producer_processor(
            process_group_id="pg-id",
            queue_name="test_queue_output",
            redis_host="localhost",
            redis_port=6379,
            redis_db=0,
        )

        self.assertEqual(processor_id, "producer-processor-id")

    def test_create_connection(self):
        """Test create_connection creates a connection between processors."""
        self.mock_post_response.json.return_value = {"id": "connection-id"}

        client = NifiClient(
            base_url="https://nifi.example.com:8443/nifi-api",
            username="admin",
            password="secret",
        )

        connection_id = client.create_connection(
            process_group_id="pg-id",
            source_id="source-processor-id",
            destination_id="dest-processor-id",
        )

        self.assertEqual(connection_id, "connection-id")

    def test_start_process_group_success(self):
        """Test start_process_group starts all processors."""
        client = NifiClient(
            base_url="https://nifi.example.com:8443/nifi-api",
            username="admin",
            password="secret",
        )

        result = client.start_process_group("pg-id")

        self.assertTrue(result)
        canvas_mod.schedule_process_group.assert_called_with("pg-id", scheduled=True)

    def test_start_process_group_failure(self):
        """Test start_process_group returns False on failure."""
        client = NifiClient(
            base_url="https://nifi.example.com:8443/nifi-api",
            username="admin",
            password="secret",
        )

        canvas_mod.schedule_process_group.side_effect = Exception("Failed to start")

        result = client.start_process_group("pg-id")

        self.assertFalse(result)

    def test_verify_flow_health_all_running(self):
        """Test verify_flow_health returns True when all processors are running."""
        client = NifiClient(
            base_url="https://nifi.example.com:8443/nifi-api",
            username="admin",
            password="secret",
        )

        mock_processor1 = MagicMock()
        mock_processor1.status.run_status = "Running"
        mock_processor1.component.name = "Processor 1"

        mock_processor2 = MagicMock()
        mock_processor2.status.run_status = "Running"
        mock_processor2.component.name = "Processor 2"

        canvas_mod.list_all_processors.return_value = [mock_processor1, mock_processor2]

        result = client.verify_flow_health("pg-id")

        self.assertTrue(result)

    def test_verify_flow_health_some_stopped(self):
        """Test verify_flow_health returns False when some processors are stopped."""
        client = NifiClient(
            base_url="https://nifi.example.com:8443/nifi-api",
            username="admin",
            password="secret",
        )

        mock_processor1 = MagicMock()
        mock_processor1.status.run_status = "Running"
        mock_processor1.component.name = "Processor 1"

        mock_processor2 = MagicMock()
        mock_processor2.status.run_status = "Stopped"
        mock_processor2.component.name = "Processor 2"

        canvas_mod.list_all_processors.return_value = [mock_processor1, mock_processor2]

        result = client.verify_flow_health("pg-id")

        self.assertFalse(result)

    def test_delete_process_group_success(self):
        """Test delete_process_group deletes a process group."""
        client = NifiClient(
            base_url="https://nifi.example.com:8443/nifi-api",
            username="admin",
            password="secret",
        )

        result = client.delete_process_group("pg-id")

        self.assertTrue(result)
        canvas_mod.delete_process_group.assert_called_once_with("pg-id")

    def test_delete_process_group_failure(self):
        """Test delete_process_group returns False on failure."""
        client = NifiClient(
            base_url="https://nifi.example.com:8443/nifi-api",
            username="admin",
            password="secret",
        )

        canvas_mod.delete_process_group.side_effect = Exception("Failed to delete")

        result = client.delete_process_group("pg-id")

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()

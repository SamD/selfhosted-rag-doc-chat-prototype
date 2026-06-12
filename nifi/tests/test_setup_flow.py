"""
Integration tests for setup_flow.py with mocked NifiClient.
"""
import types
import unittest
from unittest.mock import MagicMock, patch

nifiapi_mock = types.ModuleType("nipyapi")
nifiapi_mock.config = MagicMock()
nifiapi_mock.utils = MagicMock()
nifiapi_mock.canvas = MagicMock()
nifiapi_mock.nifi = MagicMock()
nifiapi_mock.versioning = MagicMock()

import sys  # noqa: E402 (mock setup must precede import)

sys.modules["nipyapi"] = nifiapi_mock
sys.modules["nipyapi.config"] = nifiapi_mock.config
sys.modules["nipyapi.utils"] = nifiapi_mock.utils
sys.modules["nipyapi.canvas"] = nifiapi_mock.canvas
sys.modules["nipyapi.nifi"] = nifiapi_mock.nifi
sys.modules["nipyapi.versioning"] = nifiapi_mock.versioning
sys.modules["requests"] = MagicMock()

nifi_client_module = types.ModuleType("nifi_client")
nifi_client_module.NifiClient = MagicMock()
sys.modules["nifi_client"] = nifi_client_module

import nifi.setup_flow as setup_flow  # noqa: E402


class TestGetQueueNames(unittest.TestCase):
    """Test get_queue_names() function."""

    def test_default_queue_names(self):
        with patch.dict("os.environ", {}, clear=True):
            queues = setup_flow.get_queue_names()
            self.assertIn("ocr_processing_job", queues)
            self.assertIn("whisper_processing_job", queues)
            self.assertIn("chunk_ingest_queue:0", queues)
            self.assertIn("chunk_ingest_queue:1", queues)

    def test_custom_queue_names(self):
        with patch.dict("os.environ", {
            "REDIS_OCR_JOB_QUEUE": "custom_ocr",
            "REDIS_WHISPER_JOB_QUEUE": "custom_whisper",
            "QUEUE_NAMES": "queue_a,queue_b,queue_c",
        }, clear=True):
            queues = setup_flow.get_queue_names()
            self.assertEqual(queues[0], "custom_ocr")
            self.assertEqual(queues[1], "custom_whisper")
            self.assertEqual(queues[2], "queue_a")
            self.assertEqual(queues[3], "queue_b")
            self.assertEqual(queues[4], "queue_c")

    def test_empty_chunk_queues(self):
        with patch.dict("os.environ", {
            "REDIS_OCR_JOB_QUEUE": "custom_ocr",
            "REDIS_WHISPER_JOB_QUEUE": "custom_whisper",
            "QUEUE_NAMES": "",
        }, clear=True):
            queues = setup_flow.get_queue_names()
            self.assertEqual(queues, ["custom_ocr", "custom_whisper"])


class TestGetRedisConfig(unittest.TestCase):
    """Test get_redis_config() function."""

    def test_default_redis_config(self):
        with patch.dict("os.environ", {}, clear=True):
            config = setup_flow.get_redis_config()
            self.assertEqual(config["host"], "localhost")
            self.assertEqual(config["port"], 6379)
            self.assertEqual(config["db"], 0)

    def test_custom_redis_config(self):
        with patch.dict("os.environ", {
            "REDIS_HOST": "redis.example.com",
            "REDIS_PORT": "6380",
            "REDIS_DB": "3",
        }, clear=True):
            config = setup_flow.get_redis_config()
            self.assertEqual(config["host"], "redis.example.com")
            self.assertEqual(config["port"], 6380)
            self.assertEqual(config["db"], 3)


class TestSetupFlow(unittest.TestCase):
    """Test setup_flow() with mocked NifiClient."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_client.wait_for_nifi.return_value = True
        self.mock_client.check_flow_exists.return_value = None
        self.mock_client.create_process_group.return_value = "test-pg-id"
        self.mock_client.create_consumer_processor.return_value = "consumer-id"
        self.mock_client.create_producer_processor.return_value = "producer-id"
        self.mock_client.create_connection.return_value = "conn-id"
        self.mock_client.start_process_group.return_value = True
        self.mock_client.verify_flow_health.return_value = True

    def _run_setup_flow(self):
        with patch.dict("os.environ", {
            "NIFI_ENDPOINT": "https://nifi.example.com:8443/nifi-api",
            "NIFI_USERNAME": "admin",
            "NIFI_PASSWORD": "secret",
            "NIFI_SSL_VERIFY": "false",
        }, clear=True), patch("nifi.setup_flow.NifiClient", return_value=self.mock_client):
            return setup_flow.setup_flow()

    def test_missing_nifi_endpoint(self):
        with patch.dict("os.environ", {
            "NIFI_USERNAME": "admin",
            "NIFI_PASSWORD": "secret",
        }, clear=True), patch("nifi.setup_flow.NifiClient", return_value=self.mock_client):
            with self.assertRaises(SystemExit) as cm:
                setup_flow.setup_flow()
            self.assertEqual(cm.exception.code, 1)

    def test_missing_nifi_username(self):
        with patch.dict("os.environ", {
            "NIFI_ENDPOINT": "https://nifi.example.com:8443/nifi-api",
            "NIFI_PASSWORD": "secret",
        }, clear=True), patch("nifi.setup_flow.NifiClient", return_value=self.mock_client):
            with self.assertRaises(SystemExit) as cm:
                setup_flow.setup_flow()
            self.assertEqual(cm.exception.code, 1)

    def test_missing_nifi_password(self):
        with patch.dict("os.environ", {
            "NIFI_ENDPOINT": "https://nifi.example.com:8443/nifi-api",
            "NIFI_USERNAME": "admin",
        }, clear=True), patch("nifi.setup_flow.NifiClient", return_value=self.mock_client):
            with self.assertRaises(SystemExit) as cm:
                setup_flow.setup_flow()
            self.assertEqual(cm.exception.code, 1)

    def test_nifi_not_available(self):
        self.mock_client.wait_for_nifi.return_value = False

        with patch.dict("os.environ", {
            "NIFI_ENDPOINT": "https://nifi.example.com:8443/nifi-api",
            "NIFI_USERNAME": "admin",
            "NIFI_PASSWORD": "secret",
        }, clear=True), patch("nifi.setup_flow.NifiClient", return_value=self.mock_client):
            with self.assertRaises(SystemExit) as cm:
                setup_flow.setup_flow()
            self.assertEqual(cm.exception.code, 1)
            self.mock_client.wait_for_nifi.assert_called_once()

    def test_flow_already_exists(self):
        self.mock_client.check_flow_exists.return_value = "existing-pg-id"

        with patch.dict("os.environ", {
            "NIFI_ENDPOINT": "https://nifi.example.com:8443/nifi-api",
            "NIFI_USERNAME": "admin",
            "NIFI_PASSWORD": "secret",
        }, clear=True), patch("nifi.setup_flow.NifiClient", return_value=self.mock_client):
            with self.assertRaises(SystemExit) as cm:
                setup_flow.setup_flow()
            self.assertEqual(cm.exception.code, 0)
            self.mock_client.check_flow_exists.assert_called_once_with("RAG Pipeline")
            self.mock_client.create_process_group.assert_not_called()

    def test_successful_flow_setup(self):
        with patch.dict("os.environ", {
            "NIFI_ENDPOINT": "https://nifi.example.com:8443/nifi-api",
            "NIFI_USERNAME": "admin",
            "NIFI_PASSWORD": "secret",
            "NIFI_SSL_VERIFY": "false",
        }, clear=True), patch("nifi.setup_flow.NifiClient", return_value=self.mock_client):
            setup_flow.setup_flow()

            self.mock_client.check_flow_exists.assert_called_once_with("RAG Pipeline")
            self.mock_client.create_process_group.assert_called_once_with("RAG Pipeline")
            self.mock_client.start_process_group.assert_called_once_with("test-pg-id")
            self.mock_client.verify_flow_health.assert_called_once_with("test-pg-id")

    def test_processor_creation_for_each_queue(self):
        with patch.dict("os.environ", {
            "NIFI_ENDPOINT": "https://nifi.example.com:8443/nifi-api",
            "NIFI_USERNAME": "admin",
            "NIFI_PASSWORD": "secret",
            "REDIS_OCR_JOB_QUEUE": "my_ocr",
            "REDIS_WHISPER_JOB_QUEUE": "my_whisper",
            "QUEUE_NAMES": "",
        }, clear=True), patch("nifi.setup_flow.NifiClient", return_value=self.mock_client):
            setup_flow.setup_flow()

            consumer_calls = self.mock_client.create_consumer_processor.call_args_list
            producer_calls = self.mock_client.create_producer_processor.call_args_list
            connection_calls = self.mock_client.create_connection.call_args_list

            self.assertEqual(len(consumer_calls), 2)
            self.assertEqual(len(producer_calls), 2)
            self.assertEqual(len(connection_calls), 2)

            self.assertEqual(consumer_calls[0][1]["queue_name"], "my_ocr_input")
            self.assertEqual(producer_calls[0][1]["queue_name"], "my_ocr_output")
            self.assertEqual(consumer_calls[1][1]["queue_name"], "my_whisper_input")
            self.assertEqual(producer_calls[1][1]["queue_name"], "my_whisper_output")

    def test_start_process_group_failure(self):
        self.mock_client.start_process_group.return_value = False

        with patch.dict("os.environ", {
            "NIFI_ENDPOINT": "https://nifi.example.com:8443/nifi-api",
            "NIFI_USERNAME": "admin",
            "NIFI_PASSWORD": "secret",
        }, clear=True), patch("nifi.setup_flow.NifiClient", return_value=self.mock_client):
            with self.assertRaises(SystemExit) as cm:
                setup_flow.setup_flow()
            self.assertEqual(cm.exception.code, 1)

    def test_verify_flow_health_failure(self):
        self.mock_client.verify_flow_health.return_value = False

        with patch.dict("os.environ", {
            "NIFI_ENDPOINT": "https://nifi.example.com:8443/nifi-api",
            "NIFI_USERNAME": "admin",
            "NIFI_PASSWORD": "secret",
        }, clear=True), patch("nifi.setup_flow.NifiClient", return_value=self.mock_client):
            with self.assertRaises(SystemExit) as cm:
                setup_flow.setup_flow()
            self.assertEqual(cm.exception.code, 1)

    def test_registry_endpoint_configures_client(self):
        with patch.dict("os.environ", {
            "NIFI_ENDPOINT": "https://nifi.example.com:8443/nifi-api",
            "NIFI_USERNAME": "admin",
            "NIFI_PASSWORD": "secret",
            "REGISTRY_ENDPOINT": "http://registry:18080/nifi-registry-api",
        }, clear=True), patch("nifi.setup_flow.NifiClient") as mock_nifi_client_cls:
            mock_nifi_client_cls.return_value = self.mock_client
            setup_flow.setup_flow()
            mock_nifi_client_cls.assert_called_once_with(
                base_url="https://nifi.example.com:8443/nifi-api",
                username="admin",
                password="secret",
                ssl_verify=False,
                registry_url="http://registry:18080/nifi-registry-api",
            )

    def test_ssl_verify_enabled(self):
        with patch.dict("os.environ", {
            "NIFI_ENDPOINT": "https://nifi.example.com:8443/nifi-api",
            "NIFI_USERNAME": "admin",
            "NIFI_PASSWORD": "secret",
            "NIFI_SSL_VERIFY": "true",
        }, clear=True), patch("nifi.setup_flow.NifiClient") as mock_nifi_client_cls:
            mock_nifi_client_cls.return_value = self.mock_client
            setup_flow.setup_flow()
            mock_nifi_client_cls.assert_called_once_with(
                base_url="https://nifi.example.com:8443/nifi-api",
                username="admin",
                password="secret",
                ssl_verify=True,
                registry_url=None,
            )

    def test_redis_config_passed_to_processors(self):
        with patch.dict("os.environ", {
            "NIFI_ENDPOINT": "https://nifi.example.com:8443/nifi-api",
            "NIFI_USERNAME": "admin",
            "NIFI_PASSWORD": "secret",
            "REDIS_HOST": "my-redis",
            "REDIS_PORT": "6380",
            "REDIS_DB": "2",
        }, clear=True), patch("nifi.setup_flow.NifiClient", return_value=self.mock_client):
            setup_flow.setup_flow()

            consumer_call = self.mock_client.create_consumer_processor.call_args
            self.assertEqual(consumer_call[1]["redis_host"], "my-redis")
            self.assertEqual(consumer_call[1]["redis_port"], 6380)
            self.assertEqual(consumer_call[1]["redis_db"], 2)

    def test_main_block_success_exit(self):
        self.mock_client.check_flow_exists.return_value = "existing-pg-id"
        self.mock_client.wait_for_nifi.return_value = True

        with patch.dict("os.environ", {
            "NIFI_ENDPOINT": "https://nifi.example.com:8443/nifi-api",
            "NIFI_USERNAME": "admin",
            "NIFI_PASSWORD": "secret",
        }, clear=True), patch("nifi.setup_flow.NifiClient", return_value=self.mock_client):
            with self.assertRaises(SystemExit) as cm:
                setup_flow.setup_flow()
            self.assertEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main()

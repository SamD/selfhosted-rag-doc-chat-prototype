"""
NiFi client wrapper using NiPyAPI for programmatic flow management.
"""
import logging
import time
from typing import Optional

import nipyapi
import requests
from nipyapi import canvas, config

logger = logging.getLogger(__name__)


class NifiClient:
    """Wrapper around NiPyAPI for NiFi flow management."""

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        ssl_verify: bool = False,
        registry_url: Optional[str] = None,
    ):
        """
        Initialize NiFi client.

        Args:
            base_url: NiFi API endpoint (must include /nifi-api suffix)
            username: NiFi username for basic auth
            password: NiFi password for basic auth
            ssl_verify: Whether to verify SSL certificates (False for self-signed)
            registry_url: Optional NiFi Registry endpoint
        """
        self.base_url = base_url
        self.username = username
        self.password = password
        self.ssl_verify = ssl_verify
        self.registry_url = registry_url
        self.verify_ssl = ssl_verify

        self._configure_nifi()
        self._configure_registry()
        self._setup_headers()

    def _configure_nifi(self):
        """Configure NiPyAPI for NiFi connection."""
        logger.info(f"Configuring NiFi connection to {self.base_url}")

        config.nifi_config.host = self.base_url
        config.nifi_config.verify_ssl = self.ssl_verify
        config.nifi_config.username = self.username
        config.nifi_config.password = self.password

        if not self.ssl_verify:
            logger.warning("SSL verification disabled - using self-signed certificates")

        try:
            nipyapi.utils.set_endpoint(
                self.base_url,
                ssl=self.base_url.startswith("https"),
                login=True,
            )
            logger.info("NiFi endpoint configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure NiFi endpoint: {e}")
            raise

    def _configure_registry(self):
        """Configure NiPyAPI for NiFi Registry connection (if provided)."""
        if not self.registry_url:
            logger.info("No NiFi Registry URL provided - skipping registry configuration")
            return

        logger.info(f"Configuring NiFi Registry connection to {self.registry_url}")

        config.registry_config.host = self.registry_url
        config.registry_config.verify_ssl = self.ssl_verify

        is_https = self.registry_url.startswith("https")
        if is_https:
            config.registry_config.username = self.username
            config.registry_config.password = self.password

        try:
            nipyapi.utils.set_endpoint(
                self.registry_url,
                ssl=is_https,
                login=is_https,
            )
            logger.info("NiFi Registry endpoint configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure NiFi Registry endpoint: {e}")
            raise

    def _setup_headers(self):
        """Set up HTTP headers for REST API calls."""
        # Get access token
        token_response = requests.post(
            f"{self.base_url}/access/token",
            data={"username": self.username, "password": self.password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            verify=self.verify_ssl
        )
        token_response.raise_for_status()
        self.token = token_response.text
        
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        logger.info("REST API headers configured")

    def wait_for_nifi(self, max_retries: int = 5) -> bool:
        """
        Wait for NiFi to become available with exponential backoff.

        Args:
            max_retries: Maximum number of retry attempts

        Returns:
            True if NiFi is available, False otherwise
        """
        logger.info(f"Waiting for NiFi at {self.base_url} (max {max_retries} retries)")

        for attempt in range(max_retries):
            try:
                root_pg_id = canvas.get_root_pg_id()
                logger.info(f"NiFi is available. Root process group ID: {root_pg_id}")
                return True
            except Exception as e:
                wait_time = 2 ** attempt
                logger.warning(
                    f"NiFi not available (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)

        logger.error(f"NiFi not available after {max_retries} attempts")
        return False

    def create_process_group(self, name: str, parent_pg_id: Optional[str] = None) -> str:
        """
        Create a new process group.

        Args:
            name: Name of the process group
            parent_pg_id: Parent process group ID (defaults to root)

        Returns:
            Process group ID
        """
        if parent_pg_id is None:
            parent_pg_id = "root"

        logger.info(f"Creating process group '{name}' in parent {parent_pg_id}")

        try:
            pg = canvas.create_process_group(parent_pg_id, name)
            pg_id = pg.id
            logger.info(f"Created process group '{name}' with ID: {pg_id}")
            return pg_id
        except Exception as e:
            logger.error(f"Failed to create process group '{name}': {e}")
            raise

    def check_flow_exists(self, name: str, parent_pg_id: Optional[str] = None) -> Optional[str]:
        """
        Check if a process group with the given name already exists.

        Args:
            name: Name of the process group to find
            parent_pg_id: Parent process group ID to search in (defaults to root)

        Returns:
            Process group ID if found, None otherwise
        """
        if parent_pg_id is None:
            parent_pg_id = "root"

        logger.debug(f"Checking if process group '{name}' exists in {parent_pg_id}")

        try:
            pgs = canvas.list_all_process_groups(parent_pg_id)
            for pg in pgs:
                if pg.component.name == name:
                    logger.info(f"Found existing process group '{name}' with ID: {pg.id}")
                    return pg.id
            logger.debug(f"Process group '{name}' not found")
            return None
        except Exception as e:
            logger.error(f"Failed to check for process group '{name}': {e}")
            raise

    def create_processor(self, process_group_id: str, processor_type: str, name: str, properties: dict, auto_terminate: list = None, bundle: dict = None) -> str:
        """
        Create a processor using REST API.
        
        Args:
            process_group_id: Process group to add processor to
            processor_type: Type of processor (e.g., "RedisQueueConsumer")
            name: Processor name
            properties: Processor properties
            auto_terminate: List of relationship names to auto-terminate
            bundle: Optional bundle dict with group, artifact, version
            
        Returns:
            Processor ID
        """
        logger.info(f"Creating processor '{name}' of type '{processor_type}'")
        
        try:
            # Create processor with revision version 0 (required for new processors)
            processor_data = {
                "revision": {"version": 0},
                "component": {
                    "type": processor_type,
                    "name": name,
                    "config": {
                        "properties": properties
                    }
                }
            }
            
            # Add bundle if specified
            if bundle:
                processor_data["component"]["bundle"] = bundle
            
            # Add auto-terminated relationships if specified
            if auto_terminate:
                processor_data["component"]["config"]["autoTerminatedRelationships"] = auto_terminate
            
            response = requests.post(
                f"{self.base_url}/process-groups/{process_group_id}/processors",
                headers=self.headers,
                json=processor_data,
                verify=self.verify_ssl
            )
            
            if response.status_code != 201:
                logger.error(f"Failed to create processor: {response.status_code} - {response.text}")
            
            response.raise_for_status()
            
            processor_id = response.json()["component"]["id"]
            logger.info(f"Created processor '{name}' with ID: {processor_id}")
            return processor_id
            
        except Exception as e:
            logger.error(f"Failed to create processor '{name}': {e}")
            raise

    def update_processor_name(self, processor_id: str, new_name: str) -> None:
        """
        Update the name of a processor using REST API.
        
        Args:
            processor_id: ID of the processor to update
            new_name: New name for the processor
        """
        logger.info(f"Updating processor '{processor_id}' name to '{new_name}'")
        
        try:
            # Get current processor to get revision
            response = requests.get(
                f"{self.base_url}/processors/{processor_id}",
                headers=self.headers,
                verify=self.verify_ssl
            )
            response.raise_for_status()
            
            processor_data = response.json()
            revision = processor_data["revision"]
            
            # Update processor name
            update_data = {
                "revision": revision,
                "component": {
                    "id": processor_id,
                    "name": new_name
                }
            }
            
            response = requests.put(
                f"{self.base_url}/processors/{processor_id}",
                headers=self.headers,
                json=update_data,
                verify=self.verify_ssl
            )
            response.raise_for_status()
            
            logger.info(f"Updated processor name to '{new_name}'")
            
        except Exception as e:
            logger.error(f"Failed to update processor name: {e}")
            raise

    def update_processor_scheduling(self, processor_id: str, scheduling_period: str = "100 ms", concurrent_tasks: int = 1) -> None:
        """
        Update the scheduling configuration of a processor.
        
        Args:
            processor_id: ID of the processor to update
            scheduling_period: Run schedule (e.g., "100 ms", "1 min")
            concurrent_tasks: Number of concurrent tasks
        """
        logger.info(f"Updating processor '{processor_id}' scheduling to {scheduling_period}")
        
        try:
            # Get current processor to get revision
            response = requests.get(
                f"{self.base_url}/processors/{processor_id}",
                headers=self.headers,
                verify=self.verify_ssl
            )
            response.raise_for_status()
            
            processor_data = response.json()
            revision = processor_data["revision"]
            
            # Update scheduling
            update_data = {
                "revision": revision,
                "component": {
                    "id": processor_id,
                    "config": {
                        "schedulingPeriod": scheduling_period,
                        "concurrentlySchedulableTaskCount": concurrent_tasks
                    }
                }
            }
            
            response = requests.put(
                f"{self.base_url}/processors/{processor_id}",
                headers=self.headers,
                json=update_data,
                verify=self.verify_ssl
            )
            response.raise_for_status()
            
            logger.info(f"Updated processor scheduling to {scheduling_period}")
            
        except Exception as e:
            logger.error(f"Failed to update processor scheduling: {e}")
            raise

    def create_consumer_processor(
        self,
        process_group_id: str,
        queue_name: str,
        redis_host: str,
        redis_port: int,
        redis_db: int = 0,
        socket_timeout: int = 5,
        connection_timeout: int = 5,
    ) -> str:
        """
        Create a RedisQueueConsumer processor.

        Args:
            process_group_id: Process group to add processor to
            queue_name: Redis queue name to consume from
            redis_host: Redis server hostname
            redis_port: Redis server port
            redis_db: Redis database number
            socket_timeout: Socket timeout in seconds
            connection_timeout: Connection timeout in seconds

        Returns:
            Processor ID
        """
        processor_name = f"RedisConsumer - {queue_name}"
        logger.info(f"Creating consumer processor '{processor_name}'")

        properties = {
            "Redis Host": redis_host,
            "Redis Port": str(redis_port),
            "Redis DB Index": str(redis_db),
            "Queue Key": queue_name,
            "Socket Timeout": str(socket_timeout),
            "Connection Timeout": str(connection_timeout),
        }

        bundle = {
            "group": "org.apache.nifi",
            "artifact": "python-extensions",
            "version": "2.0.0"
        }
        return self.create_processor(process_group_id, "RedisQueueConsumer", processor_name, properties, bundle=bundle)

    def create_producer_processor(
        self,
        process_group_id: str,
        queue_name: str,
        redis_host: str,
        redis_port: int,
        redis_db: int = 0,
        push_operation: str = "LPUSH",
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """
        Create a RedisQueueProducer processor.

        Args:
            process_group_id: Process group to add processor to
            queue_name: Redis queue name to produce to
            redis_host: Redis server hostname
            redis_port: Redis server port
            redis_db: Redis database number
            push_operation: Redis push operation (LPUSH or RPUSH)
            ttl_seconds: Optional TTL for queue keys

        Returns:
            Processor ID
        """
        processor_name = f"RedisProducer - {queue_name}"
        logger.info(f"Creating producer processor '{processor_name}'")

        properties = {
            "Redis Host": redis_host,
            "Redis Port": str(redis_port),
            "Redis DB": str(redis_db),
            "Redis List Key": queue_name,
            "Push Operation": push_operation,
        }

        if ttl_seconds is not None:
            properties["TTL Seconds"] = str(ttl_seconds)

        bundle = {
            "group": "org.apache.nifi",
            "artifact": "python-extensions",
            "version": "2.0.0"
        }
        return self.create_processor(process_group_id, "RedisQueueProducer", processor_name, properties, auto_terminate=["success"], bundle=bundle)

    def create_log_processor(
        self,
        process_group_id: str,
        name: str,
        log_level: str = "error",
        log_prefix: str = "[QUEUE FAILURE]",
        log_message: str = "${queue.error.message}",
    ) -> str:
        """
        Create a LogMessage processor for error logging.

        Args:
            process_group_id: Process group to add processor to
            name: Processor name
            log_level: Log level (error, warn, info, debug)
            log_prefix: Prefix for log messages
            log_message: Log message template (supports NiFi Expression Language)

        Returns:
            Processor ID
        """
        processor_name = name
        logger.info(f"Creating log processor '{processor_name}'")

        properties = {
            "Log Level": log_level,
            "Log Prefix": log_prefix,
            "Log Message": log_message,
        }

        return self.create_processor(process_group_id, "org.apache.nifi.processors.standard.LogMessage", processor_name, properties, auto_terminate=["success"])

    def create_connection(
        self,
        process_group_id: str,
        source_id: str,
        destination_id: str,
        relationships: Optional[list] = None,
    ) -> str:
        """
        Create a connection between two processors using direct REST API.

        Args:
            process_group_id: Process group containing the processors
            source_id: Source processor ID
            destination_id: Destination processor ID
            relationships: List of relationships to connect (defaults to ["success"])

        Returns:
            Connection ID
        """
        if relationships is None:
            relationships = ["success"]

        logger.info(f"Creating connection from {source_id} to {destination_id} with relationships {relationships}")

        try:
            # Verify processors exist
            requests.get(
                f"{self.base_url}/processors/{source_id}",
                headers=self.headers,
                verify=self.verify_ssl
            ).raise_for_status()

            requests.get(
                f"{self.base_url}/processors/{destination_id}",
                headers=self.headers,
                verify=self.verify_ssl
            ).raise_for_status()

            # Create connection using REST API
            connection_data = {
                "revision": {"version": 0},
                "sourceType": "PROCESSOR",
                "destinationType": "PROCESSOR",
                "component": {
                    "source": {
                        "id": source_id,
                        "groupId": process_group_id,
                        "type": "PROCESSOR"
                    },
                    "destination": {
                        "id": destination_id,
                        "groupId": process_group_id,
                        "type": "PROCESSOR"
                    },
                    "selectedRelationships": relationships
                }
            }

            response = requests.post(
                f"{self.base_url}/process-groups/{process_group_id}/connections",
                headers=self.headers,
                json=connection_data,
                verify=self.verify_ssl
            )

            if response.status_code != 201:
                logger.error(f"Failed to create connection: {response.status_code} - {response.text}")

            response.raise_for_status()

            connection_id = response.json()["id"]
            logger.info(f"Created connection with ID: {connection_id}")
            return connection_id

        except Exception as e:
            logger.error(f"Failed to create connection from {source_id} to {destination_id}: {e}")
            raise

    def start_process_group(self, process_group_id: str) -> bool:
        """
        Start all processors in a process group.

        Args:
            process_group_id: Process group to start

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Starting process group {process_group_id}")

        try:
            canvas.schedule_process_group(process_group_id, scheduled=True)
            logger.info(f"Started process group {process_group_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to start process group {process_group_id}: {e}")
            return False

    def verify_flow_health(self, process_group_id: str) -> bool:
        """
        Verify that all processors in a process group are running.

        Args:
            process_group_id: Process group to verify

        Returns:
            True if all processors are running, False otherwise
        """
        logger.info(f"Verifying flow health for process group {process_group_id}")

        try:
            processors = canvas.list_all_processors(process_group_id)
            all_running = True

            for processor in processors:
                state = processor.status.run_status
                name = processor.component.name

                if state != "Running":
                    logger.error(f"Processor '{name}' is not running (state: {state})")
                    all_running = False
                else:
                    logger.debug(f"Processor '{name}' is running")

            if all_running:
                logger.info(f"All processors in process group {process_group_id} are running")
            else:
                logger.error(f"Some processors in process group {process_group_id} are not running")

            return all_running
        except Exception as e:
            logger.error(f"Failed to verify flow health for process group {process_group_id}: {e}")
            return False

    def delete_process_group(self, process_group_id: str) -> bool:
        """
        Delete a process group and all its contents.

        Args:
            process_group_id: Process group to delete

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Deleting process group {process_group_id}")

        try:
            canvas.delete_process_group(process_group_id)
            logger.info(f"Deleted process group {process_group_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete process group {process_group_id}: {e}")
            return False

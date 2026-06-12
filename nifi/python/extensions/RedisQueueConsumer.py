import json
import time

from nifiapi.__jvm__ import JvmHolder
from nifiapi.flowfilesource import FlowFileSource, FlowFileSourceResult
from nifiapi.properties import PropertyDescriptor
from nifiapi.relationship import Relationship

try:
    import redis
    from redis.exceptions import ConnectionError, RedisError, TimeoutError
except ImportError:
    redis = None


class RedisQueueConsumer(FlowFileSource):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileSource']

    class ProcessorDetails:
        version = '2.0.0'
        description = 'Non-blocking LPOP from Redis LIST. Schedule at 100ms for near-real-time processing.'
        tags = ['redis', 'queue', 'lpop', 'source', 'python']
        dependencies = ['redis>=5.0.0']

    REDIS_HOST = PropertyDescriptor(
        name="Redis Host",
        description="Redis server hostname or IP",
        required=True,
        default_value="localhost"
    )

    REDIS_PORT = PropertyDescriptor(
        name="Redis Port",
        description="Redis server port",
        required=True,
        default_value="6379"
    )

    REDIS_PASSWORD = PropertyDescriptor(
        name="Redis Password",
        description="Redis password (if required)",
        required=False,
        sensitive=True
    )

    REDIS_DB = PropertyDescriptor(
        name="Redis DB Index",
        description="Redis database number (0-15)",
        required=True,
        default_value="0"
    )

    QUEUE_KEY = PropertyDescriptor(
        name="Queue Key",
        description="Redis LIST key to consume from",
        required=True
    )

    SOCKET_TIMEOUT = PropertyDescriptor(
        name="Socket Timeout",
        description="Socket timeout in seconds (read/write)",
        required=True,
        default_value="5"
    )

    CONNECTION_TIMEOUT = PropertyDescriptor(
        name="Connection Timeout",
        description="Connection timeout in seconds",
        required=True,
        default_value="5"
    )

    property_descriptors = [REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB, QUEUE_KEY, SOCKET_TIMEOUT, CONNECTION_TIMEOUT]
    
    SUCCESS = Relationship(
        name="success",
        description="Messages successfully consumed from Redis"
    )

    def __init__(self, jvm=None, **kwargs):
        super().__init__()
        self.redis_client = None
        self.queue_key = None
        self.java_success_relationship = None

    def getPropertyDescriptors(self):
        return self.property_descriptors
    
    def getRelationships(self):
        return [self.SUCCESS]

    def onScheduled(self, context):
        """Initialize Redis connection when processor starts"""
        try:
            self.redis_host = context.getProperty(self.REDIS_HOST).getValue()
            self.redis_port = int(context.getProperty(self.REDIS_PORT).getValue())
            self.redis_password = context.getProperty(self.REDIS_PASSWORD).getValue()
            self.redis_db = int(context.getProperty(self.REDIS_DB).getValue() or "0")
            self.queue_key = context.getProperty(self.QUEUE_KEY).getValue()
            self.socket_timeout = float(context.getProperty(self.SOCKET_TIMEOUT).getValue() or "5")
            self.connection_timeout = float(context.getProperty(self.CONNECTION_TIMEOUT).getValue() or "5")
            
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                password=self.redis_password if self.redis_password else None,
                db=self.redis_db,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.connection_timeout,
                decode_responses=True,
                retry_on_timeout=True,
                socket_keepalive=True,
                health_check_interval=30
            )
            
            # Create Java Relationship object for FlowFileSourceResult
            self.java_success_relationship = self.SUCCESS.to_java_relationship(JvmHolder.gateway)
            
            self.redis_client.ping()
            self.logger.info(f"Successfully connected to Redis at {self.redis_host}:{self.redis_port}")
            self.logger.info(f"Consuming from queue key: {self.queue_key}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis connection: {str(e)}")
            raise

    def create(self, context):
        """Non-blocking LPOP from Redis. Returns FlowFileSourceResult or None"""
        try:
            message = self.redis_client.lpop(self.queue_key)
            
            if message is None:
                return None
            
            self.logger.debug(f"Received message from {self.queue_key}")
            
            attributes = {
                "redis.queue": self.queue_key,
                "redis.consumer.timestamp": str(int(time.time())),
                "redis.message.length": str(len(message)),
            }

            # Extract trace_id from JSON payload for provenance indexing
            try:
                data = json.loads(message)
                trace_id = data.get('trace_id')
                if trace_id:
                    attributes["trace_id"] = str(trace_id)
            except (json.JSONDecodeError, TypeError):
                pass
            
            return FlowFileSourceResult(
                relationship="success",
                contents=message.encode('utf-8'),
                attributes=attributes
            )
            
        except ConnectionError as e:
            self.logger.error(f"Redis connection error: {str(e)}")
            return None
            
        except TimeoutError as e:
            self.logger.warning(f"Redis timeout error: {str(e)}")
            return None
            
        except RedisError as e:
            self.logger.error(f"Redis error: {str(e)}")
            return None
            
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return None

    def onStopped(self, context):
        """Clean up Redis connection when processor stops"""
        if self.redis_client:
            try:
                self.redis_client.close()
                self.logger.info("Redis connection closed")
            except Exception as e:
                self.logger.warning(f"Error closing Redis connection: {str(e)}")
        
        self.redis_client = None

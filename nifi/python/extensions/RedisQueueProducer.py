import redis
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult
from nifiapi.properties import PropertyDescriptor, StandardValidators
from nifiapi.relationship import Relationship


class RedisQueueProducer(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']

    class ProcessorDetails:
        version = '2.0.0'
        description = 'Pushes FlowFile content to a Redis list using LPUSH/RPUSH.'
        tags = ['redis', 'queue', 'lpush', 'rpush', 'sink', 'python']
        dependencies = ['redis>=5.0.0']

    REDIS_HOST = PropertyDescriptor(
        name="Redis Host",
        description="The hostname or IP address of the Redis server.",
        required=True,
        default_value="localhost",
        validators=[StandardValidators.NON_EMPTY_VALIDATOR]
    )

    REDIS_PORT = PropertyDescriptor(
        name="Redis Port",
        description="The port of the Redis server.",
        required=True,
        default_value="6379",
        validators=[StandardValidators.PORT_VALIDATOR]
    )

    REDIS_DB = PropertyDescriptor(
        name="Redis DB",
        description="The Redis database number.",
        required=False,
        default_value="0",
        validators=[StandardValidators.NON_NEGATIVE_INTEGER_VALIDATOR]
    )

    REDIS_LIST_KEY = PropertyDescriptor(
        name="Redis List Key",
        description="The name of the Redis list/queue to push to.",
        required=True,
        validators=[StandardValidators.NON_EMPTY_VALIDATOR]
    )

    PUSH_OPERATION = PropertyDescriptor(
        name="Push Operation",
        description="The Redis push operation to use.",
        allowable_values=["LPUSH", "RPUSH"],
        default_value="LPUSH",
        required=True
    )

    TTL_SECONDS = PropertyDescriptor(
        name="TTL Seconds",
        description="Optional TTL in seconds to apply to the queue key after push.",
        required=False,
        validators=[StandardValidators.POSITIVE_INTEGER_VALIDATOR]
    )

    property_descriptors = [REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_LIST_KEY, PUSH_OPERATION, TTL_SECONDS]
    
    SUCCESS = Relationship(
        name="success",
        description="All FlowFiles will go to this relationship after being successfully processed"
    )
    FAILURE = Relationship(
        name="failure",
        description="All FlowFiles that fail to process will go to this relationship"
    )

    def __init__(self, jvm=None, **kwargs):
        super().__init__()
        self.redis_client = None
        self.list_key = None
        self.push_operation = None
        self.ttl_seconds = None

    def getPropertyDescriptors(self):
        return self.property_descriptors
    
    def getRelationships(self):
        return [self.SUCCESS, self.FAILURE]
    
    def onUnscheduled(self):
        pass
    
    def getAutoTerminatedRelationships(self):
        return [self.SUCCESS, self.FAILURE]

    def onScheduled(self, context):
        host = context.getProperty(self.REDIS_HOST).getValue()
        port = int(context.getProperty(self.REDIS_PORT).getValue())
        db = int(context.getProperty(self.REDIS_DB).getValue() or "0")
        self.list_key = context.getProperty(self.REDIS_LIST_KEY).getValue()
        self.push_operation = context.getProperty(self.PUSH_OPERATION).getValue()
        ttl_val = context.getProperty(self.TTL_SECONDS).getValue()
        self.ttl_seconds = int(ttl_val) if ttl_val else None

        pool = redis.ConnectionPool(host=host, port=port, db=db, decode_responses=True)
        self.redis_client = redis.Redis(connection_pool=pool)
        self.logger.info(f"RedisQueueProducer scheduled: {self.push_operation} to {self.list_key} on {host}:{port}/{db}")

    def transform(self, context, flow_file):
        attributes = flow_file.getAttributes()

        try:
            content = flow_file.getContentsAsBytes()
            content_str = content.decode('utf-8')

            if self.push_operation == "RPUSH":
                self.redis_client.rpush(self.list_key, content_str)
            else:
                self.redis_client.lpush(self.list_key, content_str)

            if self.ttl_seconds:
                self.redis_client.expire(self.list_key, self.ttl_seconds)

            self.logger.info(f"Pushed FlowFile to {self.list_key} ({len(content)} bytes)")

            return FlowFileTransformResult(
                relationship="success",
                contents=content,
                attributes=attributes
            )

        except redis.ConnectionError as e:
            self.logger.error(f"Redis connection error in RedisQueueProducer: {e}")
            attributes["queue.error.message"] = f"Redis connection error: {str(e)}"
            return FlowFileTransformResult(
                relationship="failure",
                contents=content if 'content' in dir() else b"",
                attributes=attributes
            )
        except redis.RedisError as e:
            self.logger.error(f"Redis error in RedisQueueProducer: {e}")
            attributes["queue.error.message"] = f"Redis error: {str(e)}"
            return FlowFileTransformResult(
                relationship="failure",
                contents=content if 'content' in dir() else b"",
                attributes=attributes
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in RedisQueueProducer: {e}")
            attributes["queue.error.message"] = f"Unexpected error: {str(e)}"
            return FlowFileTransformResult(
                relationship="failure",
                contents=content if 'content' in dir() else b"",
                attributes=attributes
            )

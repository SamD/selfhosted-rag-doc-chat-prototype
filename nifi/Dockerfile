FROM apache/nifi:2.0.0

# Copy Python processors to NiFi's extensions directory
COPY nifi/python/extensions/RedisQueueConsumer.py /opt/nifi/nifi-current/python/extensions/
COPY nifi/python/extensions/RedisQueueProducer.py /opt/nifi/nifi-current/python/extensions/

# Ensure Python extensions are enabled
USER root
RUN echo "nifi.python.extensions.source.directory.default=./python/extensions" >> /opt/nifi/nifi-current/conf/nifi.properties && \
    echo "nifi.python.max.tasks=2" >> /opt/nifi/nifi-current/conf/nifi.properties && \
    chown -R nifi:nifi /opt/nifi/nifi-current/python/extensions/
USER nifi

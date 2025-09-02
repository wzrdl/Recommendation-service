# consumer.py
import json
from kafka import KafkaConsumer

KAFKA_TOPIC = 'click-stream'

# Note: We connect to 'kafka:29092' because we are inside the Docker network.
consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers='18.224.96.184:9092',
    auto_offset_reset='earliest', # Start reading from the beginning of the topic
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("Listening for messages on topic 'click-stream'...")
for message in consumer:
    print(f"Received: {message.value}")

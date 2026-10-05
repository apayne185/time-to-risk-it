"""Kafka (Redpanda) adapters around the transport-agnostic ``StreamScorer``.

Delivery is at-least-once: consumer offsets are committed only after an event has been processed
and any scores it triggered have been produced. Malformed messages go to a dead-letter topic
instead of stopping the consumer.

Event-time ordering: the demo topic has one partition, so the stream is globally ordered. With
several partitions the watermark would be the minimum event time across assigned partitions.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Any

from confluent_kafka import Consumer, KafkaError, KafkaException, Producer
from confluent_kafka.admin import AdminClient
from confluent_kafka.cimpl import NewTopic
from pydantic import ValidationError

from ttr.stream.events import ACTIVITY_TOPIC, SCORES_TOPIC, ActivityEvent
from ttr.stream.processor import StreamScorer, StreamStats

log = logging.getLogger(__name__)

DLQ_TOPIC = f"{ACTIVITY_TOPIC}.dlq"


def ensure_topics(bootstrap: str, topics: Iterable[str], partitions: int = 1) -> None:
    admin = AdminClient({"bootstrap.servers": bootstrap})
    existing = set(admin.list_topics(timeout=10).topics)
    new = [
        NewTopic(t, num_partitions=partitions, replication_factor=1)
        for t in topics
        if t not in existing
    ]
    for topic, future in admin.create_topics(new).items() if new else []:
        try:
            future.result()
            log.info("created topic %s", topic)
        except KafkaException as exc:  # created concurrently by another client
            if exc.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
                raise


def replay(
    events: Iterable[ActivityEvent],
    bootstrap: str,
    topic: str = ACTIVITY_TOPIC,
    rate: float | None = None,
) -> int:
    """Publish events in order. ``rate`` limits events per second (None = as fast as possible)."""
    ensure_topics(bootstrap, [topic, SCORES_TOPIC, DLQ_TOPIC])
    producer = Producer(
        {"bootstrap.servers": bootstrap, "enable.idempotence": True, "linger.ms": 20}
    )
    sent = 0
    start = time.monotonic()
    for event in events:
        producer.produce(topic, key=event.key, value=event.to_bytes())
        sent += 1
        producer.poll(0)
        if rate:
            ahead = sent / rate - (time.monotonic() - start)
            if ahead > 0:
                time.sleep(ahead)
    producer.flush(30)
    log.info("replayed %d events to %s", sent, topic)
    return sent


def consume_and_score(
    processor: StreamScorer,
    bootstrap: str,
    group: str = "ttr-scorer",
    in_topic: str = ACTIVITY_TOPIC,
    out_topic: str = SCORES_TOPIC,
    idle_seconds: float = 10.0,
    flush_at_end: bool = True,
    commit_every: int = 500,
) -> StreamStats:
    """Consume activity, publish scores. Stops after ``idle_seconds`` without messages
    (a batch-style demo); a long-running deployment would pass ``idle_seconds=inf``."""
    ensure_topics(bootstrap, [in_topic, out_topic, DLQ_TOPIC])
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": group,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    producer = Producer({"bootstrap.servers": bootstrap, "enable.idempotence": True})
    consumer.subscribe([in_topic])
    dead_letters = 0

    def publish(scores: list[Any]) -> None:
        for s in scores:
            producer.produce(out_topic, key=str(s.user_id).encode(), value=s.to_bytes())
        producer.poll(0)

    def checkpoint(msg: Any) -> None:
        producer.flush(30)  # scores are durable before the offset moves
        consumer.commit(message=msg, asynchronous=False)

    last_message = time.monotonic()
    uncommitted, last_msg = 0, None
    try:
        while time.monotonic() - last_message < idle_seconds:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())
            last_message, last_msg = time.monotonic(), msg
            fired = False
            try:
                event = ActivityEvent.from_bytes(msg.value() or b"")
            except ValidationError:
                producer.produce(DLQ_TOPIC, key=msg.key(), value=msg.value())
                dead_letters += 1
            else:
                scores = processor.process(event)
                publish(scores)
                fired = bool(scores)
            uncommitted += 1
            if fired or uncommitted >= commit_every:
                checkpoint(msg)
                uncommitted = 0
        if last_msg is not None and uncommitted:
            checkpoint(last_msg)
        if flush_at_end:
            publish(processor.flush())
            producer.flush(30)
    finally:
        consumer.close()
    if dead_letters:
        log.warning("%d malformed messages sent to %s", dead_letters, DLQ_TOPIC)
    return processor.stats

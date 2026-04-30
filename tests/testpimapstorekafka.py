"""Unittest file for pimapstorekafka.py

This test file relies on Kafka downloaded and in the path. To download Kafka go to
kafka.apache.org/downloads and download the latest release and place the uncompressed
folder in the top level directory (the directory that contains the pimap directory and
Tests directory). Then create a symbolic link in the same directory to create a new
directory named kafka (ln -s kafka_version kafka).

To run the tests in the top level directory type the command:
PYTHONPATH=. python tests/testpimapstorekafka.py

Note: Be careful when running these tests as it will connect to Kafka on localhost:9092,
the default Kafka host and port, and delete all topics!!!

License:
Author: Sam Mansfield
"""

import ast
import subprocess
import time
import unittest
from confluent_kafka.admin import AdminClient
from confluent_kafka import KafkaException
from pimap import pimapstorekafka as pstk
from pimap import pimaputilities as pu


class PimapStoreKafkaTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        zookeeper = "kafka/bin/zookeeper-server-start.sh"
        zookeeper_properties = "kafka/config/zookeeper.properties"
        self.zookeeper_process = subprocess.Popen(
            [zookeeper, zookeeper_properties], stdout=subprocess.DEVNULL
        )
        time.sleep(10)
        kafka = "kafka/bin/kafka-server-start.sh"
        kafka_properties = "kafka/config/server.properties"
        self.kafka_process = subprocess.Popen(
            [kafka, kafka_properties], stdout=subprocess.DEVNULL
        )
        time.sleep(10)

        # Delete all Kafka topics.
        kafka_host = "localhost"
        kafka_port = "9092"
        broker = kafka_host + ":" + kafka_port
        admin = AdminClient({"bootstrap.servers": broker})
        cluster_metadata = admin.list_topics()
        while len(cluster_metadata.topics) > 0:
            admin.delete_topics(list(cluster_metadata.topics))
            cluster_metadata = admin.list_topics()

    @classmethod
    def tearDownClass(self):
        # Clean up all Kafka topics created during tests
        kafka_host = "localhost"
        kafka_port = "9092"
        broker = kafka_host + ":" + kafka_port
        try:
            admin = AdminClient({"bootstrap.servers": broker})
            cluster_metadata = admin.list_topics(timeout=5.0)
            topics_to_delete = [
                t for t in cluster_metadata.topics if not t.startswith("__")
            ]
            if topics_to_delete:
                admin.delete_topics(topics_to_delete)
                time.sleep(2)
        except Exception:
            pass

        self.kafka_process.kill()
        self.kafka_process.wait()
        self.zookeeper_process.kill()
        self.zookeeper_process.wait()

    def test_creating_incorrect_usage(self):
        # Test using invalid host and port.
        host = "nonsense"
        port = ""
        self.assertRaises(KafkaException, pstk.PimapStoreKafka, host, port)

    def test_store_correct_usage(self):
        # Test using system_samples with an empty list.
        store = pstk.PimapStoreKafka(system_samples=True)
        system_samples = store.store([])
        while len(system_samples) == 0:
            system_samples = store.store([])

        throughput = ast.literal_eval(pu.get_data(system_samples[0]))["throughput"]
        correct_throughput = 0.0
        self.assertEqual(throughput, correct_throughput)

    def test_store_incorrect_usage(self):
        # Test using a PIMAP sample instead of a list of PIMAP samples.
        sample_type = "store_test"
        patient_id = "1"
        device_id = "2"
        sample = {"store": 3}
        pimap_sample = pu.create_pimap_sample(
            sample_type, patient_id, device_id, sample
        )
        store = pstk.PimapStoreKafka()
        self.assertRaises(TypeError, store.store, pimap_sample)

        # Test passing a list of non-PIMAP data.
        invalid_pimap_data = ["nonsense", "invalid", "non-pimap"]
        store = pstk.PimapStoreKafka()
        self.assertRaises(ValueError, store.store, invalid_pimap_data)

    def test_retrieve_correct_usage(self):
        # Test subscribing to a topic that does not yet exist.
        store = pstk.PimapStoreKafka()
        sample_type = "non-existent"
        pimap_data = store.retrieve(sample_type)
        correct_pimap_data = []
        self.assertEqual(pimap_data, correct_pimap_data)

    def test_store_retrieve_correct_usage(self):
        # Test storing and retrieving multiple sample_types, metric_types, and patient_ids.
        sample_types = ["stest1", "stest2"]
        metric_types = ["mtest1", "mtest2"]
        patient_ids = ["alice", "bob", "cat"]
        device_id = "1"
        pimap_data = []
        for sample_type in sample_types:
            for patient_id in patient_ids:
                sample = {sample_type: patient_id}
                pimap_sample = pu.create_pimap_sample(
                    sample_type, patient_id, device_id, sample
                )
                pimap_data.append(pimap_sample)
        for metric_type in metric_types:
            for patient_id, pimap_sample in zip(patient_ids, pimap_data):
                metric = {metric_type: patient_id}
                pimap_metric = pu.create_pimap_metric(metric_type, pimap_sample, metric)
                pimap_data.append(pimap_metric)

        store = pstk.PimapStoreKafka()
        store.store(pimap_data)

        retrieved_pimap_data = []
        while len(retrieved_pimap_data) < len(pimap_data):
            for sample_type, metric_type in zip(sample_types, metric_types):
                retrieved_pimap_data.extend(store.retrieve(sample_type))
                retrieved_pimap_data.extend(store.retrieve(metric_type))

        pimap_data.sort()
        retrieved_pimap_data.sort()
        for retrieved_pimap_datum, pimap_datum in zip(retrieved_pimap_data, pimap_data):
            self.assertEqual(retrieved_pimap_datum, pimap_datum)


if __name__ == "__main__":
    unittest.main()

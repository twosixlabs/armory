"""

"""

import unittest
from armory.docker.management import ManagementInstance
from docker.models.containers import Container


class ContainerTest(unittest.TestCase):
    def test_creation(self):
        manager = ManagementInstance(image_name="twosixarmory/tf1:0.3.3")
        instance = manager.start_armory_instance()
        self.assertIsInstance(instance.docker_container, Container)
        self.assertIn(instance.docker_container.short_id, manager.instances)

    def test_deletion(self):
        manager = ManagementInstance(image_name="twosixarmory/tf1:0.3.3")
        instance = manager.start_armory_instance()
        manager.stop_armory_instance(instance)
        self.assertEqual(manager.instances, {})

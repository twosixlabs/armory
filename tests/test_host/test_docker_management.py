"""
Test docker container management
"""

import unittest

from docker.models.containers import Container

from armory.docker.management import ManagementInstance
from armory.docker import images


class ContainerTest(unittest.TestCase):
    def test_creation(self):
        manager = ManagementInstance(image_name=images.TF1)
        instance = manager.start_armory_instance()
        self.assertIsInstance(instance.docker_container, Container)
        self.assertIn(instance.docker_container.short_id, manager.instances)

    def test_deletion(self):
        manager = ManagementInstance(image_name=images.TF1)
        instance = manager.start_armory_instance()
        manager.stop_armory_instance(instance)
        self.assertEqual(manager.instances, {})

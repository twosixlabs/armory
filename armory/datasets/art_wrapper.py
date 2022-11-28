from armory.datasets import generator

from art.data_generators import DataGenerator


class WrappedDataGenerator(DataGenerator):
    """
    Wrap an ArmoryDataGenerator in the ART interface
    """

    def __init__(self, gen: generator.ArmoryDataGenerator):
        if gen.output_as_dict or len(gen.output_tuple) != 2:
            raise ValueError("gen must output (x, y) tuples")
        super().__init__(gen.size, gen.batch_size)
        self._iterator = gen

    def __iter__(self):
        return iter(self._iterator)

    def __next__(self):
        return next(self._iterator)

    def __len__(self):
        return len(self._iterator)

    def get_batch(self):
        return next(self._iterator)

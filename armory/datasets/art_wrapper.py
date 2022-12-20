from art.data_generators import DataGenerator


class WrappedDataGenerator(DataGenerator):
    """
    Wrap an ArmoryDataGenerator in the ART interface
    """

    def __init__(self, gen):
        super().__init__(gen.size, gen.batch_size)
        self._iterator = gen
        self.context = gen.context

    def __iter__(self):
        return iter(self._iterator)

    def __next__(self):
        return next(self._iterator)

    def __len__(self):
        return len(self._iterator)

    def get_batch(self):
        return next(self._iterator)

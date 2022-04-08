import numpy as np
from art.attacks.poisoning.gradient_matching_attack import GradientMatchingAttack


class GradientMatchingWrapper(GradientMatchingAttack):

    def __init__(self, **kwargs):
        self.attack_kwargs = kwargs
        estimator = kwargs["estimator"]
        percent_poison = kwargs["percent_poison"]
        epsilon = 0.01 # TODO: this depended on the dataset, in the notebook



        super().__init__(estimator,
                percent_poison,
                max_trials=1,
                max_epochs=500,
                # clip_values=(min_,max_),
                learning_rate_schedule=(np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5]), [250, 350, 400, 430, 460]),
                epsilon=epsilon,
                verbose=1)


    def poison(self, filepath, x_trigger, y_trigger, x_train, y_train):
        """
        if filepath doesn't exist, data should be provided
        if data is none, filepath should be path to pregenerated data
        """

        # check if file exists to return
        if os.path.exists(filepath):

            return # TODO load data from npz

        else:

            x_poison, y_poison = super().poison(x_trigger, y_trigger, x_train, y_train)
            poison_index = np.where([np.any(p != o) for (p,o) in zip(x_poison,x_train)])[0]

            # TODO then save data to filepath as npz
        return x_poison, y_poison, poison_index



from art.attacks.evasion import ProjectedGradientDescent
import numpy as np


class CustomAttack(ProjectedGradientDescent):
    def generate(self, x, y):

        x_adv = []
        for x_sample, y_sample in zip(x, y):
            for target in range(10):

                # Do not target correct class
                if target == y_sample:
                    continue

                # Generate sample targeting `target` class
                y_target = np.zeros((1, 10), dtype=np.int64)
                y_target[0, target] = 1
                x_adv_sample = super().generate(
                    np.expand_dims(x_sample, axis=0), y_target
                )

                # Check - does this example fool the classifier?
                x_adv_pred = np.argmax(self.estimator.predict(x_adv_sample))
                if x_adv_pred != y_sample:
                    break
            x_adv.append(x_adv_sample)

        x_adv = np.concatenate(x_adv, axis=0)
        return x_adv

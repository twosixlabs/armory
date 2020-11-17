import numpy as np


class KenansvilleDFT:
    def __init__(
        self,
        estimator,
        sample_rate=16000,
        snr_db=100,
        partial_attack=False,
        attack_len=500,
        attack_prob=0.5,
        targeted=False,
    ):
        """
        This DFT attack is a variant of the one described in https://arxiv.org/abs/1910.05262.
        In the paper, the attack assumed to have knowledge of word or phoneme locations
        in the input. The attack implemented here assumes complete blackbox knowledge,
        so the only options are: 1) modified the whole input or 2) modified subsequences
        of the input with some probability.

        param sample_rate: sample rate in Hz of inputs
        param estimator: not used but necessary for interoperability with Armory/ART
        param snr_db: the minimum SNR (in dB) to maintain
        type snr_db: 'float'
        param partial_attack: boolean to indicate if subsequences of the input are to be modified
        param attack_len: length of subsequences to attack. Valid when partial_attack = True
        param attack_prob: probability each subsequence will be attacked. Valid when partial_attack = True
        param targeted: not used but necessary for interoperability with Armory
        """
        self.sample_rate = sample_rate
        self.snr_db = snr_db
        self.targeted = targeted
        self.partial_attack = partial_attack
        self.attack_len = attack_len
        self.attack_prob = attack_prob

        if targeted:
            raise Warning("'targeted' argument is not used in Kenansville attack")

        if snr_db < 0:
            raise ValueError("Negative SNR is not allowed")

    def _attack(self, x):
        x_len = len(x)
        x_fft = np.fft.fft(x)
        x_psd = np.abs(x_fft) ** 2
        # sort by frequencies with increasing power
        x_psd_ind = np.argsort(x_psd)
        dc_ind = np.where(x_psd_ind == 0)[0][0]
        signal_db = 10 * np.log10(np.sum(x_psd))

        """
        The goal of the following search is to find all the
        low power frequencies that can be discarded while
        maintaining a minimum SNR.

        If desired, the following coarse and fine search could
        be replaced with binary search for faster convergence.
        """
        # coarse search
        id = 2
        noise = np.sum(x_psd[x_psd_ind[:id]])
        noise_db = 10 * np.log10(noise)
        while signal_db - noise_db > self.snr_db:
            id *= 2
            noise = np.sum(x_psd[x_psd_ind[: min(id, x_len)]])
            noise_db = 10 * np.log10(noise)

        if id == 2:
            return x

        # fine search
        id = int(id / 2)
        noise = np.sum(x_psd[x_psd_ind[:id]])
        noise_db = 10 * np.log10(noise)
        while signal_db - noise_db > self.snr_db:
            id += 2
            noise = np.sum(x_psd[x_psd_ind[: min(id, x_len)]])
            noise_db = 10 * np.log10(noise)

        id -= 2

        # make sure the only non-paired frequencies are DC and, if x is even, x_len/2
        if (dc_ind in x_psd_ind[:id]) ^ (
            x_len % 2 == 0 and x_len / 2 in x_psd_ind[:id]
        ):
            id -= 1

        # zero out low power frequencies
        x_fft[x_psd_ind[:id]] = 0

        x_ifft = np.fft.ifft(x_fft)

        return np.real(x_ifft).astype(np.float32)

    def generate(self, x):
        x_out = np.empty((len(x),), dtype=object)
        for i, x_example in enumerate(x):
            if self.partial_attack:
                # split input into multiple segments and attack each with some probability
                x_adv = np.zeros_like(x_example)
                seg_len = self.attack_len
                for j in range(int(np.ceil(len(x_example) / seg_len))):
                    xs = x_example[seg_len * j : min((j + 1) * seg_len, len(x_example))]
                    if np.random.rand(1) < self.attack_prob:
                        xs = self._attack(xs)
                    x_adv[seg_len * j : min((j + 1) * seg_len, len(x_example))] = xs
            else:
                x_adv = self._attack(x_example)
            x_out[i] = x_adv

        return x_out

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
        self.threshold = 10 ** (-self.snr_db / 10)
        self.targeted = targeted
        self.partial_attack = partial_attack
        self.attack_len = attack_len
        self.attack_prob = attack_prob

        if targeted:
            raise Warning("'targeted' argument is not used in Kenansville attack")

        if snr_db < 0:
            raise ValueError("Negative SNR (dB) is not allowed")

    def _attack(self, x):
        if not np.isreal(x).all():
            raise ValueError("Input must be real")
        if not len(x):
            return np.copy(x)

        # Determine power spectral density using real FFT
        #     Double power spectral density for paired frequencies (non-DC, non-nyquist)
        x_rfft = np.fft.rfft(x)
        x_psd = np.abs(x_rfft) ** 2
        if len(x) % 2:  # odd: DC frequency
            x_psd[1:] *= 2
        else:  # even: DC and Nyquist frequencies
            x_psd[1:-1] *= 2

        # Scale the threshold based on the power of the signal
        # Find frequencies in order with cumulative perturbation less than threshold
        #     Sort frequencies by power density in ascending order
        x_psd_index = np.argsort(x_psd)
        reordered = x_psd[x_psd_index]
        cumulative = np.cumsum(reordered)
        norm_threshold = self.threshold * cumulative[-1]
        i = np.searchsorted(cumulative, norm_threshold, side="right")

        # Zero out low power frequencies and invert to time domain
        x_rfft[x_psd_index[:i]] = 0
        return np.fft.irfft(x_rfft, len(x)).astype(x.dtype)

    def generate(self, x, y=None):
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

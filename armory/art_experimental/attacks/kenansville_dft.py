import numpy as np

class KenansvilleDFT():
    def __init__(
        self,
        estimator,
        snr_db=100,
        partial_attack=False,
        attack_len=500,
        attack_prob=0.5,
        targeted=False
    ):
        '''

        '''
        self.sample_rate = 16000
        self.snr_db = snr_db
        self.targeted = targeted
        self.partial_attack = partial_attack
        self.attack_len = attack_len
        self.attack_prob = attack_prob

    def _attack(self, x):
        x_fft = np.fft.fft(x)
        x_psd = np.abs(x_fft)**2
        # sort by frequencies with increasing power
        x_psd_ind = np.argsort(x_psd)
        signal_db = 10*np.log10(np.sum(x_psd))

        # coarse search
        id = 2
        noise = np.sum(x_psd[x_psd_ind[:id]])
        noise_db = 10*np.log10(noise)
        while signal_db - noise_db > self.snr_db:
            id *= 2
            noise = np.sum(x_psd[x_psd_ind[:id]])
            noise_db = 10*np.log10(noise)

        # fine search
        id = int(id/2)
        noise = np.sum(x_psd[x_psd_ind[:id]])
        noise_db = 10*np.log10(noise)
        while signal_db - noise_db > self.snr_db:
            id += 2 # real signal has symmetry in positive and negative frequencies
            noise = np.sum(x_psd[x_psd_ind[:id]])
            noise_db = 10*np.log10(noise)

        # zero out frequencies with lowest powers
        x_fft[x_psd_ind[:id-2]] = 0
        x_ifft = np.fft.ifft(x_fft)

        return np.real(x_ifft).astype(np.float32)

    def generate(self, x):
        x_out = np.empty((len(x),), dtype=object)
        for i, x_example in enumerate(x):
            if self.partial_attack:
                # split input into multiple segments and attack each with some probability
                x_adv = np.zeros_like(x_example)
                seg_len = self.attack_len
                for j in range(len(x_example)//seg_len+1):
                    xs = x_example[seg_len*j:min((j+1)*seg_len, len(x_example))]
                    if np.random.rand(1) < self.attack_prob:
                        xs = self._attack(xs)
                    x_adv[seg_len*j:min((j+1)*seg_len, len(x_example))] = xs
            else:
                x_adv = self._attack(x_example)
            x_out[i] = x_adv

        return x_out

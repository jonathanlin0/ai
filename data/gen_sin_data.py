import numpy as np

class GenSinData:
    """
    Generates sine wave data for RNN model training.
    """
    
    @staticmethod
    def gen_data(seq_length=100, num_samples=1000):
        X = []
        y = []
        for i in range(num_samples):
            x = np.linspace(i * 2 * np.pi, (i + 1) * 2 * np.pi, seq_length + 1)
            sine_wave = np.sin(x)
            X.append(sine_wave[:-1])  # input sequence
            y.append(sine_wave[1:])   # target sequence
        return np.array(X), np.array(y)
import numpy as np

def bandpower(data, sf, band):
    low, high = band
    freqs, psd = np.fft.rfftfreq(len(data), 1/sf), np.abs(np.fft.rfft(data))**2
    idx = (freqs >= low) & (freqs <= high)
    return psd[idx].mean()

def extract_features(raw):
    data = raw.get_data()
    sf = raw.info["sfreq"]

    features = []

    for channel in data:
        features.append([
            bandpower(channel, sf, (1,4)),   # delta
            bandpower(channel, sf, (4,8)),   # theta
            bandpower(channel, sf, (8,12)),  # alpha
            bandpower(channel, sf, (12,30))  # beta
        ])

    return np.mean(features, axis=0)
import mne

def preprocess(file_path):

    raw = mne.io.read_raw_bdf(
        file_path,
        preload=True
    )

    raw.filter(1., 40.)
    raw.notch_filter(50)

    raw.set_eeg_reference("average")

    return raw
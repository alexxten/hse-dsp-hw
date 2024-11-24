import numpy as np


def ks_note(sr, note, duration, decay):
    """
    Generates a sound wave using the Karplus-Strong algorithm.

    Parameters:
    sr (int): The sample rate.
    note (float): The note value, where 0 corresponds to A4 (440 Hz).
    duration (float): The duration of the note in seconds.
    decay (float): The decay factor for the algorithm, controlling the damping of the sound.

    Returns:
    numpy.ndarray: The generated sound wave as a numpy array.
    """

    # Calculate the frequency of the note
    freq = 440 * 2 ** (note / 12.0)

    # Calculate the length of the delay line
    T = int(sr / freq)

    # Initialize the delay line with random noise
    delay_line = np.random.rand(T)

    # Initialize the output buffer
    total_samples = int(sr * duration)
    output = np.zeros(total_samples)
    output[:T] = delay_line

    # Fill the output buffer using the Karplus-Strong algorithm
    for i in range(T, total_samples):
        output[i] = decay * (output[i - T] + output[i - T - 1]) / 2

    return output


def make_melody(filename, sixteenth_len, sr, note_function):
    """
    Parameters
    ----------
    filename: string
        Path to file containing the tune.  Consists of
        rows of <note number> <note duration>, where
        the note number 0 is a 440hz concert A, and the
        note duration is in factors of 16th notes
    sixteenth_len : float
        Duration of a sixteenth note in seconds. This parameter is used to convert the note durations
        from the file into actual time durations.
    sr: int
        Sample rate
    note_function : function
        Function to generate audio samples for a single note. It should take three parameters:
        sr (sample rate), note (note number or frequency), and duration (duration of the note in seconds),
        and return a NumPy array of audio samples.

    Returns
    -------
    np.ndarray
        Array of audio samples representing the generated melody.
    """
    melody = np.loadtxt(filename)
    notes = melody[:, 0]
    durations = sixteenth_len * melody[:, 1]

    # Initialize an empty list to store the audio samples
    audio_samples = []

    # Generate audio samples for each note and append to the list
    for note, duration in zip(notes, durations):
        if np.isnan(note):
            audio_samples.append(np.zeros(int(sr * duration)))
        else:
            audio_samples.append(note_function(sr, note, duration))

    # Concatenate all the audio samples into a single array
    melody = np.concatenate(audio_samples)

    return melody


def fm_note(
    sr,
    note,
    duration,
    ratio=2,
    I=2,
    envelope=lambda N, sr: np.ones(N),
    amplitude=lambda N, sr: np.ones(N),
):
    """
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    ratio: float
        Ratio of modulation frequency to carrier frequency
    I: float
        Modulation index (ratio of peak frequency deviation to
        modulation frequency)
    envelope: function (N, sr) -> ndarray(N)
        A function for generating an envelope profile
    amplitude: function (N, sr) -> ndarray(N)
        A function for generating a time-varying amplitude

    Returns
    -------
    ndarray(N): Audio samples for this note
    """
    # Calculate the number of samples
    num_samples = int(sr * duration)

    # Calculate the carrier frequency (fc)
    fc = 440 * 2 ** (note / 12.0)

    # Calculate the modulation frequency (fm)
    fm = fc * ratio

    # Generate the time array
    time_array = np.linspace(0, duration, num_samples)

    # Generate the envelopes for amplitude and modulation index
    amplitude_envelope = amplitude(num_samples, sr)
    modulation_envelope = envelope(num_samples, sr)

    # Generate the FM waveform
    mod_signal = modulation_envelope * np.sin(2 * np.pi * fm * time_array)
    fm_waveform = amplitude_envelope * np.sin(
        2 * np.pi * fc * time_array + I * mod_signal
    )

    return fm_waveform


def exp_env(N, sr, mu=3):
    """
    Make an exponential envelope
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate
    mu: float
        Exponential decay rate: e^{-mu*t}

    Returns
    -------
    ndarray(N): Envelope samples
    """
    return np.exp(-mu * np.arange(N) / sr)


def fm_string_note(sr, note, duration, mu=3):
    """
    Make a string of a particular length
    using FM synthesis
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    mu: float
        The decay rate of the note

    Returns
    -------
    ndarray(N): Audio samples for this note
    """
    envelope = lambda N, sr: exp_env(N, sr, mu)
    return fm_note(
        sr, note, duration, ratio=1, I=8, envelope=envelope, amplitude=envelope
    )


def fm_el_guitar_note(sr, note, duration, mu=3):
    """
    Make an electric guitar string of a particular length by
    passing along the parameters to fm_plucked_string note
    and then turning the samples into a square wave

    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    mu: float
        The decay rate of the note

    Return
    ------
    ndarray(N): Audio samples for this note
    """
    # Generate the plucked string sound
    string_sound = fm_string_note(sr, note, duration, mu)

    # Convert the plucked string sound to a square wave
    square_wave = np.where(string_sound > 0, 1, -1)

    return square_wave


def fm_bell_note(sr, note, duration):
    """
    Make a bell note of a particular length
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio

    Returns
    -------
    ndarray(N): Audio samples for this note
    """
    envelope = lambda N, sr: exp_env(N, sr, 0.8)
    return fm_note(
        sr, note, duration, ratio=1.4, I=2, envelope=envelope, amplitude=envelope
    )


def brass_env(N, sr):
    """
    Make the brass ADSR envelope from Chowning's paper

    Parameters
    ----------
    N : int
        The number of samples in the envelope.
    sr : int
        The sample rate, which is the number of samples per second.

    Returns
    -------
    ndarray
        An array of length N containing the envelope samples.
    """

    # Calculate the number of frames for each phase
    if N / sr < 0.3:
        num_sustain = 0
        num_attack, num_decay, num_release = int(N / 3), int(N / 3), int(N / 3)
    else:
        num_sustain = int(N * 0.7)
        num_attack, num_decay, num_release = int(N * 0.1), int(N * 0.1), int(N * 0.1)

    # Initialize the ADSR envelope array with nan
    adsr = np.full(N, np.nan)

    # Attack phase: linearly increase values from 0 to 1
    adsr[:num_attack] = np.linspace(0, 1, num_attack)

    # Decay phase: linearly decrease from 1 to 0.8
    adsr[num_attack : num_attack + num_decay] = np.linspace(1, 0.8, num_decay)

    # Sustain phase
    if num_sustain:
        adsr[num_attack + num_decay : num_attack + num_decay + num_sustain] = 0.8

    # Release phase: linearly decrease values from sustain to 0
    adsr[N - num_release :] = np.linspace(0.8, 0, num_release)

    return adsr


def fm_brass_note(sr, note, duration):
    """
    Make a brass note of a particular length
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio

    Return
    ------
    ndarray(N): Audio samples for this note
    """
    envelope = lambda N, sr: brass_env(N, sr)
    return fm_note(
        sr, note, duration, ratio=1, I=10, envelope=envelope, amplitude=envelope
    )


# изменила сигнатуру, чтобы переиспользовать для малого барабана
def drum_env(N, sr, mu: int = 35):
    """
    Make a drum envelope, according to Chowning's paper
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate

    Returns
    -------
    ndarray(N): Envelope samples
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ

    mu = 35
    t = np.linspace(0, N / sr, N)
    envelope = t**2 * np.exp(-mu * t)
    return envelope


def fm_drum_sound(sr, note, duration, fixed_note=-14):
    """
    Make what Chowning calls a "drum-like sound"
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number (which is ignored)
    duration: float
        Seconds of audio
    fixed_note: int
        Note number of the fixed note for this drum

    Returns
    ------
    ndarray(N): Audio samples for this drum hit
    """

    envelope = lambda N, sr: drum_env(N, sr)
    return fm_note(
        sr, fixed_note, duration, ratio=1.4, I=2, envelope=envelope, amplitude=envelope
    )


def snare_drum_sound(sr, note, duration):
    """
    Make a snare drum sound by shaping noise
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number (which is ignored)
    duration: float
        Seconds of audio

    Returns
    -------
    ndarray(N): Audio samples for this drum hit
    """
    noise_array = np.random.rand(int(sr * duration))
    return noise_array * drum_env(int(sr * duration), sr, mu=20)


def wood_drum_env(N, sr):
    """
    Make the wood-drum envelope from Chowning's paper
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate

    Returns
    -------
    ndarray(N): Envelope samples
    """
    adsr = np.full(N, np.nan)
    decay_num = int(N * 0.05)
    adsr[:decay_num] = np.linspace(1, 0, decay_num)
    adsr[decay_num:] = 0
    return adsr


def fm_wood_drum_sound(sr, note, duration, fixed_note=-14):
    """
    Make what Chowning calls a "wood drum sound"
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number (which is ignored)
    duration: float
        Seconds of audio
    fixed_note: int
        Note number of the fixed note for this drum

    Returns
    -------
    ndarray(N): Audio samples for this drum hit
    """
    envelope = lambda N, sr: wood_drum_env(N, sr)
    return fm_note(
        sr, fixed_note, duration, ratio=1.4, I=10, envelope=envelope, amplitude=envelope
    )


def dirty_bass_env(N, sr):
    """
    Make the "dirty bass" envelope

    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate

    Returns
    -------
    ndarray(N): Envelope samples
    """
    t = np.linspace(0, N / sr, int(N / 2))
    mu = 10

    adsr1 = np.exp(-mu * t)
    adsr2 = 1 - np.exp(-mu * t)
    adsr = np.concatenate((adsr1, adsr2))
    return adsr


def fm_dirty_bass_note(sr, note, duration):
    """
    Make a "dirty bass" note

    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number (which is ignored)
    duration: float
        Seconds of audio

    Returns
    -------
    ndarray(N): Audio samples for this drum hit
    """
    envelope = lambda N, sr: dirty_bass_env(N, sr)
    return fm_note(
        sr, note, duration, ratio=1, I=18, envelope=envelope, amplitude=envelope
    )

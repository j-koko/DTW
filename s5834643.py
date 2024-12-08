from scipy.fft import fft, dct
import numpy as np
import scipy
import os


def dtw(template, test):
    # initialize an empty matrix with zeroes for local distance computation
    local_distances = np.zeros((len(test), len(template)))

    for i in range(len(template)):
        for j in range(len(test)):
            # Compute local distance and create a matrix with results
            local_distances[j, i] = compute_local_distance(template[i], test[j])

    # Initialize the global distance matrix
    global_distance_matrix = np.zeros((len(test), len(template)))

    # Calculating global distance
    for i in range(len(test)):
        for j in range(len(template)):
            if i == 0 and j == 0:
                # Top-left corner (starting point)
                global_distance_matrix[i, j] = local_distances[i, j]
            elif i == 0 and j > 0:
                # First row (move only from the left)
                global_distance_matrix[i, j] = local_distances[i, j] + global_distance_matrix[i, j - 1]
            elif j == 0 and i > 0:
                # First column (move only from the top)
                global_distance_matrix[i, j] = local_distances[i, j] + global_distance_matrix[i - 1, j]
            else:
                # Rest of the matrix (top, left and diagonal moves)
                global_distance_matrix[i, j] = local_distances[i, j] + min(
                    global_distance_matrix[i - 1, j],      # From the top
                    global_distance_matrix[i, j - 1],      # From the left
                    global_distance_matrix[i - 1, j - 1]   # From the diagonal
                )

    # Initialize a matrix with indices to use it in final best path output
    indices_matrix = np.zeros((len(test), len(template)), dtype=object)

    for i in range(len(test)):
        for j in range(len(template)):
            indices_matrix[i, j] = (j+1, i+1)  # add ones to indices to get the format required by the exercise

    best_path = backtracking(global_distance_matrix, indices_matrix)
    final_global_distance = float(np.round(global_distance_matrix[-1, -1], 2))
    result_tuple = (final_global_distance, best_path)

    return result_tuple


def compute_local_distance(vector1, vector2):
    """Compute the Euclidean distance"""
    return np.linalg.norm(vector1 - vector2)


def backtracking(global_distance_matrix, indices_matrix):
    """Compute the path with the smallest global distance"""
    path = [indices_matrix[-1, -1]] # first element in the path is bottom-right corner (-1,-1)
    # we start backtracking with i and j equal to indices of bottom-right corner
    i, j = len(global_distance_matrix) - 1, len(global_distance_matrix[0]) - 1

    # while loop to go on until we reach the top-left corner (both j and i reach 0)
    while i > 0 or j > 0:
        # if we reach first row, there is only one path to the end = moving back
        if i == 0:
            path.append(indices_matrix[i, j - 1])
            j -= 1
        # if we reach first column, there is only one path to the end = moving up
        elif j == 0:
            path.append(indices_matrix[i - 1, j])
            i -= 1
        # otherwise three possible steps, I used tuples to associate indices with global distance matrix
        else:
            move_up = (global_distance_matrix[i - 1, j], indices_matrix[i - 1, j])
            move_diagonal = (global_distance_matrix[i - 1, j - 1], indices_matrix[i - 1, j - 1])
            move_back = (global_distance_matrix[i, j - 1], indices_matrix[i, j - 1])

            # pick one cell with the minimum global distance
            min_cost = min(move_up[0], move_diagonal[0], move_back[0])

            # move diagonal is the first condition, if it is equal to min, diagonal move will be added to the path
            if move_diagonal[0] == min_cost:
                path.append(move_diagonal[1])
                i -= 1
                j -= 1
            # if moving back is associated with the smallest global distance, choose those path indices
            elif move_back[0] == min_cost:
                path.append(move_back[1])
                j -= 1
            # otherwise moving up is the move with the lowest global distance
            else:
                path.append(move_up[1])
                i -= 1

    path.reverse()  # Reverse to get the order required by the specifications
    return path


def dtw_match(template_dict=None,
              test_dir=None):
    """Function that performs template matching"""

    # if not specified otherwise, use this default directory
    if test_dir is None:
        test_dir = "./test"

    # if not specified otherwise, use this dictionary of templates
    if template_dict is None:
        template_dict = {"hey_android": "./template/hey_android.wav",
                         "hey_snapdragon": "./template/hey_snapdragon.wav",
                         "hi_lumina": "./template/hi_lumina.wav",
                         "hi_galaxy": "./template/hi_galaxy.wav"}

    # Extract MFCCs for templates
    templates = {}
    for name, path in template_dict.items():
        # print(f"Loading template {name} from {path}")
        templates[name] = extract_mfcc_manual(path, 13) # dict with templates and their mfccs

    # Extract MFCCs for test files
    tests = {}
    for file_name in os.listdir(test_dir):
        file_path = os.path.join(test_dir, file_name)
        if os.path.isfile(file_path):
            #print(f"Loading test file {file_name} from {file_path}")
            tests[file_name] = extract_mfcc_manual(file_path, 13) # dict with test files and their mfccs

    # Compare each test file to all 4 templates
    results = {}
    for test_file, test_mfcc in tests.items():
        matches = {}

        for template_file, template_mfcc in templates.items():
            # print(f"Comparing Test File: {test_file} with Template: {template_file}")
            distance, path = dtw(template_mfcc, test_mfcc)
            matches[template_file] = distance # dict with template file and the associated global distance

        # Find the best match
        best_template = min(matches, key=matches.get) # choose the one with the lowest global distance

        # print(f"Test File: {test_file} -> Best Match: {best_template} (Distance: {best_distance})")
        results[test_file] = best_template # returns dict with test file and predicted template name as specified
        results = dict(sorted(results.items())) # sorts the dict to comply with specifications

    return results

# Deactivated function using librosa for mfcc extraction

# def extract_mfcc_librosa(file_path, n_mfcc=13):
#     y, sr = librosa.load(file_path, sr=None)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     return mfcc

def extract_mfcc_manual(file_path, num_ceps=13):
    sr, y = scipy.io.wavfile.read(file_path)
    # parameters to set before mfcc extraction
    pre_emphasis_param = 0.97
    frame_size = 0.025 # default value of 25ms
    frame_step = 0.01 # default value of 10ms
    n_fft = 512
    n_mels = 26
    # Pre-emphasis
    emphasized_signal = pre_emphasis(y, pre_emphasis_param)
    # Framing
    frames = frame_signal(emphasized_signal, frame_size, frame_step, sr)
    # Compute power spectrum
    power_spectrum = compute_power_spectrum(frames, n_fft)
    # Create Mel filter bank
    filter_bank = mel_filter_bank(sr, n_fft, n_mels)
    # Apply filter bank
    mel_energies = apply_filter_bank(power_spectrum, filter_bank)
    # Take log (sound intensity perception)
    log_mel_energies = np.log(mel_energies + 1e-10)  # To avoid undefined log(0)
    mfccs = compute_mfccs(log_mel_energies, num_ceps)

    return mfccs

def pre_emphasis(y, pre_emphasis_constant=0.97):
    """Vectorized pre-emphasis with default alpha constant set as 0.97"""
    emphasized_y = np.append(y[0], y[1:] - pre_emphasis_constant * y[:-1])
    return emphasized_y


def frame_signal(signal, frame_size, frame_step, sample_rate):
    """Creates frames and applies windowing (Hamming window)"""
    # Convert frame size and step from seconds to samples
    frame_length = int(frame_size * sample_rate)
    frame_step_samples = int(frame_step * sample_rate)

    # Calculate the number of frames
    num_frames = 1 + int((len(signal) - frame_length) / frame_step_samples)

    # Pad the signal if the last frame is incomplete
    pad_signal_length = num_frames * frame_step_samples + frame_length
    pad_signal = np.append(signal, np.zeros(pad_signal_length - len(signal)))

    # Create the frames using slicing and striding
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step_samples, frame_step_samples), (frame_length, 1)).T
    frames = pad_signal[indices]

    # Apply a Hamming window to each frame to reduce spectral leakage
    hamming_window = np.hamming(frame_length)
    frames *= hamming_window

    return frames


def compute_power_spectrum(frames, n_fft):
    """Compute the power spectrum of each frame."""
    fft_frames = np.abs(fft(frames, n=n_fft)) # Convert signal to the magnitude spectrum
    power_spectrum = (1.0 / n_fft) * (fft_frames ** 2)  # Power spectrum and normalization
    return power_spectrum[:, :n_fft // 2 + 1]  # Keep only positive frequencies


def mel_filter_bank(sample_rate, n_fft, n_mels=26):
    """ Create Mel filter bank"""
    # Convert frequencies (in Hz) to Mel scale
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700.0)
    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595.0) - 1)

    # Mel scale points
    mel_points = np.linspace(hz_to_mel(0), hz_to_mel(sample_rate / 2), n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Bin frequencies
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    filter_bank = np.zeros((n_mels, n_fft // 2 + 1))

    # Create triangular filters
    for m in range(1, n_mels + 1):
        f_m_minus = bins[m - 1]  # Left
        f_m = bins[m]  # Center
        f_m_plus = bins[m + 1]  # Right

        for k in range(f_m_minus, f_m):
            filter_bank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            filter_bank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    # The function return numpy array: filter bank matrix of the following shape (n_mels, n_fft//2 + 1)

    return filter_bank


def apply_filter_bank(power_spectrum, filter_bank):
    """Apply the Mel filter bank to the power spectrum"""
    return np.dot(power_spectrum, filter_bank.T)


def compute_mfccs(log_mel_energies, num_ceps=13):
    """Compute MFCCs using DCT. Number of cepstral coefficients (num_ceps) to return is set by default to 13"""
    mfccs = dct(log_mel_energies, type=2, axis=1, norm='ortho')[:, :num_ceps] # discard high-order coefficients
    return mfccs












import wave
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from windowing import hann, inverse_hann

"""This is my code for my ECEN4293 final project, which I have decided to call Sp3cktral. Sp3cktral is a spectral
gate, which is a very niche audio processing device that simply removes all frequency components that fall below 
a user-set threshold.

The code currently only works on 8- and 16-bit .wav files. In case you have any difficulty finding useable audio
files, I have included bassman.wav and encounter.wav, which are both 16-bit, 48kHz clips of songs I have released
that I have confirmed to properly load into the program. Also, attack and release are not working as intended, so
for the purest result, set these values to 0."""

def read_file(filename):
    """This function takes in a .wav file filename as a string and returns the metadata of the .wav file,
    as well as the data of the .wav file in a ndarray. This function was copied from 
    https://www.w3reference.com/blog/python-write-a-wav-file-into-numpy-float-array/ and slightly modified."""

    with wave.open(filename, mode='rb') as input_file:
        # Read wave file attributes
        num_channels = input_file.getnchannels()
        bit_depth = input_file.getsampwidth()
        sample_rate = input_file.getframerate()
        num_samples = input_file.getnframes()

        # Read raw binary
        raw_data = input_file.readframes(num_samples)

        # Convert bytes to integer array
        if bit_depth == 1:
            dtype = np.uint8
        elif bit_depth == 2:
            dtype = np.int16
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth} bytes")
        audio_int = np.frombuffer(raw_data, dtype=dtype)
            
        # Reshape for channels
        audio_int = np.reshape(audio_int, (-1, num_channels))

        # Normalize to [-1.0, 1.0]  
        if bit_depth == 1:  
            audio_float = (audio_int - 128) / 128.0  
        else:  
            max_val = np.iinfo(dtype).max  
            audio_float = audio_int.astype(np.float32) / max_val

        return audio_float, [num_channels, bit_depth, sample_rate, num_samples]
    
def write_file(filename, audio_float, metadata):
    """This function takes in a filename in string format, a ndarray of audio data, and some metadata
    pertaining to how the audio should be saved. The metadata is set and the audio data is formatted properly
    before being saved to a .wav file. This function was constructed as the reverse of the read_file
    function."""
    with wave.open(filename, mode='wb') as output_file:
        # Set wave file attributes
        output_file.setnchannels(metadata[0])
        output_file.setsampwidth(metadata[1])
        output_file.setframerate(metadata[2])
        # number of samples is not set here as this will be determined by the data written

        # Determine bit depth data type
        if output_file.getsampwidth() == 1:
            dtype = np.uint8
        elif output_file.getsampwidth() == 2:
            dtype = np.int16
        else:
            raise ValueError(f"Unsupported bit depth: {output_file.getsampwidth()} bytes")
        
        # Denormalize audio data
        if output_file.getsampwidth() == 1:
            audio_int = (128 * audio_float) + 128
        else:
            max_val = np.iinfo(dtype).max
            audio_int = audio_float * max_val
        audio_int = audio_int.astype(dtype)

        # Reshape 2 audio channels into 1-dimensional dataset, then convert to raw bytes
        audio_int = np.reshape(audio_int, (-1), order='F').tobytes()

        # Save raw data to file
        output_file.writeframes(audio_int)

def cooley_tukey(time_data: ndarray):
    """Performs the Cooley-Tukey FFT given some input time-domain data, based on the Wikipedia pseudocode
    and hashed out with assistance from Connor. This function is directly taken from my Lab 5 submission."""
    if not (float(np.log2(time_data.size)).is_integer()):
        raise ZeroDivisionError("Input time series is not a power of 2.")
    if (time_data.size == 1): # Base case
        return time_data
    else:
        frequency_data_evens = cooley_tukey(time_data[::2])
        frequency_data_odds = cooley_tukey(time_data[1::2])
        frequency_data = np.zeros_like(time_data, dtype="complex")
        for k in range(time_data.size // 2):
            p = frequency_data_evens[k]
            q = frequency_data_odds[k] * np.exp((-2j * np.pi * k) / time_data.size)
            frequency_data[k] = p + q
            frequency_data[k + (time_data.size // 2)] = p - q
        return frequency_data

if __name__ == "__main__":
    print("Welcome to Sp3cktral!")

    # File reading
    file_read = False
    while not file_read:
        input_filename = input("Please enter the name of the .wav file to be processed (file extension NOT included): ")
        try:
            filename_with_extension = input_filename + '.wav'
            input_data, input_metadata = read_file(filename_with_extension)
            print("File read successfully.")
            file_read = True
        except (FileNotFoundError, ValueError):
            print("Invalid filename provided. If the filename is correct, please check if it is the proper directory.")

    # Receive user input for threshold, attack, and release values, and whether or not to plot spectrograms
    threshold_read = False
    while not threshold_read:
        try:
            threshold = int(input("Please enter your desired threshold value in dB: "))
            # Recommend 5 dB for testing zero file rejection, -1000 for nearly no gating, and -10 for an interesting spot in the middle
            threshold_read = True
        except ValueError:
            print("Invalid threshold provided. Please provide a numerical value for the threshold.")
    
    attack_read = False
    while not attack_read:
        try:
            attack = float(input("Please enter your desired attack value in ms: "))
            # Recommend 20 ms for minimal attack, 500 ms for more extreme value
            attack_read = True
        except ValueError:
            print("Invalid attack provided. Please provide a numerical value for the attack.")

    release_read = False
    while not release_read:
        try:
            release = float(input("Please enter your desired release value in ms: "))
            # Recommend 50 ms for minimal release, 500 ms for more extreme value
            release_read = True
        except ValueError:
            print("Invalid release provided. Please provide a numerical value for the release.")

    plot_original_read = False
    while not plot_original_read:
        try:
            plot_original_yn = input("Would you like to plot the spectrogram of the original signal? (y/n): ").strip()[0].lower()
            if plot_original_yn == "y":
                plot_original = True
            else: # If the input is a string and is not y, assume n
                plot_original = False
            plot_original_read = True
        except:
            print("Invalid input provided. Please input either \"y\" or \"n\".")

    plot_processed_read = False
    while not plot_processed_read:
        try:
            plot_processed_yn = input("Would you like to plot the spectrogram of the original signal? (y/n): ").strip()[0].lower()
            if plot_processed_yn == "y":
                plot_processed = True
            else: # If the input is a string and is not y, assume n
                plot_processed = False
            plot_processed_read = True
        except:
            print("Invalid input provided. Please input either \"y\" or \"n\".")

    nuke_factor = 1e-20 # This controls what magnitude below-threshold components are set to

    # Portions of the following code were copied from Lab 5 and heavily edited. As outlined in my Lab 5 submission,
    # portions of the original code were written with assistance from ChatGPT.
    psds_original_l = [] # Spectrogram chunks of original signal's left channel
    psds_processed_l = [] # Spectrogram chunks of processed signal's left channel
    N = 1024 # samples per chunk of windowed audio
    overlap = N // 2 # number of samples that overlap between windows. Hardcoded to the integer half of N to facilitate proper signal recovery
    
    threshold_scaled = N * (10 ** (threshold / 20)) # Threshold in absolute units scaled by the number of samples per chunk
    attack_frames = round((input_metadata[2] / N) * (attack * (10 ** -3))) # (Sample rate / samples per frame) * attack time. Used to convert attack from ms to # of frames
    release_frames = round((input_metadata[2] / N) * (release * (10 ** -3))) # (Sample rate / samples per frame) * release time. Used to convert release from ms to # of frames
    print(f"Attack frames: {attack_frames}, Release frames: {release_frames}")

    # scale attack and release frame lengths to allow for transition frames in the gradients below
    attack_frames += 1
    release_frames += 1

    if (attack_frames > 1):
        attack_gradient = (np.arange(attack_frames) + 1) / attack_frames # Calculate transition values for attack fades
    if (release_frames > 1):
        release_gradient = (np.arange(release_frames) + 1) / release_frames # Calculate transition values for release fades

    fourier_chunks_l = [] # Left channel chunks after they've been FFT'd
    fourier_chunks_r = [] # Right channel chunks after they've been FFT'd
    inversed_chunks_l = [] # Left channel chunks after they've been IFFT'd
    inversed_chunks_r = [] # Right channel chunks after they've been IFFT'd

    # Windowing, FFT, and gating
    # LEFT CHANNEL
    num_chunks = 0
    for chunk in hann(input_data[:, 0], N):
        num_chunks += 1
        print(f"Transforming and gating left chunk {num_chunks}...")
        # Compute the fast Fourier transform (FFT) of this chunk
        X = cooley_tukey(chunk)

        # Compute the power spectral density (PSD) of this chunk
        # PSD is 10*log10 of the square of the real part of the FFT
        psd = 20*np.log10(np.abs(X[0:N//2]))
        psds_original_l.append(psd)

        # The below two lines are inspired by some Claude-written code (link to conversation:
        # https://claude.ai/share/53b0d7b8-95f5-43e0-acf7-cdbd656ae1bc). The first line simply takes the
        # magnitude of the chunk's FFT, while the second uses that magnitude along with the threshold to generate
        # a mask of all values where the magnitude is less than the threshold. This mask is then used to set all
        # the magnitudes of all bins with magnitudes less than the threshold to the nuke factor 1e-20, or virtually
        # zero.
        magnitude = np.abs(X)
        X[magnitude < threshold_scaled] = nuke_factor

        fourier_chunks_l.append(X)
    psds_original_l = np.array(psds_original_l).transpose()

    # RIGHT CHANNEL
    num_chunks = 0
    for chunk in hann(input_data[:, 1], N):
        num_chunks += 1
        print(f"Transforming and gating right chunk {num_chunks}...")
        # Compute the fast Fourier transform (FFT) of this chunk
        X = cooley_tukey(chunk)

        # The below two lines are inspired by some Claude-written code (link to conversation:
        # https://claude.ai/share/53b0d7b8-95f5-43e0-acf7-cdbd656ae1bc). The first line simply takes the
        # magnitude of the chunk's FFT, while the second uses that magnitude along with the threshold to generate
        # a mask of all values where the magnitude is less than the threshold. This mask is then used to set all
        # the magnitudes of all bins with magnitudes less than the threshold to the nuke factor 1e-20, or virtually
        # zero.
        magnitude = np.abs(X)
        X[magnitude < threshold_scaled] = nuke_factor

        fourier_chunks_r.append(X)
    print("Transforming and gating completed.")

    # Apply attack and release fades
    print("Applying attack and release fades...")

    # Create copies of the chunk lists to prevent implemented processing from interfering with later processing
    faded_chunks_l = fourier_chunks_l.copy()
    faded_chunks_r = fourier_chunks_r.copy()

    if (attack_frames > 1) or (release_frames > 1): # If meaningfully long attack and release lengths have been provided
        for frame in range(len(fourier_chunks_l) - 1): # Run loop on all frames but last
            for bin in range(fourier_chunks_l[0].size): # Run loop on each bin in the frame

                # LEFT CHANNEL
                # If sudden transition from nuke factor to any greater magnitude, apply attack fade
                if (fourier_chunks_l[frame][bin] <= nuke_factor) and (fourier_chunks_l[frame + 1][bin] > nuke_factor) and (attack_frames > 1):
                    for scale_frame in range(attack_frames):
                        try:
                            faded_chunks_l[frame + scale_frame - 1][bin] *= attack_gradient[scale_frame]
                        except IndexError: # Ignore index errors
                            pass
                # If sudden transition from greater magnitude to any nuke factor, apply release fade
                if (fourier_chunks_l[frame][bin] > nuke_factor) and (fourier_chunks_l[frame + 1][bin] <= nuke_factor) and (release_frames > 1):
                    for scale_frame in range(release_frames):
                        try:
                            faded_chunks_l[frame - scale_frame + 1][bin] *= release_gradient[scale_frame]
                        except IndexError:
                            pass

                # RIGHT CHANNEL
                if (fourier_chunks_r[frame][bin] <= nuke_factor) and (fourier_chunks_r[frame + 1][bin] > nuke_factor) and (attack_frames > 1):
                    for scale_frame in range(attack_frames):
                        try:
                            faded_chunks_r[frame + scale_frame - 1][bin] *= attack_gradient[scale_frame]
                        except IndexError:
                            pass
                if (fourier_chunks_r[frame][bin] > nuke_factor) and (fourier_chunks_r[frame + 1][bin] <= nuke_factor) and (release_frames > 1):
                    for scale_frame in range(release_frames):
                        try:
                            faded_chunks_r[frame - scale_frame + 1][bin] *= release_gradient[scale_frame]
                        except IndexError:
                            pass

    # Save faded chunks back to original chunks
    fourier_chunks_l = faded_chunks_l
    fourier_chunks_r = faded_chunks_r
    print("Attack and release applied.")

    # Inverse Fourier Transform and combine chunks
    # LEFT CHANNEL
    num_chunks = 0
    for chunk in fourier_chunks_l:
        num_chunks += 1
        print(f"Inverse transforming left chunk {num_chunks}...")
        # Compute the IFFT
        X_inv = np.real(np.fft.ifft(chunk))
        inversed_chunks_l.append(X_inv)

        # Compute the power spectral density (PSD) of this chunk
        # PSD is 10*log10 of the square of the real part of the FFT
        psd = 10*np.log10(np.abs(chunk[0:N//2])**2)
        psds_processed_l.append(psd)
    psds_processed_l = np.array(psds_processed_l).transpose()

    # RIGHT CHANNEL
    num_chunks = 0
    for chunk in fourier_chunks_r:
        num_chunks += 1
        print(f"Inverse transforming right chunk {num_chunks}...")
        # Compute the IFFT
        X_inv = np.real(np.fft.ifft(chunk))
        inversed_chunks_r.append(X_inv)
    print("Inverse transforming completed.")

    # Revert the effects of windowing on both channels
    processed_l = inverse_hann(inversed_chunks_l)
    processed_r = inverse_hann(inversed_chunks_r)
    print("Chunks recombined.")
    # Recombine left and right channels
    processed_data = np.concatenate((np.reshape(processed_l, (1, -1)), np.reshape(processed_r, (1, -1))), axis=0)
    print("Channels recombined.")
    print("Processing completed.")

    if processed_data.max() <= nuke_factor: # If the output signal has virtually zero amplitude
        print("The output audio signal has zero amplitude and will not be saved. Thank you for using Sp3cktral!")
    else:
        # Save output signal to .wav file
        file_written = False
        while not file_written:
            output_filename = input("Please enter the desired name to save the processed audio file under (file extension NOT included): ")
            try:
                write_file(output_filename + '.wav', processed_data, input_metadata)
                file_written = True
            except PermissionError as pe:
                print("Permission Error:", pe)

        print(f"File {output_filename}.wav saved successfully.")
        print("Thank you for using Sp3cktral!")

        # Plot spectrograms of input and processed left channels if requested by user
        if plot_original or plot_processed:
            if not plot_processed:
                fig, ax = plt.subplots()

                # Set up extents for frequencies and times to be properly notated on the spectrogram
                freqs = np.linspace(0, input_metadata[2]/2, N//2)
                times = np.linspace(0, input_metadata[3]/input_metadata[2], psds_original_l.shape[1])

                ax.imshow(psds_original_l, aspect='auto', origin='lower', extent=[times[0], times[-1], freqs[0], freqs[-1]])

                ax.set_title("Input Left Channel Spectrogram")
                ax.set_ylabel("Frequency (Hz)")
                ax.set_xlabel("Time (s)")
                plt.show()

            elif not plot_original:
                fig, ax = plt.subplots()

                freqs = np.linspace(0, input_metadata[2]/2, N//2)
                times = np.linspace(0, input_metadata[3]/input_metadata[2], psds_original_l.shape[1])

                ax.imshow(psds_processed_l, aspect='auto', origin='lower', extent=[times[0], times[-1], freqs[0], freqs[-1]])

                ax.set_title("Processed Left Channel Spectrogram")
                ax.set_ylabel("Frequency (Hz)")
                ax.set_xlabel("Time (s)")
                plt.show()

            else:
                fig, ax = plt.subplots(ncols=2, sharex='all', sharey='all') # Share x and y coordinates for ease of comparison

                freqs = np.linspace(0, input_metadata[2]/2, N//2)
                times = np.linspace(0, input_metadata[3]/input_metadata[2], psds_original_l.shape[1])

                ax[0].imshow(psds_original_l, aspect='auto', origin='lower', extent=[times[0], times[-1], freqs[0], freqs[-1]])
                ax[1].imshow(psds_processed_l, aspect='auto', origin='lower', extent=[times[0], times[-1], freqs[0], freqs[-1]])

                ax[0].set_title("Input Left Channel Spectrogram")
                ax[1].set_title("Processed Left Channel Spectrogram")
                ax[0].set_ylabel("Frequency (Hz)")
                ax[0].set_xlabel("Time (s)")
                ax[1].set_xlabel("Time (s)")
                plt.show()
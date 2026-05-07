"""Windowing functions for digital signal processing"""

import numpy as np

# I was having some issues with periodic high amplitude peaks appearing in my signal, so I had Claude troubleshoot
# both this file and the main.py file (link to conversation: https://claude.ai/share/14dc1d9c-e904-4afe-a757-e4c8ceedc883).
# The primary issues with the high amplitude peaks came for the windowing and reverse windowing functions here.
# The main issue was improper normalization to account for the overlap size. This was resolved by changing the overlap size
# to half of the window_length and directly adding the windowed chunks together instead of dividing by the windowing function.
# The original faulty functions are included (commented out) for reference.

def hann(x, window_length=1024, zero_pad=True):
    overlap = window_length // 2 # Hardcode the overlap to half the window length
    if overlap < 0: overlap = 0 # Ignore negative overlap (unnecessary but not harmful)
    stride = window_length - overlap  # correct stride regardless of overlap

    x = x.reshape((x.size,))
    if zero_pad:
        x = np.concatenate((np.zeros(overlap), x, np.zeros(stride + overlap))) # Add window_length + overlap zeros to the signal

    W = 0.5 * (1 - np.cos(2*np.pi*np.arange(window_length)/window_length))

    # The below while loop is identical in operation to the for loop in the faulty function; it's just more readable.
    i = 0
    while True:
        t_start = stride * i
        t_end = t_start + window_length
        if t_end > x.size:
            break
        yield W * x[t_start:t_end]
        i += 1

# def hann(x, window_length=1024, overlap=512, zero_pad=True):
#     """Perform Hann windowing on the sampled signal x.

#     Given a discrete time series x, yield windowed samples from the time series. The Hann window
#     function is applied to reduce tail effects.

#     :param x: A numpy array of time-series data. Should have shape (N,1), (1,N), or (N,).
#     :param window_length: How many samples to yield in each window. Default 1024.
#     :param overlap: How many samples to overlap from one window to the next. Default 256.
#     :param zero_pad: Whether to pad the input signal with zeros (front and back). Default True.
#     """
#     # Silently ignore negative overlap values
#     if overlap < 0:
#         overlap = 0
#     kernel_length = window_length - 2*overlap

#     # Zero-pad the data (if requested)
#     x = x.reshape((x.size,))
#     if zero_pad:
#         x = np.concatenate((np.zeros(overlap), x, np.zeros(kernel_length + overlap)))

#     # Precompute the Hann windowing function
#     W = 1/2 * (1 - np.cos(2*np.pi*np.arange(window_length)/window_length))

#     # Yield windowed data segments
#     for i in range(x.size // (kernel_length + overlap)):
#         t_start = (kernel_length+overlap)*i
#         t_end = t_start + window_length
#         yield W * x[t_start:t_end]

def inverse_hann(hann_x, zero_pad=True):
    window_length = hann_x[0].shape[0]
    overlap = window_length // 2 # Hardcoded overlap relative to window length
    stride = window_length - overlap
    output_length = stride * len(hann_x) + overlap
    
    x   = np.zeros(output_length) # Identical to the corresponding statement in the faulty function, just more readable
    # The norm is simply all of the window functions at each window added together, such that they can be removed from the recombined chunks
    norm = np.zeros(output_length) 
    W = 0.5 * (1 - np.cos(2 * np.pi * np.arange(window_length) / window_length))
    
    for i, chunk in enumerate(hann_x):
        start = i * stride # Find start point by multiplying current chunk number and stride amount
        x[start:start + window_length]    += np.real(chunk) # Add chunk to recombined signal (use np.real to remove negligible complex part)
        norm[start:start + window_length] += W
    
    # Avoid dividing by zero at the very edges where norm=0. This does mean that the edges do exhibit some slightly erratic behavior sometimes.
    norm = np.where(norm < 1e-8, 1.0, norm)
    x /= norm # Remove norm by division
    
    if zero_pad:
        return x[overlap:]
    return x

# def inverse_hann(hann_x, overlap=256, zero_pad=True):
#     """Revert the effects of Hann windowing on a list of windowed signals hann_x."""
#     window_length = hann_x[0].shape[0]
#     x = np.zeros((len(hann_x) * window_length - (len(hann_x) - 1) * overlap), dtype='complex128')
#     W = 1/2 * (1 - np.cos(2*np.pi*np.arange(window_length)/window_length)) # Provided Hann windowing function
#     for chunk in range(len(hann_x)):
#         window_x = hann_x[chunk] / W # Divide chunk by windowing function
#         window_x[window_length - overlap:] = np.zeros(overlap) # Zero the overlap portion of the chunk
#         prior_space = np.zeros(((window_length - overlap) * chunk))
#         post_space = np.zeros((x.size - (window_length * (chunk + 1)) + (overlap * chunk)))
#         extended_window_x = np.concatenate((prior_space, window_x, post_space)) # Zero-pad the chunk to place it in the right location in the signal
#         x += extended_window_x # Add the chunk to the reconstructed signal
#         if chunk != 0: # Fill in null points created by zero division with average of two neighboring points
#             x[(window_length - overlap) * chunk] = (x[(window_length - overlap) * chunk - 1] + x[(window_length - overlap) * chunk + 1]) / 2
#     if zero_pad:
#         return x[overlap:] # (x.size - window_length + overlap)] # Remove beginning zero-padding
#     else:
#         return x

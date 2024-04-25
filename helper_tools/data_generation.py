import numpy as  np

# Define functions for generating sine waves

# Generate time series data
def low_freq_low_amp(time_stamps):
    data = np.sin(2 * np.pi * time_stamps * 0.5) * 0.2
    return data


def high_freq_low_amp(time_stamps):
    return np.sin(2 * np.pi * time_stamps * 5) * 0.1


def low_freq_high_amp(time_stamps):
    return np.sin(2 * np.pi * time_stamps) * 2  # Increased frequency for more cycles


def high_freq_high_amp(time_stamps):
    return np.sin(2 * np.pi * time_stamps * 5) * 1


# Define functions for increasing amplitude with similar shape
def increasing_amp_smooth(time_stamps):
    return np.sin(2 * np.pi * time_stamps * 5) * (time_stamps ** 2 + 0.2)  # Smooth increase


def increasing_amp_linear(time_stamps):
    return np.sin(2 * np.pi * time_stamps * 5) * (time_stamps * 0.1 + 1)  # Linear increase


def parabola(time_stamps):
    return -4 * (time_stamps - 0.5) ** 2 + 1  # Increased coefficient for stronger curvature

def generate_curve_variants(base_function, num_variants, time_stamps):
    """
    This function generates `num_variants` minor variations of the provided `base_function`.

    Args:
        base_function: A function that takes time stamps as input and returns a curve.
        num_variants: The number of variants to generate.

    Returns:
        A list of functions, where each function generates a minor variant of the base curve.
    """
    variants = []
    for _ in range(num_variants):
        # Introduce minor variations through scaling, shifting, adding noise, etc.
        # Here are some examples:
        # - Scale amplitude: new_amplitude = base_amplitude * (1 + random.uniform(-0.1, 0.1))
        # - Shift frequency: new_frequency = base_frequency * (1 + random.uniform(-0.05, 0.05))
        # - Add white noise: new_curve = base_function(t) + random.gauss(0, 0.01)

        # You can customize the type and magnitude of variations based on your needs.
        new_curve = base_function(time_stamps) * (1 + np.random.uniform(-0.1, 0.1))  # Example: Scale amplitude slightly
        variants.append(new_curve)
    return variants


def generate_all_data(time_stamps, extend_dims = True):
    """
    This function generates all curves and their variants.

    Args:
        time_stamps: A list of time stamps.

    Returns:
        A NumPy array containing all generated curves.
    """
    data = []
    base_functions = [low_freq_low_amp, high_freq_low_amp, low_freq_high_amp,
                      high_freq_high_amp, increasing_amp_smooth, increasing_amp_linear, parabola]
    for func in base_functions:
        # Generate variants for each base curve
        variants = generate_curve_variants(func, num_variants=20, time_stamps=time_stamps)
        data.extend([func(time_stamps)] + variants)
    if extend_dims:
        return np.expand_dims(np.vstack(data), axis=-1)
    else:
        return np.vstack(data)


# Generate and save data
time_stamps = np.linspace(0, 1, 20)
data = generate_all_data(time_stamps, extend_dims=True)
np.save('../data/sample.npy', data.astype(np.float32))

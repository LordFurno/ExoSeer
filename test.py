import torch

def min_max_scaling(data, min_val, max_val):
    """
    Perform Min-Max scaling on the input data.

    Parameters:
    - data: Torch tensor or numpy array containing the original data.
    - min_val: Minimum value for normalization.
    - max_val: Maximum value for normalization.

    Returns:
    - Normalized data tensor.
    """
    scaled_data = (data - data.min()) / (data.max() - data.min())  # Min-Max scaling formula
    scaled_data = scaled_data * (max_val - min_val) + min_val  # Scale to the desired range
    return scaled_data

# Example usage:
original_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
min_val = 0.0
max_val = 1.0

normalized_data = min_max_scaling(original_data, min_val, max_val)
print(normalized_data)

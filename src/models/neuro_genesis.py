import torch
import random
import numpy as np
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)
def get_phi(m, D, which_phi='performer', device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Function that returns the random feature map, phi.
    Since our neuron-astrocyte model is equivalent to using Random Feature Attention,
    we use this representation for simplicity. Different phi functions lead to different feature maps.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Random weight matrix for random feature map
    W = torch.randn((m, D), device=device)

    if which_phi == 'cosine':
        # Random biases for cosine feature map
        rand_b = torch.rand(m, device=device) * 2 * torch.pi

        def phi(x, c=0):
            """Uses a cosine random feature map to approximate softmax attention."""
            return torch.sqrt(2 / m) * torch.cos(W @ x + rand_b) * torch.exp(0.5 * (torch.norm(x) ** 2) - c)

    elif which_phi == 'performer':
        def phi(x, c=0):
            """Uses an exponential random feature map to approximate softmax attention."""
            return torch.exp(-0.5 * torch.log(torch.tensor(m, device=device)) + W @ x - 0.5 * (torch.norm(x) ** 2))

    elif which_phi == 'linear':
        def phi(x, c=0):
            """Uses a linear random feature map to approximate softmax attention."""
            h = -0.5 * torch.log(torch.tensor(m, device=device)) + W @ x - 0.5 * (torch.norm(x) ** 2)
            return 1 + h

    elif which_phi == 'truncated_performer':
        def phi(x, thresh=150):
            """Uses an exponential random feature map to approximate softmax attention."""
            scaling_factors = torch.exp(-0.5 * torch.log(torch.tensor(m, device=device)) - 0.5 * (torch.norm(x) ** 2))
            h = torch.exp(W @ x)
            return scaling_factors * torch.clamp(h, min=0, max=thresh)

    elif which_phi == 'positive_cosine':
        # Random biases for cosine feature map
        rand_b = torch.rand(m, device=device) * 2 * torch.pi

        def phi(x, thresh=10):
            """Uses a positive cosine random feature map to approximate softmax attention."""
            scaling_factors = torch.sqrt(2 / (torch.pi * m)) * torch.exp(0.5 * (torch.norm(x) ** 2))
            h = torch.cos(W @ x + rand_b)
            return torch.clamp(scaling_factors * h, min=0)

    elif which_phi == 'dima_sin':
        # Random biases for cosine feature map
        rand_b = torch.rand(m, device=device) * 2 * torch.pi

        def clipped_sin(x):
            """Clips the sine values."""
            return torch.where(x > torch.pi / 2, 1, torch.where(x < -torch.pi / 2, -1, torch.sin(x)))

        def phi(x, thresh=10):
            """Uses a sine-based random feature map to approximate softmax attention."""
            scaling_factors = torch.sqrt(2 / m) * torch.exp(0.5 * (torch.norm(x) ** 2))
            h = clipped_sin(W @ x + rand_b)
            return scaling_factors * h

    else:
        raise ValueError(f"Unknown phi type: {which_phi}")

    return phi




def get_astro_responses(query_layer, key_layer, nhead, phi):
    """
    Computes astrocyte response given a random feature map, queries, and keys.

    Args:
        query_layer: Tensor of shape (n_sample, ntokens, dim)
        key_layer: Tensor of shape (nhead, ntokens, dim)
        nhead: Integer index for the current head
        phi: Function to apply to the keys and queries

    Returns:
        Tensor of shape (n_sample, ntokens, ntokens) representing astro_pulses.
    """
    # Get the device of the query_layer
    device = query_layer.device

    # Apply phi to the key layer for the specified head
    # key_layer[nhead] has shape (ntokens, dim)
    # rfa_normalized_keys will have shape (ntokens, m)
    rfa_normalized_keys = phi(key_layer[nhead])

    # Apply phi to the query layer
    # query_layer has shape (n_sample, ntokens, dim)
    # transformed_queries will have shape (n_sample, ntokens, m)
    transformed_queries = phi(query_layer)


    rfa_normalized_keys = rfa_normalized_keys.T

    # Perform batched matrix multiplication
    astro_pulses = torch.matmul(transformed_queries, rfa_normalized_keys)

    return astro_pulses


def neurogenesis(head_size, query_layer, key_layer, nhead):
    # Normalize Q and K matrices appropriately
    query_layer = query_layer / head_size ** (1/4)
    key_layer = key_layer / head_size ** (1/4)

    # Ensure tensors are on GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    query_layer = query_layer.to(device)
    key_layer = key_layer.to(device)


    phi_low_m = get_phi(m=512, D=head_size, which_phi='performer')


    astro_ps_low_m = get_astro_responses(query_layer, key_layer, 0, phi_low_m)

    # Move the result back to CPU for further processing or conversion to NumPy
    return astro_ps_low_m.cpu().detach().numpy()
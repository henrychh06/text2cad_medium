# cad_processing/utils/utils.py
import numpy as np
import os

def create_point_from_array(arr):
    """Create a point from a numpy array."""
    return arr

def point_distance(point1, point2, type="l2"):
    """Calculate distance between two points."""
    if type == "l1":
        return np.sum(np.abs(point1 - point2))
    else:
        return np.sqrt(np.sum((point1 - point2)**2))

def get_orientation(p1, p2, p3):
    """Calculate the orientation of three points (clockwise or counterclockwise)."""
    val = (p2[1] - p1[1]) * (p3[0] - p2[0]) - (p2[0] - p1[0]) * (p3[1] - p2[1])
    
    if val == 0:
        return "collinear"
    elif val > 0:
        return "clockwise"
    else:
        return "counterclockwise"

def quantize(coords, n_bits=8, min_range=-1, max_range=1):
    """Quantize floating point coordinates to integer values."""
    size = 2**n_bits - 1
    quantized = (coords - min_range) / (max_range - min_range) * size
    return np.round(quantized).astype(np.int32)

def dequantize_verts(verts, n_bits=8, min_range=-1, max_range=1):
    """Convert quantized values back to floating point."""
    size = 2**n_bits - 1
    dequantized = verts / size * (max_range - min_range) + min_range
    return dequantized

def float_round(arr, decimals=6):
    """Round floating point values to specified decimals."""
    if isinstance(arr, (list, tuple)):
        return [round(float(x), decimals) for x in arr]
    elif isinstance(arr, np.ndarray):
        return np.round(arr.astype(float), decimals)
    else:
        return round(float(arr), decimals)

def int_round(arr):
    """Round to integers."""
    if isinstance(arr, (list, tuple)):
        return [int(round(x)) for x in arr]
    elif isinstance(arr, np.ndarray):
        return np.round(arr).astype(np.int32)
    else:
        return int(round(arr))

def ensure_dir(directory):
    """Ensure a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def split_array(arr, val):
    """Split array based on a specific value."""
    result = []
    temp = []
    
    for item in arr:
        if isinstance(item, list) and item[0] == val:
            if temp:
                result.append(temp)
                temp = []
        else:
            temp.append(item)
    
    if temp:
        result.append(temp)
    
    return result

def merge_list(lists):
    """Merge overlapping lists."""
    if not lists:
        return []
    
    result = []
    current = lists[0]
    
    for i in range(1, len(lists)):
        if lists[i][0] in current:
            current.extend([x for x in lists[i] if x not in current])
        else:
            result.append(current)
            current = lists[i]
    
    result.append(current)
    return result

def flatten(list_of_lists):
    """Flatten a list of lists."""
    return [item for sublist in list_of_lists for item in sublist]

def random_sample_points(points, n_points):
    """Randomly sample n_points from the given points."""
    if len(points) <= n_points:
        return points, np.arange(len(points))
    
    indices = np.random.choice(len(points), n_points, replace=False)
    return points[indices], indices

def merge_end_tokens_from_loop(vec):
    """Merge tokens with end tokens."""
    result = []
    temp = []
    
    for item in vec:
        if isinstance(item, list) and item[0] >= 5:  # END_CURVE or higher
            if temp:
                result.append(temp)
                temp = []
        else:
            temp.append(item)
    
    if temp:
        result.append(temp)
    
    return result, vec

def generate_attention_mask(seq_len):
    """Generate a causal attention mask."""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask
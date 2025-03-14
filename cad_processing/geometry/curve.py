# cad_processing/geometry/curve.py
import numpy as np
import matplotlib.pyplot as plt

class Curve:
    """Base class for all curve types in the CAD representation."""
    
    def __init__(self, metadata=None):
        self.metadata = metadata or {}
        self.is_numerical = False
        
    @property
    def curve_type(self):
        """Return the type of curve."""
        raise NotImplementedError("Subclass must implement curve_type")
    
    @property
    def start_point(self):
        """Return the start point of the curve."""
        raise NotImplementedError("Subclass must implement start_point")
    
    @property
    def bbox(self):
        """Return the bounding box of the curve as [[min_x, min_y], [max_x, max_y]]."""
        raise NotImplementedError("Subclass must implement bbox")
        
    def sample_points(self, n_points=32):
        """Sample points along the curve."""
        raise NotImplementedError("Subclass must implement sample_points")
        
    def transform(self, translate, scale=1.0):
        """Transform the curve by translation and scaling."""
        raise NotImplementedError("Subclass must implement transform")
        
    def to_dict(self):
        """Convert to dictionary for serialization."""
        raise NotImplementedError("Subclass must implement to_dict")
    
    def curve_distance(self, other_curve, scale=1.0):
        """Calculate distance between two curves."""
        # Default implementation using sampling
        points_self = self.sample_points(n_points=64)
        points_other = other_curve.sample_points(n_points=64)
        
        # Calculate minimum distances between point sets
        min_distances = []
        for p1 in points_self:
            distances = np.sqrt(np.sum((points_other - p1)**2, axis=1))
            min_distances.append(np.min(distances))
        
        return np.mean(min_distances) / scale
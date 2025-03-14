# cad_processing/geometry/line.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from .curve import Curve

class Line(Curve):
    """A line segment defined by start and end points."""
    
    def __init__(self, metadata=None):
        super().__init__(metadata)
        if metadata is None:
            self.metadata = {
                "start_point": np.zeros(2, dtype=np.float32),
                "end_point": np.zeros(2, dtype=np.float32)
            }
    
    @staticmethod
    def from_dict(line_dict):
        """Create a Line from a dictionary representation."""
        metadata = {}
        metadata["start_point"] = np.array(
            [line_dict["start_point"]["x"], line_dict["start_point"]["y"]]
        )
        metadata["end_point"] = np.array(
            [line_dict["end_point"]["x"], line_dict["end_point"]["y"]]
        )
        return Line(metadata)
    
    @property
    def curve_type(self):
        return "line"
    
    @property
    def start_point(self):
        return self.metadata["start_point"]
    
    @property
    def end_point(self):
        return self.metadata["end_point"]
    
    def get_point(self, point_type):
        """Get a specific point from the metadata."""
        return self.metadata[point_type]
    
    @property
    def bbox(self):
        """Return the bounding box as min/max points."""
        points = np.stack([self.start_point, self.end_point], axis=0)
        return np.stack([np.min(points, axis=0), np.max(points, axis=0)], axis=0)
    
    @property
    def bbox_size(self):
        """Return the maximum dimension of the bounding box."""
        bbox_size = np.max(np.abs(self.bbox[1] - self.bbox[0]))
        return max(bbox_size, 1.0)  # Avoid division by zero
    
    def transform(self, translate, scale=1.0):
        """Transform the line by translation and scaling."""
        self.metadata["start_point"] = (self.metadata["start_point"] + translate) * scale
        self.metadata["end_point"] = (self.metadata["end_point"] + translate) * scale
    
    def sample_points(self, n_points=32):
        """Sample points along the line."""
        return np.linspace(self.start_point, self.end_point, num=n_points)
    
    def reverse(self):
        """Reverse the line direction."""
        self.metadata["start_point"], self.metadata["end_point"] = (
            self.metadata["end_point"],
            self.metadata["start_point"]
        )
    
    def draw(self, ax=None, color="black"):
        """Draw the line on a matplotlib axis."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        xdata = [self.start_point[0], self.end_point[0]]
        ydata = [self.start_point[1], self.end_point[1]]
        line = mlines.Line2D(xdata, ydata, lw=1, color=color)
        ax.add_line(line)
        return ax
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "type": "line",
            "start_point": {"x": float(self.start_point[0]), "y": float(self.start_point[1])},
            "end_point": {"x": float(self.end_point[0]), "y": float(self.end_point[1])}
        }
    
    def __repr__(self):
        return f"Line: Start({self.start_point.round(4)}), End({self.end_point.round(4)})"
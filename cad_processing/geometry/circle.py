# cad_processing/geometry/circle.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .curve import Curve

class Circle(Curve):
    """A circle defined by center point and radius."""
    
    def __init__(self, metadata=None):
        super().__init__(metadata)
        if metadata is None:
            self.metadata = {
                "center": np.zeros(2, dtype=np.float32),
                "radius": 0.0,
            }
            self._compute_points()
    
    def _compute_points(self):
        """Compute points on the circle for representation."""
        self.metadata["pt1"] = np.array(
            [self.metadata["center"][0], self.metadata["center"][1] + self.metadata["radius"]]
        )
        self.metadata["pt2"] = np.array(
            [self.metadata["center"][0], self.metadata["center"][1] - self.metadata["radius"]]
        )
        self.metadata["pt3"] = np.array(
            [self.metadata["center"][0] + self.metadata["radius"], self.metadata["center"][1]]
        )
        self.metadata["pt4"] = np.array(
            [self.metadata["center"][0] - self.metadata["radius"], self.metadata["center"][1]]
        )
    
    @staticmethod
    def from_dict(circle_dict):
        """Create a Circle from a dictionary representation."""
        metadata = {
            "center": np.array(
                [circle_dict["center_point"]["x"], circle_dict["center_point"]["y"]]
            ),
            "radius": circle_dict["radius"],
            "normal": np.array(
                [
                    circle_dict.get("normal", {}).get("x", 0),
                    circle_dict.get("normal", {}).get("y", 0),
                    circle_dict.get("normal", {}).get("z", 1),
                ]
            ),
        }
        circle = Circle(metadata)
        circle._compute_points()
        return circle
    
    @property
    def curve_type(self):
        return "circle"
    
    @property
    def start_point(self):
        """Return the bounding box min point as a reference point."""
        return self.bbox[0]
    
    @property
    def end_point(self):
        """Return a point on the right side of the circle."""
        return np.array(
            [
                self.metadata["center"][0] + self.metadata["radius"],
                self.metadata["center"][1],
            ]
        )
    
    def get_point(self, point_type):
        """Get a specific point from the metadata."""
        return self.metadata[point_type]
    
    @property
    def bbox(self):
        """Return the bounding box as min/max points."""
        return np.stack(
            [
                self.metadata["center"] - self.metadata["radius"],
                self.metadata["center"] + self.metadata["radius"],
            ],
            axis=0,
        )
    
    @property
    def bbox_size(self):
        """Return the maximum dimension of the bounding box."""
        bbox_size = np.max(np.abs(self.bbox[1] - self.bbox[0]))
        return max(bbox_size, 1.0)  # Avoid division by zero
    
    def transform(self, translate, scale=1.0):
        """Transform the circle by translation and scaling."""
        self.metadata["center"] = (self.metadata["center"] + translate) * scale
        self.metadata["radius"] = self.metadata["radius"] * scale
        self._compute_points()
    
    def sample_points(self, n_points=32):
        """Sample points along the circle."""
        angles = np.linspace(0, np.pi * 2, num=n_points, endpoint=False)
        points = (
            np.stack([np.cos(angles), np.sin(angles)], axis=1) * self.metadata["radius"]
            + self.metadata["center"]
        )
        return points
    
    def draw(self, ax=None, color="black"):
        """Draw the circle on a matplotlib axis."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        circle_patch = patches.Circle(
            (self.metadata["center"][0], self.metadata["center"][1]),
            self.metadata["radius"],
            lw=1,
            fill=None,
            color=color,
        )
        ax.add_patch(circle_patch)
        return ax
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "type": "circle",
            "center_point": {"x": float(self.metadata["center"][0]), "y": float(self.metadata["center"][1])},
            "radius": float(self.metadata["radius"]),
            "normal": {"x": 0.0, "y": 0.0, "z": 1.0}
        }
    
    def __repr__(self):
        return f"Circle: Center({self.metadata['center'].round(4)}), Radius({round(self.metadata['radius'], 4)})"
# cad_processing/sequence/sketch/coord_system.py
import numpy as np

class CoordinateSystem:
    """Represents a coordinate system with origin and axes."""
    
    def __init__(self, metadata=None):
        """
        Initialize a coordinate system.
        
        Args:
            metadata: Dictionary containing origin and axes
        """
        self.metadata = metadata or {
            "origin": np.zeros(3),
            "x_axis": np.array([1, 0, 0]),
            "y_axis": np.array([0, 1, 0]),
            "z_axis": np.array([0, 0, 1])
        }
    
    @staticmethod
    def from_dict(transform_dict):
        """Create a CoordinateSystem from a dictionary representation."""
        metadata = {}
        
        # Extract origin and axes from transform dictionary
        origin = transform_dict.get("origin", {})
        metadata["origin"] = np.array([
            origin.get("x", 0.0),
            origin.get("y", 0.0),
            origin.get("z", 0.0)
        ])
        
        x_axis = transform_dict.get("x_axis", {})
        metadata["x_axis"] = np.array([
            x_axis.get("x", 1.0),
            x_axis.get("y", 0.0),
            x_axis.get("z", 0.0)
        ])
        
        y_axis = transform_dict.get("y_axis", {})
        metadata["y_axis"] = np.array([
            y_axis.get("x", 0.0),
            y_axis.get("y", 1.0),
            y_axis.get("z", 0.0)
        ])
        
        z_axis = transform_dict.get("z_axis", {})
        metadata["z_axis"] = np.array([
            z_axis.get("x", 0.0),
            z_axis.get("y", 0.0),
            z_axis.get("z", 1.0)
        ])
        
        return CoordinateSystem(metadata)
    
    def get_property(self, prop_name):
        """Get a specific property of the coordinate system."""
        return self.metadata.get(prop_name, None)
    
    @property
    def normal(self):
        """Return the normal vector (z-axis) of the coordinate system."""
        return self.metadata["z_axis"]
    
    def rotate_vec(self, points_2d):
        """
        Transform 2D points to 3D using the coordinate system.
        
        Args:
            points_2d: 2D points in the sketch plane
            
        Returns:
            3D points in the global coordinate system
        """
        # Handle both single points and arrays
        is_single = len(points_2d.shape) == 1
        if is_single:
            points_2d = points_2d.reshape(1, -1)
        
        # Ensure points are 2D
        if points_2d.shape[1] != 2:
            raise ValueError("Input points must be 2D")
        
        # Convert to 3D by adding a zero z-coordinate
        n_points = points_2d.shape[0]
        points_3d = np.zeros((n_points, 3))
        
        # Transform each point
        for i in range(n_points):
            point = points_2d[i]
            # Transform to global coordinates using the coordinate system
            transformed = (
                self.metadata["origin"] + 
                point[0] * self.metadata["x_axis"] + 
                point[1] * self.metadata["y_axis"]
            )
            points_3d[i] = transformed
        
        # Return in the original shape
        if is_single:
            return points_3d[0]
        return points_3d
    
    def __repr__(self):
        """String representation of the coordinate system."""
        origin = self.metadata["origin"].round(4)
        x_axis = self.metadata["x_axis"].round(4)
        y_axis = self.metadata["y_axis"].round(4)
        z_axis = self.metadata["z_axis"].round(4)
        
        return f"CoordinateSystem(origin={origin}, x_axis={x_axis}, y_axis={y_axis}, z_axis={z_axis})"
    
    def to_dict(self):
        """Convert to dictionary representation."""
        return {
            "origin": {
                "x": float(self.metadata["origin"][0]),
                "y": float(self.metadata["origin"][1]),
                "z": float(self.metadata["origin"][2])
            },
            "x_axis": {
                "x": float(self.metadata["x_axis"][0]),
                "y": float(self.metadata["x_axis"][1]),
                "z": float(self.metadata["x_axis"][2])
            },
            "y_axis": {
                "x": float(self.metadata["y_axis"][0]),
                "y": float(self.metadata["y_axis"][1]),
                "z": float(self.metadata["y_axis"][2])
            },
            "z_axis": {
                "x": float(self.metadata["z_axis"][0]),
                "y": float(self.metadata["z_axis"][1]),
                "z": float(self.metadata["z_axis"][2])
            }
        }
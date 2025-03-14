# cad_processing/sequence/sketch/sketchsequence.py
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from .face import FaceSequence
from .coord_system import CoordinateSystem
from ...utils.macro import SKETCH_TOKEN, N_BIT

class SketchSequence:
    """A sketch consists of one or more faces and a coordinate system."""
    
    def __init__(self, facedata: List[FaceSequence], coordsystem: CoordinateSystem = None, reorder: bool = True):
        """
        Initialize a sketch sequence.
        
        Args:
            facedata: List of faces that make up the sketch
            coordsystem: The coordinate system for the sketch
            reorder: Whether to reorder faces
        """
        self.facedata = facedata
        self.coordsystem = coordsystem or CoordinateSystem()
        
        if reorder:
            self.reorder()
    
    @property
    def token_index(self):
        """Return the token index for END_SKETCH."""
        return SKETCH_TOKEN.index("END_SKETCH")
    
    def reorder(self):
        """Reorder faces based on their bounding box position."""
        if len(self.facedata) <= 1:
            return
        
        # Get min bbox coordinates for each face
        all_faces_bbox_min = np.stack([face.bbox[0] for face in self.facedata], axis=0).round(6)
        
        # Sort based on y then x coordinates
        ind = np.lexsort(all_faces_bbox_min.transpose()[[1, 0]])
        
        # Reorder faces
        self.facedata = [self.facedata[i] for i in ind]
    
    @staticmethod
    def from_dict(all_stat, profile_uid_list):
        """Create a SketchSequence from a dictionary representation."""
        facedata = []
        
        # Create coordinate system from the first sketch entity
        coordsystem = CoordinateSystem.from_dict(
            all_stat["entities"][profile_uid_list[0][0]]["transform"]
        )
        
        # Process each sketch entity
        for i in range(len(profile_uid_list)):
            sketch_entity = all_stat["entities"][profile_uid_list[i][0]]
            
            # Ensure it's a sketch
            if sketch_entity["type"] != "Sketch":
                raise ValueError(f"Entity {profile_uid_list[i][0]} is not a Sketch")
            
            # Create face from the sketch entity
            facedata.append(
                FaceSequence.from_dict(sketch_entity, profile_uid_list[i][1])
            )
        
        return SketchSequence(facedata=facedata, coordsystem=coordsystem, reorder=True)
    
    def sample_points(self, n_points=32, point_dimension=3):
        """Sample points across all faces in the sketch."""
        if not self.facedata:
            return np.zeros((n_points, point_dimension))
        
        all_points = []
        for fc in self.facedata:
            all_points.append(fc.sample_points(n_points=n_points))
        
        all_points = np.vstack(all_points)
        
        # Randomly sample n_points from all sampled points
        indices = np.random.choice(len(all_points), min(n_points, len(all_points)), replace=False)
        random_points = all_points[indices]
        
        # Convert to 3D if requested
        if random_points.shape[-1] == 2 and point_dimension == 3:
            random_points = self.coordsystem.rotate_vec(random_points)
        
        return random_points
    
    def __repr__(self):
        """String representation of the sketch."""
        s = "Sketch:"
        s += f"\n       - {self.coordsystem.__repr__()}"
        for face in self.facedata:
            s += f"\n       - {face.__repr__()}"
        return s
    
    def to_vec(self):
        """Convert the sketch to a vector representation for tokenization."""
        coord_token = []
        for fc in self.facedata:
            vec = fc.to_vec()
            coord_token.extend(vec)
        
        coord_token.append([self.token_index, 0])
        return coord_token
    
    @property
    def bbox(self):
        """Get the bounding box of the sketch."""
        if not self.facedata:
            return np.array([[0, 0], [0, 0]])
        
        all_min_box = []
        all_max_box = []
        
        for fc in self.facedata:
            bbox = fc.bbox
            all_min_box.append(bbox[0])
            all_max_box.append(bbox[1])
        
        return np.array([np.min(all_min_box, axis=0), np.max(all_max_box, axis=0)])
    
    @property
    def length(self):
        """Get the length of the sketch (x-dimension)."""
        bbox_min, bbox_max = self.bbox
        return abs(bbox_max[0] - bbox_min[0])
    
    @property
    def width(self):
        """Get the width of the sketch (y-dimension)."""
        bbox_min, bbox_max = self.bbox
        return abs(bbox_max[1] - bbox_min[1])
    
    @property
    def dimension(self):
        """Get the dimensions of the sketch."""
        return self.length, self.width
    
    @property
    def all_loops(self):
        """Return all loops in all faces of the sketch."""
        all_loops = []
        for fc in self.facedata:
            all_loops.extend(fc.all_loops)
        return all_loops
    
    @property
    def bbox_size(self):
        """Calculate the maximum dimension of the bounding box."""
        bbox_min, bbox_max = self.bbox
        bbox_size = np.max(
            np.abs(
                np.concatenate(
                    [bbox_max - self.start_point, bbox_min - self.start_point]
                )
            )
        )
        return max(bbox_size, 1.0)  # Avoid division by zero
    
    def transform(self, translate=None, scale=1):
        """Transform all faces in the sketch."""
        for fc in self.facedata:
            fc.transform(translate=translate, scale=scale)
    
    @property
    def all_curves(self):
        """Return all curves in all faces of the sketch."""
        curves = []
        for fc in self.facedata:
            curves.extend(fc.all_curves)
        return curves
    
    @property
    def start_point(self):
        """Return the start point (minimum of bounding box)."""
        return self.bbox[0]
    
    @property
    def sketch_position(self):
        """Get the 3D position of the sketch origin."""
        return (
            self.start_point[0] * self.coordsystem.get_property("x_axis") +
            self.start_point[1] * self.coordsystem.get_property("y_axis") +
            self.coordsystem.get_property("origin")
        )
    
    def normalize(self, translate=None, bit=N_BIT):
        """
        Normalize the sketch to fit within the quantization range.
        
        Args:
            translate: Optional translation to apply
            bit: Number of bits used for quantization
        """
        size = 2**bit
        scale = (size - 1) / self.bbox_size
        
        if translate is None:
            self.transform(-self.start_point, scale)
        else:
            self.transform(translate, scale)
    
    def numericalize(self, bit=N_BIT):
        """Convert coordinates to quantized integer values."""
        for fc in self.facedata:
            if hasattr(fc, 'numericalize'):
                fc.numericalize(bit=bit)
    
    def denumericalize(self, bit=N_BIT):
        """Convert quantized integer values back to float coordinates."""
        for fc in self.facedata:
            if hasattr(fc, 'denumericalize'):
                fc.denumericalize(bit=bit)
    
    def draw(self, ax=None, colors=None):
        """Draw the sketch on a matplotlib axis."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        if colors is None:
            colors = ["red", "blue", "green", "brown", "pink", "yellow", "purple", "black"] * 10
        
        for i, face in enumerate(self.facedata):
            face.draw(ax, colors[i % len(colors)])
        
        return ax
    
    def to_dict(self):
        """Convert to dictionary representation."""
        sketch_dict = {}
        for i, face in enumerate(self.facedata):
            sketch_dict[f"face_{i+1}"] = face.to_dict()
        
        sketch_dict["coordinate_system"] = self.coordsystem.to_dict()
        return sketch_dict
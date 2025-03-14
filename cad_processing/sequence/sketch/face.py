# cad_processing/sequence/sketch/face.py
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from .loop import LoopSequence
from ...utils.macro import SKETCH_TOKEN

class FaceSequence:
    """A face is composed of one or more loops (typically one outer loop and zero or more inner loops)."""
    
    def __init__(self, loopdata: List[LoopSequence], reorder: bool = True):
        """
        Initialize a face sequence.
        
        Args:
            loopdata: List of loops that make up the face
            reorder: Whether to reorder loops
        """
        self.loopdata = loopdata
        
        if reorder:
            self.reorder()
    
    @property
    def token_index(self):
        """Return the token index for END_FACE."""
        return SKETCH_TOKEN.index("END_FACE")
    
    @staticmethod
    def from_dict(face_dict, loop_uid: str):
        """Create a FaceSequence from a dictionary representation."""
        loopdata = []
        
        loop_entity = face_dict.get("profiles", {}).get(loop_uid, {})
        for lp in loop_entity.get("loops", []):
            loopdata.append(LoopSequence.from_dict(lp))
        
        return FaceSequence(loopdata, True)
    
    def to_vec(self):
        """Convert the face to a vector representation for tokenization."""
        coord_token = []
        for lp in self.loopdata:
            vec = lp.to_vec()
            coord_token.extend(vec)
        
        coord_token.append([self.token_index, 0])
        return coord_token
    
    def reorder(self):
        """Reorder loops based on their bounding box position."""
        if len(self.loopdata) <= 1:
            return
        
        # Get min bbox coordinates for each loop
        all_loops_bbox_min = np.stack([loop.bbox[0] for loop in self.loopdata], axis=0).round(6)
        
        # Sort based on y then x coordinates
        ind = np.lexsort(all_loops_bbox_min.transpose()[[1, 0]])
        
        # Reorder loops
        self.loopdata = [self.loopdata[i] for i in ind]
    
    def __repr__(self):
        """String representation of the face."""
        s = "Face:"
        for loop in self.loopdata:
            s += f"\n          - {loop.__repr__()}"
        return s + "\n"
    
    def transform(self, translate=None, scale=1):
        """Transform all loops in the face."""
        if translate is None:
            translate = np.zeros(2)
        
        for loop in self.loopdata:
            loop.transform(translate=translate, scale=scale)
    
    def sample_points(self, n_points=32):
        """Sample points across all loops in the face."""
        if not self.loopdata:
            return np.zeros((n_points, 2))
        
        all_points = []
        for loop in self.loopdata:
            all_points.append(loop.sample_points(n_points=n_points))
        
        all_points = np.vstack(all_points)
        
        # Randomly sample n_points from all sampled points
        indices = np.random.choice(len(all_points), min(n_points, len(all_points)), replace=False)
        return all_points[indices]
    
    @property
    def all_curves(self):
        """Return all curves in all loops of the face."""
        all_curves = []
        for lp in self.loopdata:
            all_curves.extend(lp.all_curves)
        return all_curves
    
    @property
    def start_point(self):
        """Return the start point of the first loop in the face."""
        if not self.loopdata:
            return np.zeros(2)
        return self.loopdata[0].start_point
    
    @property
    def all_loops(self):
        """Return all loops in the face."""
        return self.loopdata
    
    @property
    def bbox(self):
        """Get the bounding box of the face."""
        if not self.loopdata:
            return np.array([[0, 0], [0, 0]])
        
        all_min_box = []
        all_max_box = []
        
        for lp in self.loopdata:
            bbox = lp.bbox
            all_min_box.append(bbox[0])
            all_max_box.append(bbox[1])
        
        return np.array([np.min(all_min_box, axis=0), np.max(all_max_box, axis=0)])
    
    def draw(self, ax=None, colors=None):
        """Draw the face on a matplotlib axis."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        if colors is None:
            colors = ["red", "blue", "green", "brown", "pink", "yellow", "purple", "black"] * 10
        else:
            colors = [colors] * 100
        
        for i, loop in enumerate(self.loopdata):
            loop.draw(ax, colors[i % len(colors)])
        
        return ax
    
    def to_dict(self):
        """Convert to dictionary representation."""
        face_dict = {}
        for i, loop in enumerate(self.loopdata):
            face_dict[f"loop_{i+1}"] = loop.to_dict()
        return face_dict
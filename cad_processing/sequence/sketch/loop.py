# cad_processing/sequence/sketch/loop.py
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from ...geometry.curve import Curve
from ...utils.macro import SKETCH_TOKEN, END_PAD, BOOLEAN_PAD

class LoopSequence:
    """A loop is a closed sequence of curves (lines, arcs, circles)."""
    
    def __init__(self, curvedata: List[Curve], is_outer=False, post_processing=True, fix_collinearity=False):
        """
        Initialize a loop sequence.
        
        Args:
            curvedata: List of curves that form the loop
            is_outer: Whether this is an outer loop
            post_processing: Whether to apply connectivity fixes
            fix_collinearity: Whether to merge collinear curves
        """
        self.curvedata = curvedata
        self.is_outer = is_outer
        self.collinear_curves = []
        
        # Reorder curves to fix connectivity and orientation
        if len(self.curvedata) > 0:
            self.reorder(orientation_fix=True, collinearity_fix=fix_collinearity)
        
        # Fix start and end points if post-processing is enabled
        if post_processing and len(self.curvedata) > 1:
            n_curve = len(self.curvedata)
            for i, cv in enumerate(self.curvedata):
                if cv.curve_type == "line":
                    # If start and end points are the same, fix them
                    if np.all(cv.metadata["start_point"] == cv.metadata["end_point"]):
                        cv.metadata["end_point"] += 1
                        # Update the start point of the next curve
                        self.curvedata[(i + 1) % n_curve].metadata["start_point"] = cv.metadata["end_point"]
    
    @property
    def token_index(self):
        """Return the token index for END_LOOP."""
        return SKETCH_TOKEN.index("END_LOOP")
    
    @staticmethod
    def from_dict(loop_dict):
        """Create a LoopSequence from a dictionary representation."""
        from ...geometry.line import Line
        from ...geometry.circle import Circle
        
        is_outer = loop_dict.get("is_outer", False)
        curvedata = []
        
        curves = loop_dict.get("profile_curves", [])
        for curve in curves:
            curve_type = curve.get("type", "")
            
            if curve_type == "Line3D":
                curvedata.append(Line.from_dict(curve))
            elif curve_type == "Circle3D":
                curvedata.append(Circle.from_dict(curve))
            # We'll add arc support later
        
        return LoopSequence(curvedata, is_outer, False)
    
    @property
    def start_point(self):
        """Return the start point of the first curve in the loop."""
        if not self.curvedata:
            return np.zeros(2)
        return self.curvedata[0].start_point
    
    @property
    def bbox(self):
        """Get the bounding box of the loop."""
        if not self.curvedata:
            return np.array([[0, 0], [0, 0]])
        
        if len(self.curvedata) == 1:
            return self.curvedata[0].bbox
        
        all_min_box = []
        all_max_box = []
        
        for curve in self.curvedata:
            if curve is not None:
                bbox = curve.bbox
                all_min_box.append(bbox[0])
                all_max_box.append(bbox[1])
        
        return np.array([np.min(all_min_box, axis=0), np.max(all_max_box, axis=0)])
    
    def to_vec(self):
        """Convert the loop to a vector representation for tokenization."""
        coord_token = []
        for cv in self.curvedata:
            if hasattr(cv, 'to_vec'):
                vec = cv.to_vec()
                coord_token.extend(vec)
        
        coord_token.append([self.token_index, 0])
        return coord_token
    
    @staticmethod
    def ensure_connectivity(curvedata: List[Curve], verbose=False):
        """Ensure that all curves in the loop are connected."""
        if len(curvedata) <= 1:
            return curvedata
        
        new_curvedata = [curvedata[0]]
        n = len(curvedata)
        
        for i, curve in enumerate(curvedata):
            if i > 0:
                last_end = new_curvedata[-1].get_point("end_point")
                
                # Check if we need to reverse the curve
                if i < n - 1 and np.allclose(last_end, curve.get_point("end_point")):
                    curve.reverse()
                    new_curvedata.append(curve)
                elif (i == n - 1 and np.allclose(last_end, curve.get_point("end_point"))) or \
                     np.allclose(curve.get_point("start_point"), new_curvedata[0].get_point("start_point")):
                    curve.reverse()
                    new_curvedata.append(curve)
                else:
                    new_curvedata.append(curve)
        
        return new_curvedata
    
    def reorder(self, orientation_fix=True, collinearity_fix=True):
        """Reorder curves to ensure proper connectivity and orientation."""
        if len(self.curvedata) <= 1:
            return
        
        # Find the leftmost point to start the loop
        start_curve_idx = -1
        sx, sy = float('inf'), float('inf')
        
        for i, curve in enumerate(self.curvedata):
            start_pt = curve.get_point("start_point")
            if start_pt[0] < sx or (start_pt[0] == sx and start_pt[1] < sy):
                start_curve_idx = i
                sx, sy = start_pt
        
        # Reorder curves to start from the leftmost point
        self.curvedata = self.curvedata[start_curve_idx:] + self.curvedata[:start_curve_idx]
        
        # Ensure connectivity
        self.curvedata = self.ensure_connectivity(self.curvedata)
        
        # TODO: Implement orientation and collinearity fixes
    
    @property
    def all_curves(self):
        """Return all curves in the loop."""
        return self.curvedata
    
    def transform(self, translate=None, scale=1):
        """Transform all curves in the loop."""
        if translate is None:
            translate = np.zeros(2)
        
        for curve in self.curvedata:
            curve.transform(translate=translate, scale=scale)
    
    def __repr__(self):
        """String representation of the loop."""
        s = f"Loop: Start Point: {list(np.round(self.start_point, 4))}"
        for curve in self.curvedata:
            s += f"\n              - {curve.__repr__()}"
        return s + "\n"
    
    def sample_points(self, n_points=32):
        """Sample points along the loop."""
        if not self.curvedata:
            return np.zeros((n_points, 2))
        
        all_points = []
        for curve in self.curvedata:
            all_points.append(curve.sample_points(n_points=n_points))
        
        all_points = np.vstack(all_points)
        
        # Randomly sample n_points from all sampled points
        indices = np.random.choice(len(all_points), min(n_points, len(all_points)), replace=False)
        return all_points[indices]
    
    def draw(self, ax=None, colors=None):
        """Draw the loop on a matplotlib axis."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        if colors is None:
            colors = ["red", "blue", "green", "brown", "pink", "yellow", "purple", "black"] * 10
        
        for i, curve in enumerate(self.curvedata):
            curve.draw(ax, colors[i % len(colors)])
        
        return ax
    
    def to_dict(self):
        """Convert to dictionary representation."""
        loop_dict = {}
        curve_num_dict = {"line": 1, "arc": 1, "circle": 1}
        
        for curve in self.curvedata:
            curve_type = curve.curve_type
            loop_dict[f"{curve_type}_{curve_num_dict[curve_type]}"] = curve.to_dict()
            curve_num_dict[curve_type] += 1
        
        return loop_dict
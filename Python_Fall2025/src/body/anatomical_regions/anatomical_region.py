from abc import ABC, abstractmethod
from typing import Literal

import trimesh

LEFT_OR_RIGHT = Literal["left", "right"]
ANATOMICAL_REGION = Literal["head", "trunk", "left arm", "right arm", "left leg", "right leg"]

class Anatomical_Region(ABC):
    """
    Abstract base class defining the interface for body part segmentation and measurement.
    
    This class establishes a contract that all anatomical region classes must follow.
    Each region (Head, Trunk, Arm, Leg) must provide:
    - The underlying mesh for that region
    - Volume and surface area properties
    - Anatomical landmarks specific to that region
    - Measurements specific to that region
    
    Purpose
    -------
    This class solves the architectural problem of ensuring consistency across body parts.
    Without this abstract base class:
    - Different regions might have different interfaces (some use .get_volume(), others .volume)
    - You couldn't iterate over body parts uniformly
    - Type hints wouldn't work for generic body part operations
    
    With this class, you can:
    - Write code that works with any body part: `for part in body.parts.values(): print(part.volume)`
    - Enforce that all body parts implement required methods
    - Use type hints: `def process_region(region: Anatomical_Region)`
    
    Class Structure and Design Choices
    -----------------------------------
    **Why ABC (Abstract Base Class)**:
    Using Python's abc module ensures that:
    - You cannot instantiate Anatomical_Region directly (it's incomplete)
    - Subclasses MUST implement all @abstractmethod methods or they will also be abstract
    - IDEs can warn you if a subclass is missing required methods
    
    **Why @property for volume/surface_area**:
    These are properties (not methods) because:
    - They represent attributes of the region, not actions
    - Syntax is cleaner: `region.volume` vs `region.get_volume()`
    - Properties can use @cache decorator in subclasses for automatic memoization
    
    **Why @abstractmethod**:
    Marking methods as abstract means:
    - Subclasses must override them
    - Attempting to create an instance without implementing them raises TypeError
    - Provides clear documentation of what subclasses must provide
    
    **Landmarks and measurements as properties**:
    These return dictionaries to allow flexible key-value storage:
    - Different regions have different landmarks (arms have shoulders, legs have ankles)
    - Dictionary keys can be descriptive strings
    - Easy to serialize to JSON or other formats
    
    **Why static methods with @cache**:
    - Before using static methods, separating body landmark and measurement algorithms into 
    encapsulated classes was impossible. The alternative was a monolithic codebase: one 
    massive class with countless logically unrelated methods. However, we couldn't simply 
    use regular instance methods either, because circular dependencies would arise during 
    construction (Trunk construction would require Arm, which would require Trunk, and so on).
    
    - The key insight was that these methods don't modify instance state—they're primarily 
    used during construction. By making them static, we can call them without instantiating 
    the class. Combined with runtime imports (which grab static methods from other classes 
    only when called), this breaks the circular dependency chain and enables a clean, 
    modular architecture.
    
    Caching (@cache) is layered on top because:
    - These computations are expensive (mesh slicing, landmark detection)
    - Multiple body parts need the same data (both arms need armpit locations)
    - The cache key is (side, mesh hash), so left and right arms are cached separately
    
    Example usage pattern in subclasses::
    
        @staticmethod
        @cache
        def _locate_landmark(mesh: trimesh.Trimesh, side: str):
            # Expensive computation here
            return landmark_point
    
    Notes
    -----
    - This is an abstract class - you cannot create instances of it directly
    - All subclasses must implement mesh, volume, surface_area, landmarks, and measurements
    - Subclasses should use @cache decorators on expensive operations
    
    See Also
    --------
    Arm : Concrete implementation for arm regions
    Leg : Concrete implementation for leg regions
    Trunk : Concrete implementation for trunk region
    Head : Concrete implementation for head region
    """
    
    @property
    @abstractmethod
    def mesh(self) -> trimesh.Trimesh:
        """
        The underlying mesh for this anatomical region.
        
        Returns
        -------
        trimesh.Trimesh
            The segmented mesh representing this body region
        """
        pass

    @property
    @abstractmethod
    def volume(self):
        """
        Volume of this anatomical region in cubic units.
        
        Returns
        -------
        float
            Volume of the segmented mesh for this region
        """
        pass

    @property
    @abstractmethod
    def surface_area(self):
        """
        Surface area of this anatomical region in square units.
        
        Returns
        -------
        float
            Surface area of the segmented mesh for this region
        """
        pass

    @property
    @abstractmethod
    def landmarks(self):
        """
        Dictionary of anatomical landmarks for this region.
        
        Returns
        -------
        dict[str, np.ndarray]
            Mapping of landmark names to 3D coordinates
            Example: {"shoulder": [x, y, z], "wrist": [x, y, z]}
        """
        pass

    @property
    @abstractmethod
    def measurements(self):
        """
        Dictionary of anthropometric measurements for this region.
        
        Returns
        -------
        dict[str, float]
            Mapping of measurement names to values
            Example: {"arm length": 45.2, "wrist girth": 15.3}
        """
        pass

    @property
    @abstractmethod
    def drawings(self):
        """
        Dictionary of 3D paths visualizing where measurements were taken.
        
        Each measurement is visualized as a trimesh.Path3D object:
        - Circumferences/girths: closed loop showing the cross-section
        - Lengths: line segment between two landmarks
        
        All paths are in the original mesh coordinate system.
        
        Returns
        -------
        dict[str, trimesh.path.Path3D]
            Mapping of measurement names to their visual representations
            Example: {"wrist girth": <Path3D closed loop>, "arm length": <Path3D line segment>}
        """
        pass
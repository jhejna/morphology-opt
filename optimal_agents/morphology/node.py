import math
import random
import copy
import numpy as np
from scipy import spatial
from dm_control import mjcf

class Node(object):
    
    def __init__(self, extent=[0.1,0,0.1], radius=0.05, attachment_point=1, joint=0, joint_limit=0, gear=50,
                       extent_range=0.4, radius_range=[0.02, 0.05], joint_range=(30, 60), gear_range=(50, 90), only_end=False):
        # Check input for validity
        assert radius >= radius_range[0] and radius <= radius_range[1], "Radius of Node was out of range" 
        assert all([abs(v) <= extent_range for v in extent]), "Extent was out of range"
        if joint != 0:
            assert gear >= gear_range[0] and gear <= gear_range[1], "Gear of Node was out of range"
            assert joint_limit >= joint_range[0] and joint_limit <= joint_range[1], "joint limit out of range"

        # Save configuration values
        self._extent_range = extent_range
        self._radius_range = radius_range
        self._joint_range = joint_range
        self._gear_range = gear_range
        self._only_end = only_end

        if self._only_end:
            attachment_point = 1

        # Physical Parameters
        self.radius = radius
        self.extent = extent
        self._attachment_point = attachment_point # Value between zero and one
        self._joint = joint
        self.joint_limit = joint_limit
        self.gear = gear

        # General Values
        self._children = []
        self.color = [random.random() for _ in range(3)]
        self.color.append(1)
        self._id = None
        self.segment = None # np array of size six giving positions relative to the ROOT of the line segment at the center of the node
                             # Used for intersection calculations.
        self.basis = np.identity(3)
    
    def construct(self, geom_kwargs={}, joint_kwargs={}, two_dim=False, one_dim=False):
        assert not self._id is None, "Must have labeled all nodes for MJCF generation"
        root = mjcf.RootElement(model=str(self._id))
        root.worldbody.add('geom', name="geom", type='capsule', fromto=[0, 0, 0]+ self.extent, size=(self.radius,), rgba=self.color,
                           **geom_kwargs)
        root.worldbody.add('site', name="end" + str(self._id), size=[0.005], pos=self.extent)
        body_length = math.sqrt(self.extent[0]**2 + self.extent[1]**2 + self.extent[2]**2)
        for child in self._children:
            # Create site for attachment along body position
            dist = child._attachment_point
            attachment_site = root.worldbody.add('site', name="attach" + str(child._id), size=[1e-6]*3, pos=[self.extent[0]*dist, self.extent[1]*dist, self.extent[2]*dist])

            if not two_dim and not one_dim: # Update the child's reference frame
                child_extent_vec = np.array([child.extent])
                rot, _ = spatial.transform.Rotation.align_vectors(np.identity(3)[:1, :], child_extent_vec)
                child.basis = rot.as_matrix()
            # Attach the child node to the specified site
            child_root = child.construct(geom_kwargs=geom_kwargs, joint_kwargs=joint_kwargs, two_dim=two_dim, one_dim=one_dim)
            attachment_frame = attachment_site.attach(child_root)
            # Add the joint between the child and the parent
            if child.joint != 0:
                axis = list(self.basis[child.joint - 1])
                joint = attachment_frame.add('joint', name="joint", axis=axis, range=(-child.joint_limit, child.joint_limit), type='hinge', limited=True, **joint_kwargs)
                child_root.actuator.add('motor', name="motor", joint=joint, gear=(child.gear,), ctrlrange=(-1, 1), ctrllimited=True)
        return root

    def compute_segment(self, parent=None):
        if parent is None:
            parent_segment = np.zeros(6)
        else:
            parent_segment = parent.segment
        # Compute the absolute starting point of this limb based on the parent.
        diff = parent_segment[3:] - parent_segment[:3]
        start_point = parent_segment[:3] + self._attachment_point * diff
        # Compute the end point based on extent.
        end_point = start_point + np.array(self.extent)
        self.segment = np.concatenate((start_point, end_point), axis=0)
        for child in self._children:
            child.compute_segment(parent=self)

    def intersects(self, other):
        '''
        Should be used to check if this.segment intersects with other.segment.
        More clearly, leaf.segment intersects with any other segment in the tree
        Should ONLY be used to on leaves.
        '''
        assert not self.segment is None, "Haven't computed segments for this joint"
        assert not other.segment is None, "Haven't computed segments for other joint"
        num_pts = 100
        num_radii_of_ignore = 1.4 #1.66 #1.75 # Multiple for how many radii to not consider around the start point of the limb.
        diff = self.segment[3:] - self.segment[:3]
        ignore_factor = min(num_radii_of_ignore*self.radius, np.linalg.norm(diff) - 1e-5)
        start = self.segment[:3] + ignore_factor * diff / np.linalg.norm(diff)
        my_pts = np.linspace(start=start, stop=self.segment[3:], num=num_pts)
        other_pts = np.linspace(start=other.segment[:3], stop=other.segment[3:], num=num_pts)
        dists = spatial.distance.cdist(my_pts, other_pts)
        min_dist = np.min(dists)
        if min_dist < self.radius:
            return True
        else:
            return False

    def mutate(self, two_dim=False, one_dim=False, geom_mut=0.3, joint_mut=0.15, joint_prob=0.8, joint_type_mut=0.5, extent_std=None,
                     radius_std=None, attachment_std=None, gear_std=None, joint_std=None):
        
        std_factor = 5
        extent_std = extent_std if not extent_std is None else (2*self._extent_range)/std_factor
        radius_std = radius_std if not radius_std is None else (self._radius_range[1] - self._radius_range[0])/std_factor
        attachment_std = attachment_std if not attachment_std is None else 1/std_factor
        gear_std = gear_std if not gear_std is None else (self._gear_range[1] - self._gear_range[0])/std_factor
        joint_std = joint_std if not joint_std is None else (self._joint_range[1] - self._joint_range[0])/std_factor

        if random.random() < geom_mut:
            extent = [max(-self._extent_range, min(self._extent_range, random.gauss(orig, extent_std))) for orig in self.extent]
            if two_dim:
                extent[1] = 0
            if one_dim:
                extent[0], extent[1], extent[2] = abs(extent[0]), 0, 0
            radius = max(self._radius_range[0], min(self._radius_range[1], random.gauss(self.radius, radius_std)))
            attachment_point = max(0, min(1, random.gauss(self._attachment_point, attachment_std)))
        else:
            extent = self.extent.copy() # Extent is a list, make sure to copy it
            radius = self.radius
            attachment_point = self._attachment_point
        
        if random.random() < joint_mut:
            if random.random() < joint_type_mut: # With prob joint_type_mut resample the joint type.
                if random.random() < joint_prob: # Give it a joint with prob joint_prob
                    if two_dim:
                        joint = 2
                    elif one_dim:
                        joint = 3
                    else:
                        joint = random.randint(2,3) # Uniformly over all possible choices.
                else:
                    joint = 0
            else:
                joint = self.joint

            joint_limit = max(self._joint_range[0], min(self._joint_range[1], random.gauss(abs(self.joint_limit), joint_std)))
            gear = max(self._gear_range[0], min(self._gear_range[1], random.gauss(self.gear, gear_std)))
        else:
            joint = self.joint
            joint_limit = self.joint_limit
            gear = self.gear
        joint = self.joint
        joint_limit = self.joint_limit
        gear = self.gear

        # Check Asserts in mutation
        assert radius >= self._radius_range[0] and radius <= self._radius_range[1], "Radius of Node was out of range" 
        assert all([abs(v) <= self._extent_range for v in extent]), "Extent was out of range"
        if joint != 0:
            assert gear >= self._gear_range[0] and gear <= self._gear_range[1], "Gear of Node was out of range" 
            assert joint_limit >= self._joint_range[0] and joint_limit <= self._joint_range[1], "Positive joint limit out of range"

        return Node(extent=extent, radius=radius, attachment_point=attachment_point, joint=joint, joint_limit=joint_limit, gear=gear,
                    extent_range=self._extent_range, radius_range=self._radius_range, joint_range=self._joint_range, gear_range=self._gear_range, 
                    only_end=self._only_end)

    def get_embedding(self):
        extent_embedding = [val / self._extent_range for val in self.extent] # Values between -1 and 1
        radius_embedding = (self.radius - self._radius_range[0]) / (self._radius_range[1] - self._radius_range[0]) # between 0 and 1
        node_embedding = np.array([*extent_embedding, radius_embedding, self._attachment_point], dtype=np.float32)
        edge_embedding = np.zeros(6)
        if self.joint != 0:
            # Assume first joint value is negative, and second is positive.
            edge_embedding[0] = (self.joint_limit - self._joint_range[0]) / (self._joint_range[1] - self._joint_range[0])
            edge_embedding[1] = (self.gear - self._gear_range[0]) / (self._gear_range[1] - self._gear_range[0]) # between 0 and 1
        else:
            edge_embedding[0], edge_embedding[1] = -1, -1 # Everything is -1 if no joint
        if self.joint != 0:
            edge_embedding[2 + self.joint] = 1.0
        edge_embedding = edge_embedding.astype(np.float32)
        return node_embedding, edge_embedding

    @classmethod
    def from_embedding(cls, node_embedding, edge_embedding=None, two_dim=False, one_dim=False,
            extent_range=0.4, radius_range=[0.02, 0.05], joint_range=(30, 60), gear_range=(50, 90), only_end=False):
        # Node embedding is size 5 (extent, radius, attachment)
        # Edge Embedding is size 6 (limit, gear, type)
        if edge_embedding is None:
            assert node_embedding.shape == (11,)
            edge_embedding = node_embedding[5:]
            node_embedding = node_embedding[:5]
        extent = list(extent_range * np.clip(node_embedding[:3], -1, 1))
        radius = max(0, min(node_embedding[3], 1)) * (radius_range[1] - radius_range[0]) + radius_range[0]
        attachment_point = max(0, min(1, node_embedding[4]))
        one_hot_joint = edge_embedding[-4:]
        if two_dim: # Enforce two dim
            extent[1] = 0
            one_hot_joint[1] = -np.inf
            one_hot_joint[3] = -np.inf
        elif one_dim:
            extent[0], extent[1], extent[2] = abs(extent[0]), 0, 0
            one_hot_joint[1] = -np.inf
            one_hot_joint[2] = -np.inf
        joint = np.argmax(one_hot_joint) # support softmax of joint type.
        if joint == 0:
            joint_limit, gear = 0, 0
        else:
            joint_limit = max(0, min(1, edge_embedding[0])) * (joint_range[1] - joint_range[0]) + joint_range[0]
            gear = max(0, min(1, edge_embedding[1])) * (gear_range[1] - gear_range[0]) + gear_range[0]

        return cls(extent=extent, radius=radius, attachment_point=attachment_point, joint=joint, joint_limit=joint_limit, gear=gear, 
                    extent_range=extent_range, radius_range=radius_range, joint_range=joint_range, gear_range=gear_range, only_end=only_end)

    @classmethod
    def generate_random_node(cls, two_dim=False, one_dim=False, extent_range=0.4, radius_range=[0.02, 0.05], joint_range=(30, 60), gear_range=(50, 90), 
                                   only_end=False, joint_prob=0.8):
        extent = [random.uniform(-extent_range, extent_range) for _ in range(3)]
        if two_dim:
                extent[1] = 0
        if one_dim:
            extent[0], extent[1], extent[2] = abs(extent[0]), 0, 0

        radius = random.uniform(*radius_range)
        attachment_point = random.random()
        gear = int(random.uniform(*gear_range))
        joint_val = random.uniform(*joint_range)
        if random.random() < joint_prob:
            if two_dim:
                joint = 2
            elif one_dim:
                joint = 3
            else:
                joint = random.randint(2,3)
        else:
            joint = 0
        return cls(extent=extent, radius=radius, attachment_point=attachment_point, joint=joint, joint_limit=joint_val, gear=gear,
                   extent_range=extent_range, radius_range=radius_range, joint_range=joint_range, gear_range=gear_range, 
                   only_end=only_end)

    @property
    def joint(self):
        return self._joint
    
    @property
    def node_id(self):
        return self._id
    
    def add_child(self, child):
        self._children.append(child)

    def __iter__(self):
        yield self
        for child in self._children:
            for node in child:
                yield node

    def __copy__(self):
        # Copy the node, but do not copy links to children etc.
        return Node(extent=self.extent.copy(), radius=self.radius, attachment_point=self._attachment_point, joint=self.joint, 
                    joint_limit=copy.copy(self.joint_limit), gear=self.gear,
                    extent_range=self._extent_range, radius_range=self._radius_range, joint_range=self._joint_range, gear_range=self._gear_range, 
                    only_end=self._only_end)

    def __str__(self, depth=0):
        out_string = ""
        node_string = "Node(extent=[%.3f, %.3f, %.3f], radius=%.3f, attachment_point=%.3f, joint=%d, joint_limit=%d, gear=%d)" % (*self.extent, self.radius, self._attachment_point, self.joint, self.joint_limit, self.gear)
        if not self._id is None:
            node_string = str(self._id) + ": " + node_string
        out_string += depth*"\t" + node_string + "\n"
        depth += 1
        for child in self._children:
            out_string += child.__str__(depth=depth+1) + "\n"
        return out_string

    def __len__(self):
        return 1 + sum([len(child) for child in self._children])

    def label(self, node_id):
        self._id = node_id
        node_id += 1
        for child in self._children:
            node_id = child.label(node_id)
        return node_id


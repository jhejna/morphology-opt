import numpy as np
import random
from .node import Node
import copy
import pickle
from dm_control import mjcf

class Morphology(object):
    
    def __init__(self, root, two_dim=False, one_dim=False, geom_kwargs={}, joint_kwargs={}, global_kwargs={}, node_kwargs={}):
        self.two_dim = two_dim
        self.one_dim = one_dim
        self.node_kwargs = node_kwargs
        self.geom_kwargs = geom_kwargs
        self.joint_kwargs = joint_kwargs
        self.global_kwargs = global_kwargs
        
        root._joint, root._gear, root._joint_limit = 0, 0, 0 # Root has no joint.
        # Check consistency of given tree with range parameters
        if two_dim:
            for node in root:
                assert node.extent[1] == 0, "Two dim but extent y value was not zero."
                assert node.joint != 1 and node.joint != 3, "Incorrect joints for two dim option"
        elif one_dim:
            for node in root:
                assert node.extent[1] == 0, "One dim but extent y value was not zero."
                assert node.extent[2] == 0, "One dim but extent z value not zero"
                assert node.joint != 1 and node.joint != 2, "Incorrect joints for two dim option"

        self.root = root

        # Label all the nodes
        self.root.label(0)
        # Compute all absolute positions
        self.root.compute_segment()
        self._length = len(self.root)

        # NOTE: We make everything static as evaluating the morphology requires the most computation
        # this means that its better to re-create a new morphology object than have everything dynamically update
        
        # Generate the embeddings
        node_embeddings = list()
        edge_embeddings = list()
        segment_embeddings = list()
        edge_list = [None for _ in range(self._length - 1)] # Tree, so one less edge than length
        edge_embeddings = [None for _ in range(self._length - 1)]
        self._joint_map = np.zeros(self._length - 1, dtype=np.long) # N - 1 edges in the tree.
        self._num_joints = 0 
        
        site_pos_idx = 0
        site_pos_list = []
        for i, node in enumerate(self.root):
            assert i == node.node_id, "Labeling order was not correct DFS"
            node_embedding, edge_embedding = node.get_embedding()
            node_embeddings.append(node_embedding)
            segment_embeddings.append(node.segment)
            site_pos_list.append(site_pos_idx)
            site_pos_idx += 1
            if node.node_id != 0: # its not the root, thus we have an edge embedding
                edge_embeddings[node.node_id - 1] = edge_embedding
                assert not edge_list[node.node_id - 1] is None # Verify that we have already visited this node 
                if node.joint != 0:
                    self._joint_map[node.node_id - 1] = 1
                    self._num_joints += 1
            for child in node._children:
                # IDs are in DFS order. Edges associated with child
                edge_list[child.node_id - 1] = [node.node_id, child.node_id]
            site_pos_idx += len(node._children)

        site_pos_offset = site_pos_list[-1] + 1
        site_pos_list = [p - site_pos_offset for p in site_pos_list]
        self._site_pos_list = np.array(site_pos_list)

        assert len(node_embeddings) == len(self), "Must be N collected node embeddings"
        assert len(edge_list) == len(self) - 1, "More or less than N-1 Edges"

        self._node_embeddings = np.array(node_embeddings, dtype=np.float32)
        self._edge_embeddings = np.array(edge_embeddings, dtype=np.float32)

        self._edge_list = np.array(edge_list, dtype=np.long)
        self._adj_matrix = np.zeros((self._length, self._length), dtype=np.long)
        for edge in self._edge_list:
            parent, child = edge
            self._adj_matrix[parent,child] = 1
        self._segment_embeddings = np.array(segment_embeddings, dtype=np.float32)

    @classmethod 
    def from_embedding(cls, node_embeddings, adj_matrix, edge_embeddings=None, two_dim=False, one_dim=False, 
                            geom_kwargs={}, joint_kwargs={}, global_kwargs={}, node_kwargs={}):
        num_nodes = adj_matrix.shape[0]
        
        assert len(node_embeddings) == num_nodes, "Adj Matrix and node embedding were different sizes"
        # Check for Cycles:
        assert np.all(np.linalg.matrix_power(adj_matrix, num_nodes) == 0), "Adj Matrix contained cycles"
        # Assert that it is a tree:
        assert np.sum(adj_matrix) == num_nodes - 1, "Adjacency matrix had more than n-1 edges, not a tree."

        if edge_embeddings is None:
            edge_embeddings = node_embeddings[:, 5:] # Assume input has provided concatenated node and edge embeddings.
            node_embeddings = node_embeddings[:, :5] # Note that the end embedding is associated with the child.
        else:
            # Edge embeddings only exist for the edges. We need to add an embedding for the root.
            assert len(edge_embeddings) == num_nodes - 1
            root_edge_embedding = np.zeros((1,6)) # Edge is associated with child.
            root_edge_embedding[-0, 4] = 1 # indicate that this is the root node.
            edge_embeddings = np.concatenate((root_edge_embedding, edge_embeddings), axis=0)

        nodes = [Node.from_embedding(node_embedding, edge_embedding=edge_embedding, two_dim=two_dim, one_dim=one_dim, **node_kwargs) 
                        for node_embedding, edge_embedding in zip(node_embeddings, edge_embeddings)]
        # Next, add children according to the adj matrix. Parent --> Child in matrix.
        # We check for intersectiosn here. Note that we start with the parent node so we're all good.
        nodes[0].compute_segment()
        for parent_ind in range(num_nodes):
            for child_ind in range(num_nodes):
                if adj_matrix[parent_ind][child_ind] == 1:
                    nodes[parent_ind].add_child(nodes[child_ind])
        
        # Now, we need to reconstruct the morphology to determine collisions.
        new_root = copy.copy(nodes[0])
        new_root.compute_segment()
        queue = [(nodes[0], new_root)]

        while len(queue) != 0:
            old_node, new_node = queue.pop()
            for old_child in old_node._children:
                new_child = copy.copy(old_child)
                new_child.compute_segment(parent=new_node)               
                if any([new_child.intersects(node) for node in new_root]):
                    return None   
                new_node.add_child(new_child)
                queue.append((old_child, new_child))
        
        # Arbitrarily? Choose the root to be the zeroth node.
        return cls(new_root, two_dim=two_dim, one_dim=one_dim, geom_kwargs=geom_kwargs, 
                                     joint_kwargs=joint_kwargs, global_kwargs=global_kwargs, node_kwargs=node_kwargs)

    @classmethod
    def generate_random_morphology(cls, two_dim=False, one_dim=False, node_kwargs={},
                geom_kwargs={}, joint_kwargs={}, global_kwargs={}, mutation_kwargs={}, child_prob=0.4):
        '''
        Note: Generating a random morphology does not randomize the following values.
            - Gear
        Nodes are generated in BFS order

        Mutation Kwargs contain information about the mutation parameters
        mutation corresponds to anything that is not stored in the embedding of a morphology (nodes features + adj matrix)
        '''
        min_nodes = mutation_kwargs['min_nodes'] if 'min_nodes' in mutation_kwargs else 2
        max_nodes = mutation_kwargs['max_nodes'] if 'max_nodes' in mutation_kwargs else 11
        max_children = mutation_kwargs['max_children'] if 'max_children' in mutation_kwargs else 3
        joint_prob = mutation_kwargs['joint_prob'] if 'joint_prob' in mutation_kwargs else 0.8
        
        def generate():
            root = Node.generate_random_node(two_dim=two_dim, one_dim=one_dim, joint_prob=joint_prob, **node_kwargs)
            root._joint = 0 # Ensure the root does not have a moveable joint.
            root.compute_segment()
            # Create the segment for the root.
            num_nodes, num_joints = 1, 0
            # Generate in BFS Order
            queue = [root]
            while len(queue) != 0 and num_nodes < max_nodes:
                cur_node = queue.pop()
                generated_children = []
                for _ in range(max_children):
                    if num_nodes == max_nodes:
                        break
                    elif random.random() < child_prob:
                        # We have elected to create a child, now must ensure that we generate a node that doesn't collide.
                        for _ in range(200): # Attempt to generate 100 different children.
                            child = Node.generate_random_node(two_dim=two_dim, one_dim=one_dim, joint_prob=joint_prob, **node_kwargs)
                            child.compute_segment(parent=cur_node)
                            if all([not child.intersects(node) for node in root]):                                
                                cur_node.add_child(child)
                                queue.append(child)
                                num_nodes += 1
                                if child.joint != 0:
                                    num_joints += 1
                                break
            return root, num_nodes, num_joints
        
        root, num_nodes, num_joints = None, 0, 0
        while num_nodes < min_nodes or num_joints < 3:# Minimum of three joints
            root, num_nodes, num_joints = generate()

        return cls(root, two_dim=two_dim, one_dim=one_dim, geom_kwargs=geom_kwargs, joint_kwargs=joint_kwargs, global_kwargs=global_kwargs, node_kwargs=node_kwargs)

    def mutate(self, max_children=2, max_nodes=10, min_nodes=3, node_prob=0.5, remove_prob=0.01, gen_prob=0.01, geom_mut=0.3,
               joint_prob=0.8, joint_mut=0.15, joint_type_mut=0.5, extent_std=None, radius_std=None, attachment_std=None, gear_std=None,
               joint_std=None):
        
        # Step 1: Traverse self.root in BFS order, if we mutate, keep mutating until we don't have collisions
        #           Note: collision detection isnt perfect as changes in upper nodes could prevent it from working.
        # Step 2: Attempt to modify the morphology by adding / removing leaf limbs
        #   first remove 
        #   next recompute leaves, then re-add
        root = self.root

        had_mutation = False
        if random.random() < node_prob and not self.one_dim:
            new_root = root.mutate(two_dim=self.two_dim, one_dim=self.one_dim, geom_mut=geom_mut, joint_prob=joint_prob, joint_mut=joint_mut, joint_type_mut=joint_type_mut, 
                                    extent_std=extent_std, radius_std=radius_std, attachment_std=attachment_std, gear_std=gear_std, joint_std=joint_std)
            had_mutation = True
        else:
            new_root = copy.copy(root)
        
        new_root.compute_segment()
        queue = [(root, new_root)]
        # Mutate everything
        while len(queue) != 0:
            old_node, new_node = queue.pop()
            for old_child in old_node._children:
                if random.random() < node_prob:
                    for _ in range(100):
                        new_child = old_child.mutate(two_dim=self.two_dim, one_dim=self.one_dim, geom_mut=geom_mut, joint_prob=joint_prob, joint_mut=joint_mut, joint_type_mut=joint_type_mut, extent_std=extent_std,
                                        radius_std=radius_std, attachment_std=attachment_std, gear_std=gear_std, joint_std=joint_std)
                        new_child.compute_segment(parent=new_node)
                        if all([not new_child.intersects(node) for node in new_root]):
                            had_mutation = True
                            break
                    else:
                        new_child = copy.copy(old_child)
                        new_child.compute_segment(parent=new_node)
                else:
                    new_child = copy.copy(old_child)
                    new_child.compute_segment(parent=new_node)
                new_node.add_child(new_child)
                queue.append((old_child, new_child))
        
        # Remove nodes
        queue = [new_root]
        while len(queue) > 0:
            cur_node = queue.pop()
            to_del = []
            for i, child in enumerate(cur_node._children):
                if len(child._children) == 0 and random.random() < remove_prob and len(new_root) > min_nodes:
                    had_mutation = True
                    to_del.append(i)
            for i in sorted(to_del, reverse=True):
                del cur_node._children[i]
            for child in cur_node._children:
                queue.append(child)
        
        # Generation: don't allow recursive so just try for single layer
        can_gen = [node for node in new_root if len(node._children) < max_children]
        for node in can_gen:
            if random.random() < gen_prob and len(new_root) < max_nodes:
                for _ in range(250): # Attempt to generate 250 different children. Only add if it doesn't intersect.
                    child = Node.generate_random_node(two_dim=self.two_dim, one_dim=self.one_dim, joint_prob=joint_prob, **self.node_kwargs)
                    child.compute_segment(parent=cur_node)
                    if all([not child.intersects(node) for node in new_root]):
                        cur_node.add_child(child)
                        had_mutation = True
                        break
        
        if had_mutation:
            return Morphology(new_root, two_dim=self.two_dim, one_dim=self.one_dim, geom_kwargs=self.geom_kwargs, joint_kwargs=self.joint_kwargs, 
                          global_kwargs=self.global_kwargs, node_kwargs=self.node_kwargs)
        else:
            return self.mutate(max_children=max_children, max_nodes=max_nodes, min_nodes=min_nodes, node_prob=node_prob, remove_prob=remove_prob, 
                                gen_prob=gen_prob, geom_mut=geom_mut, joint_prob=joint_prob, joint_mut=joint_mut, joint_type_mut=joint_type_mut, extent_std=extent_std,
                                radius_std=radius_std, attachment_std=attachment_std, gear_std=gear_std, joint_std=joint_std)

    def construct(self, arena=None, morphology_height=None):
        if morphology_height is None:
            # Determine the morphology height.
            # segments contains the morphology positions. Find the minimum z, add morphology height to make it > thres
            min_z = min(np.min(self.segment_embeddings[:, 2]), np.min(self.segment_embeddings[:, 5]))
            morphology_height = max(0.1, 0.1 - min_z) # Make morphology height at least 1, larger if needed.
        if arena is None:
            arena = mjcf.RootElement(model="morphology")
            arena.asset.add('texture', name="texplane", builtin='checker', height=300, rgb1=[0.1, 0.2, 0.3], rgb2=[0.2, 0.3, 0.4], type="2d", width=300)
            arena.asset.add('material', name="MatPlane", reflectance=0.5, shininess=1, specular=1, texrepeat=[20, 20], texture="texplane")
            arena.worldbody.add('light', cutoff=100, diffuse=[1,1,1], dir=[-0, 0, -1.3], directional=True, exponent=1, pos=[0,0,1.5], specular=[.1,.1,.1])
            arena.worldbody.add('geom', name='ground', type='plane', size=[15, 15, .5], rgba=(1, 1, 1, 1), material="MatPlane")
        else:
            if hasattr(arena, "morphology_height"):
                morphology_height += arena.morphology_height
            arena = arena.construct()

        # TODO: Set global kwargs. Find a better way to do this.
        if 'compiler.settotalmass' in self.global_kwargs:
            arena.compiler.settotalmass = self.global_kwargs['compiler.settotalmass']
        if 'compiler.angle' in self.global_kwargs:
            arena.compiler.angle = self.global_kwargs['compiler.angle']
        if 'option.timestep' in self.global_kwargs:
            arena.option.timestep = self.global_kwargs['option.timestep']
        if 'option.integrator' in self.global_kwargs:
            arena.option.integrator = self.global_kwargs['option.integrator']
        if 'compiler.angle' in self.global_kwargs:
            arena.compiler.angle = self.global_kwargs['compiler.angle']
        if 'compiler.coordinate' in self.global_kwargs:
            arena.compiler.coordinate = self.global_kwargs['compiler.coordinate']
        if 'compiler.inertiafromgeom' in self.global_kwargs:
            arena.compiler.inertiafromgeom = self.global_kwargs['compiler.inertiafromgeom']

        morphology_attachment_site = arena.worldbody.add('site', name='body', size=[1e-6]*3, pos=[0, 0, morphology_height])
        morphology = self.root.construct(geom_kwargs=self.geom_kwargs, joint_kwargs=self.joint_kwargs, two_dim=self.two_dim)

        # attach camera to morphology
        if self.one_dim:
            morphology.worldbody.add('camera', name="top", pos=[1.2, 0, 3.0], quat=[1, 0, 0, 0])
        elif self.two_dim:
            morphology.worldbody.add('camera', name="side", pos=[0.4, -4, 0.2], quat=[0.707, 0.707, 0, 0], mode="trackcom")
        else: # its three dimensional
            morphology.worldbody.add('camera', name="iso", pos=[-2.3, -2.3, 1.0], xyaxes=[0.45, -0.45, 0, 0.3, 0.15, 0.94], mode="trackcom")

        attachment_frame = morphology_attachment_site.attach(morphology)
        if self.two_dim:
            attachment_frame.add('joint', name='rootx', type='slide', axis=[1, 0, 0], limited=False, damping=0, armature=0, stiffness=0)
            attachment_frame.add('joint', name='rootz', type='slide', axis=[0, 0, 1], limited=False, damping=0, armature=0, stiffness=0)
            attachment_frame.add('joint', name='rooty', type='hinge', axis=[0, 1, 0], limited=False, damping=0, armature=0, stiffness=0)
        elif self.one_dim:
            pass
        else:
            attachment_frame.add('freejoint')

        # Add sensors, assume the last body in worldbody is the morphology root
        arena.sensor.add("subtreelinvel", name="velocity", body=arena.worldbody.body[-1])

        return arena
    
    def save_xml(self, path, arena=None):
        arena = self.construct(arena=arena)
        xml = arena.to_xml_string()
        if not path.endswith(".xml"):
            path += ".xml"
        with open(path, 'w+') as f:
            f.write(xml)
        print("Wrote XML to", path)

    def save(self, path):
        if not path.endswith(".pkl"):
            path += ".pkl"
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path):
        if not path.endswith(".pkl"):
            path += ".pkl"
        with open(path, 'rb') as f:
            return pickle.load(f)

    @property
    def node_embeddings(self):
        return self._node_embeddings

    @property
    def edge_embeddings(self):
        return self._edge_embeddings

    @property
    def segment_embeddings(self):
        return self._segment_embeddings
    
    @property
    def edge_list(self):
        return self._edge_list

    @property
    def adj_matrix(self):
        return self._adj_matrix

    @property
    def joint_map(self):
        return self._joint_map
    
    @property
    def num_joints(self):
        return self._num_joints

    @property
    def end_site_indices(self):
        return self._site_pos_list

    def __str__(self):
        return self.root.__str__()

    def __len__(self):
        return self._length

    def expand(self, arr):
        # return a new array of shape (nodes,) that zero-pads nodes without joints.
        assert arr.shape == (self.num_joints,), "Incorrect shape passed to expand"
        new_arr = np.zeros(len(self) - 1) # Expand to num edges
        new_arr[self._joint_map == 1] = arr
        return new_arr

    def shrink(self, arr):
        # return a new array of shape (nodes with joints,) from an array of shape (nodes)
        # This is most likely used in action outputs
        assert arr.shape == (len(self) - 1,), "Input to shrink must be equal to number of edges"
        return arr[self._joint_map == 1]

    def get_kwargs(self):
        return {
            'one_dim' : self.one_dim,
            'two_dim' : self.two_dim,
            'geom_kwargs' : self.geom_kwargs,
            'joint_kwargs': self.joint_kwargs,
            'global_kwargs': self.global_kwargs,
            'node_kwargs' : self.node_kwargs
        }
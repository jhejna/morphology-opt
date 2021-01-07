
from .morphology import Node, Morphology
import random

def random2d(**kwargs):
    global_kwargs = {"option.timestep": 0.01}
    geom_kwargs = {"contype": 1,
                   "conaffinity": 1,
                   "condim": 3,
                   "friction": [0.6, 0.1, 0.1]}
    joint_kwargs = {"damping" : 3.5,
                    "armature" : 0.2,
                    "stiffness": 100}
    node_kwargs = {
        "extent_range" : 0.75,
        "radius_range" : [0.04, 0.07],
        "joint_range" : (30, 70),
        "gear_range" : (50, 90),
    }
    morphology = Morphology.generate_random_morphology(two_dim=True, node_kwargs=node_kwargs,
                geom_kwargs=geom_kwargs, joint_kwargs=joint_kwargs, global_kwargs=global_kwargs, 
                mutation_kwargs=kwargs['mutation_kwargs'], child_prob=0.3)
    while morphology.num_joints < 2: # Randomly generated agents should have at least two joints.
        morpholgoy = Morphology.generate_random_morphology(two_dim=True, node_kwargs=node_kwargs,
                geom_kwargs=geom_kwargs, joint_kwargs=joint_kwargs, global_kwargs=global_kwargs, 
                mutation_kwargs=kwargs['mutation_kwargs'], child_prob=0.3)
    return morphology

def random3d(**kwargs):
    global_kwargs = {"option.timestep": 0.01}
    geom_kwargs = {"contype": 1,
                   "conaffinity": 1,
                   "condim": 3,
                   "friction": [0.6, 0.1, 0.1]}
    joint_kwargs = {"damping" : 3.5,
                    "armature" : 0.2,
                    "stiffness": 100}
    node_kwargs = {
        "extent_range" : 0.75,
        "radius_range" : [0.035, 0.07],
        "joint_range" : (30, 70),
        "gear_range" : (50, 100), # Was 50-80
    }

    morphology = Morphology.generate_random_morphology(two_dim=False, node_kwargs=node_kwargs,
                geom_kwargs=geom_kwargs, joint_kwargs=joint_kwargs, global_kwargs=global_kwargs, 
                mutation_kwargs=kwargs['mutation_kwargs'], child_prob=0.32)
    while morphology.num_joints < 2:
        morpholgoy = Morphology.generate_random_morphology(two_dim=False, node_kwargs=node_kwargs,
                geom_kwargs=geom_kwargs, joint_kwargs=joint_kwargs, global_kwargs=global_kwargs, 
                mutation_kwargs=kwargs['mutation_kwargs'], child_prob=0.32)
    return morphology

def random_arm(**kwargs):
    global_kwargs = {"option.timestep": 0.01,
                     "option.integrator": "RK4"}
    geom_kwargs = {"contype": 1,
                   "conaffinity": 1,
                   "condim": 3,
                   "friction": [0.7, 0.1, 0.1]}
    joint_kwargs = {"damping" : 1,
                    "armature" : 1}
    node_kwargs = {
        "extent_range" : 0.75,
        "radius_range" : [0.04, 0.07],
        "joint_range" : (90, 175),
        "gear_range" : (70, 80),
        "only_end" : True
    }

    min_nodes = 3
    max_nodes = kwargs['mutation_kwargs']['max_nodes'] if 'max_nodes' in kwargs['mutation_kwargs'] else 6
    num_nodes = random.randint(min_nodes, max_nodes)
    root = Node(extent=[0.001, 0, 0], radius=0.0699, joint=0, **node_kwargs)
    morphology = Morphology(root, two_dim=False, one_dim=True, geom_kwargs=geom_kwargs, joint_kwargs=joint_kwargs, global_kwargs=global_kwargs, node_kwargs=node_kwargs)
    # Mutate until we hit num nodes
    while len(morphology) < num_nodes:
        test_morphology = morphology.mutate(**kwargs['mutation_kwargs'])
        if len(test_morphology) > len(morphology):
            morphology = test_morphology
    return morphology

def reacher(**kwargs):
    global_kwargs = {"option.timestep": 0.01,
                     "option.integrator": "RK4"}
    geom_kwargs = {"contype": 1,
                   "conaffinity": 1,
                   "condim": 3,
                   "friction": [0.7, 0.1, 0.1]}
    joint_kwargs = {"damping" : 1,
                    "armature" : 1}
    node_kwargs = {
        "extent_range" : 0.75,
        "radius_range" : [0.04, 0.07],
        "joint_range" : (90, 175),
        "gear_range" : (70, 80),
        "only_end" : True
    }

    root = Node(extent=[0.001, 0, 0], radius=0.0699, joint=0, **node_kwargs)
    link1 = Node(extent=[0.7, 0.0, 0.0], radius=0.04, joint=3, joint_limit=140, gear=80, **node_kwargs)
    root.add_child(link1)
    link2 = Node(extent=[0.7, 0.0, 0.0], radius=0.04, joint=3, joint_limit=140, gear=80, **node_kwargs)
    link1.add_child(link2)
    end = Node(extent=[0.001, 0.0, 0.0], radius=0.05, joint=3, joint_limit=140, gear=80, **node_kwargs)
    link2.add_child(end)

    return Morphology(root, two_dim=False, one_dim=True, geom_kwargs=geom_kwargs, joint_kwargs=joint_kwargs, global_kwargs=global_kwargs, node_kwargs=node_kwargs)

def cheetah(**kwargs):
    if 'two_dim' in kwargs:
        assert kwargs['two_dim'], "When using cheetah must be two dim"
    
    global_kwargs = {
        "compiler.settotalmass" : 14,
        "option.timestep": 0.01
    }
    geom_kwargs = {
        "contype": 1,
        "conaffinity": 1,
        "condim": 3,
        "friction": [0.4, 0.1, 0.1]
    }

    joint_kwargs = {
        "damping" : 3.75,
        "armature" : 0.1,
        "stiffness": 160
    }

    node_kwargs = {
        "extent_range" : 1.25,
        "radius_range" : [0.025, 0.06],
        "joint_range" : (30, 100),
        "gear_range" : (50, 110),
    }

    root = Node(extent=[0.999, 0, 0], radius=0.046, joint=0, **node_kwargs)
    head = Node(extent=[0.1928, 0, 0.2298], radius=0.046, attachment_point=1, joint=0, **node_kwargs)
    root.add_child(head)
    
    bthigh = Node(extent=[0.17854, 0, -.2285], radius=0.046, attachment_point=0, joint=2, joint_limit=45, gear=110, **node_kwargs)
    bshin = Node(extent=[-0.26964, 0, -0.13152], radius=0.046, attachment_point=1, joint=2, joint_limit=50, gear=90, **node_kwargs)
    bthigh.add_child(bshin)
    bfoot = Node(extent=[0.0491756, 0, -0.183526], radius=0.046, attachment_point=1, joint=2, joint_limit=100, gear=60, **node_kwargs)
    bshin.add_child(bfoot)
    root.add_child(bthigh)

    fthigh = Node(extent=[-0.133, 0, -0.23036], radius=0.046, attachment_point=1, joint=2, joint_limit=30, gear=90, **node_kwargs)
    fshin = Node(extent=[0.118548, 0, -0.17576], radius=0.046, attachment_point=1, joint=2, joint_limit=60, gear=60, **node_kwargs)
    fthigh.add_child(fshin)
    ffoot = Node(extent=[0.078288, 0, -0.116066], radius=0.046, attachment_point=1, joint=2, joint_limit=30, gear=50, **node_kwargs)
    fshin.add_child(ffoot)
    root.add_child(fthigh)

    return Morphology(root, two_dim=True, geom_kwargs=geom_kwargs, joint_kwargs=joint_kwargs, global_kwargs=global_kwargs, node_kwargs=node_kwargs)

def cheetah_like(**kwargs):
    morphology = cheetah(**kwargs)
    morphology = morphology.mutate(**kwargs['mutation_kwargs'])
    morphology = morphology.mutate(**kwargs['mutation_kwargs'])
    return morphology
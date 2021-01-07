from dm_control import mjcf
import random
import numpy as np
import scipy

class GM_Terrain(object):

    def __init__(self, n=8, max_height=0.2):
        self.n = n
        self.max_height = max_height
        self.morphology_height = 0.16

    def construct(self):
        max_spread = 0.5
        min_var, max_var = 0.2, 0.8
        min_pos, max_pos = -4, 4
        samples = 60
        pdfs = [scipy.stats.norm(random.uniform(min_pos, max_pos), random.uniform(min_var, max_var)) for _ in range(self.n)]
        x = np.linspace(min_pos, max_pos, num=samples)
        y = np.zeros(x.shape)
        for pdf in pdfs:
            y += pdf.pdf(x)
        y /= np.max(y)
        y *= self.max_height

        arena = mjcf.RootElement(model="morphology")
        arena.asset.add('texture', name="texplane", builtin='checker', height=300, rgb1=[0.1, 0.2, 0.3], rgb2=[0.2, 0.3, 0.4], type="2d", width=300)
        arena.asset.add('material', name="MatPlane", reflectance=0.5, shininess=1, specular=1, texrepeat=[20, 20], texture="texplane")
        arena.worldbody.add('light', cutoff=100, diffuse=[1, 1, 1], dir=[-0, 0, -1.3], directional=True, exponent=1, pos=[0,0,1.5], specular=[.1,.1,.1])
        arena.worldbody.add('geom', name='ground', type='plane', size=[15, 15, .5], rgba=(1, 1, 1, 1), material="MatPlane")

        width = (max_pos - min_pos)/(2*samples)
        for i, xval, yval in zip(range(samples), x, y):
            geom_name = "terrain" + str(i)
            size = [width, 0.25, yval/2]
            pos = [xval, 0, yval/2]
            arena.worldbody.add('geom', name=geom_name, type='box', size=size, pos=pos, rgba=(.4, .4, .4, 1))

        return arena

class GM_Terrain3D(object):

    def __init__(self, n=10, max_height=0.25):
        self.n = n
        self.max_height = max_height
        self.morphology_height = 0.18

    def construct(self):
        max_spread = 0.5
        min_var, max_var = 0.3, 1.2
        min_pos, max_pos = -4, 4
        samples = 30
        
        pdfs = []
        for _ in range(self.n):
            mean = np.array([random.uniform(min_pos, max_pos), random.uniform(min_pos, max_pos)])
            v1, v2, cov = 0, 0, 100
            while abs(cov) > v1 or abs(cov) > v2:
                v1, v2, cov = random.uniform(min_var, max_var), random.uniform(min_var, max_var), (random.random()-0.5) * random.uniform(min_var, max_var)
            
            cov_mat = np.array([[v1, cov], [cov, v2]])
            pdfs.append(scipy.stats.multivariate_normal(mean, cov_mat))
        
        x = np.linspace(min_pos, max_pos, num=samples)
        y = np.linspace(min_pos, max_pos, num=samples)
        xx, yy = np.meshgrid(x, y, sparse=False)
        z = np.zeros((x.shape[0], y.shape[0]))
        pos = np.dstack((xx, yy))
        for pdf in pdfs:
            z += pdf.pdf(pos)
        z /= np.max(z)
        z *= self.max_height
        width = (max_pos - min_pos)/(2*samples)

        arena = mjcf.RootElement(model="morphology")
        arena.asset.add('texture', name="texplane", builtin='checker', height=300, rgb1=[0.1, 0.2, 0.3], rgb2=[0.2, 0.3, 0.4], type="2d", width=300)
        arena.asset.add('material', name="MatPlane", reflectance=0.5, shininess=1, specular=1, texrepeat=[20, 20], texture="texplane")
        arena.worldbody.add('light', cutoff=100, diffuse=[1, 1, 1], dir=[-0, 0, -1.3], directional=True, exponent=1, pos=[0,0,1.5], specular=[.1,.1,.1])
        arena.worldbody.add('geom', name='ground', type='plane', size=[15, 15, .5], rgba=(1, 1, 1, 1), material="MatPlane")

        for i, xval, yval, zval in zip(range(samples**2), xx.flatten(), yy.flatten(), z.flatten()):
            geom_name = "terrain" + str(i)
            size = [width, width, zval/2]
            pos = [xval, yval, zval/2]
            arena.worldbody.add('geom', name=geom_name, type='box', size=size, pos=pos, rgba=(.4, .4, .4, 1))
        
        return arena

class ReachTarget(object):

    def construct(self):
        arena = mjcf.RootElement(model="morphology")
        arena.asset.add('texture', name="texplane", builtin='checker', height=300, rgb1=[0.1, 0.2, 0.3], rgb2=[0.2, 0.3, 0.4], type="2d", width=300)
        arena.asset.add('material', name="MatPlane", reflectance=0.5, shininess=1, specular=1, texrepeat=[20, 20], texture="texplane")
        arena.worldbody.add('light', cutoff=100, diffuse=[1, 1, 1], dir=[-0, 0, -1.3], directional=True, exponent=1, pos=[0,0,1.5], specular=[.1,.1,.1])
        arena.worldbody.add('geom', name='ground', type='plane', size=[15, 15, .5], rgba=(1, 1, 1, 1), material="MatPlane")
        arena.worldbody.add('geom', name='target', type='sphere', size=(0.08,), pos=[0, 0, 0.04], rgba=(.8, .4, .4, .6), conaffinity=0, contype=0)
        return arena

class ReachBox(object):

    def construct(self):
        arena = mjcf.RootElement(model="morphology")
        arena.asset.add('texture', name="texplane", builtin='checker', height=300, rgb1=[0.1, 0.2, 0.3], rgb2=[0.2, 0.3, 0.4], type="2d", width=300)
        arena.asset.add('material', name="MatPlane", reflectance=0.5, shininess=1, specular=1, texrepeat=[20, 20], texture="texplane")
        arena.worldbody.add('light', cutoff=100, diffuse=[1, 1, 1], dir=[-0, 0, -1.3], directional=True, exponent=1, pos=[0,0,1.5], specular=[.1,.1,.1])
        arena.worldbody.add('geom', name='ground', type='plane', size=[15, 15, .5], rgba=(1, 1, 1, 1), material="MatPlane")
        
        block1 = arena.worldbody.add('body', name='block1', pos=[0.9, 0.65, 0.0])
        block1.add('geom', name='block1', type='box', size=[0.1, 0.1, 0.1], pos=[0, 0, 0.1], rgba=(.8, .3, .3, 1), contype=1, conaffinity=1, condim=3)
        block1.add('joint', name='block1x', type='slide', axis=[1, 0, 0], limited=False, damping=0, armature=0, stiffness=0)
        block1.add('joint', name='block1y', type='slide', axis=[0, 1, 0], limited=False, damping=0, armature=0, stiffness=0)
        block1.add('joint', name='block1z', type='slide', axis=[0, 0, 1], limited=True, damping=0, armature=0, stiffness=0, range=(-0.02, 0.02))

        block2 = arena.worldbody.add('body', name='block2', pos=[0.9, -0.65, 0])
        block2.add('geom', name='block2', type='box', size=[0.1, 0.1, 0.1], pos=[0, 0, 0.1], rgba=(.8, .3, .3, 1), contype=1, conaffinity=1, condim=3)
        block2.add('joint', name='block2x', type='slide', axis=[1, 0, 0], limited=False, damping=0, armature=0, stiffness=0)
        block2.add('joint', name='block2y', type='slide', axis=[0, 1, 0], limited=False, damping=0, armature=0, stiffness=0)
        block2.add('joint', name='block2z', type='slide', axis=[0, 0, 1], limited=True, damping=0, armature=0, stiffness=0, range=(-0.02, 0.02))

        return arena

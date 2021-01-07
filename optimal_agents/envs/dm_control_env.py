'''
Environments based off of DM Control for testing with DM Control Viewer.
They are not used for RL (observations are None)
'''
from dm_control.rl import control
from dm_control import mjcf
from dm_control.suite import base
from dm_control.utils import rewards

class Physics(mjcf.Physics):
    
    def speed(self):
        return self.named.data.sensordata['velocity'][0]

class MoveX(base.Task):
    ''' Task for Moving along positive x-axis.'''
    _RUN_SPEED = 10

    def __init__(self, random=None):
        super(MoveX, self).__init__(random=random)

    def get_observation(self, physics):
        return None
        
    def get_reward(self, physics):
        return rewards.tolerance(physics.speed(),
                             bounds=(self._RUN_SPEED, float('inf')),
                             margin=self._RUN_SPEED,
                             value_at_margin=0,
                             sigmoid='linear')

def dm_control_test_env(morphology, arena=None):
    physics = Physics.from_mjcf_model(morphology.construct(arena=arena))
    physics.morphology = morphology
    task = MoveX()
    return control.Environment(physics, task, time_limit=20)

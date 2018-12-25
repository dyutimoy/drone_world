from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class VistCustomer(MiniGridEnv):
    """
    Enivornment in which vehicle vists multiple nodes
    and collects  reward
    """

    def __init__(
        self,
        size=10,
        numObjs=4
    ):
        self.numObjs=numObjs

        super().__init__(
            grid_size=size,
            max_steps=20*size,
            see_through_walls=True   #see what it means
        )

    def _gen_grid(self, width,height):
        self.grid = Grid(width,height)

        #generate surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height-1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width-1, 0)

        types=['customer','goal']


        objs = []
        objPos = []

        def near_obj(env, p1):
            for p2 in objPos:
                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                if abs(dx) <= 1 and abs(dy) <= 1:
                    return True
            return False


        objType = 'customer'
        objColor = 'grey'

                # If this object already exists, try again
        obj = Customer(objColor)
        """
        pos = self.place_obj(obj)
        pos = self.place_obj(obj)
        pos = self.place_obj(obj)
        """
        pos = self.place_obj_det(obj,3,3)
        pos = self.place_obj_det(obj,2,7)
        pos = self.place_obj_det(obj,4,6)


        objType='goal'
        objColor = 'green'
        obj = Goal()
        pos = self.place_obj_det(obj,8,8)




        self.place_agent()

        self.mission='Pick all objects'

    def step(self, action):


        obs, reward, done,info = super().step(action)



        return obs, reward, done, info

register(
    id='MiniGrid-vrp-v0',
    entry_point='gym_minigrid.envs:VistCustomer'
)
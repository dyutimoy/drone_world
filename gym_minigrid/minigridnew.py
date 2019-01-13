import math
import gym
from enum import IntEnum
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

# Size in pixels of a cell in the full-scale human view
CELL_PIXELS = 32

# Number of cells (width and height) in the agent view
AGENT_VIEW_SIZE = 30



# Map of color names to RGB values
COLORS = {
    'red'   : np.array([255, 0, 0]),
    'green' : np.array([0, 255, 0]),
    'blue'  : np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey'  : np.array([100, 100, 100])
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {
    'red'   : 0,
    'green' : 1,
    'blue'  : 2,
    'purple': 3,
    'yellow': 4,
    'grey'  : 5
}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of object type to integers
OBJECT_TO_IDX = {
    'empty'         : 0,
    'wall'          : 1,
    'floor'         : 2,
    'door'          : 3,
    'locked_door'   : 4,
    'key'           : 5,
    'ball'          : 6,
    'box'           : 7,
    'goal'          : 8,
    'customer'      : 9
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

# Map of agent direction indices to vectors


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type, color):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_flyover(self):
        """Can a=the agent fly over? """
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False
    def can_visit(self):
        """can collect it"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        return False

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError

    def _set_color(self, r):
        """Set the color of this object as the active drawing color"""
        c = COLORS[self.color]
        r.setLineColor(c[0], c[1], c[2])
        r.setColor(c[0], c[1], c[2])

class Goal(WorldObj):
    def __init__(self):
        super().__init__('goal', 'green')

    def can_overlap(self):
        return True

    def render(self, r):
        self._set_color(r)
        r.drawPolygon([
            (0          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           0),
            (0          ,           0)
        ])

class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, color='blue'):
        super().__init__('floor', color)

    def can_overlap(self):
        return True

    def render(self, r):
        # Give the floor a pale color
        c = COLORS[self.color]
        r.setLineColor(100, 100, 100, 0)
        r.setColor(*c/2)
        r.drawPolygon([
            (1          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           1),
            (1          ,           1)
        ])
class House(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, color='yellow'):
        super().__init__('floor', color)

    def can_overlap(self):
        return False

    def render(self, r):
        # Give the floor a pale color
        c = COLORS[self.color]
        r.setLineColor(100, 100, 100, 0)
        r.setColor(*c)
        r.drawPolygon([
            (1          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           1),
            (1          ,           1)
        ])
class Wall(WorldObj):
    def __init__(self, color='grey'):
        super().__init__('wall', color)

    def see_behind(self):
        return False

    def render(self, r):
        self._set_color(r)
        r.drawPolygon([
            (0          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           0),
            (0          ,           0)
        ])
class Door(WorldObj):
    def __init__(self, color, is_open=False):
        super().__init__('door', color)
        self.is_open = is_open

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos):
        self.is_open = not self.is_open
        return True

    def render(self, r):
        c = COLORS[self.color]
        r.setLineColor(c[0], c[1], c[2])
        r.setColor(0, 0, 0)

        if self.is_open:
            r.drawPolygon([
                (CELL_PIXELS-2, CELL_PIXELS),
                (CELL_PIXELS  , CELL_PIXELS),
                (CELL_PIXELS  ,           0),
                (CELL_PIXELS-2,           0)
            ])
            return

        r.drawPolygon([
            (0          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           0),
            (0          ,           0)
        ])
        r.drawPolygon([
            (2          , CELL_PIXELS-2),
            (CELL_PIXELS-2, CELL_PIXELS-2),
            (CELL_PIXELS-2,           2),
            (2          ,           2)
        ])
        r.drawCircle(CELL_PIXELS * 0.75, CELL_PIXELS * 0.5, 2)
class LockedDoor(WorldObj):
    def __init__(self, color, is_open=False):
        super(LockedDoor, self).__init__('locked_door', color)
        self.is_open = is_open

    def toggle(self, env, pos):
        # If the player has the right key to open the door
        if isinstance(env.carrying, Key) and env.carrying.color == self.color:
            self.is_open = True
            # The key has been used, remove it from the agent
            env.carrying = None
            return True
        return False

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    def render(self, r):
        c = COLORS[self.color]
        r.setLineColor(c[0], c[1], c[2])
        r.setColor(c[0], c[1], c[2], 50)

        if self.is_open:
            r.drawPolygon([
                (CELL_PIXELS-2, CELL_PIXELS),
                (CELL_PIXELS  , CELL_PIXELS),
                (CELL_PIXELS  ,           0),
                (CELL_PIXELS-2,           0)
            ])
            return

        r.drawPolygon([
            (0          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           0),
            (0          ,           0)
        ])
        r.drawPolygon([
            (2          , CELL_PIXELS-2),
            (CELL_PIXELS-2, CELL_PIXELS-2),
            (CELL_PIXELS-2,           2),
            (2          ,           2)
        ])
        r.drawLine(
            CELL_PIXELS * 0.55,
            CELL_PIXELS * 0.5,
            CELL_PIXELS * 0.75,
            CELL_PIXELS * 0.5
        )
class Key(WorldObj):
    def __init__(self, color='blue'):
        super(Key, self).__init__('key', color)

    def can_pickup(self):
        return True

    def render(self, r):
        self._set_color(r)

        # Vertical quad
        r.drawPolygon([
            (16, 10),
            (20, 10),
            (20, 28),
            (16, 28)
        ])

        # Teeth
        r.drawPolygon([
            (12, 19),
            (16, 19),
            (16, 21),
            (12, 21)
        ])
        r.drawPolygon([
            (12, 26),
            (16, 26),
            (16, 28),
            (12, 28)
        ])

        r.drawCircle(18, 9, 6)
        r.setLineColor(0, 0, 0)
        r.setColor(0, 0, 0)
        r.drawCircle(18, 9, 2)
class Ball(WorldObj):
    def __init__(self, color='blue'):
        super(Ball, self).__init__('ball', color)

    def can_pickup(self):
        return True

    def render(self, r):
        self._set_color(r)
        r.drawCircle(CELL_PIXELS * 0.5, CELL_PIXELS * 0.5, 10)

class Customer(WorldObj):
    def __init__(self, color='red'):
        super(Customer, self).__init__('customer', color)

    def can_overlap(self):
        return True

    def render(self, r):
        self._set_color(r)
        r.drawCircle(CELL_PIXELS * 0.5, CELL_PIXELS * 0.5, 10)
        r.drawPolygon([
            (4            , CELL_PIXELS-4),
            (CELL_PIXELS-4, CELL_PIXELS-4),
            (CELL_PIXELS-4,             4),
            (4            ,             4)
        ])

class Box(WorldObj):
    def __init__(self, color, contains=None):
        super(Box, self).__init__('box', color)
        self.contains = contains

    def can_pickup(self):
        return True

    def render(self, r):
        c = COLORS[self.color]
        r.setLineColor(c[0], c[1], c[2])
        r.setColor(0, 0, 0)
        r.setLineWidth(2)

        r.drawPolygon([
            (4            , CELL_PIXELS-4),
            (CELL_PIXELS-4, CELL_PIXELS-4),
            (CELL_PIXELS-4,             4),
            (4            ,             4)
        ])

        r.drawLine(
            4,
            CELL_PIXELS / 2,
            CELL_PIXELS - 4,
            CELL_PIXELS / 2
        )

        r.setLineWidth(1)

    def toggle(self, env, pos):
        # Replace the box by its contents
        env.grid.set(*pos, self.contains)
        return True

class Grid:
    """
    Represent a grid and operations on it
    """

    def __init__(self, width, height):
        assert width >= 4
        assert height >= 4

        self.width = width
        self.height = height

        self.grid = [None] * width * height

    def __contains__(self, key):
        if isinstance(key, WorldObj):
            for e in self.grid:
                if e is key:
                    return True
        elif isinstance(key, tuple):
            for e in self.grid:
                if e is None:
                    continue
                if (e.color, e.type) == key:
                    return True
                if key[0] is None and key[1] == e.type:
                    return True
        return False

    def __eq__(self, other):
        grid1 = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other):
        return not self == other

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def set(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i] = v

    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.grid[j * self.width + i]

    def horz_wall(self, x, y, length=None):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, Wall())

    def vert_wall(self, x, y, length=None):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, Wall())

    def wall_rect(self, x, y, w, h):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y+h-1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x+w-1, y, h)

    def rotate_left(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = Grid(self.width, self.height)

        for j in range(0, self.height):
            for i in range(0, self.width):
                v = self.get(self.width - 1 - j, i)
                grid.set(i, j, v)

        return grid

    def slice(self, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = Grid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and \
                   y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall()

                grid.set(i, j, v)

        return grid

    def render(self, r, tile_size):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        assert r.width == self.width * tile_size
        assert r.height == self.height * tile_size

        # Total grid size at native scale
        widthPx = self.width * CELL_PIXELS
        heightPx = self.height * CELL_PIXELS

        r.push()

        # Internally, we draw at the "large" full-grid resolution, but we
        # use the renderer to scale back to the desired size
        r.scale(tile_size / CELL_PIXELS, tile_size / CELL_PIXELS)

        # Draw the background of the in-world cells black
        r.fillRect(
            0,
            0,
            widthPx,
            heightPx,
            0, 0, 0
        )

        # Draw grid lines
        r.setLineColor(100, 100, 100)
        for rowIdx in range(0, self.height):
            y = CELL_PIXELS * rowIdx
            r.drawLine(0, y, widthPx, y)
        for colIdx in range(0, self.width):
            x = CELL_PIXELS * colIdx
            r.drawLine(x, 0, x, heightPx)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)
                if cell == None:
                    continue
                r.push()
                r.translate(i * CELL_PIXELS, j * CELL_PIXELS)
                cell.render(r)
                r.pop()

        r.pop()

    def encode(self,agent_pos,drone_pos):
        """
        Produce a compact numpy encoding of the grid
        """

        codeSize = self.width * self.height * 3

        array = np.zeros(shape=(self.width, self.height, 3), dtype='uint8')

        for j in range(0, self.height):
            for i in range(0, self.width):

                v = self.get(i, j)

                if v == None:
                    continue

                array[i, j, 0] = OBJECT_TO_IDX[v.type]
                array[i, j, 1] = COLOR_TO_IDX[v.color]

        array[agent_pos[0],agent_pos[1],0]=10
        array[agent_pos[0],agent_pos[1],1]=10
        array[drone_pos[0],drone_pos[1],0]=15
        array[drone_pos[0],drone_pos[1],1]=15



        return array




class MiniGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['human', 'rgb_array', 'pixmap'],
        'video.frames_per_second' : 10
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2
        downward=3
        stop=5
        # Pick up an object
        #pickup = 3
        # Drop an object
        #drop = 4
        # Toggle/activate an object
        #toggle = 5
        #visit Customer
        #visit=4

        # Done completing task
        #done = 5

    def __init__(
        self,
        grid_size=16,
        max_steps=100,
        max_endu=40,
        see_through_walls=False,
        seed=1337
    ):
        # Action enumeration for this environment
        self.actions = MiniGridEnv.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(grid_size,grid_size,3),
            dtype=np.uint8
        )
        #print(self.observation_space.shape)
        #self.oservation_space = spaces.Dict({
        #    'image': self.observation_space
        #})

        # Range of possible rewards
        self.reward_range = (-4, 1)

        # Renderer object used to render the whole grid (full-scale)
        self.grid_render = None


        # Environment configuration
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.max_endu=max_endu
        self.see_through_walls = see_through_walls

        # Starting position and direction for the agent
        self.start_pos = None
        # Starting point of dorne
        self.start_dpos = None
        self.drone_endu = self.max_steps
        # Initialize the RNG

        # Drone package limit
        self.pack=0
        self.seed(seed=seed)

        self.done_truck=None
        self.done_drone=None
        # Initialize the state
        self.reset()

    def reset(self):
        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.grid_size, self.grid_size)

        # These fields should be defined by _gen_grid
        assert self.start_pos is not None
        #assert self.start_dir is not None
        assert self.start_dpos is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.start_pos)
        assert start_cell is None or start_cell.can_overlap()

        start_dcell= self.grid.get(*self.start_dpos)
        assert start_dcell is None or start_dcell.can_overlap()

        # Place the agent in the starting position and direction
        self.agent_pos = self.start_pos
        self.drone_pos= self.start_dpos
        #self.agent_dir = self.start_dir

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        self.drone_endu = self.max_endu
        # Initialize the RNG

        # Drone package limit
        self.pack=0

        self.done_truck=0
        self.done_drone=0
        # Return first observation
        obs = self.gen_obs()
        return obs

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def _gen_grid(self, width, height):
        assert False, "_gen_grid needs to be implemented by each environment"

    def _reward(self):
        """
        Compute the reward to be given upon success
        """

        return 1-self.step_count/ self.max_steps

    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high[
        """

        return self.np_random.randint(low, high)

    def _rand_float(self, low, high):
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self):
        """
        Generate random boolean value
        """

        return (self.np_random.randint(0, 2) == 0)

    def _rand_elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable, num_elems):
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self):
        """
        Generate a random color name (string)
        """

        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(self, xLow, xHigh, yLow, yHigh):
        """
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.randint(xLow, xHigh),
            self.np_random.randint(yLow, yHigh)
        )

    def place_obj_det(self,obj,i,j):
        pos = np.array((i,j ))

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def place_obj(self,
        obj,
        top=None,
        size=None,
        reject_fn=None,
        max_tries=math.inf
    ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            pos = np.array((
                self._rand_int(top[0], top[0] + size[0]),
                self._rand_int(top[1], top[1] + size[1])
            ))

            # Don't place the object on top of another object
            if self.grid.get(*pos) != None:
                continue

            # Don't place the object where the agent is
            if np.array_equal(pos, self.start_pos):
                continue

            if np.array_equal(pos, self.start_dpos):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def place_agent(
        self,
        top=None,
        size=None,
        rand_dir=True,
        max_tries=math.inf
    ):
        """
        Set the agent's starting point at an empty position in the grid
        """
        self.start_dpos = None
        self.start_pos = None
        pos = self.place_obj(None, top, size, max_tries=max_tries)
        self.start_pos = pos
        self.start_dpos = self.start_pos

        return pos



    def step(self, action):
        self.step_count += 1

        reward=0
        done = False

        if ((self.step_count%3)==0):
            if(self.done_truck==0):
        # Get the position in front of the agent
                fwd_pos =  self.agent_pos + np.array((0,1))
                dwn_pos=fwd_pos+np.array((0,-2))
                rgt_pos=fwd_pos+np.array((1,-1))
                lft_pos=fwd_pos+np.array((-1,-1))
                # Get the contents of the cell in front of the agent
                fwd_cell = self.grid.get(*fwd_pos)
                dwn_cell=self.grid.get(*dwn_pos)
                rgt_cell=self.grid.get(*rgt_pos)
                lft_cell=self.grid.get(*lft_pos)
                # Rotate left
                if action == self.actions.left:
                    if lft_cell == None or lft_cell.can_overlap():
                        self.agent_pos = lft_pos
                    if lft_cell != None and lft_cell.type == 'goal':
                        #done = True
                        self.done_truck=1
                        reward = self._reward()*0.2
                    if lft_cell != None and lft_cell.type == 'customer':
                        self.grid.set(*lft_pos, None)
                        #done = True
                        reward = self._reward()

                # Rotate right
                elif action == self.actions.right:
                    if rgt_cell == None or rgt_cell.can_overlap():
                        self.agent_pos = rgt_pos
                    if rgt_cell != None and rgt_cell.type == 'goal':
                        #done = True
                        self.done_truck=1
                        reward = self._reward()*0.2
                    if rgt_cell != None and rgt_cell.type == 'customer':
                        self.grid.set(*rgt_pos, None)
                        #done = True
                        reward = self._reward()

                # Move forward
                elif action == self.actions.forward:
                    if fwd_cell == None or fwd_cell.can_overlap():
                        self.agent_pos = fwd_pos
                    if fwd_cell != None and fwd_cell.type == 'goal':
                        #done = True
                        self.done_truck=1
                        reward = self._reward()*0.2
                    if fwd_cell != None and fwd_cell.type == 'customer':
                        #done = True
                        self.grid.set(*fwd_pos, None)
                        reward = self._reward()

                elif action == self.actions.downward:
                    if dwn_cell == None or dwn_cell.can_overlap():
                        self.agent_pos = dwn_pos
                    if dwn_cell != None and dwn_cell.type == 'goal':
                        #done = True
                        self.done_truck=1
                        reward = self._reward()*0.2
                    if dwn_cell != None and dwn_cell.type == 'customer':
                        #done = True
                        self.grid.set(*dwn_pos, None)

                        reward = self._reward()
        else :
            if(self.done_drone==0):
                fwd_pos =  self.drone_pos + np.array((0,1))
                dwn_pos=fwd_pos+np.array((0,-2))
                rgt_pos=fwd_pos+np.array((1,-1))
                lft_pos=fwd_pos+np.array((-1,-1))
                # Get the contents of the cell in front of the agent
                fwd_cell = self.grid.get(*fwd_pos)
                dwn_cell=self.grid.get(*dwn_pos)
                rgt_cell=self.grid.get(*rgt_pos)
                lft_cell=self.grid.get(*lft_pos)
                # Rotate left
                if action == self.actions.left:
                    if lft_cell == None or lft_cell.can_overlap():
                        self.drone_pos = lft_pos
                    if lft_cell != None and lft_cell.type == 'goal':
                        #done = True
                        self.done_drone=1
                        self.pack=0
                        reward = self._reward()*0.2
                    if lft_cell != None and lft_cell.type == 'customer' and self.pack==0:
                        self.grid.set(*lft_pos, None)
                        self.pack=1
                        #done = True
                        reward = self._reward()

                # Rotate right
                elif action == self.actions.right:
                    if rgt_cell == None or rgt_cell.can_overlap():
                        self.drone_pos = rgt_pos
                    if rgt_cell != None and rgt_cell.type == 'goal':
                        #done = True
                        self.done_drone=1
                        self.pack=0
                        reward = self._reward()*0.2
                    if rgt_cell != None and rgt_cell.type == 'customer' and self.pack==0:
                        self.grid.set(*rgt_pos, None)
                        self.pack=1

                        #done = True
                        reward = self._reward()

                # Move forward
                elif action == self.actions.forward:
                    if fwd_cell == None or fwd_cell.can_overlap():
                        self.drone_pos = fwd_pos
                    if fwd_cell != None and fwd_cell.type == 'goal':
                        #done = True
                        self.done_drone=1
                        self.pack=0
                        reward = self._reward()*0.2
                    if fwd_cell != None and fwd_cell.type == 'customer' and self.pack==0:
                        #done = True
                        self.grid.set(*fwd_pos, None)
                        self.pack=1
                        reward = self._reward()

                elif action == self.actions.downward:
                    if dwn_cell == None or dwn_cell.can_overlap():
                        self.drone_pos = dwn_pos
                    if dwn_cell != None and dwn_cell.type == 'goal':
                        #done = True
                        self.done_drone=1
                        self.pack=0
                        reward = self._reward()*0.2
                    if dwn_cell != None and dwn_cell.type == 'customer' and self.pack==0:
                        #done = True
                        self.grid.set(*dwn_pos, None)
                        self.pack=1

                        reward = self._reward()
        # Pick up an object
        if ((self.step_count%30)==0):

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

            pos_temp = self.place_obj(obj,reject_fn=near_obj)

        if(self.agent_pos[0]==self.drone_pos[0] and self.agent_pos[1]==self.drone_pos[1]):
            self.drone_endu=self.max_endu
            self.pack=0
        else :
            self.drone_endu -=1

        if(self.drone_endu<0  and self.done_drone==0):
            reward =self.drone_endu/self.max_endu *0.5

        if(self.done_truck==1 and self.done_drone==1):
            done=True
        """
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
                    reward = self._reward()
        elif action ==self.actions.visit:
            if fwd_cell and fwd_cell.can_visit():
                self.grid.set(*fwd_pos, None)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"
        """
        if self.step_count >= self.max_steps:
            reward = -3#self._reward()-1
            done = True

        print(reward )



        obs = self.gen_obs()

        return obs, reward, done,{}


    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        #grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = self.grid.encode(self.agent_pos,self.drone_pos)

        #assert hasattr(self, 'mission'), "environments must define a textual mission string"

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {
            'image': image,
            #'direction': self.agent_dir,
            'mission': self.mission
        }
        #obs=image
        #print(obs.shape)
        #return self.render(mode='rgb_array')
        return obs


    def render(self, mode='human', close=False):
        """
        Render the whole-grid human view
        """

        if close:
            if self.grid_render:
                self.grid_render.close()
            return

        if self.grid_render is None:
            from gym_minigrid.rendering import Renderer
            self.grid_render = Renderer(
                self.grid_size * CELL_PIXELS,
                self.grid_size * CELL_PIXELS,
                True if mode == 'human' else False
            )

        r = self.grid_render

        r.beginFrame()

        # Render the whole grid
        self.grid.render(r, CELL_PIXELS)

        # Draw the agent\
        if (self.agent_pos[0]==self.drone_pos[0] and self.agent_pos[1]==self.drone_pos[1] ):
            r.push()
            r.translate(
                CELL_PIXELS * (self.agent_pos[0] + 0.5),
                CELL_PIXELS * (self.agent_pos[1] + 0.5)
            )
            #r.rotate(self.agent_dir * 90)
            r.setLineColor(255, 0, 0)
            r.setColor(255, 0, 100)

            r.drawPolygon([
                    (-12, 12),
                    ( 6,  12),
                    (6,-12),
                    (-12, -12)
            ])

            r.drawPolygon([
                (6, 8),
                ( 12,  8),
                (12,-8),
                (6, -8)
            ])
            r.pop()
        else :
            r.push()
            r.translate(
                CELL_PIXELS * (self.agent_pos[0] + 0.5),
                CELL_PIXELS * (self.agent_pos[1] + 0.5)
            )
            #r.rotate(self.agent_dir * 90)
            r.setLineColor(255, 0, 0)
            r.setColor(255, 0, 0)
            r.drawPolygon([
                    (-12, 12),
                    ( 6,  12),
                    (6,-12),
                    (-12, -12)
            ])

            r.drawPolygon([
                (6, 8),
                ( 12,  8),
                (12,-8),
                (6, -8)
            ])
            r.pop()

            r.push()
            r.translate(
                CELL_PIXELS * (self.drone_pos[0] + 0.5),
                CELL_PIXELS * (self.drone_pos[1] + 0.5)
            )
            #r.rotate(self.agent_dir * 90)
            r.setLineColor(255, 0, 0)
            r.setColor(255, 100, 0)
            r.drawPolygon([
                (-12, 12),
                ( 12,  12),
                (-12,-12),
                (12, -12)
            ])
            r.pop()


        r.endFrame()

        if mode == 'rgb_array':
            return r.getArray()
        elif mode == 'pixmap':
            return r.getPixmap()

        return r

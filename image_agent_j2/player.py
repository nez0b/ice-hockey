from .models import Detector, load_model
from os import path
import numpy as np
import torch


class Team:
    agent_type = 'image'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.team = None
        self.num_players = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print('device: ', self.device)
        #self.model = Detector()
        self.model = load_model().eval().to(self.device)
        #self.model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))
        #self.model = self.model.eval().to(device)
        print('load model success ! ')

    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        self.team, self.num_players = team, num_players
        return ['tux'] * num_players

    def act(self, player_state, player_image):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             See HW5 for some inspiration on how to use the camera information.
                             camera:  Camera info for each player
                               - aspect:     Aspect ratio
                               - fov:        Field of view of the camera
                               - mode:       Most likely NORMAL (0)
                               - projection: float 4x4 projection matrix
                               - view:       float 4x4 view matrix
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param player_image: list[np.array] showing the rendered image from the viewpoint of each kart. Use
                             player_state[i]['camera']['view'] and player_state[i]['camera']['projection'] to find out
                             from where the image was taken.

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """
        # TODO: Change me. I'm just cruising straight
        imgs = player_image
        img1 = torch.movedim(torch.from_numpy(imgs[0]), 2, 0)
        img2 = torch.movedim(torch.from_numpy(imgs[1]), 2, 0)
        #print('imgs: ', img1.unsqueeze(0).shape) 
        #modle output
        output1 = self.model(img1.unsqueeze(0))
        output1 = output1.detach().numpy()
        output1 = output1[0]*129
        # player 2
        output2 = self.model(img2.unsqueeze(0))
        output2 = output2.detach().numpy()
        output2 = output2[0]*129
        #print('output: ', output)
        x1p, y1p, x1g, y1g = list(output1)
        x2p, y2p, x2g, y2g = list(output2)

        d1p = np.sqrt(x1p**2 + y1p**2)
        d1g = np.sqrt(x1g**2 + y1g**2)

        d2p = np.sqrt(x2p**2 + y2p**2)
        d2g = np.sqrt(x2g**2 + y2g**2)

        a1 = 0.2
        a2 = 0.2
        b1 = False
        b2 = False

        if d1p > 1:
            if y1p > 0:
              steer1 = np.cos(np.arctan2(y1p, x1p))
            else:
              a1 = 0.
              b1 = True
              steer1 = 0.95
        else:
            if y1g > 0:
              steer1 = np.cos(np.arctan2(y1g, x1g))
            else:
              a1 = 0.
              b1 = True
              steer1 = 0.95

        if d2p > 1.:
            if y2p > 0:
              steer2 = np.cos(np.arctan2(y2p, x2p))
            else:
              a2 = 0.
              b2 = True
              steer2 = 0.95
        else:
            if y2g > 0:
              steer2 = np.cos(np.arctan2(y2g, x2g))
            else:
              a2 = 0.1
              b2 = True
              steer2 = 0.95

        
        print('steer1: ', steer1, 'steer2: ', steer2)
        print('brake: ', b1, ' ', b2)
        
        return [dict(acceleration=a1, steer=steer1, brake=b1), dict(acceleration=a2, steer=steer2, brake=b2) ] 

        #return [dict(acceleration=1, steer=0)] * self.num_players

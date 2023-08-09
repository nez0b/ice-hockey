from .models import Detector, load_model
from os import path
import numpy as np
import torch
from .utils import _to_image, _in_view, obj_vec

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
        
        img1=img1.to(device=self.device)
        pred = self.model.detect(img1, max_pool_ks=7, min_score= 0.3)
        # print("pred",len(pred))
        # print("puck",(pred[0]))
        # print("goal1",(pred[1]))
        # print("goal2",(pred[2]))

        #pick the highest score as the prediction for puck location
        if len(pred[0]) >0:
          puck_det = (pred[0][0][1],pred[0][0][2])
          goal1_det =  (pred[0][1][1],pred[0][1][2])
          #print(puck_det)

        #print(np.float32(player_state[0]['kart']['location']))
        
        kart1_proj = np.float32(player_state[0]['camera']['projection'])
        kart1_view = np.float32(player_state[0]['camera']['view'])

        kart1_loc =  _to_image(np.float32(player_state[0]['kart']['location']), kart1_proj,kart1_view)
        #print(kart_loc)
        puck_loc = puck_det[0]/400
        #print(puck_loc)
        #angle = np.clip((puck_loc) , -1, 1)
        #print(angle)
        #steer1 = angle
        a1=1
        kart1_front =  _to_image(np.float32(player_state[0]['kart']['front']), kart1_proj,kart1_view)
        goal1_dir = [goal1_det[0]/400, goal1_det[1]/300]
        goal_angle = (np.clip(np.dot(kart1_front, goal1_dir), -1, 1))
        angle = np.clip((goal_angle) , -1, 1)
        steer1 = angle
        return [dict(acceleration=a1, steer=steer1, brake=False), dict(acceleration=1 , steer=0, brake=False) ] 

        #return [dict(acceleration=1, steer=0)] * self.num_players

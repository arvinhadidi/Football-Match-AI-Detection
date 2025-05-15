import sys
sys.path.append("../")
from utils import get_center_of_bbox, measure_distance_between

class PlayerBallAssigner():
    def __init__(self):
        # The maximum distance the ball can be from a player for it to be considered as "in possesion"
        # If the distance from ball to EVERY player is more than this distance, no one has the ball.
        self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self,players,ball_bbox):
        # record position of ball (centre of its bounding box)
        ball_position = get_center_of_bbox(ball_bbox)

        miniumum_distance = 99999
        # placeholder value for if no player is assigned 
        assigned_player = -1

        # for eack player, check their distance from ball.
        # whoever is closest is assigned with possession of ball
        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance_between((player_bbox[0],player_bbox[-1]),ball_position)
            distance_right = measure_distance_between((player_bbox[2],player_bbox[-1]),ball_position)
            distance = min(distance_left,distance_right)

            if distance < self.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        return assigned_player


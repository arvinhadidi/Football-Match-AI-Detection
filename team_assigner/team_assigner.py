import cv2
from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {} # dict of player id and the team they play for
    
    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self,frame,bbox):

        # crop image
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        # # crop to get the image's top half
        # top_half_image = image[0:int(image.shape[0]/2),:]

        # # Get Clustering model
        # kmeans = self.get_clustering_model(top_half_image)

        # # Get the cluster labels for each pixel
        # labels = kmeans.labels_

        # # Reshape the labels to the image shape
        # clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # # Get the player cluster
        # corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        # non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)

        # # this works since there are 2 clusters
        # player_cluster = 1 - non_player_cluster

        # # assign the correct colour as player colour
        # player_color = kmeans.cluster_centers_[player_cluster]

        # return player_color

        H = image.shape[0]
        # Define torso slice (25% to 50% height)
        top = int(0.25 * H)
        bottom = int(0.50 * H)
        torso = image[top:bottom, :]

        # Convert to HSV, mask low-saturation
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        mask = sat > 50
        torso_pixels = torso[mask]
        
        # fallback if less than 10% of pixels remain after filter
        if torso_pixels.size < (torso.size // 10):
            torso_pixels = torso.reshape(-1, 3)

        # Perform clustering on torso_pixels
        kmeans = self.get_clustering_model(torso_pixels)

        # Determine cluster for each corner by direct prediction
        h, w = torso.shape[:2]
        corners = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]
        corner_clusters = []
        for y, x in corners:
            pixel = torso[y, x].reshape(1, 3)
            corner_clusters.append(kmeans.predict(pixel)[0])

        non_player = max(set(corner_clusters), key=corner_clusters.count)
        player_cl = 1 - non_player

        return kmeans.cluster_centers_[player_cl]



    def assign_team_color(self,frame, player_detections):
        
        # for each player detected, classify them as team 1 or 2
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color =  self.get_player_color(frame,bbox)
            player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]


    # Assign players to teams (and not just colours)

    def get_player_team(self,frame,player_bbox,player_id):

        # gets the team values if already done
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame,player_bbox)

        # use image of shirt to predict team id
        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        # ensures team id is 1 or 0
        team_id+=1

        if player_id ==91:
            team_id=1

        # add team id to dictionary
        self.player_team_dict[player_id] = team_id

        return team_id

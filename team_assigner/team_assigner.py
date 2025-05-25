import cv2
from sklearn.cluster import KMeans
import numpy as np

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

    # def extract_chest_color(self, image, bbox):
    #     """
    #     Extracts dominant chest color from a player's bounding box using clustering.

    #     Parameters:
    #         image (np.ndarray): The full frame image (BGR).
    #         bbox (tuple): Bounding box (x1, y1, x2, y2) of the player.
    #         clustering_model (Callable): A function that returns a fitted clustering model (e.g., KMeans).

    #     Returns:
    #         int: The predicted player cluster label (int).
    #     """
    #     x1, y1, x2, y2 = bbox
    #     player_crop = image[y1:y2, x1:x2]

    #     # Define chest region as 30% to 55% of height, and center 40% of width
    #     H = player_crop.shape[0]
    #     top = int(0.30 * H)
    #     bottom = int(0.55 * H)

    #     W = player_crop.shape[1]
    #     left = int(0.30 * W)
    #     right = int(0.70 * W)

    #     chest = player_crop[top:bottom, left:right]

    #     # Convert to HSV, mask low-saturation
    #     hsv = cv2.cvtColor(chest, cv2.COLOR_BGR2HSV)
    #     sat = hsv[:, :, 1]
    #     mask = sat > 50
    #     chest_pixels = chest[mask]

    #     # fallback if too few pixels after saturation filter
    #     if chest_pixels.size < (chest.size // 10):
    #         chest_pixels = chest.reshape(-1, 3)

    #     # Perform clustering on chest pixels
    #     kmeans = self.get_clustering_model(chest_pixels)

    #     # Determine cluster of each corner pixel
    #     h, w = chest.shape[:2]
    #     corners = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]
    #     corner_clusters = []
    #     for y, x in corners:
    #         pixel = chest[y, x].reshape(1, 3)
    #         corner_clusters.append(kmeans.predict(pixel)[0])

    #     non_player = max(set(corner_clusters), key=corner_clusters.count)
    #     player_cl = 1 - non_player  # assume two clusters

    #     return player_cl

    # def get_player_color(self,frame,bbox):

    #     # Make sure bbox values are cast to int before unpacking
    #     x1, y1, x2, y2 = map(int, bbox)

    #     # Crop the player image from the frame
    #     player_crop = frame[y1:y2, x1:x2]
    #     H = player_crop.shape[0]
    #     W = player_crop.shape[1]

    #     # Define chest region: vertically 30% to 55%, horizontally center 40%
    #     top = int(0.30 * H)
    #     bottom = int(0.50 * H)
    #     left = int(0.45 * W)
    #     right = int(0.55 * W)

    #     chest = player_crop[top:bottom, left:right]

    #     # Convert to HSV and apply saturation mask
    #     hsv = cv2.cvtColor(chest, cv2.COLOR_BGR2HSV)
    #     sat = hsv[:, :, 1]
    #     mask = sat > 40
    #     chest_pixels = chest[mask]

    #     # Fallback if too few pixels after filtering
    #     if chest_pixels.size < (chest.size // 10):
    #         chest_pixels = chest.reshape(-1, 3)

    #     # Perform k-means clustering (assuming self.get_clustering_model exists)
    #     kmeans = self.get_clustering_model(chest_pixels)

    #     # Determine background cluster by checking corners
    #     h, w = chest.shape[:2]
    #     corners = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]
    #     corner_clusters = []
    #     for y, x in corners:
    #         pixel = chest[y, x].reshape(1, 3)
    #         corner_clusters.append(kmeans.predict(pixel)[0])

    #     non_player = max(set(corner_clusters), key=corner_clusters.count)
    #     player_cl = 1 - non_player  # Assume 2 clusters

    #     # Return the dominant chest color cluster
    #     return kmeans.cluster_centers_[player_cl]

    def get_player_color(self, frame, bbox, n_clusters=3):
        """
        Extracts the dominant chest color using K-means clustering.
        :param frame: the full video frame (BGR)
        :param bbox: (x1, y1, x2, y2) of the player bounding box
        :param n_clusters: 2 or 3, number of clusters to fit
        :return: (B, G, R) tuple of the dominant cluster center
        """
        # 1) Crop player
        x1, y1, x2, y2 = map(int, bbox)
        player_crop = frame[y1:y2, x1:x2]
        H, W = player_crop.shape[:2]

        # 2) Define chest region (vert 30–50%, horiz center 45–55%)
        top    = int(0.30 * H)
        bottom = int(0.50 * H)
        left   = int(0.45 * W)
        right  = int(0.55 * W)
        chest  = player_crop[top:bottom, left:right]

        # 3) Filter out low-saturation pixels
        hsv = cv2.cvtColor(chest, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        mask = sat > 40
        chest_pixels = chest[mask]

        # 4) Fallback if too few remain
        if chest_pixels.size < (chest.size // 10):
            chest_pixels = chest.reshape(-1, 3)

        kmeans = self.make_kmeans(chest_pixels, n_clusters)

        # if isinstance(kmeans, np.ndarray):  # early exit: single color fallback
        #     return kmeans[0]

        # 6) Choose background cluster as centroid closest to white
        centers = kmeans.cluster_centers_  # shape (n_clusters, 3)
        # compute Euclidean distance to white [255,255,255] in BGR space
        dists_to_white = np.linalg.norm(centers - np.array([255,255,255]), axis=1)

        bg_cluster = int(np.argmin(dists_to_white))

        # 7) Player cluster is the one farthest from white
        player_cluster = int(np.argmax(dists_to_white))

        # 8) Return that cluster center
        center = centers[player_cluster]
        return center


    def make_kmeans(self, pixels, n_clusters):
        """
        Internal helper to create and fit a KMeans model.
        """
        # ensure a 2D array
        data = pixels.reshape(-1, 3)
        unique_colors = np.unique(data, axis=0)

        if unique_colors.shape[0] < n_clusters:
            # Not enough colors to form clusters — return the one dominant color
            # Here we assume the most frequent color is best
            values, counts = np.unique(data, axis=0, return_counts=True)
            most_common = values[np.argmax(counts)]
            return np.array(most_common, dtype=np.float32)  # early return with a color array
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=10,
            random_state=42
        )
        kmeans.fit(data)
        return kmeans

    # convenience wrappers
    def get_player_color_2(self, frame, bbox):
        return self.get_player_color(frame, bbox, n_clusters=2)

    def get_player_color_3(self, frame, bbox):
        return self.get_player_color(frame, bbox, n_clusters=3)



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

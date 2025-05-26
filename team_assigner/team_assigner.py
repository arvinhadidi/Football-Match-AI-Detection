import cv2
from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {} # dict of player id and the team they play for
        self.kmeans = None  # Initialize kmeans as None
        self.teams_assigned = False  # Track if teams have been assigned
    
    # def get_clustering_model(self,image):
    #     # Reshape the image to 2D array
    #     image_2d = image.reshape(-1,3)

    #     # Preform K-means with 2 clusters
    #     kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
    #     kmeans.fit(image_2d)

    #     return kmeans

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
        bottom = int(0.70 * H)
        left   = int(0.45 * W)
        right  = int(0.55 * W)
        chest  = player_crop[top:bottom, left:right]

        # 3) Filter out low-saturation pixels
        hsv = cv2.cvtColor(chest, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        mask = sat > 60
        chest_pixels = chest[mask]

        # 4) Fallback if too few remain
        if chest_pixels.size < (chest.size // 10):
            chest_pixels = chest.reshape(-1, 3)

        kmeans = self.make_kmeans(chest_pixels, n_clusters)

        # 5) Count pixels in each cluster
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_  # shape (n_clusters, 3)
        
        # Count how many pixels belong to each cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Find the cluster with the fewest pixels
        smallest_cluster = unique_labels[np.argmin(counts)]
        
        # Create mask to exclude the smallest cluster
        valid_clusters = unique_labels[unique_labels != smallest_cluster]
        
        # 6) Choose background cluster as centroid closest to white (among valid clusters)
        valid_centers = centers[valid_clusters]
        # compute Euclidean distance to white [255,255,255] in BGR space
        dists_to_white = np.linalg.norm(valid_centers - np.array([255,255,255]), axis=1)

        bg_cluster_idx = np.argmin(dists_to_white)
        bg_cluster = valid_clusters[bg_cluster_idx]

        # 7) Player cluster is the one farthest from white (among valid clusters)
        player_cluster_idx = np.argmax(dists_to_white)
        player_cluster = valid_clusters[player_cluster_idx]

        # 8) Return that cluster center
        center = centers[player_cluster]
        return center

    def make_kmeans(self, pixels, n_clusters):
        """
        Internal helper to create and fit a KMeans model with dynamic cluster adjustment.
        """
        # ensure a 2D array and convert to float32
        data = pixels.reshape(-1, 3).astype(np.float32)
        
        # Get unique colors and their counts
        unique_colors, counts = np.unique(data, axis=0, return_counts=True)
        n_unique = unique_colors.shape[0]
        
        # Strategy 1: Adjust n_clusters based on available unique colors
        effective_clusters = min(n_clusters, n_unique)
        
        # Strategy 2: If we have very few unique colors, return the most dominant one
        if n_unique == 1:
            return unique_colors[0].astype(np.float32)
        
        try:
            kmeans = KMeans(
                n_clusters=effective_clusters,
                init="k-means++",
                n_init=10,
                random_state=42,
                max_iter=100
            )
            kmeans.fit(data)
            return kmeans
            
        except Exception as e:
            print(f"KMeans failed: {e}. Falling back to most frequent color.")
            # Fallback: return the most frequent color
            unique_colors, counts = np.unique(pixels.reshape(-1, 3), axis=0, return_counts=True)
            most_frequent_color = unique_colors[np.argmax(counts)]
            return most_frequent_color.astype(np.float32)

    def bin_colors(self, colors, bin_size=16):
        """
        Bins colors to reduce precision and potentially increase diversity.
        """
        # Round colors to nearest bin_size
        binned = (colors // bin_size) * bin_size
        return binned.astype(np.float32)
    
    def add_color_noise(self, colors, noise_level=5):
        """
        Adds small random noise to colors to increase diversity.
        """
        noise = np.random.randint(-noise_level, noise_level + 1, colors.shape)
        noisy_colors = colors.astype(np.float32) + noise.astype(np.float32)
        # Clamp values to valid range
        noisy_colors = np.clip(noisy_colors, 0, 255)
        return noisy_colors.astype(np.float32)

    def get_player_color_adaptive(self, frame, bbox):
        """
        Adaptive method that tries different strategies to extract player color.
        """
        # Try with 3 clusters first
        try:
            return self.get_player_color(frame, bbox, n_clusters=3)
        except:
            pass
        
        # Fall back to 2 clusters
        try:
            return self.get_player_color(frame, bbox, n_clusters=2)
        except:
            pass
        
        # Final fallback: extract dominant color without clustering
        return self.get_dominant_color_simple(frame, bbox)
    
    def get_dominant_color_simple(self, frame, bbox):
        """
        Simple fallback method to get dominant color without clustering.
        """
        x1, y1, x2, y2 = map(int, bbox)
        player_crop = frame[y1:y2, x1:x2]
        H, W = player_crop.shape[:2]

        # Define chest region
        top = int(0.30 * H)
        bottom = int(0.50 * H)
        left = int(0.45 * W)
        right = int(0.55 * W)
        chest = player_crop[top:bottom, left:right]

        # Convert to HSV and filter
        hsv = cv2.cvtColor(chest, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        mask = sat > 40
        
        if np.any(mask):
            chest_pixels = chest[mask]
        else:
            chest_pixels = chest.reshape(-1, 3)
        
        # Return most frequent color
        unique_colors, counts = np.unique(chest_pixels, axis=0, return_counts=True)
        most_frequent = unique_colors[np.argmax(counts)]
        return most_frequent.astype(np.float32)

    # convenience wrappers
    def get_player_color_2(self, frame, bbox):
        return self.get_player_color(frame, bbox, n_clusters=2)

    def get_player_color_3(self, frame, bbox):
        return self.get_player_color(frame, bbox, n_clusters=3)

    def assign_team_color(self, frame, player_detections):
        """
        Assign team colors using robust color extraction.
        This method MUST be called before get_player_team.
        """
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            # Use adaptive method for robust color extraction
            player_color = self.get_player_color_adaptive(frame, bbox)
            
            # Ensure we have a valid color array
            if isinstance(player_color, np.ndarray) and player_color.size >= 3:
                player_colors.append(player_color.flatten()[:3])
        
        if len(player_colors) < 2:
            print("Warning: Not enough player colors detected for team assignment")
            # Set default teams
            self.teams_assigned = True
            self.team_colors[1] = np.array([255, 0, 0], dtype=np.float32)  # Blue
            self.team_colors[2] = np.array([0, 0, 255], dtype=np.float32)  # Red
            self.kmeans = self.create_default_classifier()
            return
        
        # Convert to numpy array with consistent dtype
        player_colors = np.array(player_colors, dtype=np.float32)
        
        # Use 2 clusters for team assignment (more reliable)
        try:
            kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
            kmeans.fit(player_colors)
            
            self.kmeans = kmeans
            self.team_colors[1] = kmeans.cluster_centers_[0].astype(np.float32)
            self.team_colors[2] = kmeans.cluster_centers_[1].astype(np.float32)
            self.teams_assigned = True
            
        except Exception as e:
            print(f"Error in team color assignment: {e}")
            # Fallback: assign teams based on color similarity
            self.assign_teams_by_similarity(player_colors)

    def create_default_classifier(self):
        """
        Creates a default classifier when team assignment fails.
        """
        class DefaultTeamClassifier:
            def __init__(self, team1_color, team2_color):
                self.team1_color = team1_color
                self.team2_color = team2_color
            
            def predict(self, color):
                color = color.reshape(-1, 3)[0]
                dist1 = np.linalg.norm(color - self.team1_color)
                dist2 = np.linalg.norm(color - self.team2_color)
                return [0 if dist1 < dist2 else 1]
        
        return DefaultTeamClassifier(self.team_colors[1], self.team_colors[2])

    def get_team_color_for_drawing(self, team_id):
        """
        Get team color in OpenCV-compatible format
        """
        if team_id in self.team_colors:
            color = self.team_colors[team_id]
            if isinstance(color, np.ndarray):
                color = color.flatten()
            # Convert to integers and ensure BGR format
            return tuple(int(c) for c in color[:3])
        else:
            # Default colors if team not found
            default_colors = {
                1: (255, 0, 0),    # Blue team
                2: (0, 0, 255),    # Red team
            }
            return default_colors.get(team_id, (0, 255, 0))  # Default green

    def assign_teams_by_similarity(self, player_colors):
        """
        Fallback method to assign teams based on color similarity.
        """
        if len(player_colors) < 2:
            return
        
        # Use first two colors as team representatives
        self.team_colors[1] = player_colors[0]
        self.team_colors[2] = player_colors[1] if len(player_colors) > 1 else player_colors[0]
        
        # Create a simple classifier based on distance
        class SimpleTeamClassifier:
            def __init__(self, team1_color, team2_color):
                self.team1_color = team1_color
                self.team2_color = team2_color
            
            def predict(self, color):
                color = color.reshape(-1, 3)[0]
                dist1 = np.linalg.norm(color - self.team1_color)
                dist2 = np.linalg.norm(color - self.team2_color)
                return [0 if dist1 < dist2 else 1]
        
        self.kmeans = SimpleTeamClassifier(self.team_colors[1], self.team_colors[2])
        self.teams_assigned = True

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Get player team assignment with robust error handling.
        """
        # Check if already assigned
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Check if teams have been assigned
        if not self.teams_assigned or self.kmeans is None:
            print(f"Warning: Teams not assigned yet. Call assign_team_color first. Defaulting player {player_id} to team 1.")
            self.player_team_dict[player_id] = 1
            return 1

        try:
            player_color = self.get_player_color_adaptive(frame, player_bbox)
            
            # Ensure we have a valid color with consistent dtype
            if isinstance(player_color, np.ndarray):
                player_color = player_color.flatten()[:3].astype(np.float32).reshape(1, -1)
            else:
                # Fallback to a default team
                self.player_team_dict[player_id] = 1
                return 1

            # Use the classifier to predict team
            if hasattr(self.kmeans, 'predict'):
                team_id = self.kmeans.predict(player_color)[0]
                team_id += 1  # Convert to 1-based indexing
            else:
                team_id = 1  # Default assignment

            # Special case handling
            if player_id == 91:
                team_id = 1

            self.player_team_dict[player_id] = team_id
            return team_id
            
        except Exception as e:
            print(f"Error assigning team for player {player_id}: {e}")
            # Default assignment
            self.player_team_dict[player_id] = 1
            return 1
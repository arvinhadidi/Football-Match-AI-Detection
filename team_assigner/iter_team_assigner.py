import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {} # dict of player id and the team they play for
        self.kmeans = None  # Initialize kmeans as None
        self.teams_assigned = False  # Track if teams have been assigned
    
    def iterative_kmeans(self, data, max_clusters=5, min_clusters=2, use_preprocessing=True):
        """
        Iterative K-means with multiple improvement techniques:
        1. Optimal cluster selection using silhouette score
        2. Data preprocessing (standardization)
        3. Multiple initializations
        4. Outlier detection and removal
        
        :param data: Input data array (n_samples, n_features)
        :param max_clusters: Maximum number of clusters to try
        :param min_clusters: Minimum number of clusters to try
        :param use_preprocessing: Whether to standardize the data
        :return: Best KMeans model and optimal number of clusters
        """
        if len(data) < min_clusters:
            # Fallback for insufficient data
            kmeans = KMeans(n_clusters=1, random_state=42)
            kmeans.fit(data)
            return kmeans, 1
        
        # Step 1: Outlier detection and removal using IQR method
        cleaned_data = self._remove_outliers_iqr(data)
        
        # Step 2: Data preprocessing
        if use_preprocessing and len(cleaned_data) > 1:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cleaned_data)
        else:
            scaled_data = cleaned_data
            scaler = None
        
        best_score = -1
        best_kmeans = None
        best_n_clusters = min_clusters
        scores = []
        
        # Step 3: Try different numbers of clusters
        max_possible_clusters = min(max_clusters, len(np.unique(scaled_data, axis=0)))
        
        for n_clusters in range(min_clusters, max_possible_clusters + 1):
            try:
                # Step 4: Multiple initializations with different strategies
                best_local_kmeans = None
                best_local_inertia = float('inf')
                
                # Try different initialization methods
                init_methods = ['k-means++', 'random']
                for init_method in init_methods:
                    kmeans = KMeans(
                        n_clusters=n_clusters,
                        init=init_method,
                        n_init=20,  # More initializations
                        max_iter=300,  # More iterations
                        random_state=42,
                        tol=1e-6  # Tighter convergence
                    )
                    kmeans.fit(scaled_data)
                    
                    if kmeans.inertia_ < best_local_inertia:
                        best_local_inertia = kmeans.inertia_
                        best_local_kmeans = kmeans
                
                # Step 5: Evaluate using silhouette score
                if n_clusters > 1 and len(scaled_data) > n_clusters:
                    labels = best_local_kmeans.labels_
                    score = silhouette_score(scaled_data, labels)
                    scores.append((n_clusters, score))
                    
                    if score > best_score:
                        best_score = score
                        best_kmeans = best_local_kmeans
                        best_n_clusters = n_clusters
                else:
                    # For single cluster case
                    if best_kmeans is None:
                        best_kmeans = best_local_kmeans
                        best_n_clusters = n_clusters
                        
            except Exception as e:
                print(f"Error with {n_clusters} clusters: {e}")
                continue
        
        # Step 6: Post-process the best model
        if best_kmeans is not None and scaler is not None:
            # Transform cluster centers back to original space
            original_centers = scaler.inverse_transform(best_kmeans.cluster_centers_)
            best_kmeans.cluster_centers_ = original_centers
        
        return best_kmeans, best_n_clusters
    
    def _remove_outliers_iqr(self, data, factor=1.5):
        """
        Remove outliers using Interquartile Range (IQR) method.
        """
        if len(data) < 4:  # Need at least 4 points for IQR
            return data
        
        try:
            # Calculate IQR for each feature
            Q1 = np.percentile(data, 25, axis=0)
            Q3 = np.percentile(data, 75, axis=0)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            # Keep only non-outlier points
            mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
            cleaned_data = data[mask]
            
            # Ensure we don't remove too many points
            if len(cleaned_data) < len(data) * 0.5:  # Keep at least 50% of data
                return data
            
            return cleaned_data
        except:
            return data
    
    def _remove_outliers_zscore(self, data, threshold=2.0):
        """
        Remove outliers using Z-score method.
        """
        if len(data) < 3:
            return data
        
        try:
            z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
            mask = np.all(z_scores < threshold, axis=1)
            cleaned_data = data[mask]
            
            if len(cleaned_data) < len(data) * 0.5:
                return data
            
            return cleaned_data
        except:
            return data

    def get_player_color(self, frame, bbox, n_clusters=3):
        """
        Extracts the dominant chest color using improved K-means clustering.
        :param frame: the full video frame (BGR)
        :param bbox: (x1, y1, x2, y2) of the player bounding box
        :param n_clusters: number of clusters to fit
        :return: (B, G, R) tuple of the dominant cluster center
        """
        # 1) Crop player
        x1, y1, x2, y2 = map(int, bbox)
        player_crop = frame[y1:y2, x1:x2]
        H, W = player_crop.shape[:2]

        # 2) Define chest region (vert 30–70%, horiz center 45–55%)
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

        # 5) Use iterative K-means for better clustering
        chest_pixels_2d = chest_pixels.reshape(-1, 3).astype(np.float32)
        kmeans, optimal_clusters = self.iterative_kmeans(
            chest_pixels_2d, 
            max_clusters=min(n_clusters, 4),
            min_clusters=2,
            use_preprocessing=False  # Colors don't need standardization
        )

        # 6) Count pixels in each cluster and exclude smallest
        if hasattr(kmeans, 'labels_'):
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_
            
            # Count how many pixels belong to each cluster
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            if len(unique_labels) > 1:
                # Find the cluster with the fewest pixels
                smallest_cluster = unique_labels[np.argmin(counts)]
                # Create mask to exclude the smallest cluster
                valid_clusters = unique_labels[unique_labels != smallest_cluster]
            else:
                valid_clusters = unique_labels
            
            # 7) Choose player cluster as the one farthest from white (among valid clusters)
            valid_centers = centers[valid_clusters]
            dists_to_white = np.linalg.norm(valid_centers - np.array([255,255,255]), axis=1)
            
            player_cluster_idx = np.argmax(dists_to_white)
            player_cluster = valid_clusters[player_cluster_idx]
            
            # 8) Return that cluster center
            center = centers[player_cluster]
            return center
        else:
            # Fallback if clustering failed
            return self.get_dominant_color_simple(frame, bbox)

    def make_kmeans(self, pixels, n_clusters):
        """
        Internal helper to create and fit a KMeans model with dynamic cluster adjustment.
        Now uses iterative K-means for better results.
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
            class SingleColorKMeans:
                def __init__(self, color):
                    self.cluster_centers_ = np.array([color])
                    self.labels_ = np.array([0] * len(data))
            return SingleColorKMeans(unique_colors[0])
        
        try:
            # Use iterative K-means for better clustering
            kmeans, _ = self.iterative_kmeans(
                data, 
                max_clusters=effective_clusters,
                min_clusters=1,
                use_preprocessing=False
            )
            return kmeans
            
        except Exception as e:
            print(f"Iterative KMeans failed: {e}. Falling back to standard KMeans.")
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
            except Exception as e2:
                print(f"Standard KMeans also failed: {e2}. Using most frequent color.")
                # Final fallback
                most_frequent_color = unique_colors[np.argmax(counts)]
                class SingleColorKMeans:
                    def __init__(self, color):
                        self.cluster_centers_ = np.array([color])
                        self.labels_ = np.array([0] * len(data))
                return SingleColorKMeans(most_frequent_color)

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
        # Try with iterative approach first
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
        bottom = int(0.70 * H)
        left = int(0.45 * W)
        right = int(0.55 * W)
        chest = player_crop[top:bottom, left:right]

        # Convert to HSV and filter
        hsv = cv2.cvtColor(chest, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        mask = sat > 60
        
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
        Assign team colors using robust color extraction with iterative K-means.
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
        
        # Use iterative K-means for better team assignment - FORCE exactly 2 clusters for teams
        try:
            kmeans, optimal_clusters = self.iterative_kmeans(
                player_colors,
                max_clusters=2,  # Force exactly 2 clusters for teams
                min_clusters=2,
                use_preprocessing=True  # Standardize for team assignment
            )
            
            self.kmeans = kmeans
            
            # Always assign exactly 2 teams
            if len(kmeans.cluster_centers_) >= 2:
                self.team_colors[1] = kmeans.cluster_centers_[0].astype(np.float32)
                self.team_colors[2] = kmeans.cluster_centers_[1].astype(np.float32)
            else:
                # Fallback to manual assignment
                self.team_colors[1] = player_colors[0]
                self.team_colors[2] = player_colors[-1] if len(player_colors) > 1 else player_colors[0]
            
            self.teams_assigned = True
            
        except Exception as e:
            print(f"Error in iterative team color assignment: {e}")
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
        
        Note: This method extracts the player's shirt color using multiple clusters 
        (to separate shirt from background/skin), but then assigns to one of only 
        2 teams since there are only 2 team colors.
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
            # Extract player's shirt color using multiple clusters for better separation
            player_color = self.get_player_color_adaptive(frame, player_bbox)
            
            # Ensure we have a valid color with consistent dtype
            if isinstance(player_color, np.ndarray):
                player_color = player_color.flatten()[:3].astype(np.float32).reshape(1, -1)
            else:
                # Fallback to a default team
                self.player_team_dict[player_id] = 1
                return 1

            # Classify into one of the 2 teams based on extracted shirt color
            if hasattr(self.kmeans, 'predict'):
                cluster_id = self.kmeans.predict(player_color)[0]
                # Since team assignment uses exactly 2 clusters, cluster_id should be 0 or 1
                # Convert to team IDs 1 and 2
                team_id = cluster_id + 1
                
                # Safety check: ensure team_id is valid (should always be 1 or 2)
                if team_id not in [1, 2]:
                    print(f"Warning: Invalid team_id {team_id} for player {player_id}, defaulting to team 1")
                    team_id = 1
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
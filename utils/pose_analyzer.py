import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
import json
from datetime import datetime
import os
import math
from ultralytics import YOLO # Import YOLO

# Global constants
LANDMARK_NAMES = [...] # Keep existing landmark names

class PoseAnalyzer:

    def __init__(self):
        """Initialize the pose analyzer with MediaPipe Pose."""
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.ball_trajectory = []
        self.ball_bbox = None
        self.frame_count = 0
        self.track_history = []  # Store tracking history
        self.track_id = 0  # Unique ID for tracked ball
        self.track_lost_frames = 0  # Count frames where ball is lost
        self.max_track_lost = 30  # Maximum frames to keep tracking after loss
        self.prev_ball_positions = []  # Store previous ball positions for trajectory
        self.max_trajectory_length = 30  # Maximum number of positions to store
        self.min_movement_threshold = 5  # Minimum movement to consider as new position
        
        # Load YOLOv8 model
        try:
            self.yolo_model = YOLO('yolov8n.pt') # Load pre-trained YOLOv8 nano model
            # Optional: Print model class names to confirm 'sports ball' ID
            # print(self.yolo_model.names)
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Please ensure 'ultralytics' and 'torch' are installed.")
            self.yolo_model = None

    def calculate_angle(self, a: np.ndarray, b: np.ndarray,
                        c: np.ndarray) -> float:
        """
        Calculate the angle between three points.
        
        Args:
            a, b, c (np.ndarray): Points in 3D space
            
        Returns:
            float: Angle in degrees
        """
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba,
                              bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    def detect_ball(self, frame):
        """Detect and track volleyball in the frame using YOLOv8."""
        if self.yolo_model is None:
            print("YOLO model not loaded. Skipping ball detection.")
            return None
        
        # Perform YOLO detection
        # results = self.yolo_model(frame, verbose=False) # Run detection
        # Force CPU inference if GPU issues arise
        results = self.yolo_model(frame, device='cpu', verbose=False)
        
        best_ball = None
        max_conf = 0.0
        sports_ball_class_id = 32 # Class ID for 'sports ball' in COCO

        # Process results
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item()) # Get class ID
                conf = boxes.conf[i].item()      # Get confidence score
                
                # Check if it's a sports ball with sufficient confidence
                if cls_id == sports_ball_class_id and conf > 0.4: # Confidence threshold
                    if conf > max_conf:
                        max_conf = conf
                        # Get bounding box in xyxy format
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                        # Convert to xywh format
                        w = x2 - x1
                        h = y2 - y1
                        best_ball = (x1, y1, w, h) # Store as xywh
                        
        if best_ball is not None:
            x, y, w, h = best_ball
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Store ball position for trajectory tracking
            if not hasattr(self, 'ball_positions'):
                self.ball_positions = []
            self.ball_positions.append((center_x, center_y))
            
            # Keep only the last 20 positions for trajectory
            if len(self.ball_positions) > 20:
                self.ball_positions.pop(0)
            
            # Draw trajectory
            if len(self.ball_positions) > 1:
                for i in range(1, len(self.ball_positions)):
                    # Use a thicker line for better visibility
                    cv2.line(frame, self.ball_positions[i-1], self.ball_positions[i], (0, 255, 0), 3)
            
            return best_ball # Return xywh format
        
        # No suitable ball detected
        return None

    def calculate_ball_metrics(self, frame: np.ndarray, pose_landmarks, ball_bbox: Optional[Tuple[int, int, int, int]]) -> Dict:
        """Calculate ball position metrics relative to the player, given a detected ball."""
        metrics = {
            "ball_detected": False,
            "toss_height": None,
            "toss_distance": None,
            "ball_position": None,
            "ball_trajectory": []
        }
        
        if ball_bbox is None or pose_landmarks is None:
            self.ball_bbox = None  # Reset stored ball bounding box 
            return metrics
            
        # Store the ball bounding box if needed elsewhere (optional)
        self.ball_bbox = ball_bbox
            
        # Get ball center
        x, y, w, h = ball_bbox
        ball_center = (x + w // 2, y + h // 2)
        
        # Get player's shoulder positions
        try:
            left_shoulder = (
                int(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1]),
                int(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0])
            )
            right_shoulder = (
                int(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1]),
                int(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0])
            )
        except IndexError:
             # Handle cases where landmarks might be missing
             return metrics

        # Calculate shoulder center
        shoulder_center = (
            (left_shoulder[0] + right_shoulder[0]) // 2,
            (left_shoulder[1] + right_shoulder[1]) // 2
        )
        
        # Calculate toss height (vertical distance from shoulder center)
        toss_height = shoulder_center[1] - ball_center[1]
        
        # Calculate toss distance (horizontal distance from shoulder center)
        toss_distance = ball_center[0] - shoulder_center[0]
        
        # Update metrics
        metrics.update({
            "ball_detected": True,
            "toss_height": toss_height,
            "toss_distance": toss_distance,
            "ball_position": ball_center
        })
        
        # Update ball trajectory (consider if this is still needed or handled by self.ball_positions)
        # Example: append current center to a trajectory list if needed for analysis
        # if hasattr(self, 'analysis_trajectory'): 
        #     self.analysis_trajectory.append(ball_center)
        # metrics["ball_trajectory"] = self.analysis_trajectory # Or similar
        
        return metrics

    def analyze_frame(self, frame):
        """Analyze a single frame for pose and ball detection."""
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect pose
        results = self.pose.process(rgb_frame)

        # Initialize metrics
        metrics = {
            "pose_detected": False,
            "keypoints": {},
            "angles": {},
            "ball_metrics": None,
            "body_alignment": 0,
            "shoulder_rotation": 0
        }
        
        # Draw pose landmarks and calculate metrics if pose is detected
        if results.pose_landmarks:
            metrics["pose_detected"] = True
            
            # Extract keypoints using landmark index
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                # Store x, y, z coordinates using the landmark index as the key
                metrics["keypoints"][idx] = (landmark.x, landmark.y, landmark.z)
            
            # Convert keypoints dict to list of tuples for angle calculation if needed
            # Or adjust calculate_angles to accept the dictionary
            # Example assuming calculate_angles is adjusted or we extract needed points:
            metrics["angles"] = self.calculate_angles_from_landmarks(results.pose_landmarks.landmark)
            
            # Calculate body alignment and shoulder rotation
            metrics["body_alignment"] = self.calculate_body_alignment(results.pose_landmarks.landmark)
            metrics["shoulder_rotation"] = self.calculate_shoulder_rotation_from_landmarks(results.pose_landmarks.landmark)

            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        
        # Detect ball ONCE per frame
        ball_bbox = self.detect_ball(frame) # This function now also draws the ball and trajectory
        
        # Calculate ball metrics using the detected bbox
        metrics["ball_metrics"] = self.calculate_ball_metrics(frame, results.pose_landmarks, ball_bbox)
        
        # Note: Drawing related to ball metrics (height/distance lines) should ideally use
        # the ball_center calculated within calculate_ball_metrics or derived from ball_bbox here.
        # If those lines are needed, add drawing logic here based on metrics["ball_metrics"]
        # Example (if lines were re-added):
        # if metrics["ball_metrics"] and metrics["ball_metrics"]["ball_detected"]:
        #     ball_center = metrics["ball_metrics"]["ball_position"]
        #     height = metrics["ball_metrics"]["toss_height"]
        #     distance = metrics["ball_metrics"]["toss_distance"]
        #     if height is not None and height > 0:
        #         cv2.line(frame, (ball_center[0], ball_center[1]), (ball_center[0], ball_center[1] - int(height)), (255, 0, 0), 2)
        #     if distance is not None: # Distance can be negative
        #         cv2.line(frame, (ball_center[0], ball_center[1]), (ball_center[0] + int(distance), ball_center[1]), (0, 0, 255), 2)
        
        return frame, metrics

    def save_metrics(self, metrics: Dict, output_dir: str) -> str:
        """
        Save pose metrics to a JSON file.
        
        Args:
            metrics (dict): Pose metrics
            output_dir (str): Directory to save the metrics
            
        Returns:
            str: Path to the saved metrics file
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pose_metrics_{timestamp}.json"
        output_path = os.path.join(output_dir, filename)

        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        return output_path

    def generate_analysis_prompt(self, metrics_list: List[Dict]) -> str:
        """Generate a structured prompt for ChatGPT to act as a volleyball coach."""
        if not metrics_list:
            return "No pose data available for analysis."

        # Calculate average metrics with proper None handling
        def safe_mean(values):
            valid_values = [v for v in values if v is not None]
            return np.mean(valid_values) if valid_values else 0

        avg_metrics = {
            "left_arm_angle": safe_mean([m.get("angles", {}).get("left_arm_angle") for m in metrics_list]),
            "right_arm_angle": safe_mean([m.get("angles", {}).get("right_arm_angle") for m in metrics_list]),
            "shoulder_rotation": safe_mean([m.get("shoulder_rotation") for m in metrics_list]),
            "body_alignment": safe_mean([m.get("body_alignment") for m in metrics_list])
        }

        # Calculate ball toss metrics with proper None handling
        ball_metrics = [m.get("ball_metrics", {}) for m in metrics_list]
        ball_metrics = [m for m in ball_metrics if m and m.get("ball_detected", False)]
        
        if ball_metrics:
            avg_toss_height = safe_mean([m.get("toss_height") for m in ball_metrics])
            avg_toss_distance = safe_mean([m.get("toss_distance") for m in ball_metrics])
            max_toss_height = max([m.get("toss_height", 0) for m in ball_metrics])
        else:
            avg_toss_height = 0
            avg_toss_distance = 0
            max_toss_height = 0

        # Get position metrics for start, contact, and end with proper None handling
        start_metrics = metrics_list[0] if metrics_list else {}
        contact_metrics = metrics_list[len(metrics_list)//2] if metrics_list else {}
        end_metrics = metrics_list[-1] if metrics_list else {}

        # Helper function to safely get nested values
        def safe_get(metrics, *keys, default=0):
            current = metrics
            for key in keys:
                if isinstance(current, dict):
                    current = current.get(key, default)
                else:
                    return default
            return current if current is not None else default

        # Generate structured prompt for ChatGPT
        prompt = """
Volleyball Serve Analysis:

1. STARTING POSITION:
   - Body alignment angle: {:.1f}° (vertical alignment of shoulders and hips)
   - Left arm angle: {:.1f}° (measured between shoulder, elbow, and wrist)
   - Right arm angle: {:.1f}° (measured between shoulder, elbow, and wrist)
   - Shoulder rotation: {:.1f}° (angle between left and right shoulders)
   - Ball position relative to body: {:.1f} pixels forward, {:.1f} pixels high

2. CONTACT POSITION:
   - Body alignment angle: {:.1f}° (vertical alignment at contact)
   - Right arm extension: {:.1f}° (angle at contact point)
   - Shoulder rotation: {:.1f}° (rotation at contact)
   - Ball contact height: {:.1f} pixels above shoulder level
   - Ball contact distance: {:.1f} pixels in front of body

3. ENDING POSITION:
   - Body alignment angle: {:.1f}° (vertical alignment after contact)
   - Right arm follow-through: {:.1f}° (angle after contact)
   - Shoulder rotation: {:.1f}° (final rotation)
   - Weight transfer: {:.1f}° (forward lean)

4. BALL TOSS ANALYSIS:
   - Average toss height: {:.1f} pixels
   - Average toss distance: {:.1f} pixels
   - Maximum toss height: {:.1f} pixels
   - Toss consistency: {:.1f}% (based on height variation)

Please provide:
1. A detailed analysis of each position and the transitions between them
2. Specific technical feedback for improvement in each position
3. Drills or exercises to address issues in each position
4. Comparison to ideal serving technique standards for each position
5. How the positions work together and where transitions can be improved

Focus on actionable feedback that will help improve the player's serving technique, with specific attention to the transitions between positions.""".format(
            # Starting position metrics
            safe_get(start_metrics, "body_alignment"),
            safe_get(start_metrics, "angles", "left_arm_angle"),
            safe_get(start_metrics, "angles", "right_arm_angle"),
            safe_get(start_metrics, "shoulder_rotation"),
            safe_get(start_metrics, "ball_metrics", "toss_distance"),
            safe_get(start_metrics, "ball_metrics", "toss_height"),
            
            # Contact position metrics
            safe_get(contact_metrics, "body_alignment"),
            safe_get(contact_metrics, "angles", "right_arm_angle"),
            safe_get(contact_metrics, "shoulder_rotation"),
            safe_get(contact_metrics, "ball_metrics", "toss_height"),
            safe_get(contact_metrics, "ball_metrics", "toss_distance"),
            
            # Ending position metrics
            safe_get(end_metrics, "body_alignment"),
            safe_get(end_metrics, "angles", "right_arm_angle"),
            safe_get(end_metrics, "shoulder_rotation"),
            safe_get(end_metrics, "body_alignment"),  # Using body alignment as proxy for weight transfer
            
            # Ball toss metrics
            abs(avg_toss_height),
            avg_toss_distance,
            abs(max_toss_height),
            100.0 if not ball_metrics else (1.0 - np.std([m.get("toss_height", 0) for m in ball_metrics]) / max(abs(max_toss_height), 1)) * 100
        )

        return prompt

    def calculate_angles_from_landmarks(self, landmarks) -> Dict:
        """Calculate angles between key points using landmarks list."""
        angles = {}
        try:
            # Calculate left arm angle
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            angles["left_arm_angle"] = self.calculate_angle(
                np.array([left_shoulder.x, left_shoulder.y]), 
                np.array([left_elbow.x, left_elbow.y]), 
                np.array([left_wrist.x, left_wrist.y]))

            # Calculate right arm angle
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            angles["right_arm_angle"] = self.calculate_angle(
                np.array([right_shoulder.x, right_shoulder.y]), 
                np.array([right_elbow.x, right_elbow.y]), 
                np.array([right_wrist.x, right_wrist.y]))
        except IndexError:
            pass # Handle missing landmarks gracefully
        return angles

    def calculate_shoulder_rotation_from_landmarks(self, landmarks) -> float:
        """Calculate shoulder rotation angle using landmarks list."""
        try:
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            ls = np.array([left_shoulder.x, left_shoulder.y])
            rs = np.array([right_shoulder.x, right_shoulder.y])
            return np.degrees(np.arctan2(rs[1] - ls[1], rs[0] - ls[0]))
        except IndexError:
            return 0.0 # Return default value if landmarks missing

    def calculate_body_alignment(self, landmarks) -> float:
        """Calculate body alignment angle using multiple reference points."""
        if not landmarks:
            return 0.0

        # Get key landmarks
        left_shoulder = np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
        right_shoulder = np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                 landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
        left_hip = np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y])
        right_hip = np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y])
        left_ankle = np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                             landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y])
        right_ankle = np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                              landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])

        # Calculate shoulder and hip lines
        shoulder_line = right_shoulder - left_shoulder
        hip_line = right_hip - left_hip

        # Calculate ankle line for base alignment
        ankle_line = right_ankle - left_ankle

        # Calculate angles between lines
        shoulder_angle = np.degrees(np.arctan2(shoulder_line[1], shoulder_line[0]))
        hip_angle = np.degrees(np.arctan2(hip_line[1], hip_line[0]))
        ankle_angle = np.degrees(np.arctan2(ankle_line[1], ankle_line[0]))

        # Calculate vertical alignment using multiple reference points
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        ankle_center = (left_ankle + right_ankle) / 2

        # Calculate vertical alignment angle
        vertical_vector = np.array([0, -1])  # Perfect vertical line
        body_vector = hip_center - shoulder_center
        vertical_alignment = np.degrees(np.arctan2(body_vector[1], body_vector[0]))

        # Calculate forward lean using hip and ankle centers
        forward_vector = ankle_center - hip_center
        forward_lean = np.degrees(np.arctan2(forward_vector[1], forward_vector[0]))

        # Combine all angles with appropriate weights
        # Shoulder and hip alignment are most important (40% each)
        # Vertical alignment and forward lean are secondary (10% each)
        final_angle = (
            0.4 * shoulder_angle +
            0.4 * hip_angle +
            0.1 * vertical_alignment +
            0.1 * forward_lean
        )

        # Normalize angle to be between -90 and 90 degrees
        final_angle = (final_angle + 180) % 360 - 180

        return final_angle

    def __del__(self):
        self.pose.close()

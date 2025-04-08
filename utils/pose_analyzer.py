import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple
import cv2
import json
from datetime import datetime
import os


class PoseAnalyzer:

    def __init__(self):
        """Initialize the pose analyzer with MediaPipe pose estimation."""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                                      model_complexity=2,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

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

    def analyze_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Analyze a single frame for pose estimation.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            tuple: (annotated frame, pose metrics)
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        # Create a copy for drawing
        annotated_frame = frame.copy()

        metrics = {
            "pose_detected": False,
            "arm_angle": None,
            "shoulder_rotation": None,
            "body_alignment": None,
            "timestamp": datetime.now().isoformat()
        }

        if results.pose_landmarks:
            metrics["pose_detected"] = True

            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(annotated_frame,
                                           results.pose_landmarks,
                                           self.mp_pose.POSE_CONNECTIONS)

            # Extract landmarks
            landmarks = results.pose_landmarks.landmark

            # Calculate arm angle (shoulder-elbow-wrist)
            shoulder = np.array([
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            ])
            elbow = np.array([
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y
            ])
            wrist = np.array([
                landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y
            ])

            metrics["arm_angle"] = self.calculate_angle(shoulder, elbow, wrist)

            # Calculate shoulder rotation
            left_shoulder = np.array([
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y
            ])
            right_shoulder = np.array([
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            ])
            metrics["shoulder_rotation"] = np.degrees(
                np.arctan2(right_shoulder[1] - left_shoulder[1],
                           right_shoulder[0] - left_shoulder[0]))

            # Calculate body alignment (vertical alignment of shoulders and hips)
            left_hip = np.array([
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y
            ])
            right_hip = np.array([
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y
            ])

            shoulder_midpoint = (left_shoulder + right_shoulder) / 2
            hip_midpoint = (left_hip + right_hip) / 2
            metrics["body_alignment"] = np.degrees(
                np.arctan2(shoulder_midpoint[1] - hip_midpoint[1],
                           shoulder_midpoint[0] - hip_midpoint[0]))

        return annotated_frame, metrics

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
        """
        Generate a structured prompt for ChatGPT analysis.
        
        Args:
            metrics_list (list): List of pose metrics from multiple frames
            
        Returns:
            str: Formatted analysis prompt
        """
        if not metrics_list:
            return "No metrics available for analysis."

        # Calculate phase-specific metrics
        preparation_phase = metrics_list[:len(metrics_list) // 3]
        approach_phase = metrics_list[len(metrics_list) // 3:2 *
                                      len(metrics_list) // 3]
        contact_phase = metrics_list[2 * len(metrics_list) // 3:]

        # Calculate metrics for each phase
        def calc_phase_metrics(phase_data):
            return {
                "arm_angle": {
                    "avg":
                    np.mean([
                        m["arm_angle"] for m in phase_data
                        if m["arm_angle"] is not None
                    ]),
                    "min":
                    min([
                        m["arm_angle"] for m in phase_data
                        if m["arm_angle"] is not None
                    ]),
                    "max":
                    max([
                        m["arm_angle"] for m in phase_data
                        if m["arm_angle"] is not None
                    ])
                },
                "shoulder_rotation": {
                    "avg":
                    np.mean([
                        m["shoulder_rotation"] for m in phase_data
                        if m["shoulder_rotation"] is not None
                    ]),
                    "range":
                    max([
                        m["shoulder_rotation"] for m in phase_data
                        if m["shoulder_rotation"] is not None
                    ]) - min([
                        m["shoulder_rotation"] for m in phase_data
                        if m["shoulder_rotation"] is not None
                    ])
                },
                "body_alignment": {
                    "avg":
                    np.mean([
                        m["body_alignment"] for m in phase_data
                        if m["body_alignment"] is not None
                    ]),
                    "variation":
                    np.std([
                        m["body_alignment"] for m in phase_data
                        if m["body_alignment"] is not None
                    ])
                }
            }

        prep_metrics = calc_phase_metrics(preparation_phase)
        approach_metrics = calc_phase_metrics(approach_phase)
        contact_metrics = calc_phase_metrics(contact_phase)

        # Overall motion analysis
        overall_arm_range = max([m["arm_angle"] for m in metrics_list if m["arm_angle"] is not None]) - \
                          min([m["arm_angle"] for m in metrics_list if m["arm_angle"] is not None])
        overall_rotation_range = max([m["shoulder_rotation"] for m in metrics_list if m["shoulder_rotation"] is not None]) - \
                               min([m["shoulder_rotation"] for m in metrics_list if m["shoulder_rotation"] is not None])

        prompt = f"""You are an experienced volleyball coach specializing in serve analysis. Based on the following comprehensive motion analysis of a player's serving technique:

1. Preparation Phase:
   - Initial arm angle: {prep_metrics['arm_angle']['avg']:.2f}° (range: {prep_metrics['arm_angle']['min']:.2f}° to {prep_metrics['arm_angle']['max']:.2f}°)
   - Shoulder rotation: {prep_metrics['shoulder_rotation']['avg']:.2f}° with {prep_metrics['shoulder_rotation']['range']:.2f}° range of motion
   - Body alignment: {prep_metrics['body_alignment']['avg']:.2f}° from vertical (stability: {prep_metrics['body_alignment']['variation']:.2f}°)

2. Approach Phase:
   - Arm angle progression: {approach_metrics['arm_angle']['avg']:.2f}° (range: {approach_metrics['arm_angle']['min']:.2f}° to {approach_metrics['arm_angle']['max']:.2f}°)
   - Shoulder rotation: {approach_metrics['shoulder_rotation']['avg']:.2f}° with {approach_metrics['shoulder_rotation']['range']:.2f}° range of motion
   - Body alignment: {approach_metrics['body_alignment']['avg']:.2f}° from vertical (stability: {approach_metrics['body_alignment']['variation']:.2f}°)

3. Contact Phase:
   - Final arm angle: {contact_metrics['arm_angle']['avg']:.2f}° (range: {contact_metrics['arm_angle']['min']:.2f}° to {contact_metrics['arm_angle']['max']:.2f}°)
   - Shoulder rotation: {contact_metrics['shoulder_rotation']['avg']:.2f}° with {contact_metrics['shoulder_rotation']['range']:.2f}° range of motion
   - Body alignment: {contact_metrics['body_alignment']['avg']:.2f}° from vertical (stability: {contact_metrics['body_alignment']['variation']:.2f}°)

Overall Motion Analysis:
- Total arm angle range: {overall_arm_range:.2f}°
- Total rotation range: {overall_rotation_range:.2f}°
- Motion fluidity: {prep_metrics['body_alignment']['variation'] + approach_metrics['body_alignment']['variation'] + contact_metrics['body_alignment']['variation']:.2f}° (lower is better)

Please provide a detailed analysis of this player's serving technique, considering:
1. Preparation stance and initial positioning
2. Movement efficiency through each phase
3. Power generation and energy transfer
4. Body coordination and timing
5. Areas of strength and opportunities for improvement

Provide specific, actionable recommendations to enhance their serve performance, focusing on the most impactful improvements first."""

        return prompt

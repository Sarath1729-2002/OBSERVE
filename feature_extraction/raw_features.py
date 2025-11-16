"""
Raw Features Extractor
======================

This module extracts raw facial features from MediaPipe face landmarks:
- Facial geometry (EAR, MAR, behavioral detections)
- Head pose (Euler angles, position, gaze direction)
- Eye tracking (pupil position, openness, movement)
- CNN model outputs (arousal, valence)
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional
import mediapipe as mp
from scipy.spatial.distance import euclidean
import math

class RawFeaturesExtractor:
    """Extract raw facial features from MediaPipe landmarks"""
    
    def __init__(self):
        """Initialize the raw features extractor"""
        # MediaPipe face mesh landmark indices
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Eye landmark points for EAR calculation
        self.LEFT_EYE_EAR_POINTS = {
            'horizontal': [33, 133],  # inner and outer corners
            'vertical_1': [159, 145],  # upper and lower eyelid
            'vertical_2': [158, 153]   # upper and lower eyelid
        }
        
        self.RIGHT_EYE_EAR_POINTS = {
            'horizontal': [362, 263],  # inner and outer corners  
            'vertical_1': [386, 374],  # upper and lower eyelid
            'vertical_2': [385, 380]   # upper and lower eyelid
        }
        
        # Mouth landmark indices
        self.MOUTH_INDICES = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        self.MOUTH_MAR_POINTS = {
            'horizontal': [61, 291],   # left and right corners
            'vertical_1': [13, 18],    # upper and lower lip center
            'vertical_2': [14, 175]    # upper and lower lip center
        }
        
        # Head pose reference points
        self.HEAD_POSE_POINTS = {
            'nose_tip': 1,
            'chin': 18,
            'left_eye_corner': 33,
            'right_eye_corner': 263,
            'left_mouth_corner': 61,
            'right_mouth_corner': 291
        }
        
        # Initialize tracking variables
        self.previous_landmarks = None
        self.blink_start_time = None
        self.eye_closure_duration = 0
        self.yawn_start_time = None
        self.baseline_ear = 0.3  # Default baseline
        self.baseline_mar = 0.5  # Default baseline
        
    def _get_landmark_coords(self, landmarks, index: int) -> Tuple[float, float, float]:
        """Get 3D coordinates of a landmark"""
        if index < len(landmarks.landmark):
            landmark = landmarks.landmark[index]
            return (landmark.x, landmark.y, landmark.z)
        return (0.0, 0.0, 0.0)
    
    def _calculate_distance(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two 3D points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)
    
    def _calculate_ear(self, landmarks, ear_points: Dict) -> float:
        """Calculate Eye Aspect Ratio (EAR)"""
        # Get landmark coordinates
        horizontal_1 = self._get_landmark_coords(landmarks, ear_points['horizontal'][0])
        horizontal_2 = self._get_landmark_coords(landmarks, ear_points['horizontal'][1])
        vertical_1_top = self._get_landmark_coords(landmarks, ear_points['vertical_1'][0])
        vertical_1_bottom = self._get_landmark_coords(landmarks, ear_points['vertical_1'][1])
        vertical_2_top = self._get_landmark_coords(landmarks, ear_points['vertical_2'][0])
        vertical_2_bottom = self._get_landmark_coords(landmarks, ear_points['vertical_2'][1])
        
        # Calculate distances
        vertical_dist_1 = self._calculate_distance(vertical_1_top, vertical_1_bottom)
        vertical_dist_2 = self._calculate_distance(vertical_2_top, vertical_2_bottom)
        horizontal_dist = self._calculate_distance(horizontal_1, horizontal_2)
        
        # Calculate EAR
        if horizontal_dist > 0:
            ear = (vertical_dist_1 + vertical_dist_2) / (2.0 * horizontal_dist)
        else:
            ear = 0.0
            
        return ear
    
    def _calculate_mar(self, landmarks) -> float:
        """Calculate Mouth Aspect Ratio (MAR)"""
        # Get landmark coordinates
        horizontal_1 = self._get_landmark_coords(landmarks, self.MOUTH_MAR_POINTS['horizontal'][0])
        horizontal_2 = self._get_landmark_coords(landmarks, self.MOUTH_MAR_POINTS['horizontal'][1])
        vertical_1_top = self._get_landmark_coords(landmarks, self.MOUTH_MAR_POINTS['vertical_1'][0])
        vertical_1_bottom = self._get_landmark_coords(landmarks, self.MOUTH_MAR_POINTS['vertical_1'][1])
        vertical_2_top = self._get_landmark_coords(landmarks, self.MOUTH_MAR_POINTS['vertical_2'][0])
        vertical_2_bottom = self._get_landmark_coords(landmarks, self.MOUTH_MAR_POINTS['vertical_2'][1])
        
        # Calculate distances
        vertical_dist_1 = self._calculate_distance(vertical_1_top, vertical_1_bottom)
        vertical_dist_2 = self._calculate_distance(vertical_2_top, vertical_2_bottom)
        horizontal_dist = self._calculate_distance(horizontal_1, horizontal_2)
        
        # Calculate MAR
        if horizontal_dist > 0:
            mar = (vertical_dist_1 + vertical_dist_2) / (2.0 * horizontal_dist)
        else:
            mar = 0.0
            
        return mar
    
    def _calculate_head_pose(self, landmarks) -> Dict[str, float]:
        """Calculate head pose (pitch, yaw, roll)"""
        # Get 3D coordinates of key facial points
        nose_tip = self._get_landmark_coords(landmarks, self.HEAD_POSE_POINTS['nose_tip'])
        chin = self._get_landmark_coords(landmarks, self.HEAD_POSE_POINTS['chin'])
        left_eye = self._get_landmark_coords(landmarks, self.HEAD_POSE_POINTS['left_eye_corner'])
        right_eye = self._get_landmark_coords(landmarks, self.HEAD_POSE_POINTS['right_eye_corner'])
        left_mouth = self._get_landmark_coords(landmarks, self.HEAD_POSE_POINTS['left_mouth_corner'])
        right_mouth = self._get_landmark_coords(landmarks, self.HEAD_POSE_POINTS['right_mouth_corner'])
        
        # Calculate vectors
        eye_vector = np.array(right_eye) - np.array(left_eye)
        mouth_vector = np.array(right_mouth) - np.array(left_mouth)
        face_vector = np.array(nose_tip) - np.array(chin)
        
        # Calculate angles (simplified approach)
        # Yaw: rotation around Y-axis (left-right head turn)
        yaw = math.atan2(eye_vector[0], eye_vector[2]) * 180 / math.pi
        
        # Pitch: rotation around X-axis (head up-down)
        pitch = math.atan2(face_vector[1], face_vector[2]) * 180 / math.pi
        
        # Roll: rotation around Z-axis (head tilt)
        roll = math.atan2(eye_vector[1], eye_vector[0]) * 180 / math.pi
        
        return {
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll
        }
    
    def _calculate_gaze_direction(self, landmarks) -> Dict[str, float]:
        """Calculate gaze direction (simplified approach)"""
        # Get eye centers
        left_eye_center = np.mean([self._get_landmark_coords(landmarks, i) for i in self.LEFT_EYE_INDICES[:6]], axis=0)
        right_eye_center = np.mean([self._get_landmark_coords(landmarks, i) for i in self.RIGHT_EYE_INDICES[:6]], axis=0)
        
        # Get pupil positions (approximated as eye centers for now)
        left_pupil = left_eye_center
        right_pupil = right_eye_center
        
        # Calculate gaze vectors (simplified)
        left_gaze = left_pupil - left_eye_center
        right_gaze = right_pupil - right_eye_center
        average_gaze = (left_gaze + right_gaze) / 2
        
        return {
            'horizontal_gaze': float(average_gaze[0]),
            'vertical_gaze': float(average_gaze[1]),
            'gaze_intensity': float(np.linalg.norm(average_gaze))
        }
    
    def _detect_behavioral_states(self, left_ear: float, right_ear: float, mar: float, 
                                 timestamp: float) -> Dict[str, Any]:
        """Detect behavioral states (blink, yawn, speech, drowsiness)"""
        average_ear = (left_ear + right_ear) / 2
        
        # Blink detection
        blink_threshold = 0.25
        blink_detected = False
        blink_duration = 0
        
        if average_ear < blink_threshold:
            if self.blink_start_time is None:
                self.blink_start_time = timestamp
            self.eye_closure_duration = timestamp - self.blink_start_time
            
            if self.eye_closure_duration > 0.1:  # Minimum blink duration
                blink_detected = True
                blink_duration = self.eye_closure_duration
        else:
            self.blink_start_time = None
            self.eye_closure_duration = 0
        
        # Yawn detection
        yawn_threshold = 0.8
        yawn_detected = False
        yawn_intensity = 0
        
        if mar > yawn_threshold:
            if self.yawn_start_time is None:
                self.yawn_start_time = timestamp
            yawn_duration = timestamp - self.yawn_start_time
            
            if yawn_duration > 0.5:  # Minimum yawn duration
                yawn_detected = True
                yawn_intensity = mar / 1.0  # Normalize by max possible MAR
        else:
            self.yawn_start_time = None
        
        # Speech detection (simplified - based on mouth movement)
        speech_threshold = 0.1
        speech_detected = abs(mar - self.baseline_mar) > speech_threshold
        
        # Drowsiness indicator
        drowsiness_score = (1 - average_ear / self.baseline_ear) * 0.5
        if blink_detected:
            drowsiness_score += 0.3
        if self.eye_closure_duration > 0.5:
            drowsiness_score += 0.2
        drowsiness_score = min(1.0, max(0.0, drowsiness_score))
        
        return {
            'blink_detected': blink_detected,
            'blink_duration': blink_duration,
            'yawn_detected': yawn_detected,
            'yawn_intensity': yawn_intensity,
            'speech_detected': speech_detected,
            'drowsiness_score': drowsiness_score
        }
    
    def _calculate_attention_states(self, gaze_data: Dict, head_pose: Dict) -> Dict[str, Any]:
        """Calculate attention states"""
        horizontal_gaze = gaze_data['horizontal_gaze']
        vertical_gaze = gaze_data['vertical_gaze']
        
        # Looking forward
        looking_forward = abs(horizontal_gaze) < 0.2 and abs(vertical_gaze) < 0.2
        attention_score = 1 - math.sqrt(horizontal_gaze**2 + vertical_gaze**2) if looking_forward else 0
        
        # Looking away
        looking_away = math.sqrt(horizontal_gaze**2 + vertical_gaze**2) > 0.5
        distraction_score = math.sqrt(horizontal_gaze**2 + vertical_gaze**2) if looking_away else 0
        
        # Looking down
        looking_down = vertical_gaze < -0.3
        downward_gaze_intensity = abs(vertical_gaze) if looking_down else 0
        
        return {
            'looking_forward': looking_forward,
            'attention_score': attention_score,
            'looking_away': looking_away,
            'distraction_score': distraction_score,
            'looking_down': looking_down,
            'downward_gaze_intensity': downward_gaze_intensity
        }
    
    def extract_features(self, landmarks, timestamp: float = 0.0) -> Dict[str, Any]:
        """
        Extract all raw features from face landmarks
        
        Args:
            landmarks: MediaPipe face landmarks
            timestamp: Current timestamp
            
        Returns:
            Dictionary containing all raw features
        """
        features = {}
        
        try:
            # Facial Geometry Features
            left_ear = self._calculate_ear(landmarks, self.LEFT_EYE_EAR_POINTS)
            right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE_EAR_POINTS)
            average_ear = (left_ear + right_ear) / 2
            
            mar = self._calculate_mar(landmarks)
            
            features['facial_geometry'] = {
                'left_ear': left_ear,
                'right_ear': right_ear,
                'average_ear': average_ear,
                'mar': mar
            }
            
            # Behavioral detections
            behavioral_states = self._detect_behavioral_states(left_ear, right_ear, mar, timestamp)
            features['behavioral_detections'] = behavioral_states
            
            # Head pose features
            head_pose = self._calculate_head_pose(landmarks)
            features['head_pose'] = head_pose
            
            # Head position (simplified - using nose tip as reference)
            nose_tip = self._get_landmark_coords(landmarks, 1)
            features['head_position'] = {
                'x': nose_tip[0],
                'y': nose_tip[1], 
                'z': nose_tip[2]
            }
            
            # Gaze direction
            gaze_data = self._calculate_gaze_direction(landmarks)
            features['gaze_direction'] = gaze_data
            
            # Attention states
            attention_states = self._calculate_attention_states(gaze_data, head_pose)
            features['attention_states'] = attention_states
            
            # Eye tracking features
            features['eye_tracking'] = {
                'left_pupil_x': 0.0,  # Placeholder - requires more sophisticated pupil detection
                'left_pupil_y': 0.0,
                'right_pupil_x': 0.0,
                'right_pupil_y': 0.0,
                'left_openness': left_ear,
                'right_openness': right_ear,
                'saccade_detected': False,  # Placeholder
                'fixation_detected': False,  # Placeholder
                'fixation_duration': 0.0
            }
            
            # CNN model outputs (placeholder - would need actual CNN model)
            features['cnn_outputs'] = {
                'arousal': 0.0,  # Placeholder
                'valence': 0.0,  # Placeholder
                'emotion_quadrant': 'Neutral'
            }
            
            # Store current landmarks for next frame
            self.previous_landmarks = landmarks
            
        except Exception as e:
            print(f"Error extracting raw features: {e}")
            # Return default values on error
            features = self._get_default_features()
        
        return features
    
    def _get_default_features(self) -> Dict[str, Any]:
        """Return default feature values when extraction fails"""
        return {
            'facial_geometry': {
                'left_ear': 0.0,
                'right_ear': 0.0,
                'average_ear': 0.0,
                'mar': 0.0
            },
            'behavioral_detections': {
                'blink_detected': False,
                'blink_duration': 0.0,
                'yawn_detected': False,
                'yawn_intensity': 0.0,
                'speech_detected': False,
                'drowsiness_score': 0.0
            },
            'head_pose': {
                'pitch': 0.0,
                'yaw': 0.0,
                'roll': 0.0
            },
            'head_position': {
                'x': 0.0,
                'y': 0.0,
                'z': 0.0
            },
            'gaze_direction': {
                'horizontal_gaze': 0.0,
                'vertical_gaze': 0.0,
                'gaze_intensity': 0.0
            },
            'attention_states': {
                'looking_forward': False,
                'attention_score': 0.0,
                'looking_away': False,
                'distraction_score': 0.0,
                'looking_down': False,
                'downward_gaze_intensity': 0.0
            },
            'eye_tracking': {
                'left_pupil_x': 0.0,
                'left_pupil_y': 0.0,
                'right_pupil_x': 0.0,
                'right_pupil_y': 0.0,
                'left_openness': 0.0,
                'right_openness': 0.0,
                'saccade_detected': False,
                'fixation_detected': False,
                'fixation_duration': 0.0
            },
            'cnn_outputs': {
                'arousal': 0.0,
                'valence': 0.0,
                'emotion_quadrant': 'Neutral'
            }
        }
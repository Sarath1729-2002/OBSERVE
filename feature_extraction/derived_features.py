"""
Derived Features Extractor
==========================

This module extracts derived features from raw facial features:
- Duchenne smile detection
- Facial asymmetry analysis
- AU velocity and acceleration
- Visual attention index
- Fatigue detection
- Granular emotion recognition
- Micro-expression detection
- Cognitive load assessment
"""

import numpy as np
import math
from typing import Dict, List, Any, Optional
from collections import deque

class DerivedFeaturesExtractor:
    """Extract derived features from raw facial features"""
    
    def __init__(self, history_size: int = 30):
        """
        Initialize derived features extractor
        
        Args:
            history_size: Number of frames to keep in history for temporal features
        """
        self.history_size = history_size
        
        # Initialize history buffers for temporal analysis
        self.ear_history = deque(maxlen=history_size)
        self.mar_history = deque(maxlen=history_size)
        self.au_history = {}
        self.gaze_history = deque(maxlen=history_size)
        self.attention_history = deque(maxlen=history_size)
        self.timestamp_history = deque(maxlen=history_size)
        
        # Baseline values for normalization
        self.baseline_ear = 0.3
        self.baseline_mar = 0.5
        self.baseline_blink_rate = 15  # blinks per minute
        
        # Thresholds
        self.micro_expression_threshold = 0.3
        self.fatigue_window_size = 900  # 30 seconds at 30fps
        
    def _calculate_duchenne_smile(self, au6_intensity: float, au12_intensity: float) -> Dict[str, float]:
        """Calculate Duchenne smile score"""
        if au6_intensity > 0.3 and au12_intensity > 0.5:
            duchenne_score = (au6_intensity * au12_intensity) / (au6_intensity + au12_intensity + 0.001)
            authenticity_bonus = min(au6_intensity, au12_intensity)
            duchenne_final = duchenne_score * (1 + authenticity_bonus * 0.5)
        else:
            duchenne_final = 0.0
        
        return {
            'duchenne_score': min(1.0, duchenne_final),
            'authenticity_level': min(au6_intensity, au12_intensity),
            'smile_balance': abs(au6_intensity - au12_intensity) / (au6_intensity + au12_intensity + 0.001)
        }
    
    def _calculate_facial_asymmetry(self, au_features: Dict) -> Dict[str, float]:
        """Calculate facial asymmetry index"""
        asymmetries = []
        
        # Define bilateral AU pairs (if available)
        bilateral_aus = {
            'AU6': ('au6_left', 'au6_right'),
            'AU12': ('au12_left', 'au12_right'),
            'AU15': ('au15_left', 'au15_right')
        }
        
        for au_name, (left_key, right_key) in bilateral_aus.items():
            if left_key in au_features and right_key in au_features:
                left_intensity = au_features[left_key]
                right_intensity = au_features[right_key]
                
                if left_intensity > 0 or right_intensity > 0:
                    asymmetry = abs(left_intensity - right_intensity) / (left_intensity + right_intensity + 0.001)
                    asymmetries.append(asymmetry)
        
        # Eye asymmetry (from raw features)
        left_ear = au_features.get('left_ear', 0)
        right_ear = au_features.get('right_ear', 0)
        
        if left_ear > 0 or right_ear > 0:
            eye_asymmetry = abs(left_ear - right_ear) / (left_ear + right_ear + 0.001)
            asymmetries.append(eye_asymmetry)
        
        overall_asymmetry = np.mean(asymmetries) if asymmetries else 0.0
        
        return {
            'overall_asymmetry': overall_asymmetry,
            'eye_asymmetry': eye_asymmetry if 'eye_asymmetry' in locals() else 0.0,
            'au_asymmetry_count': len(asymmetries) - (1 if 'eye_asymmetry' in locals() else 0)
        }
    
    def _calculate_au_dynamics(self, current_aus: Dict, timestamp: float) -> Dict[str, Any]:
        """Calculate AU velocity and acceleration"""
        dynamics = {}
        
        for au_name, current_intensity in current_aus.items():
            if au_name not in self.au_history:
                self.au_history[au_name] = deque(maxlen=self.history_size)
            
            self.au_history[au_name].append((timestamp, current_intensity))
            
            if len(self.au_history[au_name]) >= 2:
                # Calculate velocity
                prev_time, prev_intensity = self.au_history[au_name][-2]
                time_diff = timestamp - prev_time
                
                if time_diff > 0:
                    velocity = (current_intensity - prev_intensity) / time_diff
                    
                    # Calculate acceleration if we have enough history
                    if len(self.au_history[au_name]) >= 3:
                        prev_prev_time, prev_prev_intensity = self.au_history[au_name][-3]
                        prev_time_diff = prev_time - prev_prev_time
                        
                        if prev_time_diff > 0:
                            prev_velocity = (prev_intensity - prev_prev_intensity) / prev_time_diff
                            acceleration = (velocity - prev_velocity) / time_diff
                        else:
                            acceleration = 0.0
                    else:
                        acceleration = 0.0
                    
                    dynamics[f'{au_name}_velocity'] = velocity
                    dynamics[f'{au_name}_acceleration'] = acceleration
        
        # Calculate peak dynamics
        velocities = [v for k, v in dynamics.items() if '_velocity' in k]
        accelerations = [v for k, v in dynamics.items() if '_acceleration' in k]
        
        dynamics['peak_velocity'] = max(velocities) if velocities else 0.0
        dynamics['peak_acceleration'] = max(accelerations) if accelerations else 0.0
        dynamics['avg_velocity'] = np.mean(velocities) if velocities else 0.0
        dynamics['avg_acceleration'] = np.mean(accelerations) if accelerations else 0.0
        
        return dynamics
    
    def _calculate_visual_attention_index(self, gaze_data: Dict, head_pose: Dict, ear_value: float) -> Dict[str, float]:
        """Calculate visual attention index"""
        horizontal_gaze = gaze_data.get('horizontal_gaze', 0)
        vertical_gaze = gaze_data.get('vertical_gaze', 0)
        pitch = head_pose.get('pitch', 0)
        yaw = head_pose.get('yaw', 0)
        roll = head_pose.get('roll', 0)
        
        # Calculate components
        gaze_focus = 1 - math.sqrt(horizontal_gaze**2 + vertical_gaze**2)
        head_alignment = 1 - (abs(pitch) + abs(yaw) + abs(roll)) / 180
        eye_openness = ear_value / self.baseline_ear
        
        # Combine components
        visual_attention_index = (gaze_focus * 0.4 + head_alignment * 0.3 + eye_openness * 0.3)
        visual_attention_index = max(0.0, min(1.0, visual_attention_index))
        
        # Store in history
        self.attention_history.append(visual_attention_index)
        
        return {
            'visual_attention_index': visual_attention_index,
            'gaze_focus_component': gaze_focus,
            'head_alignment_component': head_alignment,
            'eye_openness_component': eye_openness,
            'attention_stability': self._calculate_stability(self.attention_history)
        }
    
    def _calculate_fatigue_detection(self, ear_value: float, blink_data: Dict, timestamp: float) -> Dict[str, float]:
        """Calculate fatigue detection score"""
        # Store EAR history
        self.ear_history.append(ear_value)
        self.timestamp_history.append(timestamp)
        
        if len(self.ear_history) < 30:  # Need minimum history
            return {
                'fatigue_score': 0.0,
                'ear_decline_trend': 0.0,
                'blink_frequency_increase': 0.0,
                'eye_closure_prolongation': 0.0
            }
        
        # Calculate EAR trend (decline indicates fatigue)
        ear_values = list(self.ear_history)
        time_values = list(self.timestamp_history)
        
        ear_trend = self._calculate_linear_trend(ear_values, time_values)
        ear_decline = max(0, -ear_trend) / 0.01  # Normalize
        
        # Blink frequency analysis
        blink_duration = blink_data.get('blink_duration', 0)
        current_blink_rate = self._estimate_blink_rate()
        
        blink_increase = max(0, (current_blink_rate - self.baseline_blink_rate) / self.baseline_blink_rate)
        
        # Eye closure prolongation
        prolonged_closure = min(1.0, blink_duration / 0.5) if blink_duration > 0.1 else 0.0
        
        # Combine fatigue indicators
        fatigue_score = (ear_decline * 0.4 + blink_increase * 0.3 + prolonged_closure * 0.3)
        fatigue_score = min(1.0, fatigue_score)
        
        return {
            'fatigue_score': fatigue_score,
            'ear_decline_trend': ear_decline,
            'blink_frequency_increase': blink_increase,
            'eye_closure_prolongation': prolonged_closure
        }
    
    def _calculate_granular_emotions(self, au_features: Dict, arousal: float, valence: float) -> Dict[str, float]:
        """Calculate granular emotion recognition"""
        emotions = {}
        
        # Extract AU intensities
        au1 = au_features.get('au1_intensity', 0)
        au2 = au_features.get('au2_intensity', 0)
        au4 = au_features.get('au4_intensity', 0)  
        au5 = au_features.get('au5_intensity', 0)
        au6 = au_features.get('au6_intensity', 0)
        au7 = au_features.get('au7_intensity', 0)
        au12 = au_features.get('au12_intensity', 0)
        au15 = au_features.get('au15_intensity', 0)
        au20 = au_features.get('au20_intensity', 0)
        au23 = au_features.get('au23_intensity', 0)
        
        # High Arousal + Positive Valence
        if arousal > 0.5 and valence > 0.5:
            joy_score = (au6 * 0.4 + au12 * 0.6) * (arousal + valence) / 2
            excitement_score = (au1 * 0.3 + au2 * 0.3 + au5 * 0.4) * arousal
        else:
            joy_score = 0.0
            excitement_score = 0.0
        
        # High Arousal + Negative Valence  
        if arousal > 0.5 and valence < -0.5:
            anger_score = (au4 * 0.5 + au7 * 0.3 + au23 * 0.2) * arousal
            fear_score = (au1 * 0.4 + au2 * 0.3 + au5 * 0.3) * arousal
        else:
            anger_score = 0.0
            fear_score = 0.0
        
        # Low Arousal + Negative Valence
        if arousal < -0.5 and valence < -0.5:
            sadness_score = (au1 * 0.3 + au4 * 0.4 + au15 * 0.3) * abs(valence)
            boredom_score = abs(arousal) * 0.5
        else:
            sadness_score = 0.0
            boredom_score = 0.0
        
        # Low Arousal + Positive Valence
        if arousal < -0.5 and valence > 0.5:
            contentment_score = (au6 * 0.3 + au12 * 0.2) * valence * abs(arousal)
        else:
            contentment_score = 0.0
        
        # Surprise (high AU1 + AU2 + AU5)
        surprise_score = min(1.0, (au1 + au2 + au5) / 3) * arousal if arousal > 0 else 0.0
        
        # Disgust (AU15 + AU20)
        disgust_score = min(1.0, (au15 + au20) / 2)
        
        emotions = {
            'joy': min(1.0, joy_score),
            'excitement': min(1.0, excitement_score),
            'anger': min(1.0, anger_score),
            'fear': min(1.0, fear_score),
            'sadness': min(1.0, sadness_score),
            'boredom': min(1.0, boredom_score),
            'contentment': min(1.0, contentment_score),
            'surprise': min(1.0, surprise_score),
            'disgust': min(1.0, disgust_score)
        }
        
        return emotions
    
    def _detect_micro_expressions(self, au_dynamics: Dict) -> Dict[str, Any]:
        """Detect micro-expressions based on AU dynamics"""
        micro_expressions = {
            'micro_detected': False,
            'micro_type': 'none',
            'micro_intensity': 0.0,
            'micro_duration': 0.0
        }
        
        # Check for rapid AU changes (characteristic of micro-expressions)
        rapid_changes = []
        for au_name, velocity in au_dynamics.items():
            if '_velocity' in au_name and abs(velocity) > self.micro_expression_threshold:
                rapid_changes.append((au_name, velocity))
        
        if rapid_changes:
            micro_expressions['micro_detected'] = True
            micro_expressions['micro_intensity'] = max([abs(v) for _, v in rapid_changes])
            
            # Classify micro-expression type based on AUs
            dominant_au = max(rapid_changes, key=lambda x: abs(x[1]))[0]
            
            if 'au12' in dominant_au or 'au6' in dominant_au:
                micro_expressions['micro_type'] = 'joy'
            elif 'au4' in dominant_au or 'au7' in dominant_au:
                micro_expressions['micro_type'] = 'anger'
            elif 'au1' in dominant_au or 'au2' in dominant_au:
                micro_expressions['micro_type'] = 'surprise'
            elif 'au15' in dominant_au:
                micro_expressions['micro_type'] = 'disgust'
            else:
                micro_expressions['micro_type'] = 'unclassified'
        
        return micro_expressions
    
    def _calculate_cognitive_load(self, blink_rate: float, fixation_duration: float, 
                                pupil_size: float, au_activity: float) -> Dict[str, float]:
        """Calculate cognitive load assessment"""
        # Normalize inputs
        normalized_blink = min(1.0, blink_rate / 30)  # Normalize by max expected blink rate
        normalized_fixation = min(1.0, fixation_duration / 2.0)  # Normalize by 2 seconds
        normalized_pupil = min(1.0, pupil_size / 8.0)  # Normalize by max pupil size
        normalized_au = min(1.0, au_activity / 5.0)  # Normalize by max AU activity
        
        # Calculate cognitive load components
        # High cognitive load typically shows: increased blink rate, longer fixations, 
        # dilated pupils, reduced facial expressions
        cognitive_load = (
            normalized_blink * 0.3 + 
            normalized_fixation * 0.3 + 
            normalized_pupil * 0.2 + 
            (1 - normalized_au) * 0.2  # Inverted AU activity
        )
        
        return {
            'cognitive_load_score': cognitive_load,
            'blink_load_component': normalized_blink,
            'fixation_load_component': normalized_fixation,
            'pupil_load_component': normalized_pupil,
            'expression_load_component': 1 - normalized_au
        }
    
    def _calculate_stability(self, values: deque) -> float:
        """Calculate stability of a time series"""
        if len(values) < 2:
            return 0.0
        
        values_array = np.array(values)
        std_dev = np.std(values_array)
        mean_val = np.mean(values_array)
        
        if mean_val == 0:
            return 0.0
        
        coefficient_of_variation = std_dev / abs(mean_val)
        stability = 1 / (1 + coefficient_of_variation)
        
        return stability
    
    def _calculate_linear_trend(self, values: List[float], times: List[float]) -> float:
        """Calculate linear trend (slope) of values over time"""
        if len(values) < 2:
            return 0.0
        
        values_array = np.array(values)
        times_array = np.array(times)
        
        # Calculate slope using least squares
        n = len(values)
        sum_xy = np.sum(times_array * values_array)
        sum_x = np.sum(times_array)
        sum_y = np.sum(values_array)
        sum_x2 = np.sum(times_array ** 2)
        
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _estimate_blink_rate(self) -> float:
        """Estimate current blink rate from EAR history"""
        if len(self.ear_history) < 10 or len(self.timestamp_history) < 10:
            return self.baseline_blink_rate
        
        # Count blinks in recent history
        ear_values = list(self.ear_history)[-30:]  # Last 1 second at 30fps
        time_values = list(self.timestamp_history)[-30:]
        
        blink_count = 0
        in_blink = False
        blink_threshold = 0.25
        
        for ear in ear_values:
            if ear < blink_threshold and not in_blink:
                blink_count += 1
                in_blink = True
            elif ear >= blink_threshold:
                in_blink = False
        
        # Convert to blinks per minute
        time_span = time_values[-1] - time_values[0] if len(time_values) > 1 else 1
        blink_rate = (blink_count / time_span) * 60 if time_span > 0 else 0
        
        return blink_rate
    
    def extract_features(self, raw_features: Dict, au_features: Dict, arousal: float, valence: float, 
                        timestamp: float = 0.0) -> Dict[str, Any]:
        """
        Extract all derived features
        
        Args:
            raw_features: Raw facial features from RawFeaturesExtractor
            au_features: Action unit features 
            arousal: Arousal value from CNN
            valence: Valence value from CNN
            timestamp: Current timestamp
            
        Returns:
            Dictionary containing all derived features
        """
        derived_features = {}
        
        try:
            # Extract component features from raw features
            facial_geometry = raw_features.get('facial_geometry', {})
            behavioral_detections = raw_features.get('behavioral_detections', {})
            head_pose = raw_features.get('head_pose', {})
            gaze_direction = raw_features.get('gaze_direction', {})
            
            # Get EAR and MAR values
            average_ear = facial_geometry.get('average_ear', 0)
            mar = facial_geometry.get('mar', 0)
            
            # Duchenne smile detection
            au6_intensity = au_features.get('au6_intensity', 0)
            au12_intensity = au_features.get('au12_intensity', 0)
            duchenne_features = self._calculate_duchenne_smile(au6_intensity, au12_intensity)
            derived_features['duchenne_smile'] = duchenne_features
            
            # Facial asymmetry
            asymmetry_features = self._calculate_facial_asymmetry({**facial_geometry, **au_features})
            derived_features['facial_asymmetry'] = asymmetry_features
            
            # AU dynamics
            au_dynamics = self._calculate_au_dynamics(au_features, timestamp)
            derived_features['au_dynamics'] = au_dynamics
            
            # Visual attention index
            attention_features = self._calculate_visual_attention_index(gaze_direction, head_pose, average_ear)
            derived_features['visual_attention'] = attention_features
            
            # Fatigue detection
            fatigue_features = self._calculate_fatigue_detection(average_ear, behavioral_detections, timestamp)
            derived_features['fatigue_detection'] = fatigue_features
            
            # Granular emotions
            emotion_features = self._calculate_granular_emotions(au_features, arousal, valence)
            derived_features['granular_emotions'] = emotion_features
            
            # Micro-expression detection
            micro_features = self._detect_micro_expressions(au_dynamics)
            derived_features['micro_expressions'] = micro_features
            
            # Cognitive load assessment
            blink_rate = self._estimate_blink_rate()
            fixation_duration = 0.0  # Placeholder - would come from eye tracking
            pupil_size = 4.0  # Placeholder - would come from eye tracking
            au_activity = np.mean([v for k, v in au_features.items() if 'intensity' in k]) if au_features else 0.0
            
            cognitive_load_features = self._calculate_cognitive_load(blink_rate, fixation_duration, pupil_size, au_activity)
            derived_features['cognitive_load'] = cognitive_load_features
            
        except Exception as e:
            print(f"Error extracting derived features: {e}")
            derived_features = self._get_default_features()
        
        return derived_features
    
    def _get_default_features(self) -> Dict[str, Any]:
        """Return default feature values when extraction fails"""
        return {
            'duchenne_smile': {
                'duchenne_score': 0.0,
                'authenticity_level': 0.0,
                'smile_balance': 0.0
            },
            'facial_asymmetry': {
                'overall_asymmetry': 0.0,
                'eye_asymmetry': 0.0,
                'au_asymmetry_count': 0
            },
            'au_dynamics': {
                'peak_velocity': 0.0,
                'peak_acceleration': 0.0,
                'avg_velocity': 0.0,
                'avg_acceleration': 0.0
            },
            'visual_attention': {
                'visual_attention_index': 0.0,
                'gaze_focus_component': 0.0,
                'head_alignment_component': 0.0,
                'eye_openness_component': 0.0,
                'attention_stability': 0.0
            },
            'fatigue_detection': {
                'fatigue_score': 0.0,
                'ear_decline_trend': 0.0,
                'blink_frequency_increase': 0.0,
                'eye_closure_prolongation': 0.0
            },
            'granular_emotions': {
                'joy': 0.0,
                'excitement': 0.0,
                'anger': 0.0,
                'fear': 0.0,
                'sadness': 0.0,
                'boredom': 0.0,
                'contentment': 0.0,
                'surprise': 0.0,
                'disgust': 0.0
            },
            'micro_expressions': {
                'micro_detected': False,
                'micro_type': 'none',
                'micro_intensity': 0.0,
                'micro_duration': 0.0
            },
            'cognitive_load': {
                'cognitive_load_score': 0.0,
                'blink_load_component': 0.0,
                'fixation_load_component': 0.0,
                'pupil_load_component': 0.0,
                'expression_load_component': 0.0
            }
        }
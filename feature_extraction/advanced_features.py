import numpy as np
import math
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeaturesExtractor:
    """Extract advanced features from facial data using only a webcam."""
    
    def __init__(self, history_size: int = 120):  # 4 seconds at 30fps
        """
        Initialize advanced features extractor.
        
        Args:
            history_size: Number of frames to keep in history.
        """
        self.history_size = history_size
        
        # Initialize history buffers for webcam-observable features
        self.emotion_history = deque(maxlen=history_size)
        self.au_history = deque(maxlen=history_size)
        self.gaze_history = deque(maxlen=history_size)
        self.head_pose_history = deque(maxlen=history_size)
        self.arousal_history = deque(maxlen=history_size)
        self.valence_history = deque(maxlen=history_size)
        self.suppression_history = deque(maxlen=history_size)
        self.blink_rate_history = deque(maxlen=history_size)
        self.facial_asymmetry_history = deque(maxlen=history_size)
        
        # Personality trait accumulators
        self.personality_scores = {
            'extraversion': 0.0,
            'agreeableness': 0.0,
            'conscientiousness': 0.0,
            'neuroticism': 0.0,
            'openness': 0.0
        }
        
        # Engagement tracking
        self.engagement_baseline = 0.5
        self.stress_baseline = 0.3
        
        # Deception detection parameters (observable indicators)
        self.deception_indicators = {
            'eye_contact_reduction': 0.0,
            'micro_expression_increase': 0.0,
            'fidgeting': 0.0,
            'gesture_suppression': 0.0,
            'facial_touching': 0.0
        }
        
    def _calculate_consistency(self, history: deque) -> float:
        """Calculate consistency score from history."""
        if len(history) < 3:
            return 0.0
        # Ensure that history contains numerical values for standard deviation
        numeric_history = [x for x in history if isinstance(x, (int, float))]
        if not numeric_history:
            return 0.0
        std_dev = np.std(numeric_history)
        return 1.0 - min(1.0, std_dev) # Cap consistency at 1.0 for better interpretation
            
    def _calculate_emotion_suppression(self, raw_emotions: Dict, micro_expressions: Dict, 
                                        au_dynamics: Dict, facial_tension: float) -> Dict[str, float]:
        """Detect emotion suppression based on mismatches and facial tension."""
        suppression_score = 0.0
        suppression_type = 'none'
        
        # Get dominant emotion
        emotions = raw_emotions.get('granular_emotions', {})
        if emotions:
            dominant_emotion = max(emotions.keys(), key=lambda k: emotions[k])
            dominant_intensity = emotions[dominant_emotion]
        else:
            dominant_emotion = 'neutral'
            dominant_intensity = 0.0
        
        # Check for micro-expressions (observable as brief, involuntary facial movements)
        micro_detected = micro_expressions.get('micro_detected', False)
        micro_type = micro_expressions.get('micro_type', 'none')
        micro_intensity = micro_expressions.get('micro_intensity', 0.0)
        
        # Suppression indicators
        if micro_detected and micro_type != dominant_emotion:
            # Mismatch between macro and micro expressions
            suppression_score = micro_intensity * 0.7
            suppression_type = f"suppressing_{micro_type}"
        
        # Check AU dynamics for suppression patterns (e.g., restricted movements)
        au_velocities = [v for k, v in au_dynamics.items() if '_velocity' in k]
        if au_velocities:
            velocity_variance = np.var(au_velocities)
            if velocity_variance > 0.5:  # High variance can indicate effort to suppress
                suppression_score += 0.3
        
        # Facial tension (directly observable)
        suppression_score += facial_tension * 0.4 # Increase weight of direct facial tension
        
        suppression_score = min(1.0, suppression_score)
        self.suppression_history.append(suppression_score)
        
        return {
            'suppression_score': suppression_score,
            'suppression_type': suppression_type,
            'suppression_consistency': self._calculate_consistency(self.suppression_history),
            'tension_level': facial_tension
        }
    
    def _assess_personality_traits(self, emotions: Dict, gaze_data: Dict, 
                                   behavioral_data: Dict, au_features: Dict) -> Dict[str, float]:
        """Assess Big Five personality traits from facial and gaze behavior."""
        traits = {}
        
        # Extract relevant features
        joy = emotions.get('joy', 0)
        excitement = emotions.get('excitement', 0)
        anger = emotions.get('anger', 0)
        fear = emotions.get('fear', 0)
        sadness = emotions.get('sadness', 0)
        surprise = emotions.get('surprise', 0)
        
        # Gaze patterns (observable)
        eye_contact_score = gaze_data.get('eye_contact_score', 0.0) # Assume a score 0-1
        attention_score = gaze_data.get('attention_score', 0.0)
        
        # Behavioral indicators (observable)
        smile_frequency = behavioral_data.get('smile_frequency', 0.0)
        expressiveness = behavioral_data.get('expressiveness', 0.0) # Based on AU intensity variance
        
        # Extraversion: High positive emotions, eye contact, expressiveness
        extraversion = (joy * 0.3 + excitement * 0.2 + 
                        eye_contact_score * 0.2 + 
                        expressiveness * 0.3)
        traits['extraversion'] = min(1.0, extraversion)
        
        # Agreeableness: Positive emotions, low anger, high smile frequency
        agreeableness = (joy * 0.4 + smile_frequency * 0.3 + 
                         (1.0 - anger) * 0.3)
        traits['agreeableness'] = min(1.0, agreeableness)
        
        # Conscientiousness: Consistent attention, low distraction
        distraction_score = gaze_data.get('distraction_score', 0.0)
        conscientiousness = (attention_score * 0.6 + 
                             (1.0 - distraction_score) * 0.4)
        traits['conscientiousness'] = min(1.0, conscientiousness)
        
        # Neuroticism: High negative emotions, stress indicators (observable ones)
        stress_score = behavioral_data.get('stress_score', 0.0) # Based on observable facial tension, micro-movements
        neuroticism = (fear * 0.3 + sadness * 0.2 + 
                       anger * 0.2 + stress_score * 0.3)
        traits['neuroticism'] = min(1.0, neuroticism)
        
        # Openness: Varied expressions, surprise, engagement
        openness = (surprise * 0.3 + expressiveness * 0.4 + 
                    attention_score * 0.3)
        traits['openness'] = min(1.0, openness)
        
        return traits
    
    def _assess_engagement_level(self, attention_data: Dict, arousal_valence: Dict, 
                                 behavioral_data: Dict) -> Dict[str, float]:
        """Assess engagement level from observable indicators."""
        
        # Attention indicators (gaze, head pose)
        attention_score = attention_data.get('attention_score', 0.0)
        looking_forward = attention_data.get('looking_forward', False)
        distraction_score = attention_data.get('distraction_score', 0.0)
        
        # Arousal-valence indicators (from facial expressions)
        arousal = arousal_valence.get('arousal', 0.0)
        valence = arousal_valence.get('valence', 0.0)
        
        # Behavioral indicators (observable)
        blink_rate = behavioral_data.get('blink_rate', 0.0)
        drowsiness = behavioral_data.get('drowsiness_score', 0.0) # Based on eye closure, head nodding
        
        # Calculate engagement components
        visual_engagement = attention_score * 0.6 + (1.0 - distraction_score) * 0.4
        emotional_engagement = (arousal + valence) / 2.0
        cognitive_engagement = 1.0 - drowsiness
        
        # Overall engagement
        overall_engagement = (visual_engagement * 0.4 + 
                              emotional_engagement * 0.3 + 
                              cognitive_engagement * 0.3)
        
        # Engagement state classification
        if overall_engagement > 0.7:
            engagement_state = 'high'
        elif overall_engagement > 0.4:
            engagement_state = 'moderate'
        else:
            engagement_state = 'low'
        
        return {
            'overall_engagement': overall_engagement,
            'visual_engagement': visual_engagement,
            'emotional_engagement': emotional_engagement,
            'cognitive_engagement': cognitive_engagement,
            'engagement_state': engagement_state,
            'engagement_consistency': self._calculate_consistency(
                deque([overall_engagement] + list(self.arousal_history)[-10:], maxlen=10)
            )
        }
    
    def _detect_stress_anxiety(self, facial_features: Dict, behavioral_data: Dict, 
                               physiological_indicators: Dict) -> Dict[str, float]:
        """Detect stress and anxiety from observable facial and behavioral indicators."""
        
        # Facial tension indicators (directly observable)
        facial_tension = facial_features.get('facial_tension', 0.0)
        asymmetry = facial_features.get('asymmetry_score', 0.0)
        
        # Behavioral stress indicators (observable)
        blink_rate = behavioral_data.get('blink_rate', 0.0)
        micro_movements = behavioral_data.get('micro_movements', 0.0) # Small, repetitive facial movements
        jaw_tension = behavioral_data.get('jaw_tension', 0.0) # Based on AU activity around the jaw/mouth
        
        # Arousal and Valence (from facial expressions)
        arousal = physiological_indicators.get('arousal', 0.0)
        valence = physiological_indicators.get('valence', 0.0)
        
        # Calculate stress score
        stress_score = (facial_tension * 0.25 + 
                        asymmetry * 0.15 + 
                        min(blink_rate / 30.0, 1.0) * 0.2 +  # Normalize blink rate
                        micro_movements * 0.15 + 
                        jaw_tension * 0.15 + 
                        arousal * 0.1)
        
        # Calculate anxiety score (different pattern from stress, but still observable)
        anxiety_score = (micro_movements * 0.3 + 
                         asymmetry * 0.2 + 
                         (1.0 - valence) * 0.25 +  # Low valence indicates negative emotional state, common with anxiety
                         arousal * 0.25)
        
        # Temporal consistency
        stress_consistency = self._calculate_consistency(
            deque([stress_score] + list(self.arousal_history)[-10:], maxlen=10)
        )
        
        return {
            'stress_score': min(1.0, stress_score),
            'anxiety_score': min(1.0, anxiety_score),
            'stress_consistency': stress_consistency,
            'stress_level': 'high' if stress_score > 0.7 else 'moderate' if stress_score > 0.4 else 'low',
            'anxiety_level': 'high' if anxiety_score > 0.7 else 'moderate' if anxiety_score > 0.4 else 'low'
        }
    
    def _detect_deception_indicators(self, gaze_data: Dict, micro_expressions: Dict, 
                                     behavioral_data: Dict) -> Dict[str, float]:
        """Detect potential deception indicators from overt webcam data."""
        
        # Gaze aversion patterns (observable)
        eye_contact_time = gaze_data.get('eye_contact_duration', 0.0) # Proportion of time looking at camera
        baseline_eye_contact = 0.6  # Normal eye contact percentage
        eye_contact_reduction = max(0.0, (baseline_eye_contact - eye_contact_time) / baseline_eye_contact)
        
        # Micro-expression frequency (observable)
        micro_frequency = micro_expressions.get('micro_frequency', 0.0)
        baseline_micro_frequency = 0.1
        micro_increase = max(0.0, (micro_frequency - baseline_micro_frequency) / baseline_micro_frequency)
        
        # Behavioral indicators (observable)
        fidgeting = behavioral_data.get('fidgeting_score', 0.0) # Observable restless movements
        gesture_suppression = behavioral_data.get('gesture_suppression', 0.0) # Reduced natural hand/arm movements
        facial_touching = behavioral_data.get('facial_touching', 0.0) # Frequency of touching face
        
        # Calculate deception likelihood
        deception_score = (eye_contact_reduction * 0.3 +  # Increased weight for eye contact
                           micro_increase * 0.25 + 
                           fidgeting * 0.2 + 
                           gesture_suppression * 0.15 + 
                           facial_touching * 0.1) # Speech hesitation removed as it's not purely facial
        
        return {
            'deception_likelihood': min(1.0, deception_score),
            'eye_contact_reduction': eye_contact_reduction,
            'micro_expression_increase': micro_increase,
            'behavioral_indicators': fidgeting + gesture_suppression + facial_touching,
            'deception_confidence': 'high' if deception_score > 0.7 else 'moderate' if deception_score > 0.4 else 'low'
        }
    
    def _analyze_temporal_dynamics(self, current_features: Dict, timestamps: List[float]) -> Dict[str, float]:
        """Analyze temporal patterns in emotional and behavioral features from webcam data."""
        
        if len(self.emotion_history) < 10:
            return {
                'emotion_stability': 0.0,
                'emotion_variability': 0.0,
                'trend_direction': 'stable',
                'peak_detection': False,
                'valley_detection': False,
                'arousal_trend': 'stable',
                'valence_trend': 'stable'
            }
            
        # Convert history to arrays for analysis
        emotion_values = list(self.emotion_history)
        arousal_values = list(self.arousal_history)
        valence_values = list(self.valence_history)
        
        # Calculate stability (inverse of variance)
        emotion_stability = 1.0 - min(1.0, np.var(emotion_values))
        
        # Calculate variability
        emotion_variability = np.std(emotion_values)
        
        # Trend analysis
        if len(timestamps) >= len(emotion_values) and len(emotion_values) > 1:
            time_window = timestamps[-len(emotion_values):]
            # Ensure time_window is increasing for linregress
            if len(set(time_window)) > 1: # Check for at least two distinct time points
                slope, _, _, _, _ = stats.linregress(time_window, emotion_values)
            else:
                slope = 0.0 # Cannot calculate slope with identical timestamps
            
            if slope > 0.1:
                trend_direction = 'increasing'
            elif slope < -0.1:
                trend_direction = 'decreasing'
            else:
                trend_direction = 'stable'
        else:
            trend_direction = 'stable'
            
        # Peak and valley detection
        recent_values = emotion_values[-5:] if len(emotion_values) >= 5 else emotion_values
        current_value = recent_values[-1] if recent_values else 0.0
        
        # Ensure enough data points for meaningful std dev calculation
        peak_detection = (current_value > np.mean(recent_values) + np.std(recent_values)) if len(recent_values) > 1 and np.std(recent_values) > 0 else False
        valley_detection = (current_value < np.mean(recent_values) - np.std(recent_values)) if len(recent_values) > 1 and np.std(recent_values) > 0 else False
        
        arousal_trend = 'increasing' if len(arousal_values) > 1 and arousal_values[-1] > arousal_values[-2] else 'stable'
        valence_trend = 'increasing' if len(valence_values) > 1 and valence_values[-1] > valence_values[-2] else 'stable'
        
        return {
            'emotion_stability': emotion_stability,
            'emotion_variability': emotion_variability,
            'trend_direction': trend_direction,
            'peak_detection': peak_detection,
            'valley_detection': valley_detection,
            'arousal_trend': arousal_trend,
            'valence_trend': valence_trend
        }
    
    def extract_features(self, raw_features: Dict, derived_features: Dict, 
                         timestamps: List[float]) -> Dict[str, Any]:
        """
        Extract advanced features from raw and derived features using only webcam data.
        
        Args:
            raw_features: Raw facial features.
            derived_features: Derived features.
            timestamps: List of timestamps for temporal analysis.
            
        Returns:
            Dictionary containing advanced features.
        """
        advanced_features = {}
        
        try:
            # Extract component features
            emotions = derived_features.get('emotions', {})
            # Renamed 'attention_states' to 'gaze_data' for clarity with webcam focus
            gaze_data = raw_features.get('gaze_data', {}) 
            behavioral_data = raw_features.get('behavioral_detections', {})
            au_features = derived_features.get('action_units', {})
            
            # Update history
            if emotions:
                dominant_emotion_score = max(emotions.values()) if emotions.values() else 0.0
                self.emotion_history.append(dominant_emotion_score)
            
            arousal = raw_features.get('cnn_outputs', {}).get('arousal', 0.0)
            valence = raw_features.get('cnn_outputs', {}).get('valence', 0.0)
            self.arousal_history.append(arousal)
            self.valence_history.append(valence)
            
            # Additional history for webcam-observable features
            blink_rate = behavioral_data.get('blink_rate', 0.0)
            self.blink_rate_history.append(blink_rate)
            facial_asymmetry = derived_features.get('facial_features', {}).get('asymmetry_score', 0.0)
            self.facial_asymmetry_history.append(facial_asymmetry)

            # Emotion suppression detection
            micro_expressions = derived_features.get('micro_expressions', {})
            au_dynamics = derived_features.get('au_dynamics', {})
            facial_tension = derived_features.get('facial_features', {}).get('facial_tension', 0.0)
            advanced_features['emotion_suppression'] = self._calculate_emotion_suppression(
                {'granular_emotions': emotions}, micro_expressions, au_dynamics, facial_tension
            )
            
            # Personality trait assessment
            advanced_features['personality_traits'] = self._assess_personality_traits(
                emotions, gaze_data, behavioral_data, au_features
            )
            
            # Engagement level assessment
            advanced_features['engagement'] = self._assess_engagement_level(
                gaze_data, {'arousal': arousal, 'valence': valence}, behavioral_data
            )
            
            # Stress and anxiety detection
            facial_features = derived_features.get('facial_features', {})
            physiological = {'arousal': arousal, 'valence': valence} # Arousal/Valence are derived from facial expressions in a webcam context
            advanced_features['stress_anxiety'] = self._detect_stress_anxiety(
                facial_features, behavioral_data, physiological
            )
            
            # Deception indicators
            # Removed 'speech_data' as it's not a facial pic/webcam feature
            advanced_features['deception_indicators'] = self._detect_deception_indicators(
                gaze_data, micro_expressions, behavioral_data
            )
            
            # Temporal dynamics analysis
            advanced_features['temporal_dynamics'] = self._analyze_temporal_dynamics(
                raw_features, timestamps
            )
            
        except Exception as e:
            print(f"Error extracting advanced features: {e}")
            advanced_features = self._get_default_advanced_features()
            
        return advanced_features
    
    def _get_default_advanced_features(self) -> Dict[str, Any]:
        """Return default advanced feature values."""
        return {
            'emotion_suppression': {
                'suppression_score': 0.0,
                'suppression_type': 'none',
                'suppression_consistency': 0.0,
                'tension_level': 0.0
            },
            'personality_traits': {
                'extraversion': 0.0,
                'agreeableness': 0.0,
                'conscientiousness': 0.0,
                'neuroticism': 0.0,
                'openness': 0.0
            },
            'engagement': {
                'overall_engagement': 0.0,
                'visual_engagement': 0.0,
                'emotional_engagement': 0.0,
                'cognitive_engagement': 0.0,
                'engagement_state': 'low',
                'engagement_consistency': 0.0
            },
            'stress_anxiety': {
                'stress_score': 0.0,
                'anxiety_score': 0.0,
                'stress_consistency': 0.0,
                'stress_level': 'low',
                'anxiety_level': 'low'
            },
            'deception_indicators': {
                'deception_likelihood': 0.0,
                'eye_contact_reduction': 0.0,
                'micro_expression_increase': 0.0,
                'behavioral_indicators': 0.0,
                'deception_confidence': 'low'
            },
            'temporal_dynamics': {
                'emotion_stability': 0.0,
                'emotion_variability': 0.0,
                'trend_direction': 'stable',
                'peak_detection': False,
                'valley_detection': False,
                'arousal_trend': 'stable',
                'valence_trend': 'stable'
            }
        }
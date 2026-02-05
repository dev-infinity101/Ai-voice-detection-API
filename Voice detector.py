import librosa
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from scipy import signal
from typing import Tuple, Dict
import io
import base64
import soundfile as sf

class VoiceDetector:
    """
    AI vs Human Voice Detection System
    Uses audio feature analysis and deep learning
    """
    
    def __init__(self):
        """Initialize the voice detector with feature extractors"""
        self.sample_rate = 16000
        
        # These are heuristic thresholds - you'll tune based on sample data
        self.ai_indicators = {
            'pitch_variance_threshold': 0.15,  # AI voices have lower variance
            'spectral_flatness_threshold': 0.25,  # AI voices are "flatter"
            'zero_crossing_threshold': 0.08,  # Different patterns
            'formant_stability_threshold': 0.9  # AI formants are too stable
        }
    
    def decode_base64_audio(self, audio_base64: str) -> Tuple[np.ndarray, int]:
        """
        Decode base64 MP3 to audio array
        
        Args:
            audio_base64: Base64 encoded MP3 string
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Decode base64
            audio_bytes = base64.b64decode(audio_base64)
            
            # Load audio from bytes
            audio_data, sr = sf.read(io.BytesIO(audio_bytes))
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample to standard rate
            if sr != self.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
            
            return audio_data, self.sample_rate
            
        except Exception as e:
            raise ValueError(f"Failed to decode audio: {str(e)}")
    
    def extract_audio_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract key features that differentiate AI from human voices
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            
        Returns:
            Dictionary of feature values
        """
        features = {}
        
        # 1. Pitch Variance (AI voices have unnaturally consistent pitch)
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 0:
            features['pitch_variance'] = np.std(pitch_values) / (np.mean(pitch_values) + 1e-8)
        else:
            features['pitch_variance'] = 0.0
        
        # 2. Spectral Flatness (AI voices have flatter spectrum)
        spectral_flatness = librosa.feature.spectral_flatness(y=audio)
        features['spectral_flatness'] = np.mean(spectral_flatness)
        
        # 3. Zero Crossing Rate (Different patterns in AI vs human)
        zcr = librosa.feature.zero_crossing_rate(audio)
        features['zero_crossing_rate'] = np.mean(zcr)
        
        # 4. Spectral Rolloff (Energy distribution)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features['spectral_rolloff'] = np.mean(rolloff)
        
        # 5. MFCC Statistics (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs)
        features['mfcc_std'] = np.std(mfccs)
        
        # 6. Formant Stability (AI voices have too-stable formants)
        # Using spectral contrast as a proxy
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features['formant_stability'] = 1.0 - np.std(contrast) / (np.mean(contrast) + 1e-8)
        
        # 7. Temporal Envelope Smoothness
        envelope = np.abs(signal.hilbert(audio))
        envelope_diff = np.diff(envelope)
        features['envelope_smoothness'] = np.std(envelope_diff)
        
        # 8. Jitter and Shimmer (voice quality measures)
        # Simplified version
        if len(pitch_values) > 1:
            jitter = np.mean(np.abs(np.diff(pitch_values))) / (np.mean(pitch_values) + 1e-8)
            features['jitter'] = jitter
        else:
            features['jitter'] = 0.0
        
        return features
    
    def calculate_ai_probability(self, features: Dict[str, float]) -> Tuple[float, str]:
        """
        Calculate probability that voice is AI-generated based on features
        
        Args:
            features: Extracted audio features
            
        Returns:
            Tuple of (confidence_score, explanation)
        """
        ai_score = 0.0
        indicators = []
        
        # Check each indicator
        if features['pitch_variance'] < self.ai_indicators['pitch_variance_threshold']:
            ai_score += 0.25
            indicators.append("unnaturally consistent pitch")
        
        if features['spectral_flatness'] > self.ai_indicators['spectral_flatness_threshold']:
            ai_score += 0.2
            indicators.append("flat spectral characteristics")
        
        if features['formant_stability'] > self.ai_indicators['formant_stability_threshold']:
            ai_score += 0.25
            indicators.append("overly stable formants")
        
        # Low jitter indicates synthetic voice
        if features['jitter'] < 0.01:
            ai_score += 0.15
            indicators.append("minimal pitch variation (jitter)")
        
        # Envelope smoothness - AI voices are often too smooth
        if features['envelope_smoothness'] < 0.05:
            ai_score += 0.15
            indicators.append("unusually smooth temporal envelope")
        
        # Generate explanation
        if ai_score > 0.5:
            explanation = f"AI-generated indicators detected: {', '.join(indicators)}"
        else:
            human_indicators = []
            if features['pitch_variance'] >= self.ai_indicators['pitch_variance_threshold']:
                human_indicators.append("natural pitch variation")
            if features['jitter'] >= 0.01:
                human_indicators.append("human-like voice quality")
            if features['formant_stability'] <= self.ai_indicators['formant_stability_threshold']:
                human_indicators.append("natural formant dynamics")
            
            explanation = f"Human voice characteristics: {', '.join(human_indicators) if human_indicators else 'natural speech patterns detected'}"
        
        return ai_score, explanation
    
    def detect(self, audio_base64: str, language: str) -> Dict:
        """
        Main detection function
        
        Args:
            audio_base64: Base64 encoded MP3 audio
            language: Language of the audio
            
        Returns:
            Detection result dictionary
        """
        try:
            # Decode audio
            audio, sr = self.decode_base64_audio(audio_base64)
            
            # Extract features
            features = self.extract_audio_features(audio, sr)
            
            # Calculate AI probability
            ai_probability, explanation = self.calculate_ai_probability(features)
            
            # Determine classification
            if ai_probability > 0.5:
                classification = "AI_GENERATED"
                confidence = ai_probability
            else:
                classification = "HUMAN"
                confidence = 1.0 - ai_probability
            
            # Ensure confidence is in valid range
            confidence = min(max(confidence, 0.0), 1.0)
            
            return {
                "status": "success",
                "language": language,
                "classification": classification,
                "confidenceScore": round(confidence, 2),
                "explanation": explanation
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Detection failed: {str(e)}"
            }


# Enhanced version with ML model (optional - use if you have time)
class EnhancedVoiceDetector(VoiceDetector):
    """
    Enhanced detector using pre-trained models
    Falls back to feature-based detection
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None
        
        # Try to load a pre-trained model (optional)
        try:
            # You can use models like:
            # - facebook/wav2vec2-base
            # - MIT/ast-finetuned-audioset-10-10-0.4593
            # For this hackathon, feature-based might be faster and more reliable
            pass
        except:
            print("Using feature-based detection (no pre-trained model)")
    
    def detect_with_model(self, audio: np.ndarray) -> float:
        """
        Use ML model for detection if available
        """
        if self.model is None:
            return None
        
        try:
            # Model inference logic here
            # Returns AI probability
            pass
        except:
            return None
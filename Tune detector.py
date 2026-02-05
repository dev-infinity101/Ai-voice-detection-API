"""
Tuning script for voice detection thresholds
Use this to optimize detection accuracy with sample data
"""

import os
import base64
import json
from voice_detector import VoiceDetector
from pathlib import Path
import numpy as np

class DetectorTuner:
    """Helper class to tune detection thresholds"""
    
    def __init__(self, samples_dir: str = "samples"):
        self.samples_dir = Path(samples_dir)
        self.detector = VoiceDetector()
        
    def load_audio_as_base64(self, file_path: str) -> str:
        """Load audio file and convert to base64"""
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def test_sample(self, file_path: str, language: str = "English", 
                    true_label: str = None) -> dict:
        """
        Test a single audio sample
        
        Args:
            file_path: Path to audio file
            language: Language of the audio
            true_label: True label (AI_GENERATED or HUMAN) if known
            
        Returns:
            Test results dictionary
        """
        audio_b64 = self.load_audio_as_base64(file_path)
        
        # Get raw features first
        audio, sr = self.detector.decode_base64_audio(audio_b64)
        features = self.detector.extract_audio_features(audio, sr)
        
        # Get detection result
        result = self.detector.detect(audio_b64, language)
        
        # Add comparison if true label provided
        if true_label:
            result['correct'] = result['classification'] == true_label
            result['true_label'] = true_label
        
        result['features'] = features
        result['file'] = str(file_path)
        
        return result
    
    def test_directory(self, label: str = None) -> list:
        """
        Test all MP3 files in a directory
        
        Args:
            label: Expected label (AI_GENERATED or HUMAN)
            
        Returns:
            List of test results
        """
        results = []
        
        if not self.samples_dir.exists():
            print(f"Warning: Directory {self.samples_dir} not found")
            return results
        
        for file_path in self.samples_dir.glob("*.mp3"):
            # Try to infer language from filename
            filename = file_path.stem.lower()
            
            if 'tamil' in filename:
                language = 'Tamil'
            elif 'hindi' in filename:
                language = 'Hindi'
            elif 'malayalam' in filename:
                language = 'Malayalam'
            elif 'telugu' in filename:
                language = 'Telugu'
            else:
                language = 'English'
            
            # Try to infer true label from filename
            true_label = None
            if label:
                true_label = label
            elif 'ai' in filename or 'generated' in filename or 'synthetic' in filename:
                true_label = 'AI_GENERATED'
            elif 'human' in filename or 'real' in filename:
                true_label = 'HUMAN'
            
            try:
                result = self.test_sample(file_path, language, true_label)
                results.append(result)
                
                # Print result
                status = "✓" if result.get('correct', None) else "✗"
                print(f"{status} {file_path.name}: {result['classification']} "
                      f"(confidence: {result['confidenceScore']:.2f})")
                
            except Exception as e:
                print(f"✗ Error testing {file_path.name}: {e}")
        
        return results
    
    def analyze_features(self, results: list):
        """
        Analyze feature distributions to suggest optimal thresholds
        
        Args:
            results: List of test results with features
        """
        if not results:
            print("No results to analyze")
            return
        
        print("\n" + "=" * 70)
        print("FEATURE ANALYSIS")
        print("=" * 70)
        
        ai_results = [r for r in results if r.get('true_label') == 'AI_GENERATED']
        human_results = [r for r in results if r.get('true_label') == 'HUMAN']
        
        if not ai_results or not human_results:
            print("Need both AI and Human samples for analysis")
            return
        
        # Analyze each feature
        features_to_analyze = [
            'pitch_variance',
            'spectral_flatness',
            'zero_crossing_rate',
            'formant_stability',
            'jitter'
        ]
        
        print("\nFeature Statistics:")
        print("-" * 70)
        
        for feature_name in features_to_analyze:
            ai_values = [r['features'][feature_name] for r in ai_results]
            human_values = [r['features'][feature_name] for r in human_results]
            
            ai_mean = np.mean(ai_values)
            ai_std = np.std(ai_values)
            human_mean = np.mean(human_values)
            human_std = np.std(human_values)
            
            print(f"\n{feature_name}:")
            print(f"  AI:    {ai_mean:.4f} ± {ai_std:.4f}")
            print(f"  Human: {human_mean:.4f} ± {human_std:.4f}")
            
            # Suggest threshold
            if ai_mean < human_mean:
                suggested = (ai_mean + ai_std + human_mean - human_std) / 2
                print(f"  Suggested threshold (AI < threshold < Human): {suggested:.4f}")
            else:
                suggested = (ai_mean - ai_std + human_mean + human_std) / 2
                print(f"  Suggested threshold (Human < threshold < AI): {suggested:.4f}")
    
    def calculate_accuracy(self, results: list) -> dict:
        """
        Calculate accuracy metrics
        
        Args:
            results: List of test results
            
        Returns:
            Dictionary of accuracy metrics
        """
        if not results:
            return {}
        
        results_with_labels = [r for r in results if 'true_label' in r]
        
        if not results_with_labels:
            return {}
        
        total = len(results_with_labels)
        correct = sum(1 for r in results_with_labels if r.get('correct', False))
        
        # Confusion matrix
        tp = sum(1 for r in results_with_labels 
                if r['true_label'] == 'AI_GENERATED' and r['classification'] == 'AI_GENERATED')
        tn = sum(1 for r in results_with_labels 
                if r['true_label'] == 'HUMAN' and r['classification'] == 'HUMAN')
        fp = sum(1 for r in results_with_labels 
                if r['true_label'] == 'HUMAN' and r['classification'] == 'AI_GENERATED')
        fn = sum(1 for r in results_with_labels 
                if r['true_label'] == 'AI_GENERATED' and r['classification'] == 'HUMAN')
        
        accuracy = correct / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total': total,
            'correct': correct,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        return metrics
    
    def print_summary(self, results: list):
        """Print summary of results"""
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        metrics = self.calculate_accuracy(results)
        
        if metrics:
            print(f"\nOverall Accuracy: {metrics['accuracy']:.2%}")
            print(f"Precision: {metrics['precision']:.2%}")
            print(f"Recall: {metrics['recall']:.2%}")
            print(f"F1 Score: {metrics['f1']:.2%}")
            print(f"\nConfusion Matrix:")
            print(f"  True Positives (AI detected as AI): {metrics['true_positives']}")
            print(f"  True Negatives (Human detected as Human): {metrics['true_negatives']}")
            print(f"  False Positives (Human detected as AI): {metrics['false_positives']}")
            print(f"  False Negatives (AI detected as Human): {metrics['false_negatives']}")
        
        # Average confidence scores
        if results:
            avg_confidence = np.mean([r['confidenceScore'] for r in results])
            print(f"\nAverage Confidence Score: {avg_confidence:.2f}")


def main():
    """Main tuning function"""
    print("=" * 70)
    print("VOICE DETECTION THRESHOLD TUNING")
    print("=" * 70)
    
    tuner = DetectorTuner(samples_dir="samples")
    
    print("\nTesting all samples...")
    print("-" * 70)
    
    # Test all samples
    results = tuner.test_directory()
    
    if results:
        # Print summary
        tuner.print_summary(results)
        
        # Analyze features
        tuner.analyze_features(results)
        
        # Save results
        with open('tuning_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: tuning_results.json")
        
        print("\n" + "=" * 70)
        print("NEXT STEPS:")
        print("=" * 70)
        print("1. Review the feature statistics above")
        print("2. Update thresholds in voice_detector.py based on suggestions")
        print("3. Re-run this script to verify improvements")
        print("4. Iterate until accuracy is satisfactory")
    else:
        print("\nNo samples found. Please add MP3 files to the 'samples' directory.")
        print("\nFile naming convention:")
        print("  - ai_english.mp3 (for AI-generated English audio)")
        print("  - human_tamil.mp3 (for human Tamil audio)")
        print("  - Include language name: tamil, english, hindi, malayalam, telugu")
        print("  - Include type: ai/generated/synthetic OR human/real")


if __name__ == "__main__":
    main()
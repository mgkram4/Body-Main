#!/usr/bin/env python3
"""
BIO-SCAN-BEST - Standalone Body Scanner
Run advanced body measurements using computer vision and AI models
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict

import cv2

# Import our body estimator
from body import AdvancedBodyEstimator


def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('body_scan.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def setup_camera() -> cv2.VideoCapture:
    """Initialize camera with optimal settings"""
    logger = logging.getLogger(__name__)
    
    # Try different camera indices
    for camera_index in [0, 1, 2]:
        logger.info(f"Trying camera index {camera_index}...")
        camera = cv2.VideoCapture(camera_index)
        
        if camera.isOpened():
            # Set camera properties for best quality
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
            # Test if camera is working
            ret, test_frame = camera.read()
            if ret and test_frame is not None:
                logger.info(f"Camera {camera_index} initialized successfully")
                logger.info(f"Camera resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
                return camera
            else:
                camera.release()
        
    raise RuntimeError("No working camera found. Please check your camera connection.")

def save_results(results: Dict[str, Any], timestamp: str) -> str:
    """Save scan results to JSON file"""
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = "scan_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    filename = f"body_scan_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        return ""

def display_results(results: Dict[str, Any]):
    """Display scan results in a formatted way"""
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*60)
    print("🔬 BIO-SCAN-BEST - BODY SCAN RESULTS")
    print("="*60)
    
    # Basic measurements
    if results.get('height_ft_in'):
        print(f"📏 Height: {results['height_ft_in']} ({results.get('height_cm', 'N/A'):.1f} cm)")
    
    if results.get('weight_lbs'):
        print(f"⚖️  Weight: {results['weight_lbs']:.1f} lbs ({results.get('weight_kg', 'N/A'):.1f} kg)")
    
    if results.get('bmi'):
        bmi = results['bmi']
        print(f"🏃 BMI: {bmi:.1f}", end="")
        if bmi < 18.5:
            print(" (Underweight)")
        elif bmi < 25:
            print(" (Normal)")
        elif bmi < 30:
            print(" (Overweight)")
        else:
            print(" (Obese)")
    
    # Body composition
    print("\n💪 BODY COMPOSITION:")
    if results.get('body_fat_percentage'):
        print(f"   Body Fat: {results['body_fat_percentage']:.1f}%")
    
    if results.get('muscle_mass_lbs'):
        print(f"   Muscle Mass: {results['muscle_mass_lbs']:.1f} lbs ({results.get('muscle_mass_kg', 'N/A'):.1f} kg)")
    
    if results.get('bone_density'):
        print(f"   Bone Density: {results['bone_density']:.2f} g/cm³")
    
    if results.get('metabolic_age'):
        print(f"   Metabolic Age: {results['metabolic_age']:.0f} years")
    
    # Additional measurements
    measurements = results.get('measurements_us', {})
    if measurements:
        print("\n📐 DETAILED MEASUREMENTS:")
        for key, value in measurements.items():
            if value:
                formatted_key = key.replace('_', ' ').title()
                unit = "in" if key.endswith('_in') else ""
                print(f"   {formatted_key}: {value:.1f} {unit}")
    
    # Scan quality
    confidence = results.get('confidence_scores', {})
    if confidence:
        print(f"\n🎯 SCAN QUALITY:")
        print(f"   Overall Confidence: {confidence.get('overall', 0):.1%}")
        print(f"   Height Confidence: {confidence.get('height', 0):.1%}")
        print(f"   Weight Confidence: {confidence.get('weight', 0):.1%}")
    
    scan_quality = results.get('scan_quality', {})
    if scan_quality:
        print(f"   Frames Processed: {scan_quality.get('frames_processed', 0)}")
        print(f"   Height Measurements: {scan_quality.get('height_measurements', 0)}")
        print(f"   Weight Measurements: {scan_quality.get('weight_measurements', 0)}")
    
    print("\n" + "="*60)

def run_body_scan(duration: int = 20, use_gpu: bool = True) -> Dict[str, Any]:
    """Run the complete body scanning process"""
    logger = logging.getLogger(__name__)
    
    print("🔬 Initializing BIO-SCAN-BEST Body Scanner...")
    print("⚡ Loading AI models (this may take a moment)...")
    
    # Initialize the body estimator
    try:
        estimator = AdvancedBodyEstimator(use_gpu=use_gpu)
        logger.info("Body estimator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize body estimator: {e}")
        raise
    
    # Setup camera
    try:
        camera = setup_camera()
        logger.info("Camera setup complete")
    except Exception as e:
        logger.error(f"Camera setup failed: {e}")
        raise
    
    try:
        print(f"\n🎯 Starting {duration}-second body scan...")
        print("📋 Instructions:")
        print("   1. Stand 6-8 feet from the camera")
        print("   2. Ensure your full body (head to feet) is visible")
        print("   3. Stand in a T-pose with arms extended")
        print("   4. Remain as still as possible during the scan")
        print("   5. Press 'q' to quit early if needed")
        print("\n⏱️  Scan will begin in 3 seconds...")
        
        # Countdown
        for i in range(3, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
        
        print("🚀 SCANNING NOW - REMAIN STILL!")
        
        # Run the scan
        results = estimator.run_scan(camera, duration=duration)
        
        # Add metadata
        results['scan_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'scanner_version': '1.0.0',
            'gpu_used': use_gpu
        }
        
        logger.info("Body scan completed successfully")
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️  Scan interrupted by user")
        logger.info("Scan interrupted by user")
        return {'error': 'Scan interrupted by user'}
        
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        return {'error': str(e)}
        
    finally:
        # Cleanup
        if 'camera' in locals():
            camera.release()
        cv2.destroyAllWindows()

def main():
    """Main application entry point"""
    logger = setup_logging()
    
    print("🔬 BIO-SCAN-BEST - Advanced Body Scanner")
    print("🤖 Powered by AI: YOLOv8 + MediaPipe + MiDaS + SAM + SMPL-X")
    print("-" * 60)
    
    # Check system requirements
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            print(f"🚀 GPU acceleration available: {torch.cuda.get_device_name(0)}")
        else:
            print("💻 Running on CPU (GPU recommended for faster processing)")
    except ImportError:
        gpu_available = False
        print("💻 PyTorch not found - running in basic mode")
    
    # Configuration
    scan_duration = 20  # seconds
    use_gpu = gpu_available
    
    try:
        # Run the scan
        results = run_body_scan(duration=scan_duration, use_gpu=use_gpu)
        
        if 'error' in results:
            print(f"\n❌ Scan failed: {results['error']}")
            logger.error(f"Scan failed: {results['error']}")
            return 1
        
        # Display results
        display_results(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_file = save_results(results, timestamp)
        
        if saved_file:
            print(f"\n💾 Results saved to: {saved_file}")
        
        print("\n✅ Body scan completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logger.error(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
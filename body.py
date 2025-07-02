# require user face and feet to be in the frame after detection of a human in the frame with yolo and media pipe

# use midas to determine user distance from the camera

# use smplx + mediapipe + midas + yolo + ratio body parts using SAM to estimate user height and weight

# use a custom algo to estimate user bmi

# use a custom algo to estimate user body fat percentage

# use a custom algo to estimate user muscle mass

# use a custom algo to estimate user bone density

# the scan should be done in a 20 secound window and save the results to a json file in collected data and dataframe

"""
Phase 2: Advanced Body Estimation
Implements comprehensive body measurements using multiple AI models
"""

import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.optim as optim
from ultralytics import YOLO

# Set up logger first
logger = logging.getLogger(__name__)

# Optional imports with graceful fallbacks
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    logger.warning("trimesh not available - volume calculations will use approximations")

try:
    from segment_anything import SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    logger.warning("segment-anything not available - will use fallback segmentation")

try:
    import smplx
    SMPLX_AVAILABLE = True
except ImportError:
    SMPLX_AVAILABLE = False
    logger.warning("smplx not available - will use pose estimation fallbacks")

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    EXIF_AVAILABLE = True
except ImportError:
    EXIF_AVAILABLE = False
    logger.warning("PIL not available - EXIF data extraction disabled")

class AdvancedBodyEstimator:
    """
    Advanced body estimator implementing 8-stage measurement pipeline:
    1. YOLOv8 person detection
    2. MediaPipe pose processing (33 body landmarks)
    3. MiDaS depth estimation for monocular depth maps
    4. SAM (Segment Anything Model) for body segmentation
    5. SMPL-X 3D body model fitting
    6. Height estimation (landmark + depth + SMPL-X)
    7. Weight estimation (volume + anthropometric + demographic)
    8. Result validation with confidence scores
    """
    
    def __init__(self, model_cache_dir: str = "models", use_gpu: bool = True):
        """Initialize the advanced body estimator"""
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.model_cache_dir = model_cache_dir
        os.makedirs(model_cache_dir, exist_ok=True)
        
        # Mixed precision for performance
        self.use_amp = torch.cuda.is_available() and use_gpu
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Stage 1: YOLOv8 Person Detection
        try:
            self.yolo_model = YOLO('yolov8n.pt')  # Will download if not present
            logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            logger.warning(f"YOLOv8 loading failed: {e}")
            self.yolo_model = None
        
        # Stage 2: MediaPipe Pose Processing
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Stage 3: MiDaS Depth Estimation
        self.midas_model = None
        self.midas_transform = None
        self.midas_device = None
        self._load_midas_model()
        
        # Stage 4: SAM (Segment Anything Model)
        self.sam_model = None
        self.sam_predictor = None
        self._load_sam_model()
        
        # Stage 5: SMPL-X 3D Body Model
        self.smplx_model = None
        self.smplx_available = False
        self._load_smplx_model()
        
        # Body measurement parameters
        self.pose_landmarks = []
        self.depth_map = None
        self.body_segmentation = None
        self.height_estimates = {}
        self.weight_estimates = {}
        
        # Confidence thresholds
        self.min_pose_confidence = 0.5
        self.min_measurement_confidence = 0.3
        
        # Performance metrics
        self.processing_times = {}
        
        # Visual feedback parameters
        self.current_metrics = {
            'height': None,
            'weight': None,
            'bmi': None,
            'body_type': None,
            'shoulders': None,
            'torso': None,
            'person_confidence': 0.0,
            'pose_quality': 0,
            'scan_stage': 'person_detection'
        }
        
        # Visual effect state
        self.scan_position = 0
        self.pulse_phase = 0
        self.particle_effects = []
        self.scan_stages = ['person_detection', 'pose_analysis', '3d_modeling', 'measurements']
        self.current_stage = 0
        self.stage_progress = 0.0
        
        # Color scheme for visual feedback (matching face.py)
        self.colors = {
            'primary': (255, 255, 0),      # Cyan
            'secondary': (255, 0, 255),    # Purple  
            'success': (0, 255, 0),        # Lime
            'warning': (0, 255, 255),      # Yellow
            'error': (0, 0, 255),          # Red
            'bg_panel': (0, 0, 0),         # Black
            'text': (255, 255, 255),       # White
            'high_conf': (0, 255, 0),      # Green (>0.8)
            'med_conf': (0, 255, 255),     # Yellow (0.5-0.8)
            'low_conf': (0, 0, 255)        # Red (<0.5)
        }
        
        # 3D wireframe for visualization
        self.wireframe_points = []
        self.wireframe_rotation = 0
        
        logger.info(f"Advanced Body Estimator initialized on {self.device}")
    
    def _load_midas_model(self):
        """Load MiDaS depth estimation model with DPT-Hybrid"""
        try:
            logger.info("Loading MiDaS v3.1 DPT-Hybrid model...")
            
            # Load MiDaS v3.1 with DPT-Hybrid transformer for best accuracy
            model_path = os.path.join(self.model_cache_dir, 'midas_v31_dpt_hybrid.pt')
            
            if os.path.exists(model_path):
                # Load cached model
                logger.info("Loading cached MiDaS model...")
                self.midas_model = torch.jit.load(model_path, map_location=self.device)
            else:
                # Download model
                self.midas_model = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid', 
                                                pretrained=True, trust_repo=True)
                # Cache the model
                torch.jit.save(torch.jit.script(self.midas_model), model_path)
            
            self.midas_model.to(self.device)
            self.midas_model.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
            self.midas_transform = midas_transforms.dpt_transform
            
            self.midas_device = self.device
            logger.info(f"MiDaS DPT-Hybrid model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.warning(f"MiDaS model loading failed: {e}. Depth estimation will use fallback.")
            self.midas_model = None
            self.midas_transform = None
    
    def _load_sam_model(self):
        """Load SAM (Segment Anything Model) with ViT-H backbone"""
        try:
            if not SAM_AVAILABLE:
                logger.warning("SAM not available - body segmentation will use fallback")
                return
                
            logger.info("Loading SAM ViT-H model...")
            
            # Download SAM checkpoint if not exists
            checkpoint_path = os.path.join(self.model_cache_dir, "sam_vit_h_4b8939.pth")
            
            if not os.path.exists(checkpoint_path):
                logger.info("Downloading SAM checkpoint...")
                import urllib.request
                url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
                urllib.request.urlretrieve(url, checkpoint_path)
                logger.info("SAM checkpoint downloaded")
            
            # Load SAM model
            self.sam_model = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
            self.sam_model.to(device=self.device)
            
            # Create predictor
            self.sam_predictor = SamPredictor(self.sam_model)
            
            logger.info(f"SAM ViT-H model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.warning(f"SAM model loading failed: {e}. Body segmentation will use fallback.")
            self.sam_model = None
            self.sam_predictor = None
    
    def _load_smplx_model(self):
        """Load SMPL-X 3D body model"""
        try:
            if not SMPLX_AVAILABLE:
                logger.warning("SMPL-X not available - 3D body modeling will use fallback")
                return
                
            logger.info("Loading SMPL-X model...")
            
            # Set up SMPL-X model directory
            smplx_model_dir = os.path.join(self.model_cache_dir, "smplx")
            os.makedirs(smplx_model_dir, exist_ok=True)
            
            # Download SMPL-X models if not present
            model_files = ["SMPLX_NEUTRAL.npz", "SMPLX_MALE.npz", "SMPLX_FEMALE.npz"]
            base_url = "https://download.is.tue.mpg.de/download.php?domain=smplx&sfile="
            
            for model_file in model_files:
                model_path = os.path.join(smplx_model_dir, model_file)
                if not os.path.exists(model_path):
                    logger.info(f"SMPL-X model {model_file} not found. Please download from https://smpl-x.is.tue.mpg.de/")
                    break
            else:
                # All models present, create SMPL-X model
                self.smplx_model = smplx.create(
                    model_path=smplx_model_dir,
                    model_type='smplx',
                    gender='neutral',
                    use_face_contour=False,
                    num_betas=10,
                    num_expression_coeffs=10,
                    ext='npz',
                    use_pca=False,
                    flat_hand_mean=True
                )
                
                if torch.cuda.is_available():
                    self.smplx_model = self.smplx_model.to(self.device)
                
                self.smplx_available = True
                logger.info(f"SMPL-X model loaded successfully on {self.device}")
                return
            
            # Fallback if models not found
            logger.warning("SMPL-X models not found. 3D body modeling will use approximations.")
            self.smplx_available = False
            
        except Exception as e:
            logger.warning(f"SMPL-X model loading failed: {e}. 3D body modeling will use fallback.")
            self.smplx_model = None
            self.smplx_available = False
    
    def detect_full_body(self, frame: np.ndarray) -> bool:
        """
        Detect if full body (including face and feet) is visible in frame
        
        Args:
            frame: Input video frame
            
        Returns:
            bool: True if full body detected
        """
        try:
            # Stage 1: YOLO person detection
            person_detected = self._detect_person_yolo(frame)
            if not person_detected:
                return False
            
            # Stage 2: MediaPipe pose detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if not results.pose_landmarks:
                return False
            
            # Check for key landmarks (head, torso, feet)
            landmarks = results.pose_landmarks.landmark
            
            # Key points for full body detection
            head_visible = landmarks[self.mp_pose.PoseLandmark.NOSE].visibility > 0.5
            torso_visible = (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].visibility > 0.5 and
                           landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility > 0.5)
            feet_visible = (landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].visibility > 0.5 and
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].visibility > 0.5)
            
            return head_visible and torso_visible and feet_visible
            
        except Exception as e:
            logger.error(f"Full body detection failed: {e}")
            return False
    
    def _detect_person_yolo(self, frame: np.ndarray) -> bool:
        """Detect person using YOLOv8"""
        try:
            if self.yolo_model is None:
                return True  # Fallback to True if YOLO unavailable
            
            results = self.yolo_model(frame, verbose=False)
            
            # Check for person class (class 0 in COCO dataset)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    classes = boxes.cls.cpu().numpy()
                    confidences = boxes.conf.cpu().numpy()
                    
                    # Look for person detections with good confidence
                    person_detections = (classes == 0) & (confidences > 0.5)
                    if np.any(person_detections):
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"YOLO person detection failed: {e}")
            return True  # Fallback
    
    def run_scan(self, camera, duration: int = 20) -> Dict[str, Any]:
        """
        Run complete body estimation scan for specified duration with optimized frame rate
        
        Args:
            camera: OpenCV camera object
            duration: Scan duration in seconds
            
        Returns:
            Dict containing all body measurements in US units
        """
        try:
            logger.info(f"Starting {duration}-second body estimation scan (optimized)")
            
            # Initialize scan variables
            start_time = time.time()
            frame_count = 0
            processed_frame_count = 0
            scan_results = {
                'height_cm': None,
                'height_ft_in': None,
                'weight_kg': None,
                'weight_lbs': None,
                'bmi': None,
                'body_fat_percentage': None,
                'muscle_mass_kg': None,
                'muscle_mass_lbs': None,
                'bone_density': None,
                'measurements': {},
                'scan_quality': {},
                'confidence_scores': {}
            }
            
            # Clear previous data
            self.pose_landmarks.clear()
            self.height_estimates.clear()
            self.weight_estimates.clear()
            
            best_frame = None
            best_pose_confidence = 0
            
            # Frame processing optimization: Skip frames for heavy processing
            process_every_n_frames = 2  # Process every 2nd frame for heavy operations
            heavy_process_every_n_frames = 5  # Process depth/SAM every 5th frame
            
            while time.time() - start_time < duration:
                ret, frame = camera.read()
                if not ret:
                    continue
                
                frame_count += 1
                current_time = time.time()
                
                # Always show visual feedback for smooth experience
                self._display_scan_progress(frame, duration - (current_time - start_time))
                
                # Process frames selectively for performance
                if frame_count % process_every_n_frames == 0:
                    # Light processing (pose estimation only)
                    frame_results = self._process_frame_pipeline_optimized(
                        frame, 
                        heavy_processing=(frame_count % heavy_process_every_n_frames == 0)
                    )
                    
                    processed_frame_count += 1
                    
                    # Track best frame for final analysis
                    if frame_results and frame_results.get('pose_confidence', 0) > best_pose_confidence:
                        best_frame = frame.copy()
                        best_pose_confidence = frame_results['pose_confidence']
                    
                    # Update scan results
                    if frame_results:
                        self._update_scan_results(scan_results, frame_results)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cv2.destroyAllWindows()
            
            # Final comprehensive analysis using best frame
            if best_frame is not None:
                final_results = self._final_analysis(best_frame, scan_results, processed_frame_count)
            else:
                final_results = self._finalize_measurements(scan_results, processed_frame_count)
            
            # Convert to US units
            final_results = self._convert_to_us_units(final_results)
            
            logger.info(f"Body estimation scan completed successfully - processed {processed_frame_count}/{frame_count} frames")
            return final_results
            
        except Exception as e:
            logger.error(f"Body estimation scan failed: {e}")
            return {'error': str(e)}
    
    def _process_frame_pipeline_optimized(self, frame: np.ndarray, heavy_processing: bool = True) -> Optional[Dict[str, Any]]:
        """
        Optimized frame processing pipeline with selective heavy operations
        
        Args:
            frame: Input video frame
            heavy_processing: Whether to run expensive operations (depth, SAM)
            
        Returns:
            Dict containing frame analysis results
        """
        try:
            start_time = time.time()
            h, w = frame.shape[:2]
            
            # Stage 1: YOLO person detection (lightweight, run every frame)
            person_detected = self._detect_person_yolo(frame)
            if not person_detected:
                return None
            
            # Stage 2: MediaPipe pose estimation (lightweight, run every frame)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(rgb_frame)
            
            if not pose_results.pose_landmarks:
                return None
            
            landmarks = pose_results.pose_landmarks.landmark
            pose_confidence = self._calculate_pose_confidence(landmarks)
            
            if pose_confidence < self.min_pose_confidence:
                return None
            
            # Heavy processing stages (run selectively for performance)
            depth_map = None
            body_mask = None
            smplx_params = None
            
            if heavy_processing:
                # Stage 3: MiDaS depth estimation (expensive)
                depth_map = self._estimate_depth_midas(frame)
                self.depth_map = depth_map
                
                # Stage 4: SAM body segmentation (expensive)
                body_mask = self._segment_body_sam(frame, landmarks, w, h)
                self.body_segmentation = body_mask
                
                # Stage 5: SMPL-X 3D model fitting (expensive)
                smplx_params = self._fit_smplx_model(landmarks, depth_map)
            else:
                # Use cached values for lightweight processing
                depth_map = self.depth_map
                body_mask = self.body_segmentation
                smplx_params = None  # Skip SMPL-X for lightweight frames
            
            # Stage 6: Multi-method measurements (always run)
            height_estimates = self._estimate_height_multi_method(landmarks, depth_map, smplx_params, w, h)
            weight_estimates = self._estimate_weight_multi_method(landmarks, body_mask, smplx_params, height_estimates)
            
            # Update current metrics for real-time display
            self._update_frame_metrics(landmarks, height_estimates, weight_estimates)
            
            # Calculate frame quality (lightweight)
            frame_quality = self._assess_frame_quality_lightweight(landmarks, pose_confidence)
            
            processing_time = time.time() - start_time
            self.processing_times['frame_pipeline'] = processing_time
            
            return {
                'landmarks': landmarks,
                'depth_map': depth_map,
                'body_mask': body_mask,
                'smplx_params': smplx_params,
                'height_estimates': height_estimates,
                'weight_estimates': weight_estimates,
                'pose_confidence': pose_confidence,
                'frame_quality': frame_quality,
                'processing_time': processing_time,
                'heavy_processing': heavy_processing
            }
            
        except Exception as e:
            logger.error(f"Optimized frame processing pipeline failed: {e}")
            return None

    def _process_frame_pipeline(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Stage 1-6: Full frame processing pipeline
        
        Args:
            frame: Input video frame
            
        Returns:
            Dict containing all frame analysis results
        """
        try:
            start_time = time.time()
            h, w = frame.shape[:2]
            
            # Stage 1: YOLO person detection
            person_detected = self._detect_person_yolo(frame)
            if not person_detected:
                return None
            
            # Stage 2: MediaPipe pose estimation
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(rgb_frame)
            
            if not pose_results.pose_landmarks:
                return None
            
            landmarks = pose_results.pose_landmarks.landmark
            pose_confidence = self._calculate_pose_confidence(landmarks)
            
            if pose_confidence < self.min_pose_confidence:
                return None
            
            # Stage 3: MiDaS depth estimation
            depth_map = self._estimate_depth_midas(frame)
            self.depth_map = depth_map  # Store for visualization
            
            # Stage 4: SAM body segmentation
            body_mask = self._segment_body_sam(frame, landmarks, w, h)
            self.body_segmentation = body_mask  # Store for visualization
            
            # Stage 5: SMPL-X 3D model fitting
            smplx_params = self._fit_smplx_model(landmarks, depth_map)
            
            # Stage 6: Multi-method measurements
            height_estimates = self._estimate_height_multi_method(landmarks, depth_map, smplx_params, w, h)
            weight_estimates = self._estimate_weight_multi_method(landmarks, body_mask, smplx_params, height_estimates)
            
            # Update current metrics for real-time display
            self._update_frame_metrics(landmarks, height_estimates, weight_estimates)
            
            # Calculate frame quality
            frame_quality = self._assess_frame_quality(frame, landmarks)
            
            processing_time = time.time() - start_time
            self.processing_times['frame_pipeline'] = processing_time
            
            return {
                'landmarks': landmarks,
                'depth_map': depth_map,
                'body_mask': body_mask,
                'smplx_params': smplx_params,
                'height_estimates': height_estimates,
                'weight_estimates': weight_estimates,
                'pose_confidence': pose_confidence,
                'frame_quality': frame_quality,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Frame processing pipeline failed: {e}")
            return None
    
    def _update_frame_metrics(self, landmarks, height_estimates: Dict[str, float], weight_estimates: Dict[str, float]):
        """Update current metrics from frame processing"""
        try:
            # Calculate shoulder width
            shoulder_width = self._calculate_shoulder_width(landmarks)
            if shoulder_width:
                self.current_metrics['shoulders'] = round(shoulder_width, 1)
            
            # Calculate torso length (rough estimate from shoulders to hips)
            if len(landmarks) > 24:
                left_shoulder = landmarks[11]
                right_shoulder = landmarks[12]
                left_hip = landmarks[23]
                right_hip = landmarks[24]
                
                shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
                hip_center_y = (left_hip.y + right_hip.y) / 2
                
                # Convert to pixels and then to approximate cm (rough conversion)
                torso_length_pixels = abs(hip_center_y - shoulder_center_y) * 480  # Assuming 480p height
                torso_length_cm = torso_length_pixels * 0.5  # Rough pixel to cm conversion
                
                if torso_length_cm > 20 and torso_length_cm < 100:  # Sanity check
                    self.current_metrics['torso'] = round(torso_length_cm, 1)
            
        except Exception as e:
            logger.error(f"Failed to update frame metrics: {e}")
    
    def _calculate_pose_confidence(self, landmarks) -> float:
        """Calculate overall pose detection confidence"""
        try:
            key_landmarks = [
                self.mp_pose.PoseLandmark.NOSE,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_HIP,
                self.mp_pose.PoseLandmark.LEFT_ANKLE,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE
            ]
            
            visibilities = [landmarks[lm].visibility for lm in key_landmarks]
            return np.mean(visibilities)
            
        except Exception:
            return 0.0
    
    def _estimate_depth_midas(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Stage 3: MiDaS depth estimation with proper preprocessing"""
        try:
            if self.midas_model is None or self.midas_transform is None:
                # Fallback depth estimation
                h, w = frame.shape[:2]
                return np.ones((h, w), dtype=np.float32) * 2.0  # 2 meters default
            
            start_time = time.time()
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Transform input
            input_batch = self.midas_transform(rgb_frame).to(self.midas_device)
            
            # Run inference with mixed precision if available
            with torch.no_grad():
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        prediction = self.midas_model(input_batch)
                else:
                    prediction = self.midas_model(input_batch)
            
            # Convert to numpy and normalize
            depth_map = prediction.squeeze().cpu().numpy()
            
            # Normalize depth values to reasonable range (0.5-5 meters)
            depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
            depth_min, depth_max = np.percentile(depth_map, [5, 95])
            depth_normalized = np.clip((depth_map - depth_min) / (depth_max - depth_min), 0, 1)
            depth_meters = 0.5 + depth_normalized * 4.5  # Scale to 0.5-5 meters
            
            self.processing_times['midas'] = time.time() - start_time
            return depth_meters.astype(np.float32)
            
        except Exception as e:
            logger.error(f"MiDaS depth estimation failed: {e}")
            # Fallback depth map
            h, w = frame.shape[:2]
            return np.ones((h, w), dtype=np.float32) * 2.0
    
    def _segment_body_sam(self, frame: np.ndarray, landmarks, width: int, height: int) -> Optional[np.ndarray]:
        """Stage 4: SAM body segmentation using pose landmarks as prompts"""
        try:
            if self.sam_predictor is None:
                # Fallback segmentation using pose landmarks
                return self._create_fallback_mask(landmarks, width, height)
            
            start_time = time.time()
            
            # Set image for SAM predictor
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(rgb_frame)
            
            # Create prompt points from pose landmarks
            input_points, input_labels = self._create_sam_prompts(landmarks, width, height)
            
            if len(input_points) == 0:
                return self._create_fallback_mask(landmarks, width, height)
            
            # Run SAM prediction
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=np.array(input_points),
                point_labels=np.array(input_labels),
                multimask_output=True
            )
            
            # Select best mask based on score
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            
            # Post-process mask
            mask = self._postprocess_sam_mask(best_mask, landmarks, width, height)
            
            self.processing_times['sam'] = time.time() - start_time
            return mask.astype(np.uint8) * 255
            
        except Exception as e:
            logger.error(f"SAM segmentation failed: {e}")
            return self._create_fallback_mask(landmarks, width, height)
    
    def _create_sam_prompts(self, landmarks, width: int, height: int) -> Tuple[List[List[int]], List[int]]:
        """Create prompt points for SAM from pose landmarks"""
        input_points = []
        input_labels = []
        
        # Positive prompts (body parts)
        positive_landmarks = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE
        ]
        
        for landmark in positive_landmarks:
            if landmarks[landmark].visibility > 0.5:
                x = int(landmarks[landmark].x * width)
                y = int(landmarks[landmark].y * height)
                # Ensure points are within frame
                if 0 <= x < width and 0 <= y < height:
                    input_points.append([x, y])
                    input_labels.append(1)  # Positive prompt
        
        # Add negative prompts (background areas)
        # Corners of the image
        margin = 10
        background_points = [
            [margin, margin],  # Top-left
            [width - margin, margin],  # Top-right
            [margin, height - margin],  # Bottom-left
            [width - margin, height - margin]  # Bottom-right
        ]
        
        for point in background_points:
            input_points.append(point)
            input_labels.append(0)  # Negative prompt
        
        return input_points, input_labels
    
    def _postprocess_sam_mask(self, mask: np.ndarray, landmarks, width: int, height: int) -> np.ndarray:
        """Post-process SAM mask to improve body segmentation"""
        try:
            # Fill holes
            mask_filled = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, 
                                         np.ones((10, 10), np.uint8))
            
            # Remove small disconnected regions
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_filled)
            
            if num_labels > 1:
                # Find largest component (excluding background)
                largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                mask_cleaned = (labels == largest_component).astype(np.uint8)
            else:
                mask_cleaned = mask_filled
            
            # Smooth edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_smooth = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
            
            return mask_smooth
            
        except Exception as e:
            logger.error(f"Mask post-processing failed: {e}")
            return mask.astype(np.uint8)
    
    def _create_fallback_mask(self, landmarks, width: int, height: int) -> np.ndarray:
        """Create fallback body mask using pose landmarks"""
        try:
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Create body outline from landmarks
            body_points = []
            
            # Get visible landmarks for body outline
            outline_landmarks = [
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_HIP,
                self.mp_pose.PoseLandmark.LEFT_HIP
            ]
            
            for lm in outline_landmarks:
                if landmarks[lm].visibility > 0.5:
                    x = int(landmarks[lm].x * width)
                    y = int(landmarks[lm].y * height)
                    body_points.append([x, y])
            
            # Add head and feet if visible
            if landmarks[self.mp_pose.PoseLandmark.NOSE].visibility > 0.5:
                head_x = int(landmarks[self.mp_pose.PoseLandmark.NOSE].x * width)
                head_y = int(landmarks[self.mp_pose.PoseLandmark.NOSE].y * height)
                body_points.insert(0, [head_x, head_y - 30])  # Extend above head
            
            # Add feet
            feet_points = []
            for ankle in [self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_ANKLE]:
                if landmarks[ankle].visibility > 0.5:
                    x = int(landmarks[ankle].x * width)
                    y = int(landmarks[ankle].y * height)
                    feet_points.append([x, y + 20])  # Extend below feet
            
            body_points.extend(feet_points)
            
            if len(body_points) >= 3:
                # Create convex hull for better body shape
                hull = cv2.convexHull(np.array(body_points))
                cv2.fillPoly(mask, [hull], 255)
            
            return mask
            
        except Exception as e:
            logger.error(f"Fallback mask creation failed: {e}")
            return np.zeros((height, width), dtype=np.uint8)
    
    def _fit_smplx_model(self, landmarks, depth_map) -> Optional[Dict[str, Any]]:
        """Stage 5: SMPL-X 3D body model fitting using MediaPipe landmarks"""
        try:
            if not self.smplx_available or self.smplx_model is None:
                # Fallback SMPL-X parameters based on landmarks
                return self._estimate_smplx_fallback(landmarks, depth_map)
            
            start_time = time.time()
            
            # Convert MediaPipe landmarks to SMPL-X joint format
            smplx_joints = self._mediapipe_to_smplx_joints(landmarks)
            
            if smplx_joints is None:
                return self._estimate_smplx_fallback(landmarks, depth_map)
            
            # Initialize SMPL-X parameters
            batch_size = 1
            device = self.device
            
            # Initialize pose parameters (63 for body pose, 3 for global orientation)
            body_pose = torch.zeros((batch_size, 63), device=device, requires_grad=True)
            global_orient = torch.zeros((batch_size, 3), device=device, requires_grad=True)
            betas = torch.zeros((batch_size, 10), device=device, requires_grad=True)
            
            # Estimate initial translation from depth
            translation = self._estimate_translation_from_depth(landmarks, depth_map)
            transl = torch.tensor(translation, device=device, requires_grad=True).unsqueeze(0)
            
            # Optimization parameters
            optimizer = torch.optim.Adam([body_pose, global_orient, betas, transl], lr=0.02)
            
            # Joint weights for optimization
            joint_weights = self._get_joint_weights()
            
            # Optimization loop (simplified)
            for iteration in range(50):  # Reduced iterations for real-time performance
                optimizer.zero_grad()
                
                # Forward pass through SMPL-X
                smplx_output = self.smplx_model(
                    body_pose=body_pose,
                    global_orient=global_orient,
                    betas=betas,
                    transl=transl
                )
                
                # Get 3D joints
                joints_3d = smplx_output.joints[:, :25, :]  # First 25 joints
                
                # Project to 2D (simplified camera model)
                joints_2d_pred = self._project_joints_to_2d(joints_3d, translation[2])
                
                # Calculate loss
                loss = self._calculate_joint_loss(joints_2d_pred, smplx_joints, joint_weights)
                
                if loss.item() < 0.01:  # Early stopping
                    break
                
                # Backward pass
                loss.backward()
                optimizer.step()
            
            # Extract final parameters
            result = {
                'body_pose': body_pose.detach().cpu().numpy().flatten(),
                'global_orient': global_orient.detach().cpu().numpy().flatten(),
                'betas': betas.detach().cpu().numpy().flatten(),
                'transl': transl.detach().cpu().numpy().flatten(),
                'vertices': smplx_output.vertices.detach().cpu().numpy(),
                'joints': smplx_output.joints.detach().cpu().numpy(),
                'faces': self.smplx_model.faces,
                'optimization_loss': loss.item()
            }
            
            self.processing_times['smplx'] = time.time() - start_time
            return result
            
        except Exception as e:
            logger.error(f"SMPL-X fitting failed: {e}")
            return self._estimate_smplx_fallback(landmarks, depth_map)
    
    def _mediapipe_to_smplx_joints(self, landmarks) -> Optional[np.ndarray]:
        """Convert MediaPipe landmarks to SMPL-X joint format"""
        try:
            # Mapping from MediaPipe to SMPL-X joints (simplified)
            mp_to_smplx = {
                0: self.mp_pose.PoseLandmark.NOSE,  # Head
                1: self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                2: self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                3: self.mp_pose.PoseLandmark.LEFT_ELBOW,
                4: self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                5: self.mp_pose.PoseLandmark.LEFT_WRIST,
                6: self.mp_pose.PoseLandmark.RIGHT_WRIST,
                7: self.mp_pose.PoseLandmark.LEFT_HIP,
                8: self.mp_pose.PoseLandmark.RIGHT_HIP,
                9: self.mp_pose.PoseLandmark.LEFT_KNEE,
                10: self.mp_pose.PoseLandmark.RIGHT_KNEE,
                11: self.mp_pose.PoseLandmark.LEFT_ANKLE,
                12: self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            }
            
            joints_2d = np.zeros((len(mp_to_smplx), 2))
            joint_confidence = np.zeros(len(mp_to_smplx))
            
            for smplx_idx, mp_landmark in mp_to_smplx.items():
                landmark = landmarks[mp_landmark]
                joints_2d[smplx_idx] = [landmark.x, landmark.y]
                joint_confidence[smplx_idx] = landmark.visibility
            
            # Filter out low-confidence joints
            valid_joints = joint_confidence > 0.5
            if np.sum(valid_joints) < 8:  # Need at least 8 good joints
                return None
            
            return joints_2d
            
        except Exception as e:
            logger.error(f"MediaPipe to SMPL-X conversion failed: {e}")
            return None
    
    def _estimate_translation_from_depth(self, landmarks, depth_map) -> np.ndarray:
        """Estimate 3D translation using depth information"""
        try:
            if depth_map is None:
                return np.array([0.0, 0.0, 2.0])  # Default 2m distance
            
            # Get torso center for depth estimation
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                # Center between shoulders
                center_x = (left_shoulder.x + right_shoulder.x) / 2
                center_y = (left_shoulder.y + right_shoulder.y) / 2
                
                # Convert to pixel coordinates
                h, w = depth_map.shape
                px_x = int(center_x * w)
                px_y = int(center_y * h)
                
                # Clamp to image bounds
                px_x = max(0, min(w-1, px_x))
                px_y = max(0, min(h-1, px_y))
                
                # Get depth value
                depth = depth_map[px_y, px_x]
                
                return np.array([0.0, 0.0, float(depth)])
            
            # Fallback to median depth
            median_depth = np.median(depth_map)
            return np.array([0.0, 0.0, float(median_depth)])
            
        except Exception as e:
            logger.error(f"Translation estimation failed: {e}")
            return np.array([0.0, 0.0, 2.0])
    
    def _get_joint_weights(self) -> torch.Tensor:
        """Get joint weights for optimization"""
        # Higher weights for more reliable joints
        weights = torch.ones(13)  # 13 main joints
        weights[0] = 0.8   # Head (less reliable)
        weights[1:3] = 1.5  # Shoulders (very reliable)
        weights[7:9] = 1.5  # Hips (very reliable)
        weights[9:11] = 1.2  # Knees (fairly reliable)
        weights[11:13] = 1.0  # Ankles (moderately reliable)
        
        return weights.to(self.device)
    
    def _project_joints_to_2d(self, joints_3d: torch.Tensor, depth: float) -> torch.Tensor:
        """Project 3D joints to 2D using simple camera model"""
        # Simplified orthographic projection
        focal_length = 500.0  # Approximate focal length in pixels
        
        # Project to 2D
        joints_2d = joints_3d[:, :, :2] / (joints_3d[:, :, 2:3] + 1e-8) * focal_length
        
        # Normalize to [0, 1] range (assuming image center at origin)
        joints_2d = (joints_2d + 320) / 640  # Assuming 640x480 image
        
        return joints_2d
    
    def _calculate_joint_loss(self, pred_joints: torch.Tensor, target_joints: np.ndarray, 
                             weights: torch.Tensor) -> torch.Tensor:
        """Calculate joint reprojection loss"""
        target_tensor = torch.tensor(target_joints, device=self.device, dtype=torch.float32)
        
        # L2 loss between predicted and target 2D joints
        diff = pred_joints.squeeze() - target_tensor
        weighted_loss = weights.unsqueeze(1) * (diff ** 2).sum(dim=1)
        
        return weighted_loss.mean()
    
    def _estimate_smplx_fallback(self, landmarks, depth_map) -> Dict[str, Any]:
        """Fallback SMPL-X parameters when full fitting is not available"""
        try:
            # Estimate basic shape parameters from landmarks
            betas = np.zeros(10)
            
            # Estimate height factor from pose
            if (landmarks[self.mp_pose.PoseLandmark.NOSE].visibility > 0.5 and
                landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].visibility > 0.5):
                
                head_y = landmarks[self.mp_pose.PoseLandmark.NOSE].y
                ankle_y = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y
                height_ratio = abs(ankle_y - head_y)
                
                # First beta controls height
                betas[0] = (height_ratio - 0.7) * 2.0  # Rough height adjustment
            
            # Estimate body width from shoulders/hips
            shoulder_width = self._calculate_shoulder_width(landmarks)
            if shoulder_width:
                betas[1] = (shoulder_width - 0.25) * 3.0  # Body width adjustment
            
            # Translation from depth
            translation = self._estimate_translation_from_depth(landmarks, depth_map)
            
            return {
                'body_pose': np.zeros(63),  # Neutral pose
                'global_orient': np.zeros(3),  # No rotation
                'betas': betas,  # Estimated shape
                'transl': translation,
                'vertices': None,  # Not available in fallback
                'joints': None,  # Not available in fallback
                'faces': None,  # Not available in fallback
                'optimization_loss': 0.0  # No optimization performed
            }
            
        except Exception as e:
            logger.error(f"SMPL-X fallback estimation failed: {e}")
            return {
                'body_pose': np.zeros(63),
                'global_orient': np.zeros(3),
                'betas': np.zeros(10),
                'transl': np.array([0, 0, 2.0]),
                'vertices': None,
                'joints': None,
                'faces': None,
                'optimization_loss': 0.0
            }
    
    def _estimate_height_multi_method(self, landmarks, depth_map, smplx_params, width: int, height: int) -> Dict[str, float]:
        """
        Stage 6: Height estimation using multiple methods with weighted combination
        - Landmark-based measurement (30% weight)
        - Depth-corrected scaling (50% weight)  
        - SMPL-X validation (20% weight)
        """
        try:
            height_estimates = {}
            
            # Method 1: Landmark-based measurement (30% weight)
            landmark_height = self._estimate_height_landmarks(landmarks, width, height)
            if landmark_height:
                height_estimates['landmark'] = landmark_height
            
            # Method 2: Depth-corrected scaling (50% weight)
            depth_height = self._estimate_height_depth_corrected(landmarks, depth_map, width, height)
            if depth_height:
                height_estimates['depth_corrected'] = depth_height
            
            # Method 3: SMPL-X validation (20% weight)
            smplx_height = self._estimate_height_smplx(smplx_params)
            if smplx_height:
                height_estimates['smplx'] = smplx_height
            
            return height_estimates
            
        except Exception as e:
            logger.error(f"Height estimation failed: {e}")
            return {}
    
    def _estimate_height_landmarks(self, landmarks, width: int, height: int) -> Optional[float]:
        """Estimate height using pose landmarks"""
        try:
            # Get head and ankle positions
            head_y = landmarks[self.mp_pose.PoseLandmark.NOSE].y * height
            left_ankle_y = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y * height
            right_ankle_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y * height
            
            # Check visibility
            head_vis = landmarks[self.mp_pose.PoseLandmark.NOSE].visibility
            left_ankle_vis = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].visibility
            right_ankle_vis = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].visibility
            
            if head_vis < 0.5 or (left_ankle_vis < 0.5 and right_ankle_vis < 0.5):
                return None
            
            # Use best visible ankle
            ankle_y = left_ankle_y if left_ankle_vis > right_ankle_vis else right_ankle_y
            
            # Calculate pixel height
            pixel_height = abs(ankle_y - head_y)
            
            # Convert to real height (assuming person occupies 60-80% of frame height)
            # This is a rough calibration - in practice would use camera calibration
            frame_height_ratio = pixel_height / height
            estimated_height_cm = 170 * (frame_height_ratio / 0.7)  # 170cm baseline, 70% frame occupation
            
            # Clamp to reasonable range
            return max(140, min(220, estimated_height_cm))
            
        except Exception as e:
            logger.error(f"Landmark height estimation failed: {e}")
            return None
    
    def _estimate_height_depth_corrected(self, landmarks, depth_map, width: int, height: int) -> Optional[float]:
        """Estimate height using depth-corrected scaling"""
        try:
            if depth_map is None:
                return None
            
            # Get head and foot positions
            head_x = int(landmarks[self.mp_pose.PoseLandmark.NOSE].x * width)
            head_y = int(landmarks[self.mp_pose.PoseLandmark.NOSE].y * height)
            
            left_ankle_x = int(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].x * width)
            left_ankle_y = int(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y * height)
            
            # Get depth values
            head_depth = depth_map[head_y, head_x] if 0 <= head_y < depth_map.shape[0] and 0 <= head_x < depth_map.shape[1] else 2.0
            ankle_depth = depth_map[left_ankle_y, left_ankle_x] if 0 <= left_ankle_y < depth_map.shape[0] and 0 <= left_ankle_x < depth_map.shape[1] else 2.0
            
            # Calculate 3D distance
            pixel_distance = math.sqrt((head_x - left_ankle_x)**2 + (head_y - left_ankle_y)**2)
            avg_depth = (head_depth + ankle_depth) / 2
            
            # Convert pixel distance to real-world distance using depth
            # Assuming standard camera focal length of ~500 pixels
            focal_length = 500
            real_height_m = (pixel_distance * avg_depth) / focal_length
            real_height_cm = real_height_m * 100
            
            # Clamp to reasonable range
            return max(140, min(220, real_height_cm))
            
        except Exception as e:
            logger.error(f"Depth-corrected height estimation failed: {e}")
            return None
    
    def _estimate_height_smplx(self, smplx_params) -> Optional[float]:
        """Estimate height using SMPL-X model"""
        try:
            if smplx_params is None:
                return None
            
            # In production, extract height from SMPL-X mesh
            # vertices = smplx_output.vertices
            # height = vertices[:, 1].max() - vertices[:, 1].min()  # Y-axis range
            # return height * 100  # Convert to cm
            
            # Placeholder calculation
            betas = smplx_params.get('betas', np.zeros(10))
            base_height = 170  # Average height in cm
            height_variation = betas[0] * 10  # First beta parameter affects height
            estimated_height = base_height + height_variation
            
            return max(140, min(220, estimated_height))
            
        except Exception as e:
            logger.error(f"SMPL-X height estimation failed: {e}")
            return None
    
    def _estimate_weight_multi_method(self, landmarks, body_mask, smplx_params, height_estimates) -> Dict[str, float]:
        """
        Stage 7: Weight estimation using multiple methods
        - Volume-based calculation from SMPL-X
        - Anthropometric formulas
        - Demographic adjustments
        """
        try:
            weight_estimates = {}
            
            # Method 1: Volume-based calculation
            volume_weight = self._estimate_weight_volume(smplx_params)
            if volume_weight:
                weight_estimates['volume'] = volume_weight
            
            # Method 2: Anthropometric formulas
            anthro_weight = self._estimate_weight_anthropometric(landmarks, height_estimates)
            if anthro_weight:
                weight_estimates['anthropometric'] = anthro_weight
            
            # Method 3: Body area estimation
            area_weight = self._estimate_weight_body_area(body_mask, height_estimates)
            if area_weight:
                weight_estimates['body_area'] = area_weight
            
            return weight_estimates
            
        except Exception as e:
            logger.error(f"Weight estimation failed: {e}")
            return {}
    
    def _estimate_weight_volume(self, smplx_params) -> Optional[float]:
        """Estimate weight using body volume from SMPL-X mesh"""
        try:
            if smplx_params is None:
                return None
            
            # Check if we have actual mesh data
            vertices = smplx_params.get('vertices')
            faces = smplx_params.get('faces')
            
            if vertices is not None and faces is not None and SMPLX_AVAILABLE:
                try:
                    # Calculate actual volume from SMPL-X mesh
                    mesh = trimesh.Trimesh(vertices=vertices.squeeze(), faces=faces)
                    
                    # Ensure mesh is watertight
                    if not mesh.is_watertight:
                        mesh.fill_holes()
                    
                    # Get volume in cubic meters
                    volume_m3 = mesh.volume
                    
                    # Human body density (kg/m³)
                    # Average density considering bone (~1900), muscle (~1060), fat (~920)
                    body_density = 985  # kg/m³
                    
                    # Calculate weight
                    weight_kg = volume_m3 * body_density
                    
                    # Sanity check and clamp to reasonable range
                    if 30 <= weight_kg <= 200:
                        return weight_kg
                    else:
                        logger.warning(f"Volume-based weight {weight_kg:.1f}kg outside reasonable range")
                        
                except Exception as mesh_error:
                    logger.warning(f"Mesh volume calculation failed: {mesh_error}")
            
            # Fallback calculation using shape parameters
            betas = smplx_params.get('betas', np.zeros(10))
            base_weight = 70  # Average weight in kg
            
            # Beta parameters influence weight
            # Beta[0] generally affects height/overall size
            # Beta[1] generally affects weight/bulk
            height_factor = betas[0] * 8 if len(betas) > 0 else 0
            weight_factor = betas[1] * 15 if len(betas) > 1 else 0
            
            # Additional factors from other betas
            if len(betas) > 2:
                additional_factor = np.sum(betas[2:6]) * 2  # Other shape variations
            else:
                additional_factor = 0
            
            estimated_weight = base_weight + height_factor + weight_factor + additional_factor
            
            return max(40, min(150, estimated_weight))
            
        except Exception as e:
            logger.error(f"Volume weight estimation failed: {e}")
            return None
    
    def _estimate_weight_anthropometric(self, landmarks, height_estimates) -> Optional[float]:
        """Estimate weight using anthropometric formulas"""
        try:
            # Get height estimate
            if not height_estimates:
                return None
            
            height_cm = list(height_estimates.values())[0]  # Use first available estimate
            height_m = height_cm / 100
            
            # Get body measurements from landmarks
            shoulder_width = self._calculate_shoulder_width(landmarks)
            hip_width = self._calculate_hip_width(landmarks)
            
            # Simplified anthropometric formula (Robinson formula variant)
            if shoulder_width and hip_width:
                # Estimate frame size
                frame_ratio = (shoulder_width + hip_width) / 2
                
                # Base weight from height
                base_weight = 52 + 1.9 * (height_cm - 152.4) / 2.54  # Robinson formula (kg)
                
                # Adjust for frame size
                frame_adjustment = (frame_ratio - 0.5) * 20  # Simplified adjustment
                estimated_weight = base_weight + frame_adjustment
                
                return max(40, min(150, estimated_weight))
            
            return None
            
        except Exception as e:
            logger.error(f"Anthropometric weight estimation failed: {e}")
            return None
    
    def _estimate_weight_body_area(self, body_mask, height_estimates) -> Optional[float]:
        """Estimate weight using body area analysis"""
        try:
            if body_mask is None or not height_estimates:
                return None
            
            # Calculate body area in pixels
            body_area_pixels = np.sum(body_mask > 0)
            total_area_pixels = body_mask.shape[0] * body_mask.shape[1]
            body_area_ratio = body_area_pixels / total_area_pixels
            
            # Get height estimate
            height_cm = list(height_estimates.values())[0]
            
            # Simplified body area to weight correlation
            # This is a very rough approximation
            baseline_weight = height_cm - 100  # Simple height-weight correlation
            area_adjustment = (body_area_ratio - 0.2) * 50  # Adjust based on visible body area
            
            estimated_weight = baseline_weight + area_adjustment
            return max(40, min(150, estimated_weight))
            
        except Exception as e:
            logger.error(f"Body area weight estimation failed: {e}")
            return None
    
    def _calculate_shoulder_width(self, landmarks) -> Optional[float]:
        """Calculate shoulder width from landmarks"""
        try:
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            if left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5:
                return None
            
            # Calculate normalized distance
            dx = abs(left_shoulder.x - right_shoulder.x)
            return dx
            
        except Exception:
            return None
    
    def _calculate_hip_width(self, landmarks) -> Optional[float]:
        """Calculate hip width from landmarks"""
        try:
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            
            if left_hip.visibility < 0.5 or right_hip.visibility < 0.5:
                return None
            
            # Calculate normalized distance
            dx = abs(left_hip.x - right_hip.x)
            return dx
            
        except Exception:
            return None
    
    def _assess_frame_quality_lightweight(self, landmarks, pose_confidence: float) -> Dict[str, float]:
        """Lightweight frame quality assessment for optimized performance"""
        try:
            quality_metrics = {}
            
            # Pose visibility score (already computed)
            quality_metrics['pose_visibility'] = pose_confidence
            
            # Landmark stability (simplified)
            key_landmarks = [0, 11, 12, 23, 24, 27, 28]  # Key body points
            stability_score = 0
            valid_landmarks = 0
            
            for lm_idx in key_landmarks:
                if len(landmarks) > lm_idx:
                    if landmarks[lm_idx].visibility > 0.5:
                        stability_score += landmarks[lm_idx].visibility
                        valid_landmarks += 1
            
            quality_metrics['stability'] = stability_score / max(1, valid_landmarks)
            quality_metrics['overall'] = (quality_metrics['pose_visibility'] + quality_metrics['stability']) / 2
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Lightweight frame quality assessment failed: {e}")
            return {'overall': 0.5}

    def _assess_frame_quality(self, frame: np.ndarray, landmarks) -> Dict[str, float]:
        """Assess frame quality for measurements"""
        try:
            # Calculate various quality metrics
            quality_metrics = {}
            
            # Pose visibility score
            visibilities = [lm.visibility for lm in landmarks]
            quality_metrics['pose_visibility'] = np.mean(visibilities)
            
            # Image sharpness (Laplacian variance)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            quality_metrics['sharpness'] = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Lighting assessment (histogram analysis)
            brightness = np.mean(gray)
            quality_metrics['brightness'] = min(1.0, brightness / 128.0)
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Frame quality assessment failed: {e}")
            return {}
    
    def _update_scan_results(self, scan_results: Dict[str, Any], frame_results: Dict[str, Any]):
        """Update scan results with frame analysis"""
        try:
            # Accumulate height estimates
            if frame_results.get('height_estimates'):
                for method, height in frame_results['height_estimates'].items():
                    if method not in self.height_estimates:
                        self.height_estimates[method] = []
                    self.height_estimates[method].append(height)
            
            # Accumulate weight estimates
            if frame_results.get('weight_estimates'):
                for method, weight in frame_results['weight_estimates'].items():
                    if method not in self.weight_estimates:
                        self.weight_estimates[method] = []
                    self.weight_estimates[method].append(weight)
                    
        except Exception as e:
            logger.error(f"Failed to update scan results: {e}")
    
    def _display_scan_progress(self, frame: np.ndarray, time_remaining: float):
        """Display comprehensive real-time visual feedback"""
        try:
            h, w = frame.shape[:2]
            
            # Update animation variables
            self.pulse_phase = (self.pulse_phase + 0.15) % (2 * math.pi)
            self.scan_position = (self.scan_position + 2) % h
            self.wireframe_rotation = (self.wireframe_rotation + 1) % 360
            
            # Create overlay for transparency effects
            overlay = frame.copy()
            
            # Process body detection and pose
            yolo_result = self._detect_person_yolo(frame)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(rgb_frame)
            
            # Update current metrics
            self._update_current_metrics(yolo_result, pose_results)
            
            # Main display components
            self._draw_person_detection_indicator(overlay, yolo_result, w, h)
            self._draw_yolo_bounding_box(overlay, frame, w, h)
            self._draw_pose_skeleton(overlay, pose_results, w, h)
            self._draw_body_metrics_panel(overlay, time_remaining, w, h)
            self._draw_depth_visualization(overlay, w, h)
            self._draw_scan_stages_progress(overlay, time_remaining, w, h)
            self._draw_scan_line_effect(overlay, w, h)
            self._draw_body_instructions(overlay, pose_results, yolo_result, w, h)
            self._draw_3d_wireframe(overlay, pose_results, w, h)
            
            # Apply overlay with transparency
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            
            # Draw title
            self._draw_glowing_text(frame, "BIO-SCAN-BEST BODY ANALYSIS", 
                                  (w//2, 30), self.colors['primary'], size=1.2)
            
            cv2.imshow('BIO-SCAN-BEST - Body Analysis', frame)
            
        except Exception as e:
            logger.error(f"Display progress failed: {e}")
            cv2.imshow('BIO-SCAN-BEST - Body Analysis', frame)
    
    def _final_analysis(self, best_frame: np.ndarray, scan_results: Dict[str, Any], frame_count: int) -> Dict[str, Any]:
        """Perform final comprehensive analysis using best frame"""
        try:
            logger.info("Performing final comprehensive analysis...")
            
            # Re-process best frame with maximum precision
            final_frame_results = self._process_frame_pipeline(best_frame)
            
            if final_frame_results:
                # Update with best frame results
                self._update_scan_results(scan_results, final_frame_results)
            
            # Finalize measurements
            return self._finalize_measurements(scan_results, frame_count)
            
        except Exception as e:
            logger.error(f"Final analysis failed: {e}")
            return self._finalize_measurements(scan_results, frame_count)
    
    def _finalize_measurements(self, scan_results: Dict[str, Any], frame_count: int) -> Dict[str, Any]:
        """Stage 8: Result validation and final calculations"""
        try:
            # Calculate weighted average height
            final_height = self._calculate_weighted_height()
            scan_results['height_cm'] = final_height
            
            # Calculate weighted average weight
            final_weight = self._calculate_weighted_weight()
            scan_results['weight_kg'] = final_weight
            
            # Calculate BMI
            if final_height and final_weight:
                height_m = final_height / 100
                bmi = final_weight / (height_m ** 2)
                scan_results['bmi'] = round(bmi, 1)
            
            # Estimate body composition
            body_composition = self._estimate_body_composition(final_height, final_weight)
            scan_results.update(body_composition)
            
            # Additional measurements
            additional_measurements = self._calculate_additional_measurements(final_height)
            scan_results['measurements'].update(additional_measurements)
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores()
            scan_results['confidence_scores'] = confidence_scores
            
            # Scan quality metrics
            scan_results['scan_quality'] = {
                'frames_processed': frame_count,
                'height_measurements': sum(len(heights) for heights in self.height_estimates.values()),
                'weight_measurements': sum(len(weights) for weights in self.weight_estimates.values())
            }
            
            return scan_results
            
        except Exception as e:
            logger.error(f"Failed to finalize measurements: {e}")
            return scan_results
    
    def _calculate_weighted_height(self) -> Optional[float]:
        """Calculate final height using weighted combination"""
        try:
            if not self.height_estimates:
                return None
            
            # Weights for different methods
            weights = {
                'landmark': 0.3,
                'depth_corrected': 0.5,
                'smplx': 0.2
            }
            
            weighted_sum = 0
            total_weight = 0
            
            for method, heights in self.height_estimates.items():
                if heights and method in weights:
                    avg_height = np.mean(heights)
                    method_weight = weights[method]
                    weighted_sum += avg_height * method_weight
                    total_weight += method_weight
            
            if total_weight > 0:
                return weighted_sum / total_weight
            
            return None
            
        except Exception as e:
            logger.error(f"Height calculation failed: {e}")
            return None
    
    def _calculate_weighted_weight(self) -> Optional[float]:
        """Calculate final weight using ensemble methods"""
        try:
            if not self.weight_estimates:
                return None
            
            # Equal weights for now - could be optimized
            weights = {
                'volume': 0.4,
                'anthropometric': 0.4,
                'body_area': 0.2
            }
            
            weighted_sum = 0
            total_weight = 0
            
            for method, weights_list in self.weight_estimates.items():
                if weights_list and method in weights:
                    avg_weight = np.mean(weights_list)
                    method_weight = weights[method]
                    weighted_sum += avg_weight * method_weight
                    total_weight += method_weight
            
            if total_weight > 0:
                return weighted_sum / total_weight
            
            return None
            
        except Exception as e:
            logger.error(f"Weight calculation failed: {e}")
            return None
    
    def _estimate_body_composition(self, height_cm: Optional[float], weight_kg: Optional[float]) -> Dict[str, Optional[float]]:
        """Estimate body fat percentage, muscle mass, and bone density using advanced methods"""
        try:
            if not height_cm or not weight_kg:
                return {
                    'body_fat_percentage': None,
                    'muscle_mass_kg': None,
                    'bone_density': None,
                    'waist_to_hip_ratio': None,
                    'body_frame_size': None,
                    'metabolic_age': None
                }
            
            height_m = height_cm / 100
            bmi = weight_kg / (height_m ** 2)
            
            # Enhanced body fat estimation using multiple methods
            body_fat_methods = []
            
            # Method 1: Navy Method (requires waist and neck measurements - estimated from landmarks)
            navy_bf = self._estimate_body_fat_navy_method(height_cm, weight_kg)
            if navy_bf:
                body_fat_methods.append(navy_bf)
            
            # Method 2: Improved BMI-based estimation
            bmi_bf = self._estimate_body_fat_from_bmi(bmi)
            body_fat_methods.append(bmi_bf)
            
            # Method 3: SMPL-X derived estimation
            smplx_bf = self._estimate_body_fat_from_shape()
            if smplx_bf:
                body_fat_methods.append(smplx_bf)
            
            # Average the methods
            if body_fat_methods:
                body_fat_pct = np.mean(body_fat_methods)
            else:
                body_fat_pct = 15.0  # Default
            
            body_fat_pct = max(5, min(50, body_fat_pct))
            
            # Enhanced muscle mass estimation
            muscle_mass_kg = self._estimate_muscle_mass_advanced(height_cm, weight_kg, body_fat_pct)
            
            # Enhanced bone density estimation
            bone_density = self._estimate_bone_density_advanced(height_cm, weight_kg, body_fat_pct)
            
            # Additional measurements
            waist_to_hip_ratio = self._calculate_waist_to_hip_ratio()
            body_frame_size = self._estimate_body_frame_size(height_cm)
            metabolic_age = self._estimate_metabolic_age(body_fat_pct, muscle_mass_kg, weight_kg)
            
            return {
                'body_fat_percentage': round(body_fat_pct, 1),
                'muscle_mass_kg': round(muscle_mass_kg, 1),
                'bone_density': round(bone_density, 2),
                'waist_to_hip_ratio': round(waist_to_hip_ratio, 2) if waist_to_hip_ratio else None,
                'body_frame_size': body_frame_size,
                'metabolic_age': round(metabolic_age) if metabolic_age else None
            }
            
        except Exception as e:
            logger.error(f"Body composition estimation failed: {e}")
            return {
                'body_fat_percentage': None,
                'muscle_mass_kg': None,
                'bone_density': None,
                'waist_to_hip_ratio': None,
                'body_frame_size': None,
                'metabolic_age': None
            }
    
    def _estimate_body_fat_navy_method(self, height_cm: float, weight_kg: float) -> Optional[float]:
        """Estimate body fat using Navy Method (requires waist/neck measurements)"""
        try:
            # This would require actual waist and neck measurements
            # For now, estimate based on typical body proportions
            
            # Estimate waist circumference from BMI and weight
            bmi = weight_kg / ((height_cm / 100) ** 2)
            estimated_waist_cm = 70 + (bmi - 22) * 2.5  # Rough estimation
            estimated_waist_cm = max(60, min(120, estimated_waist_cm))
            
            # Estimate neck circumference (typically 2x wrist circumference)
            estimated_neck_cm = 35 + (weight_kg - 70) * 0.1  # Rough estimation
            estimated_neck_cm = max(30, min(50, estimated_neck_cm))
            
            # Navy Method formula (simplified for male - would need gender detection)
            # BF% = 86.010 * log10(waist - neck) - 70.041 * log10(height) + 36.76
            log_waist_neck = math.log10(max(1, estimated_waist_cm - estimated_neck_cm))
            log_height = math.log10(height_cm)
            
            body_fat_pct = 86.010 * log_waist_neck - 70.041 * log_height + 36.76
            
            return max(5, min(50, body_fat_pct))
            
        except Exception as e:
            logger.error(f"Navy method estimation failed: {e}")
            return None
    
    def _estimate_body_fat_from_bmi(self, bmi: float) -> float:
        """Improved BMI-based body fat estimation"""
        # Enhanced formula based on research
        if bmi < 18.5:
            return 8 + (18.5 - bmi) * 1.2
        elif bmi < 25:
            return 8 + (bmi - 18.5) * 2.3
        elif bmi < 30:
            return 23 + (bmi - 25) * 3.2
        else:
            return 39 + (bmi - 30) * 1.8
    
    def _estimate_body_fat_from_shape(self) -> Optional[float]:
        """Estimate body fat from shape parameters if available"""
        try:
            # This would use the latest SMPL-X fit if available
            # For now, return None as placeholder
            return None
        except Exception:
            return None
    
    def _estimate_muscle_mass_advanced(self, height_cm: float, weight_kg: float, body_fat_pct: float) -> float:
        """Advanced muscle mass estimation using lean body mass formulas"""
        try:
            # Calculate fat-free mass
            fat_mass_kg = weight_kg * (body_fat_pct / 100)
            lean_body_mass = weight_kg - fat_mass_kg
            
            # Muscle mass is approximately 75-85% of lean body mass
            # The rest is organs, bones, water, etc.
            muscle_percentage = 0.8  # 80% of lean mass is muscle
            
            # Adjust based on height (taller people tend to have slightly higher muscle %)
            height_factor = 1 + (height_cm - 170) * 0.001
            muscle_percentage *= height_factor
            
            muscle_mass = lean_body_mass * muscle_percentage
            
            return max(15, min(80, muscle_mass))
            
        except Exception as e:
            logger.error(f"Advanced muscle mass estimation failed: {e}")
            return weight_kg * 0.4  # Fallback
    
    def _estimate_bone_density_advanced(self, height_cm: float, weight_kg: float, body_fat_pct: float) -> float:
        """Advanced bone density estimation"""
        try:
            # Base bone density adjusted for height, weight, and body composition
            base_density = 1.2
            
            # Height factor (taller people generally have lower bone density)
            height_factor = (170 - height_cm) * 0.002
            
            # Weight factor (heavier people generally have higher bone density)
            weight_factor = (weight_kg - 70) * 0.003
            
            # Body fat factor (higher muscle mass typically correlates with higher bone density)
            muscle_factor = (30 - body_fat_pct) * 0.005
            
            bone_density = base_density + height_factor + weight_factor + muscle_factor
            
            return max(0.8, min(1.6, bone_density))
            
        except Exception as e:
            logger.error(f"Advanced bone density estimation failed: {e}")
            return 1.2  # Average
    
    def _calculate_waist_to_hip_ratio(self) -> Optional[float]:
        """Calculate waist-to-hip ratio from landmark measurements"""
        try:
            if not hasattr(self, 'pose_landmarks') or not self.pose_landmarks:
                return None
            
            # Get the latest landmarks
            latest_landmarks = self.pose_landmarks[-1] if self.pose_landmarks else None
            if not latest_landmarks:
                return None
            
            # Estimate waist and hip measurements from landmarks
            waist_width = self._estimate_waist_width(latest_landmarks)
            hip_width = self._calculate_hip_width(latest_landmarks)
            
            if waist_width and hip_width:
                return waist_width / hip_width
            
            return None
            
        except Exception as e:
            logger.error(f"Waist-to-hip ratio calculation failed: {e}")
            return None
    
    def _estimate_waist_width(self, landmarks) -> Optional[float]:
        """Estimate waist width from landmarks"""
        try:
            # Waist is approximately midway between shoulders and hips
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            
            if all(lm.visibility > 0.5 for lm in [left_shoulder, right_shoulder, left_hip, right_hip]):
                # Estimate waist width as slightly narrower than hip width
                shoulder_width = abs(right_shoulder.x - left_shoulder.x)
                hip_width = abs(right_hip.x - left_hip.x)
                
                # Waist is typically 70-80% of hip width
                waist_width = hip_width * 0.75
                
                return waist_width
            
            return None
            
        except Exception:
            return None
    
    def _estimate_body_frame_size(self, height_cm: float) -> Optional[str]:
        """Estimate body frame size (small/medium/large)"""
        try:
            if not hasattr(self, 'pose_landmarks') or not self.pose_landmarks:
                return None
            
            # Get the latest landmarks
            latest_landmarks = self.pose_landmarks[-1] if self.pose_landmarks else None
            if not latest_landmarks:
                return None
            
            # Use wrist size relative to height (approximated from landmarks)
            # In practice, this would measure wrist circumference
            
            # Estimate frame from shoulder width relative to height
            shoulder_width = self._calculate_shoulder_width(latest_landmarks)
            if not shoulder_width:
                return None
            
            # Convert normalized shoulder width to approximate measurement
            # This is a rough approximation
            frame_ratio = shoulder_width * height_cm
            
            if frame_ratio < 240:
                return "small"
            elif frame_ratio < 280:
                return "medium"
            else:
                return "large"
                
        except Exception as e:
            logger.error(f"Body frame estimation failed: {e}")
            return None
    
    def _estimate_metabolic_age(self, body_fat_pct: float, muscle_mass_kg: float, weight_kg: float) -> Optional[float]:
        """Estimate metabolic age based on body composition"""
        try:
            # Base metabolic age calculation
            # Lower body fat % and higher muscle mass = younger metabolic age
            
            # Baseline age (would typically use chronological age)
            baseline_age = 30  # Assuming average adult
            
            # Body fat adjustment (higher BF% increases metabolic age)
            bf_adjustment = (body_fat_pct - 15) * 0.8  # 15% is considered optimal
            
            # Muscle mass adjustment (higher muscle mass decreases metabolic age)
            muscle_ratio = muscle_mass_kg / weight_kg
            muscle_adjustment = (0.45 - muscle_ratio) * 40  # 45% muscle is considered good
            
            metabolic_age = baseline_age + bf_adjustment + muscle_adjustment
            
            return max(18, min(80, metabolic_age))
            
        except Exception as e:
            logger.error(f"Metabolic age estimation failed: {e}")
            return None
    
    def _calculate_additional_measurements(self, height_cm: Optional[float]) -> Dict[str, Optional[float]]:
        """Calculate additional body measurements from landmarks"""
        try:
            measurements = {}
            
            if not hasattr(self, 'pose_landmarks') or not self.pose_landmarks:
                return measurements
            
            # Get the latest landmarks
            latest_landmarks = self.pose_landmarks[-1] if self.pose_landmarks else None
            if not latest_landmarks:
                return measurements
            
            # Arm span estimation
            arm_span = self._estimate_arm_span(latest_landmarks, height_cm)
            if arm_span:
                measurements['arm_span_cm'] = round(arm_span, 1)
            
            # Shoulder to waist ratio
            shoulder_waist_ratio = self._calculate_shoulder_waist_ratio(latest_landmarks)
            if shoulder_waist_ratio:
                measurements['shoulder_waist_ratio'] = round(shoulder_waist_ratio, 2)
            
            # Chest/bust circumference estimation
            chest_circumference = self._estimate_chest_circumference(latest_landmarks, height_cm)
            if chest_circumference:
                measurements['chest_circumference_cm'] = round(chest_circumference, 1)
            
            # Leg length estimation
            leg_length = self._estimate_leg_length(latest_landmarks, height_cm)
            if leg_length:
                measurements['leg_length_cm'] = round(leg_length, 1)
            
            # Torso length estimation
            torso_length = self._estimate_torso_length(latest_landmarks, height_cm)
            if torso_length:
                measurements['torso_length_cm'] = round(torso_length, 1)
            
            return measurements
            
        except Exception as e:
            logger.error(f"Additional measurements calculation failed: {e}")
            return {}
    
    def _estimate_arm_span(self, landmarks, height_cm: Optional[float]) -> Optional[float]:
        """Estimate arm span from wrist-to-wrist distance"""
        try:
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            
            if left_wrist.visibility > 0.5 and right_wrist.visibility > 0.5:
                # Calculate normalized distance between wrists
                wrist_distance = math.sqrt(
                    (right_wrist.x - left_wrist.x) ** 2 + 
                    (right_wrist.y - left_wrist.y) ** 2
                )
                
                # Convert to real measurement using height as reference
                if height_cm:
                    # Arm span is typically close to height (0.95-1.05 ratio)
                    # Estimate based on the proportion in the image
                    estimated_arm_span = height_cm * (wrist_distance / 0.8)  # Assume 80% frame width
                    
                    # Clamp to reasonable range
                    return max(height_cm * 0.9, min(height_cm * 1.1, estimated_arm_span))
            
            return None
            
        except Exception:
            return None
    
    def _calculate_shoulder_waist_ratio(self, landmarks) -> Optional[float]:
        """Calculate shoulder to waist ratio"""
        try:
            shoulder_width = self._calculate_shoulder_width(landmarks)
            waist_width = self._estimate_waist_width(landmarks)
            
            if shoulder_width and waist_width:
                return shoulder_width / waist_width
            
            return None
            
        except Exception:
            return None
    
    def _estimate_chest_circumference(self, landmarks, height_cm: Optional[float]) -> Optional[float]:
        """Estimate chest/bust circumference from depth and landmarks"""
        try:
            # Get chest area landmarks
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                # Calculate chest width
                chest_width_normalized = abs(right_shoulder.x - left_shoulder.x)
                
                # Estimate depth from available depth map or use approximation
                chest_depth_factor = 0.4  # Chest depth is typically ~40% of width
                
                if height_cm:
                    # Convert normalized measurements to real measurements
                    chest_width_cm = chest_width_normalized * height_cm * 0.3  # Rough scaling
                    chest_depth_cm = chest_width_cm * chest_depth_factor
                    
                    # Estimate circumference using ellipse approximation
                    # Circumference ≈ π * (3(a+b) - √((3a+b)(a+3b))) for ellipse
                    a = chest_width_cm / 2
                    b = chest_depth_cm / 2
                    circumference = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
                    
                    return max(70, min(150, circumference))
            
            return None
            
        except Exception:
            return None
    
    def _estimate_leg_length(self, landmarks, height_cm: Optional[float]) -> Optional[float]:
        """Estimate leg length from hip to ankle"""
        try:
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            
            if left_hip.visibility > 0.5 and left_ankle.visibility > 0.5:
                # Calculate normalized leg length
                leg_length_normalized = math.sqrt(
                    (left_ankle.x - left_hip.x) ** 2 + 
                    (left_ankle.y - left_hip.y) ** 2
                )
                
                if height_cm:
                    # Leg length is typically 45-50% of total height
                    estimated_leg_length = height_cm * (leg_length_normalized / 0.9)  # Rough scaling
                    
                    # Clamp to reasonable range (40-55% of height)
                    return max(height_cm * 0.4, min(height_cm * 0.55, estimated_leg_length))
            
            return None
            
        except Exception:
            return None
    
    def _estimate_torso_length(self, landmarks, height_cm: Optional[float]) -> Optional[float]:
        """Estimate torso length from shoulder to hip"""
        try:
            # Use average of left and right shoulders/hips
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            
            if all(lm.visibility > 0.5 for lm in [left_shoulder, right_shoulder, left_hip, right_hip]):
                # Calculate center points
                shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
                hip_center_y = (left_hip.y + right_hip.y) / 2
                
                # Calculate normalized torso length
                torso_length_normalized = abs(hip_center_y - shoulder_center_y)
                
                if height_cm:
                    # Torso length is typically 30-35% of total height
                    estimated_torso_length = height_cm * (torso_length_normalized / 0.6)  # Rough scaling
                    
                    # Clamp to reasonable range (25-40% of height)
                    return max(height_cm * 0.25, min(height_cm * 0.4, estimated_torso_length))
            
            return None
            
        except Exception:
            return None
    
    def _extract_focal_length_from_exif(self, image_path: str) -> Optional[float]:
        """Extract focal length from image EXIF data if available"""
        try:
            if not EXIF_AVAILABLE:
                return None
            
            with Image.open(image_path) as img:
                exif_data = img._getexif()
                
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        
                        if tag == "FocalLength":
                            # Convert to pixels using sensor size (approximate)
                            # This is a rough approximation
                            focal_length_mm = float(value)
                            sensor_width_mm = 23.6  # Typical APS-C sensor width
                            image_width_px = img.width
                            
                            focal_length_px = (focal_length_mm * image_width_px) / sensor_width_mm
                            return focal_length_px
            
            return None
            
        except Exception as e:
            logger.error(f"EXIF focal length extraction failed: {e}")
            return None
    
    def _apply_perspective_correction(self, landmarks, focal_length: float, distance: float) -> Dict[str, Any]:
        """Apply perspective correction to measurements"""
        try:
            # This would implement proper perspective correction
            # For now, return simple scaling factor
            
            # Standard focal length for perspective correction
            standard_focal_length = 500.0
            
            # Distance-based scaling
            standard_distance = 2.0  # meters
            
            # Calculate correction factors
            focal_correction = focal_length / standard_focal_length
            distance_correction = distance / standard_distance
            
            return {
                'focal_correction': focal_correction,
                'distance_correction': distance_correction,
                'combined_correction': focal_correction * distance_correction
            }
            
        except Exception as e:
            logger.error(f"Perspective correction failed: {e}")
            return {'combined_correction': 1.0}
    
    def _convert_to_us_units(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metric measurements to US units"""
        try:
            # Convert height from cm to feet and inches
            if results.get('height_cm'):
                height_cm = results['height_cm']
                height_inches = height_cm / 2.54
                feet = int(height_inches // 12)
                inches = height_inches % 12
                results['height_ft_in'] = f"{feet}'{inches:.1f}\""
                results['height_inches'] = round(height_inches, 1)
            
            # Convert weight from kg to pounds
            if results.get('weight_kg'):
                weight_kg = results['weight_kg']
                weight_lbs = weight_kg * 2.20462
                results['weight_lbs'] = round(weight_lbs, 1)
            
            # Convert muscle mass from kg to pounds
            if results.get('muscle_mass_kg'):
                muscle_kg = results['muscle_mass_kg']
                muscle_lbs = muscle_kg * 2.20462
                results['muscle_mass_lbs'] = round(muscle_lbs, 1)
            
            # Convert measurements to inches
            measurements = results.get('measurements', {})
            us_measurements = {}
            
            for key, value in measurements.items():
                if value and key.endswith('_cm'):
                    # Convert cm measurements to inches
                    us_key = key.replace('_cm', '_in')
                    us_measurements[us_key] = round(value / 2.54, 1)
                elif value:
                    us_measurements[key] = value
            
            results['measurements_us'] = us_measurements
            
            return results
            
        except Exception as e:
            logger.error(f"US units conversion failed: {e}")
            return results

    def _calculate_confidence_scores(self) -> Dict[str, float]:
        """Calculate confidence scores for all measurements"""
        try:
            confidence_scores = {}
            
            # Height confidence based on method agreement
            if self.height_estimates:
                height_values = []
                for heights in self.height_estimates.values():
                    height_values.extend(heights)
                
                if len(height_values) > 1:
                    height_std = np.std(height_values)
                    height_confidence = max(0.1, 1.0 - (height_std / 20.0))  # Lower std = higher confidence
                else:
                    height_confidence = 0.5
                
                confidence_scores['height'] = height_confidence
            
            # Weight confidence based on method agreement
            if self.weight_estimates:
                weight_values = []
                for weights in self.weight_estimates.values():
                    weight_values.extend(weights)
                
                if len(weight_values) > 1:
                    weight_std = np.std(weight_values)
                    weight_confidence = max(0.1, 1.0 - (weight_std / 15.0))  # Lower std = higher confidence
                else:
                    weight_confidence = 0.5
                
                confidence_scores['weight'] = weight_confidence
            
            # Overall confidence
            individual_confidences = list(confidence_scores.values())
            if individual_confidences:
                confidence_scores['overall'] = np.mean(individual_confidences)
            else:
                confidence_scores['overall'] = 0.0
            
            return confidence_scores
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return {'overall': 0.0}
    
    def _update_current_metrics(self, yolo_result: bool, pose_results):
        """Update current metrics for real-time display"""
        try:
            # Update person confidence
            if yolo_result and self.yolo_model:
                results = self.yolo_model(pose_results, verbose=False)
                for result in results:
                    for box in result.boxes:
                        if box.cls == 0:  # Person class
                            self.current_metrics['person_confidence'] = box.conf.item()
                            break
            
            # Update height and weight estimates
            if self.height_estimates:
                avg_height = np.mean([np.mean(heights) for heights in self.height_estimates.values()])
                self.current_metrics['height'] = round(avg_height, 1)
                
                # Calculate BMI if we have weight too
                if self.weight_estimates:
                    avg_weight = np.mean([np.mean(weights) for weights in self.weight_estimates.values()])
                    self.current_metrics['weight'] = round(avg_weight, 1)
                    
                    if avg_height > 0:
                        bmi = avg_weight / ((avg_height / 100) ** 2)
                        self.current_metrics['bmi'] = round(bmi, 1)
                        
                        # Determine body type
                        if bmi < 18.5:
                            self.current_metrics['body_type'] = "Underweight"
                        elif bmi < 25:
                            self.current_metrics['body_type'] = "Normal"
                        elif bmi < 30:
                            self.current_metrics['body_type'] = "Overweight"
                        else:
                            self.current_metrics['body_type'] = "Obese"
            
        except Exception as e:
            logger.error(f"Failed to update current metrics: {e}")
    
    def _draw_person_detection_indicator(self, frame: np.ndarray, person_detected: bool, w: int, h: int):
        """Draw full body detection indicator with silhouette"""
        # Draw human silhouette outline
        silhouette_points = self._get_silhouette_points(w, h)
        
        if person_detected:
            # Person detected - draw in green
            color = self.colors['success']
            # Add pulsing effect
            pulse_intensity = int(50 + 30 * math.sin(self.pulse_phase))
            glow_color = tuple(min(255, c + pulse_intensity) for c in color)
            
            # Draw silhouette with glow
            cv2.drawContours(frame, [silhouette_points], -1, glow_color, 3, cv2.LINE_AA)
            cv2.drawContours(frame, [silhouette_points], -1, color, 2, cv2.LINE_AA)
            
            # Check for full body visibility
            self._check_body_completeness(frame, w, h)
        else:
            # No person detected - draw in red with dashed effect
            self._draw_dashed_contour(frame, silhouette_points, self.colors['error'], 3)
    
    def _get_silhouette_points(self, w: int, h: int) -> np.ndarray:
        """Generate human silhouette points"""
        center_x, center_y = w // 2, h // 2
        head_radius = min(w, h) // 16
        
        # Simple human silhouette
        points = [
            [center_x, center_y - h//3],  # Head top
            [center_x + head_radius, center_y - h//3 + head_radius * 2],  # Head right
            [center_x + w//8, center_y - h//6],  # Shoulder right
            [center_x + w//8, center_y],  # Arm right
            [center_x + w//12, center_y + h//6],  # Waist right
            [center_x + w//10, center_y + h//3],  # Hip right
            [center_x + w//16, center_y + h//2.5],  # Leg right
            [center_x, center_y + h//2.5],  # Feet center
            [center_x - w//16, center_y + h//2.5],  # Leg left
            [center_x - w//10, center_y + h//3],  # Hip left
            [center_x - w//12, center_y + h//6],  # Waist left
            [center_x - w//8, center_y],  # Arm left
            [center_x - w//8, center_y - h//6],  # Shoulder left
            [center_x - head_radius, center_y - h//3 + head_radius * 2],  # Head left
        ]
        
        return np.array(points, dtype=np.int32)
    
    def _check_body_completeness(self, frame: np.ndarray, w: int, h: int):
        """Check and display missing body parts"""
        margin = 50
        
        # Check head visibility (top area)
        if self._is_area_empty(frame[:margin, :]):
            cv2.putText(frame, "❌ Head not visible", (20, margin + 20), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['error'], 2, cv2.LINE_AA)
            # Draw arrow pointing up
            cv2.arrowedLine(frame, (w//2, margin + 30), (w//2, 10), 
                           self.colors['error'], 3, tipLength=0.3)
        
        # Check feet visibility (bottom area)
        if self._is_area_empty(frame[h-margin:, :]):
            cv2.putText(frame, "❌ Feet not in frame", (20, h - 10), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['error'], 2, cv2.LINE_AA)
            # Draw arrow pointing down
            cv2.arrowedLine(frame, (w//2, h - margin - 30), (w//2, h - 10), 
                           self.colors['error'], 3, tipLength=0.3)
        else:
            cv2.putText(frame, "✅ Full body detected - Stand still", (20, h - 10), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['success'], 2, cv2.LINE_AA)
    
    def _is_area_empty(self, area: np.ndarray) -> bool:
        """Check if area has minimal content (placeholder)"""
        # Simplified check - in real implementation would analyze pixel content
        return np.mean(area) < 50
    
    def _draw_yolo_bounding_box(self, frame: np.ndarray, original_frame: np.ndarray, w: int, h: int):
        """Draw sleek YOLO bounding box with confidence"""
        if not self.yolo_model:
            return
        
        try:
            results = self.yolo_model(original_frame, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    if box.cls == 0 and box.conf > 0.5:  # Person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = box.conf.item()
                        
                        # Update current metrics
                        self.current_metrics['person_confidence'] = confidence
                        
                        # Draw gradient border with glow
                        self._draw_gradient_box(frame, (x1, y1, x2, y2), confidence)
                        
                        # Confidence label
                        label = f"Person ({confidence:.0%})"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0]
                        
                        # Label background
                        cv2.rectangle(frame, (x2 - label_size[0] - 10, y1 - 30), 
                                     (x2, y1), self.colors['primary'], -1)
                        cv2.putText(frame, label, (x2 - label_size[0] - 5, y1 - 10), 
                                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                        break
        except Exception as e:
            logger.error(f"YOLO visualization failed: {e}")
    
    def _draw_gradient_box(self, frame: np.ndarray, box: tuple, confidence: float):
        """Draw gradient bounding box with glow"""
        x1, y1, x2, y2 = box
        
        # Calculate gradient colors based on confidence
        primary_intensity = int(255 * confidence)
        secondary_intensity = int(255 * (1 - confidence))
        
        # Draw multiple border lines for glow effect
        for i in range(5, 0, -1):
            alpha = int(100 / i)
            color = (primary_intensity // i, secondary_intensity // i, 255 // i)
            cv2.rectangle(frame, (x1 - i, y1 - i), (x2 + i, y2 + i), color, 2)
        
        # Main border
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['primary'], 3)
    
    def _draw_pose_skeleton(self, frame: np.ndarray, results, w: int, h: int):
        """Draw MediaPipe pose skeleton with color-coded confidence"""
        if not results.pose_landmarks:
            return
        
        landmarks = results.pose_landmarks.landmark
        connections = self.mp_pose.POSE_CONNECTIONS
        
        # Calculate average pose confidence
        pose_confidences = [lm.visibility for lm in landmarks if hasattr(lm, 'visibility')]
        avg_confidence = np.mean(pose_confidences) if pose_confidences else 0
        self.current_metrics['pose_quality'] = int(avg_confidence * 5)
        
        # Draw skeleton connections
        for connection in connections:
            start_idx, end_idx = connection
            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]
            
            # Get confidence for this connection
            confidence = min(getattr(start_lm, 'visibility', 1.0), 
                           getattr(end_lm, 'visibility', 1.0))
            
            # Choose color based on confidence
            if confidence > 0.8:
                color = self.colors['high_conf']
            elif confidence > 0.5:
                color = self.colors['med_conf']
            else:
                color = self.colors['low_conf']
            
            # Draw connection
            start_x, start_y = int(start_lm.x * w), int(start_lm.y * h)
            end_x, end_y = int(end_lm.x * w), int(end_lm.y * h)
            
            # Thick gradient line
            cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 4, cv2.LINE_AA)
        
        # Draw landmark points with pulsing effect
        for i, landmark in enumerate(landmarks):
            if hasattr(landmark, 'visibility') and landmark.visibility > 0.5:
                x, y = int(landmark.x * w), int(landmark.y * h)
                confidence = getattr(landmark, 'visibility', 1.0)
                
                # Pulsing effect for key joints
                key_joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # Major joints
                if i in key_joints:
                    pulse_radius = int(8 + 3 * math.sin(self.pulse_phase))
                    cv2.circle(frame, (x, y), pulse_radius, self.colors['primary'], -1, cv2.LINE_AA)
                    cv2.circle(frame, (x, y), pulse_radius + 2, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.circle(frame, (x, y), 5, self.colors['primary'], -1, cv2.LINE_AA)
        
        # Draw measurement lines
        self._draw_measurement_lines(frame, landmarks, w, h)
    
    def _draw_measurement_lines(self, frame: np.ndarray, landmarks, w: int, h: int):
        """Draw measurement lines between key points"""
        # Head to toe line
        if len(landmarks) > 28:
            nose = landmarks[0]
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]
            
            nose_x, nose_y = int(nose.x * w), int(nose.y * h)
            ankle_x = int((left_ankle.x + right_ankle.x) * w / 2)
            ankle_y = int((left_ankle.y + right_ankle.y) * h / 2)
            
            # Draw height measurement line
            cv2.line(frame, (nose_x, nose_y), (ankle_x, ankle_y), 
                    self.colors['warning'], 2, cv2.LINE_AA)
            
            # Height label
            mid_x = (nose_x + ankle_x) // 2
            mid_y = (nose_y + ankle_y) // 2
            cv2.putText(frame, "Height", (mid_x + 10, mid_y), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, self.colors['warning'], 1, cv2.LINE_AA)
        
        # Shoulder width line
        if len(landmarks) > 12:
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            
            left_x, left_y = int(left_shoulder.x * w), int(left_shoulder.y * h)
            right_x, right_y = int(right_shoulder.x * w), int(right_shoulder.y * h)
            
            cv2.line(frame, (left_x, left_y), (right_x, right_y), 
                    self.colors['secondary'], 2, cv2.LINE_AA)
            
            # Shoulder width label
            mid_x = (left_x + right_x) // 2
            cv2.putText(frame, "Shoulders", (mid_x - 30, left_y - 10), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, self.colors['secondary'], 1, cv2.LINE_AA)
    
    def _draw_body_metrics_panel(self, frame: np.ndarray, time_remaining: float, w: int, h: int):
        """Draw real-time body metrics panel"""
        panel_w, panel_h = 300, 180
        panel_x = w - panel_w - 20
        panel_y = 60
        
        # Semi-transparent background
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     self.colors['bg_panel'], -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     self.colors['primary'], 2)
        
        # Metrics text
        y_offset = panel_y + 25
        line_height = 20
        
        # Height (US Units)
        height_text = f"📏 Height: Calculating..."
        if self.current_metrics['height'] and isinstance(self.current_metrics['height'], (int, float)):
            height_cm = self.current_metrics['height']
            height_inches = height_cm / 2.54
            feet = int(height_inches // 12)
            inches = height_inches % 12
            height_text = f"📏 Height: {feet}'{inches:.1f}\""
        cv2.putText(frame, height_text, (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, self.colors['text'], 1, cv2.LINE_AA)
        y_offset += line_height
        
        # Weight (US Units)
        weight_text = f"⚖️ Weight: Estimating..."
        if self.current_metrics['weight'] and isinstance(self.current_metrics['weight'], (int, float)):
            weight_kg = self.current_metrics['weight']
            weight_lbs = weight_kg * 2.20462
            weight_text = f"⚖️ Weight: {weight_lbs:.1f} lbs"
        cv2.putText(frame, weight_text, (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, self.colors['text'], 1, cv2.LINE_AA)
        y_offset += line_height
        
        # BMI
        bmi_text = f"🏃 BMI: {self.current_metrics['bmi'] or '--'}"
        if self.current_metrics['bmi'] and isinstance(self.current_metrics['bmi'], (int, float)):
            bmi_text = f"🏃 BMI: {self.current_metrics['bmi']:.1f}"
        cv2.putText(frame, bmi_text, (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, self.colors['text'], 1, cv2.LINE_AA)
        y_offset += line_height
        
        # Body type
        body_type_text = f"💪 Body Type: {self.current_metrics['body_type'] or 'Analyzing...'}"
        cv2.putText(frame, body_type_text, (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, self.colors['text'], 1, cv2.LINE_AA)
        y_offset += line_height + 5
        
        # Measurements section
        cv2.putText(frame, "📐 Measurements:", (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, self.colors['warning'], 1, cv2.LINE_AA)
        y_offset += line_height
        
        # Shoulders
        shoulders_text = f"Shoulders: {self.current_metrics['shoulders'] or '--'} cm"
        cv2.putText(frame, shoulders_text, (panel_x + 20, y_offset), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.4, self.colors['text'], 1, cv2.LINE_AA)
        y_offset += line_height - 5
        
        # Torso
        torso_text = f"Torso: {self.current_metrics['torso'] or '--'} cm"
        cv2.putText(frame, torso_text, (panel_x + 20, y_offset), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.4, self.colors['text'], 1, cv2.LINE_AA)
        
        # Scan timer
        timer_text = f"⏱️ Body Scan: {time_remaining:.1f}s"
        cv2.putText(frame, timer_text, (panel_x + 10, panel_y + panel_h - 10), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['warning'], 2, cv2.LINE_AA)
    
    def _draw_depth_visualization(self, frame: np.ndarray, w: int, h: int):
        """Draw depth map as subtle heat overlay"""
        if self.depth_map is None:
            return
        
        try:
            # Resize depth map to match frame
            depth_resized = cv2.resize(self.depth_map, (w, h))
            
            # Normalize depth for visualization (0-255)
            depth_norm = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Apply colormap (COLORMAP_JET for heat effect)
            depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            
            # Blend with frame (subtle overlay)
            cv2.addWeighted(frame, 0.9, depth_colored, 0.1, 0, frame)
            
            # Add depth legend
            cv2.putText(frame, "Depth Map", (10, h - 40), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, self.colors['text'], 1, cv2.LINE_AA)
            
        except Exception as e:
            logger.error(f"Depth visualization failed: {e}")
    
    def _draw_scan_stages_progress(self, frame: np.ndarray, time_remaining: float, w: int, h: int):
        """Draw multi-stage progress bar at top"""
        # Calculate overall progress
        overall_progress = (20 - time_remaining) / 20
        
        # Update current stage based on progress
        stage_duration = 1.0 / len(self.scan_stages)
        self.current_stage = min(len(self.scan_stages) - 1, int(overall_progress / stage_duration))
        self.stage_progress = (overall_progress % stage_duration) / stage_duration
        
        # Draw stage indicators
        stage_width = w // len(self.scan_stages)
        stage_height = 30
        stage_y = 70
        
        for i, stage in enumerate(self.scan_stages):
            stage_x = i * stage_width
            
            # Background
            cv2.rectangle(frame, (stage_x + 10, stage_y), 
                         (stage_x + stage_width - 10, stage_y + stage_height), 
                         (64, 64, 64), -1)
            
            # Progress fill
            if i < self.current_stage:
                # Completed stage
                cv2.rectangle(frame, (stage_x + 10, stage_y), 
                             (stage_x + stage_width - 10, stage_y + stage_height), 
                             self.colors['success'], -1)
                symbol = "✓"
            elif i == self.current_stage:
                # Current stage
                progress_width = int((stage_width - 20) * self.stage_progress)
                cv2.rectangle(frame, (stage_x + 10, stage_y), 
                             (stage_x + 10 + progress_width, stage_y + stage_height), 
                             self.colors['primary'], -1)
                symbol = "▓▓▓░"[int(self.stage_progress * 4)]
            else:
                # Future stage
                symbol = "░░░░"
            
            # Stage label
            stage_text = stage.replace('_', ' ').title()
            text_size = cv2.getTextSize(stage_text, cv2.FONT_HERSHEY_DUPLEX, 0.4, 1)[0]
            text_x = stage_x + (stage_width - text_size[0]) // 2
            text_y = stage_y + stage_height // 2 + 3
            
            cv2.putText(frame, f"[{symbol}] {stage_text}", (text_x - 20, text_y), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.4, self.colors['text'], 1, cv2.LINE_AA)
        
        # Overall timer with circular progress
        center = (w - 100, 130)
        radius = 30
        
        # Background circle
        cv2.circle(frame, center, radius, (64, 64, 64), 3)
        
        # Progress arc
        if overall_progress > 0:
            end_angle = int(360 * overall_progress)
            cv2.ellipse(frame, center, (radius, radius), 0, -90, -90 + end_angle, 
                       self.colors['primary'], 3)
        
        # Timer text
        timer_text = f"{time_remaining:.1f}s"
        text_size = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
        text_x = center[0] - text_size[0] // 2
        text_y = center[1] + text_size[1] // 2
        
        cv2.putText(frame, timer_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1, cv2.LINE_AA)
        
        cv2.putText(frame, "/ 20s", (text_x, text_y + 15), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.4, self.colors['text'], 1, cv2.LINE_AA)
    
    def _draw_scan_line_effect(self, frame: np.ndarray, w: int, h: int):
        """Draw animated scan line effect"""
        line_y = self.scan_position
        
        # Horizontal scan line with fade effect
        for i in range(-15, 16):
            if 0 <= line_y + i < h:
                alpha = max(0, 1 - abs(i) / 15)
                intensity = int(255 * alpha)
                cv2.line(frame, (0, line_y + i), (w, line_y + i), 
                        (0, intensity, intensity), 1, cv2.LINE_AA)
    
    def _draw_body_instructions(self, frame: np.ndarray, pose_results, yolo_result: bool, w: int, h: int):
        """Draw animated instructions"""
        instruction_y = h - 100
        
        if not yolo_result:
            instruction = "👤 Step into frame"
            color = self.colors['error']
        elif not pose_results.pose_landmarks:
            instruction = "🤸 Stand in T-pose for detection"
            color = self.colors['warning']
        else:
            # Check pose quality
            landmarks = pose_results.pose_landmarks.landmark
            pose_quality = self._calculate_pose_confidence(landmarks)
            
            if pose_quality > 0.8:
                instruction = "✅ Perfect pose! Remain still"
                color = self.colors['success']
            elif pose_quality > 0.5:
                instruction = "📏 Adjust posture for better scan"
                color = self.colors['warning']
            else:
                instruction = "🤸 Stand straight with arms extended"
                color = self.colors['warning']
        
        # Draw instruction with glow effect
        self._draw_glowing_text(frame, instruction, (w//2, instruction_y), color, size=1.0)
    
    def _draw_3d_wireframe(self, frame: np.ndarray, pose_results, w: int, h: int):
        """Draw rotating 3D wireframe of detected pose"""
        if not pose_results.pose_landmarks:
            return
        
        try:
            # Create simplified 3D wireframe in bottom-right corner
            wireframe_size = 120
            wireframe_x = w - wireframe_size - 20
            wireframe_y = h - wireframe_size - 20
            
            # Background for wireframe
            cv2.rectangle(frame, (wireframe_x - 10, wireframe_y - 10), 
                         (wireframe_x + wireframe_size + 10, wireframe_y + wireframe_size + 10), 
                         (0, 0, 0), -1)
            cv2.rectangle(frame, (wireframe_x - 10, wireframe_y - 10), 
                         (wireframe_x + wireframe_size + 10, wireframe_y + wireframe_size + 10), 
                         self.colors['primary'], 2)
            
            # Draw grid floor for perspective
            grid_y = wireframe_y + wireframe_size - 20
            for i in range(0, wireframe_size, 20):
                cv2.line(frame, (wireframe_x + i, grid_y), 
                        (wireframe_x + i, grid_y + 20), (64, 64, 64), 1)
            for i in range(0, 21, 10):
                cv2.line(frame, (wireframe_x, grid_y + i), 
                        (wireframe_x + wireframe_size, grid_y + i), (64, 64, 64), 1)
            
            # Draw simplified 3D stick figure (front view with rotation effect)
            landmarks = pose_results.pose_landmarks.landmark
            rotation_factor = math.sin(math.radians(self.wireframe_rotation)) * 0.3
            
            # Key body points
            key_points = {
                'head': landmarks[0],
                'left_shoulder': landmarks[11],
                'right_shoulder': landmarks[12],
                'left_hip': landmarks[23],
                'right_hip': landmarks[24],
                'left_knee': landmarks[25],
                'right_knee': landmarks[26],
                'left_ankle': landmarks[27],
                'right_ankle': landmarks[28]
            }
            
            # Convert to wireframe coordinates with rotation effect
            wireframe_points = {}
            for name, lm in key_points.items():
                # Apply simple rotation effect
                x_offset = int(lm.x * wireframe_size * (1 + rotation_factor))
                y_offset = int(lm.y * wireframe_size)
                wireframe_points[name] = (wireframe_x + x_offset//2 + wireframe_size//4, 
                                        wireframe_y + y_offset//2 + 10)
            
            # Draw connections
            connections = [
                ('head', 'left_shoulder'), ('head', 'right_shoulder'),
                ('left_shoulder', 'right_shoulder'),
                ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
                ('left_hip', 'right_hip'),
                ('left_hip', 'left_knee'), ('right_hip', 'right_knee'),
                ('left_knee', 'left_ankle'), ('right_knee', 'right_ankle')
            ]
            
            for start, end in connections:
                if start in wireframe_points and end in wireframe_points:
                    cv2.line(frame, wireframe_points[start], wireframe_points[end], 
                            self.colors['secondary'], 2, cv2.LINE_AA)
            
            # Draw points
            for point in wireframe_points.values():
                cv2.circle(frame, point, 3, self.colors['primary'], -1)
            
            # Label
            cv2.putText(frame, "3D Pose", (wireframe_x, wireframe_y - 15), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.4, self.colors['text'], 1, cv2.LINE_AA)
            
        except Exception as e:
            logger.error(f"3D wireframe drawing failed: {e}")
    
    def _draw_glowing_text(self, frame: np.ndarray, text: str, pos: Tuple[int, int], 
                          color: Tuple[int, int, int], size: float = 1.0):
        """Draw text with glowing effect"""
        font = cv2.FONT_HERSHEY_DUPLEX
        thickness = max(1, int(2 * size))
        
        # Calculate centered position
        text_size = cv2.getTextSize(text, font, size, thickness)[0]
        text_x = pos[0] - text_size[0] // 2
        text_y = pos[1] + text_size[1] // 2
        
        # Draw glow (larger, lighter text)
        glow_color = tuple(min(255, c + 100) for c in color)
        cv2.putText(frame, text, (text_x, text_y), font, size, glow_color, thickness + 2, cv2.LINE_AA)
        
        # Draw main text
        cv2.putText(frame, text, (text_x, text_y), font, size, color, thickness, cv2.LINE_AA)
    
    def _draw_dashed_contour(self, frame: np.ndarray, contour: np.ndarray, 
                            color: Tuple[int, int, int], thickness: int):
        """Draw dashed contour"""
        perimeter = cv2.arcLength(contour, True)
        dash_length = 10
        gap_length = 5
        
        # Approximate contour for simpler dashed drawing
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        for i in range(len(approx)):
            start_point = tuple(approx[i][0])
            end_point = tuple(approx[(i + 1) % len(approx)][0])
            
            # Draw dashed line between points
            self._draw_dashed_line(frame, start_point, end_point, color, thickness)
    
    def _draw_dashed_line(self, frame: np.ndarray, start: Tuple[int, int], 
                         end: Tuple[int, int], color: Tuple[int, int, int], thickness: int):
        """Draw dashed line between two points"""
        distance = np.linalg.norm(np.array(end) - np.array(start))
        dash_length = 8
        gap_length = 4
        total_length = dash_length + gap_length
        
        if distance < total_length:
            cv2.line(frame, start, end, color, thickness)
            return
        
        direction = (np.array(end) - np.array(start)) / distance
        
        current_pos = np.array(start, dtype=float)
        remaining_distance = distance
        
        while remaining_distance > 0:
            dash_end = current_pos + direction * min(dash_length, remaining_distance)
            cv2.line(frame, tuple(current_pos.astype(int)), tuple(dash_end.astype(int)), 
                    color, thickness)
            
            current_pos = dash_end + direction * gap_length
            remaining_distance -= (dash_length + gap_length)


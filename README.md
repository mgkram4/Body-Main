# 🔬 BIO-SCAN-BEST Body Scanner

Advanced AI-powered body measurement system using computer vision and machine learning.

## 🚀 Features

- **Full Body Measurements**: Height, weight, BMI, body composition
- **AI-Powered Analysis**: Uses YOLOv8, MediaPipe, MiDaS, SAM, and SMPL-X
- **Real-time Processing**: Live camera feed with visual feedback
- **Detailed Metrics**: Body fat %, muscle mass, bone density, metabolic age
- **US & Metric Units**: Results in both measurement systems
- **High Accuracy**: Multi-method validation for reliable measurements

## 🎯 What It Measures

### Basic Measurements
- Height (feet/inches and cm)
- Weight (lbs and kg)
- BMI and body type classification

### Body Composition
- Body fat percentage
- Muscle mass
- Bone density
- Metabolic age

### Detailed Measurements
- Shoulder width
- Chest/bust circumference
- Waist-to-hip ratio
- Arm span
- Leg length
- Torso length

## 🔧 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux
- **Python**: 3.8 or higher
- **RAM**: 8GB (16GB recommended)
- **Camera**: USB webcam or built-in camera
- **Storage**: 5GB free space for models

### Recommended for Best Performance
- **GPU**: NVIDIA GPU with CUDA support
- **RAM**: 16GB or more
- **Camera**: 1080p webcam with good lighting

## 📦 Installation

### Step 1: Clone or Download Files

Make sure you have these files in your directory:
```
body_scanner/
├── body.py              # Core body estimation class
├── body_run.py          # Standalone runner
├── requirements_body.txt # Python dependencies
└── README_body_scanner.md # This file
```

### Step 2: Set Up Python Virtual Environment

**Windows:**
```bash
# Create virtual environment
python -m venv bioscan_env

# Activate virtual environment
bioscan_env\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv bioscan_env

# Activate virtual environment
source bioscan_env/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements_body.txt

# Install Segment Anything Model (optional but recommended)
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Step 4: Download AI Models

The system will automatically download most models on first run:
- **YOLOv8**: Downloads automatically (~6MB)
- **MiDaS**: Downloads automatically (~1.3GB)
- **SAM**: Downloads automatically (~2.4GB)

**Optional SMPL-X Setup** (for enhanced 3D body modeling):
1. Visit https://smpl-x.is.tue.mpg.de/
2. Register and download SMPL-X models
3. Create `models/smplx/` directory
4. Place model files in the directory

## 🎮 Usage

### Basic Usage

1. **Activate Virtual Environment** (if not already active):
   ```bash
   # Windows
   bioscan_env\Scripts\activate
   
   # macOS/Linux
   source bioscan_env/bin/activate
   ```

2. **Run Body Scanner**:
   ```bash
   python body_run.py
   ```

3. **Follow On-Screen Instructions**:
   - Stand 6-8 feet from camera
   - Ensure full body is visible (head to feet)
   - Stand in T-pose with arms extended
   - Remain still during 20-second scan

### Sample Output

```
🔬 BIO-SCAN-BEST - BODY SCAN RESULTS
============================================================
📏 Height: 5'10.2" (178.3 cm)
⚖️  Weight: 165.4 lbs (75.0 kg)
🏃 BMI: 23.6 (Normal)

💪 BODY COMPOSITION:
   Body Fat: 15.2%
   Muscle Mass: 132.1 lbs (59.9 kg)
   Bone Density: 1.24 g/cm³
   Metabolic Age: 25 years

📐 DETAILED MEASUREMENTS:
   Arm Span: 71.2 in
   Chest Circumference: 38.5 in
   Shoulder Width: 17.8 in
   Leg Length: 32.1 in

🎯 SCAN QUALITY:
   Overall Confidence: 87.3%
   Height Confidence: 91.2%
   Weight Confidence: 83.4%
============================================================
```

## 📁 Output Files

Results are automatically saved to:
- **JSON File**: `scan_results/body_scan_YYYYMMDD_HHMMSS.json`
- **Log File**: `body_scan.log`

## 🔧 Troubleshooting

### Camera Issues
```bash
# List available cameras
python -c "import cv2; [print(f'Camera {i}: {cv2.VideoCapture(i).isOpened()}') for i in range(3)]"
```

### GPU Not Detected
```bash
# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Memory Issues
- Close other applications
- Reduce scan duration: Edit `scan_duration = 10` in `body_run.py`
- Use CPU mode: Set `use_gpu = False` in `body_run.py`

### Model Download Issues
- Ensure stable internet connection
- Check firewall settings
- Manually download models if needed

## ⚡ Performance Optimization

### For Better Speed
1. **Use GPU**: Install CUDA-compatible PyTorch
2. **Good Lighting**: Ensure well-lit environment
3. **Stable Internet**: For initial model downloads
4. **Close Apps**: Free up RAM and CPU

### For Better Accuracy
1. **Camera Quality**: Use 1080p camera if available
2. **Lighting**: Even, bright lighting without shadows
3. **Background**: Plain, contrasting background
4. **Distance**: Stand 6-8 feet from camera
5. **Stillness**: Remain completely still during scan

## 🔄 Updates and Maintenance

### Update Dependencies
```bash
pip install --upgrade -r requirements_body.txt
```

### Clear Model Cache (if needed)
```bash
# Remove downloaded models to force re-download
rm -rf models/
```

## ⚠️ Important Notes

### Privacy
- All processing is done locally on your device
- No data is sent to external servers
- Images are processed in memory only

### Accuracy
- Results are estimates for health/fitness reference
- Not intended for medical diagnosis
- Professional medical assessment recommended for health decisions

### Hardware Requirements
- Webcam with at least 720p resolution
- Sufficient lighting for clear body detection
- Stable camera mount or tripod recommended

## 🆘 Getting Help

### Common Issues
1. **"No camera found"**: Check camera connections and permissions
2. **"Model loading failed"**: Check internet connection and storage space
3. **"Low confidence results"**: Improve lighting and ensure full body visibility
4. **"GPU out of memory"**: Use CPU mode or close other applications

### Support
- Check the log file `body_scan.log` for detailed error messages
- Ensure all dependencies are properly installed
- Verify camera functionality with other applications

## 📊 Technical Details

### AI Models Used
- **YOLOv8**: Person detection and bounding boxes
- **MediaPipe**: 33-point body landmark detection
- **MiDaS**: Monocular depth estimation
- **SAM**: Precise body segmentation
- **SMPL-X**: 3D human body modeling (optional)

### Processing Pipeline
1. Person detection with YOLO
2. Pose estimation with MediaPipe
3. Depth mapping with MiDaS
4. Body segmentation with SAM
5. 3D modeling with SMPL-X
6. Multi-method measurement validation
7. Result fusion and confidence scoring

---

🔬 **BIO-SCAN-BEST** - Advanced AI Body Analysis System 

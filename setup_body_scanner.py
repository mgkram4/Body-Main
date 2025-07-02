#!/usr/bin/env python3
"""
BIO-SCAN-BEST Body Scanner Setup Script
Automates the installation and setup process
"""

import os
import platform
import subprocess
import sys
from pathlib import Path


def print_banner():
    """Print setup banner"""
    print("=" * 60)
    print("🔬 BIO-SCAN-BEST Body Scanner Setup")
    print("=" * 60)
    print("This script will set up your body scanning environment.")
    print("")

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required.")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        print("   Please upgrade Python and try again.")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def check_system_requirements():
    """Check system requirements"""
    print("\n🔍 Checking system requirements...")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check available space (rough estimate)
    try:
        import shutil
        free_space_gb = shutil.disk_usage('.').free / (1024**3)
        if free_space_gb < 5:
            print(f"⚠️  Warning: Low disk space ({free_space_gb:.1f}GB available)")
            print("   At least 5GB recommended for AI models")
        else:
            print(f"✅ Disk space: {free_space_gb:.1f}GB available")
    except:
        print("⚠️  Could not check disk space")
    
    # Check if camera is available
    try:
        import cv2
        camera = cv2.VideoCapture(0)
        if camera.isOpened():
            print("✅ Camera detected")
            camera.release()
        else:
            print("⚠️  No camera detected - please connect a webcam")
    except:
        print("⚠️  Cannot check camera (OpenCV not installed yet)")
    
    return True

def setup_virtual_environment():
    """Set up Python virtual environment"""
    print("\n🐍 Setting up virtual environment...")
    
    venv_name = "bioscan_env"
    venv_path = Path(venv_name)
    
    if venv_path.exists():
        print(f"📁 Virtual environment '{venv_name}' already exists")
        return True
    
    try:
        # Create virtual environment
        print(f"   Creating virtual environment: {venv_name}")
        subprocess.run([sys.executable, "-m", "venv", venv_name], check=True)
        
        # Get activation script path
        system = platform.system().lower()
        if system == "windows":
            activate_script = venv_path / "Scripts" / "activate.bat"
            pip_path = venv_path / "Scripts" / "pip.exe"
        else:
            activate_script = venv_path / "bin" / "activate"
            pip_path = venv_path / "bin" / "pip"
        
        print(f"✅ Virtual environment created successfully")
        print(f"   Activation script: {activate_script}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Check if requirements file exists
    req_file = "requirements_body.txt"
    if not os.path.exists(req_file):
        print(f"❌ Requirements file '{req_file}' not found")
        return False
    
    try:
        # Get pip path in virtual environment
        venv_name = "bioscan_env"
        system = platform.system().lower()
        if system == "windows":
            pip_path = os.path.join(venv_name, "Scripts", "pip.exe")
        else:
            pip_path = os.path.join(venv_name, "bin", "pip")
        
        if not os.path.exists(pip_path):
            print("❌ Virtual environment pip not found")
            print("   Please run this script after creating the virtual environment")
            return False
        
        # Upgrade pip first
        print("   Upgrading pip...")
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        print("   Installing core dependencies...")
        subprocess.run([pip_path, "install", "-r", req_file], check=True)
        
        print("✅ Core dependencies installed successfully")
        
        # Optional: Install SAM
        try:
            print("   Installing Segment Anything Model (optional)...")
            subprocess.run([
                pip_path, "install", 
                "git+https://github.com/facebookresearch/segment-anything.git"
            ], check=True, timeout=300)
            print("✅ SAM installed successfully")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print("⚠️  SAM installation failed (optional - will use fallback)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "models",
        "scan_results", 
        "models/smplx"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   Created: {directory}")
    
    print("✅ Directories created successfully")

def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")
    
    try:
        # Get python path in virtual environment
        venv_name = "bioscan_env"
        system = platform.system().lower()
        if system == "windows":
            python_path = os.path.join(venv_name, "Scripts", "python.exe")
        else:
            python_path = os.path.join(venv_name, "bin", "python")
        
        # Test core imports
        test_script = """
import cv2
import mediapipe as mp
import numpy as np
import torch
from ultralytics import YOLO
print("✅ All core dependencies imported successfully")
print(f"   OpenCV: {cv2.__version__}")
print(f"   MediaPipe: {mp.__version__}")
print(f"   NumPy: {np.__version__}")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
"""
        
        result = subprocess.run([python_path, "-c", test_script], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"❌ Import test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  Import test timed out")
        return False
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def print_usage_instructions():
    """Print final usage instructions"""
    system = platform.system().lower()
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("=" * 60)
    print("\n📋 Next steps:")
    print("\n1. Activate the virtual environment:")
    
    if system == "windows":
        print("   bioscan_env\\Scripts\\activate")
    else:
        print("   source bioscan_env/bin/activate")
    
    print("\n2. Run the body scanner:")
    print("   python body_run.py")
    
    print("\n3. For best results:")
    print("   - Use good lighting")
    print("   - Stand 6-8 feet from camera")
    print("   - Ensure full body is visible")
    print("   - Remain still during scan")
    
    print("\n📖 For detailed instructions, see README_body_scanner.md")
    print("\n🔬 Happy scanning with BIO-SCAN-BEST!")

def main():
    """Main setup process"""
    print_banner()
    
    # Check system requirements
    if not check_system_requirements():
        print("\n❌ System requirements not met. Please resolve issues and try again.")
        return 1
    
    # Set up virtual environment
    if not setup_virtual_environment():
        print("\n❌ Virtual environment setup failed.")
        return 1
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Dependency installation failed.")
        return 1
    
    # Create directories
    create_directories()
    
    # Test installation
    if not test_installation():
        print("\n⚠️  Installation test failed, but setup may still work.")
        print("   Try running the scanner manually to verify.")
    
    # Print usage instructions
    print_usage_instructions()
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1) 
#!/usr/bin/env python3
"""Test script to verify the advanced dehazing algorithm"""

import os
import sys
import cv2
import numpy as np
from dehazing.dehaze import dehaze_image, get_dehazing_model

def test_dehazing():
    """Test the dehazing algorithm on a sample image"""
    print("Testing advanced dehazing algorithm...")

    # Check if sample images exist
    uploads_dir = "static/uploads"
    if not os.path.exists(uploads_dir):
        print("✗ Uploads directory not found")
        return False

    # Find a sample image
    sample_images = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not sample_images:
        print("✗ No sample images found")
        return False

    sample_image = os.path.join(uploads_dir, sample_images[0])
    print(f"Testing with sample image: {sample_image}")

    # Create output directory
    output_dir = "static/test_outputs"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Test the dehazing model initialization
        print("Initializing dehazing model...")
        model = get_dehazing_model()
        print("✓ Model initialized successfully")

        # Test dehazing
        print("Running dehazing algorithm...")
        output_path, steps = dehaze_image(sample_image, output_dir)
        print(f"✓ Dehazing completed. Output: {output_path}")

        # Check if output files were created
        if os.path.exists(output_path):
            print("✓ Output file created successfully")
        else:
            print("✗ Output file not created")
            return False

        # Check if step files were created
        step_files_created = sum(1 for step_path in steps.values() if os.path.exists(step_path))
        print(f"✓ {step_files_created}/{len(steps)} step visualization files created")

        # Verify image quality
        original = cv2.imread(sample_image)
        dehazed = cv2.imread(output_path)

        if dehazed is not None:
            print(f"✓ Output image dimensions: {dehazed.shape}")
            print("✓ Dehazing algorithm working correctly")
            return True
        else:
            print("✗ Failed to read output image")
            return False

    except Exception as e:
        print(f"✗ Error during dehazing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dehazing()
    if success:
        print("\n🎉 Advanced dehazing algorithm test PASSED!")
        sys.exit(0)
    else:
        print("\n❌ Advanced dehazing algorithm test FAILED!")
        sys.exit(1)

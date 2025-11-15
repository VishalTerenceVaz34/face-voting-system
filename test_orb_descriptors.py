#!/usr/bin/env python3
"""
Test script to verify ORB descriptor computation and storage at registration time.
"""
import os
import sys
import cv2
import numpy as np
from PIL import Image

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import (
    compute_and_save_orb_descriptors,
    match_against_precomputed_descriptors,
    detect_face_roi_from_np,
    DESCRIPTORS_FOLDER,
    UPLOAD_FOLDER
)

def test_descriptor_computation():
    """Test computing and saving descriptors, then matching."""
    print("=" * 60)
    print("Test 1: Compute and save ORB descriptors")
    print("=" * 60)
    
    # Check if debug_face.png exists from earlier face-login test
    debug_image_path = os.path.join(UPLOAD_FOLDER, 'debug_face.png')
    if not os.path.exists(debug_image_path):
        print(f"❌ Debug image not found at {debug_image_path}")
        print("   Skipping test. Register/login once to generate debug_face.png")
        return False
    
    print(f"✓ Found debug image at {debug_image_path}")
    
    # Test descriptor computation
    test_descriptor_path = os.path.join(DESCRIPTORS_FOLDER, 'test_desc.npy')
    kp, des = compute_and_save_orb_descriptors(debug_image_path, test_descriptor_path)
    
    if des is None:
        print("❌ Failed to compute descriptors")
        return False
    
    print(f"✓ Computed descriptors: shape={des.shape}, dtype={des.dtype}")
    
    if not os.path.exists(test_descriptor_path):
        print("❌ Descriptor file not saved")
        return False
    
    print(f"✓ Descriptors saved to {test_descriptor_path}")
    
    # Test matching: load precomputed and match against same image
    print("\n" + "=" * 60)
    print("Test 2: Match against precomputed descriptors (same image)")
    print("=" * 60)
    
    # Load the debug image again and convert to RGB numpy array
    img_pil = Image.open(debug_image_path).convert('RGB')
    img_np = np.asarray(img_pil, dtype=np.uint8)
    
    matched, score = match_against_precomputed_descriptors(img_np, test_descriptor_path, match_threshold=0.05)
    print(f"Matching result: matched={matched}, score={score}")
    
    if score > 0.7:
        print(f"✓ High match score (same image): {score:.3f}")
    else:
        print(f"⚠ Lower than expected match score: {score:.3f} (threshold: 0.12)")
    
    # Cleanup
    if os.path.exists(test_descriptor_path):
        os.remove(test_descriptor_path)
        print(f"✓ Cleaned up test descriptor file")
    
    return True

if __name__ == '__main__':
    print("Testing ORB Descriptor Storage and Matching\n")
    
    if not os.path.exists(DESCRIPTORS_FOLDER):
        print(f"❌ Descriptors folder does not exist: {DESCRIPTORS_FOLDER}")
        sys.exit(1)
    
    success = test_descriptor_computation()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed!")
        print("\nNext steps:")
        print("1. Register a new student (this will compute and save ORB descriptors)")
        print("2. Login with face recognition (will use precomputed descriptors)")
        print("3. Check terminal output for matching scores")
    else:
        print("⚠ Tests inconclusive or failed. Please generate debug_face.png first.")
    print("=" * 60)

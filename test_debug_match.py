#!/usr/bin/env python3
"""
Quick script to test the /debug-match endpoint.
Usage: python test_debug_match.py <image1_path> <image2_path>
"""
import sys
import requests
import json

def test_debug_match(image1_path, image2_path, server_url="http://127.0.0.1:5000"):
    """Test the debug-match endpoint."""
    print(f"Testing /debug-match endpoint:")
    print(f"  Image 1: {image1_path}")
    print(f"  Image 2: {image2_path}")
    print(f"  Server: {server_url}")
    print()
    
    try:
        with open(image1_path, 'rb') as f1, open(image2_path, 'rb') as f2:
            files = {
                'image1': f1,
                'image2': f2
            }
            response = requests.post(f"{server_url}/debug-match", files=files, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return False
        
        data = response.json()
        
        print("=" * 70)
        print("MATCHING CONFIGURATION")
        print("=" * 70)
        for key, val in data.get('config', {}).items():
            print(f"  {key}: {val}")
        
        print("\n" + "=" * 70)
        print("MATCHING RESULTS")
        print("=" * 70)
        
        results = data.get('results', {})
        
        # ORB Results
        orb = results.get('orb', {})
        print(f"\nORB Matching:")
        print(f"  Score: {orb.get('score', 0):.3f} (threshold: {orb.get('threshold', 0):.3f})")
        print(f"  Matched: {orb.get('matched', False)}")
        if orb.get('score', 0) < orb.get('threshold', 0):
            print(f"  → Score below threshold by {orb.get('threshold', 0) - orb.get('score', 0):.3f}")
            print(f"    Recommendation: Increase 'orb_match_threshold' to {orb.get('score', 0) + 0.02:.3f} or higher")
        
        # Histogram Results
        hist = results.get('histogram', {})
        print(f"\nHistogram Matching:")
        print(f"  Score: {hist.get('score', 0):.3f} (threshold: {hist.get('threshold', 0):.3f})")
        if hist.get('score', 0) < hist.get('threshold', 0):
            print(f"  → Score below threshold by {hist.get('threshold', 0) - hist.get('score', 0):.3f}")
            print(f"    Recommendation: Decrease 'hist_threshold' to {hist.get('score', 0) - 0.02:.3f} or lower")
        
        # Precomputed Results
        precomp = results.get('precomputed_orb', {})
        print(f"\nPrecomputed ORB Matching:")
        print(f"  Score: {precomp.get('score', 0):.3f} (threshold: {precomp.get('threshold', 0):.3f})")
        print(f"  Matched: {precomp.get('matched', False)}")
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        any_matched = (orb.get('matched', False) or 
                      hist.get('score', 0) >= hist.get('threshold', 0) or
                      precomp.get('matched', False))
        if any_matched:
            print("✓ At least one matching method succeeded!")
        else:
            print("❌ No matching method succeeded.")
            print("\nSuggested adjustments to MATCHING_CONFIG:")
            print(f"  - Increase 'orb_match_threshold' from {orb.get('threshold')} to {orb.get('score') + 0.05:.3f}")
            print(f"  - Decrease 'hist_threshold' from {hist.get('threshold')} to {hist.get('score') - 0.05:.3f}")
            print("\nThen restart the server and test again.")
        
        print("\n✓ Test completed successfully!")
        return True
    
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to server at {server_url}")
        print("   Make sure Flask app is running: python app.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python test_debug_match.py <image1_path> <image2_path> [server_url]")
        print("\nExample:")
        print("  python test_debug_match.py static/uploads/user1.png static/uploads/user2.png")
        print("  python test_debug_match.py user1.png user1.png http://localhost:5000")
        sys.exit(1)
    
    image1 = sys.argv[1]
    image2 = sys.argv[2]
    server = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:5000"
    
    success = test_debug_match(image1, image2, server)
    sys.exit(0 if success else 1)

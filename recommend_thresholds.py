#!/usr/bin/env python3
"""
Threshold recommendation script.
Analyzes matching scores and suggests optimal MATCHING_CONFIG values.
"""
import sys

def recommend_thresholds(orb_score, hist_score, print_details=True):
    """
    Recommend MATCHING_CONFIG adjustments based on observed scores.
    
    Args:
        orb_score: ORB matching score (0.0-1.0)
        hist_score: Histogram matching score (0.0-1.0)
        print_details: Whether to print detailed analysis
    
    Returns:
        dict: Recommended MATCHING_CONFIG values
    """
    
    current = {
        'orb_ratio_test': 0.70,
        'orb_match_threshold': 0.08,
        'hist_threshold': 0.45,
        'orb_nfeatures': 500,
    }
    
    recommended = current.copy()
    
    if print_details:
        print("=" * 70)
        print("THRESHOLD RECOMMENDATION ANALYSIS")
        print("=" * 70)
        print(f"\nObserved Scores:")
        print(f"  ORB Score: {orb_score:.3f} (current threshold: {current['orb_match_threshold']:.3f})")
        print(f"  Histogram Score: {hist_score:.3f} (current threshold: {current['hist_threshold']:.3f})")
    
    # Analyze ORB
    orb_gap = current['orb_match_threshold'] - orb_score
    if orb_gap > 0.02:
        print(f"\n⚠ ORB is FAILING (gap of {orb_gap:.3f})")
        print(f"  Current threshold {current['orb_match_threshold']:.3f} is too strict for score {orb_score:.3f}")
        # Recommend adjustment
        new_thresh = min(orb_score + 0.03, 0.20)  # Don't go above 0.20
        recommended['orb_match_threshold'] = new_thresh
        print(f"  → Recommend increasing 'orb_match_threshold' to {new_thresh:.3f}")
        
        # Also try increasing ratio_test
        if current['orb_ratio_test'] < 0.80:
            recommended['orb_ratio_test'] = min(current['orb_ratio_test'] + 0.05, 0.85)
            print(f"  → Also try increasing 'orb_ratio_test' to {recommended['orb_ratio_test']:.3f}")
        
        # Maybe more features help
        if current['orb_nfeatures'] < 800:
            recommended['orb_nfeatures'] = 800
            print(f"  → Try increasing 'orb_nfeatures' to {recommended['orb_nfeatures']}")
    elif orb_gap > 0:
        print(f"\n⚠ ORB is BORDERLINE (gap of {orb_gap:.3f})")
        print(f"  Score {orb_score:.3f} is close to threshold {current['orb_match_threshold']:.3f}")
        new_thresh = min(orb_score + 0.01, 0.15)
        recommended['orb_match_threshold'] = new_thresh
        print(f"  → Recommend slightly increasing 'orb_match_threshold' to {new_thresh:.3f}")
    else:
        print(f"\n✓ ORB is PASSING (score {orb_score:.3f} >= threshold {current['orb_match_threshold']:.3f})")
    
    # Analyze Histogram
    hist_gap = current['hist_threshold'] - hist_score
    if hist_gap > 0.05:
        print(f"\n⚠ HISTOGRAM is FAILING (gap of {hist_gap:.3f})")
        print(f"  Current threshold {current['hist_threshold']:.3f} is too strict for score {hist_score:.3f}")
        new_thresh = max(hist_score - 0.03, 0.25)  # Don't go below 0.25
        recommended['hist_threshold'] = new_thresh
        print(f"  → Recommend decreasing 'hist_threshold' to {new_thresh:.3f}")
    elif hist_gap > 0:
        print(f"\n⚠ HISTOGRAM is BORDERLINE (gap of {hist_gap:.3f})")
        print(f"  Score {hist_score:.3f} is close to threshold {current['hist_threshold']:.3f}")
        new_thresh = max(hist_score - 0.01, 0.30)
        recommended['hist_threshold'] = new_thresh
        print(f"  → Recommend slightly decreasing 'hist_threshold' to {new_thresh:.3f}")
    else:
        print(f"\n✓ HISTOGRAM is PASSING (score {hist_score:.3f} >= threshold {current['hist_threshold']:.3f})")
    
    # Summary
    print("\n" + "=" * 70)
    print("RECOMMENDED MATCHING_CONFIG")
    print("=" * 70)
    print("\nCurrent:")
    for k, v in current.items():
        marker = " (unchanged)" if v == recommended[k] else " ← CHANGED"
        print(f"  '{k}': {v}{marker}")
    
    print("\nRecommended:")
    for k, v in recommended.items():
        if v != current[k]:
            print(f"  '{k}': {v},  # was {current[k]}")
        else:
            print(f"  '{k}': {v},")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. Edit app.py and update MATCHING_CONFIG with the recommended values
2. Save the file
3. Restart the server: python app.py
4. Test registration and login again
5. Check terminal output for matching scores
6. If still not working, run this script again with the new scores
""")
    
    return recommended

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python recommend_thresholds.py <orb_score> <hist_score>")
        print("\nExample:")
        print("  python recommend_thresholds.py 0.05 0.62")
        print("\nYou can get these scores from:")
        print("  1. Running: python test_debug_match.py <img1> <img2>")
        print("  2. Checking terminal output during login")
        sys.exit(1)
    
    try:
        orb = float(sys.argv[1])
        hist = float(sys.argv[2])
        
        if not (0 <= orb <= 1 and 0 <= hist <= 1):
            print("Error: Scores must be between 0 and 1")
            sys.exit(1)
        
        recommend_thresholds(orb, hist, print_details=True)
    
    except ValueError:
        print(f"Error: Could not parse scores as floats")
        print(f"  Got: '{sys.argv[1]}' and '{sys.argv[2]}'")
        sys.exit(1)

# Face Recognition Accuracy Tuning Guide

## Problem
The face recognition is not accurate enough. Users are having trouble logging in with face recognition.

## Solution: Calibrate Thresholds

The matching accuracy depends on 3 configurable thresholds in `app.py`:

```python
MATCHING_CONFIG = {
    'orb_ratio_test': 0.70,           # ORB Lowe's ratio test (0.0-1.0, lower=stricter)
    'orb_match_threshold': 0.08,      # Min fraction of good matches (0.0-1.0, lower=stricter)
    'hist_threshold': 0.45,            # Histogram correlation threshold (0.0-1.0, lower=more lenient)
    'orb_nfeatures': 500,              # Number of ORB features to detect
}
```

### Current Defaults
- `orb_ratio_test: 0.70` — Stricter Lowe's ratio test
- `orb_match_threshold: 0.08` — Very strict good-match fraction (8% of keypoints)
- `hist_threshold: 0.45` — Moderate histogram threshold
- `orb_nfeatures: 500` — Standard number of features

## Step 1: Use the Debug Endpoint to Test Scores

The `/debug-match` endpoint lets you upload two images and see exact matching scores.

### How to Use

1. **Start the server:**
   ```bash
   python app.py
   ```

2. **Upload two images to test matching:**
   ```bash
   # Using curl (Windows PowerShell):
   $form = @{
       image1 = Get-Item -Path "path/to/image1.png"
       image2 = Get-Item -Path "path/to/image2.png"
   }
   Invoke-WebRequest -Uri "http://127.0.0.1:5000/debug-match" -Method POST -Form $form
   ```

   Or use Postman:
   - POST to: `http://127.0.0.1:5000/debug-match`
   - Select "form-data"
   - Add two files: `image1` and `image2`
   - Send

3. **Review the response:**
   ```json
   {
     "config": {
       "orb_ratio_test": 0.70,
       "orb_match_threshold": 0.08,
       "hist_threshold": 0.45,
       "orb_nfeatures": 500
     },
     "results": {
       "orb": {
         "matched": false,
         "score": 0.05,
         "threshold": 0.08
       },
       "histogram": {
         "score": 0.62,
         "threshold": 0.45
       },
       "precomputed_orb": {
         "matched": false,
         "score": 0.04,
         "threshold": 0.08
       }
     },
     "recommendation": "Adjust MATCHING_CONFIG thresholds based on these scores..."
   }
   ```

## Step 2: Interpret the Scores

- **ORB matched**: If `false` but `score` is close to `threshold`, increase `orb_match_threshold`
- **Histogram score**: If less than `hist_threshold`, histogram is the bottleneck
- **Both low scores**: Face detection ROI may be inconsistent; try increasing `orb_nfeatures`

## Step 3: Adjust Thresholds

Edit `app.py` and update `MATCHING_CONFIG`:

```python
MATCHING_CONFIG = {
    'orb_ratio_test': 0.75,           # Increase to 0.75-0.80 (more lenient)
    'orb_match_threshold': 0.12,      # Increase to 0.12-0.15 (accept lower scores)
    'hist_threshold': 0.40,            # Decrease to 0.35-0.40 (more lenient)
    'orb_nfeatures': 800,              # Increase to 800-1000 (more features)
}
```

### Tuning Strategy

**If ORB is failing (score too low):**
- Increase `orb_match_threshold` to 0.10-0.15
- Increase `orb_ratio_test` to 0.75-0.80
- Increase `orb_nfeatures` to 800-1000

**If histogram is failing (score too low):**
- Decrease `hist_threshold` to 0.35-0.40

**If both are failing:**
- Both thresholds are too strict
- Try increasing both `orb_match_threshold` and decreasing `hist_threshold`

## Step 4: Test Registration → Login Flow

1. Register a new student with face capture
2. Check the terminal: `ORB descriptors computed and saved for {username}...`
3. Try logging in with face recognition
4. Check the terminal for matching scores: `Precomputed ORB match {username}: matched={matched}, score={score}`
5. If still not matching, adjust thresholds and restart the server

## Recommended Thresholds by Scenario

### Scenario A: Very Lenient (accept more faces, risk false positives)
```python
MATCHING_CONFIG = {
    'orb_ratio_test': 0.85,
    'orb_match_threshold': 0.18,
    'hist_threshold': 0.30,
    'orb_nfeatures': 1000,
}
```

### Scenario B: Balanced (default, reasonable accuracy)
```python
MATCHING_CONFIG = {
    'orb_ratio_test': 0.75,
    'orb_match_threshold': 0.12,
    'hist_threshold': 0.40,
    'orb_nfeatures': 500,
}
```

### Scenario C: Very Strict (reject questionable matches, risk false negatives)
```python
MATCHING_CONFIG = {
    'orb_ratio_test': 0.60,
    'orb_match_threshold': 0.05,
    'hist_threshold': 0.50,
    'orb_nfeatures': 500,
}
```

## Terminal Output Examples

**Successful precomputed match:**
```
Precomputed ORB match john_smith: matched=True, score=0.35
```

**Failed precomputed match (score too low):**
```
Precomputed ORB match john_smith: matched=False, score=0.06
```
→ Increase `orb_match_threshold` to accept score 0.06

**Successful histogram fallback:**
```
OpenCV fallback compare john_smith: matched=True, score=0.62
```

## Quick Adjustment Checklist

- [ ] Register a test user
- [ ] Use `/debug-match` to compare registration photo with a test photo
- [ ] Record the ORB and histogram scores
- [ ] If scores < thresholds, adjust thresholds accordingly
- [ ] Restart server (`python app.py`)
- [ ] Test login again
- [ ] Check terminal for matching scores
- [ ] Repeat until satisfied with accuracy

## Next Steps (if still inaccurate)

If even after threshold tuning the system is still inaccurate:

1. **Try precomputing face_recognition encodings** (Option 2) — may be more accurate if dlib/face_recognition gets fixed
2. **Reinstall face_recognition/dlib** (Option 4) — the original algorithm may work better if environment is fixed
3. **Use a different algorithm** — consider integrating MediaPipe or TensorFlow-based face matching

---
**Note:** The `/debug-match` endpoint is for testing only. Don't leave it enabled in production.

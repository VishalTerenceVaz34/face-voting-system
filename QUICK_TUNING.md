# Quick Start: Improve Face Recognition Accuracy

## What Changed
✅ Added configurable thresholds in `MATCHING_CONFIG`
✅ Added `/debug-match` endpoint to test matching scores
✅ Added tuning guide with recommended values

## Current Thresholds
```python
MATCHING_CONFIG = {
    'orb_ratio_test': 0.70,
    'orb_match_threshold': 0.08,      # ← Main threshold to adjust
    'hist_threshold': 0.45,            # ← Secondary threshold
    'orb_nfeatures': 500,
}
```

## 3-Step Tuning Process

### Step 1: Test Current Matching Scores
```bash
# Terminal 1: Start the server
python app.py

# Terminal 2: Test matching
python test_debug_match.py static/uploads/user1.png static/uploads/user1.png
```

Expected output:
```
ORB Matching:
  Score: 0.05 (threshold: 0.08) ← Score is below threshold!
Histogram Matching:
  Score: 0.62 (threshold: 0.45) ✓
```

### Step 2: Identify Which Threshold to Adjust
- **ORB score too low**: Increase `orb_match_threshold` (e.g., 0.08 → 0.12)
- **Histogram score too low**: Decrease `hist_threshold` (e.g., 0.45 → 0.35)

### Step 3: Update and Restart
Edit `app.py`, find `MATCHING_CONFIG`, and adjust:
```python
MATCHING_CONFIG = {
    'orb_ratio_test': 0.75,           # Increase from 0.70
    'orb_match_threshold': 0.12,      # Increase from 0.08 ← make more lenient
    'hist_threshold': 0.40,            # Decrease from 0.45 ← make more lenient
    'orb_nfeatures': 500,
}
```

Save, restart server: `python app.py`

## Test Full Flow

1. **Register** a new student with face capture
   - Terminal will show: `ORB descriptors computed and saved for {username}...`

2. **Login** with face recognition
   - Terminal will show: `Precomputed ORB match {username}: matched=True, score=0.25`

3. **If still not matching**:
   - Go back to Step 1, adjust thresholds more, restart, test again

## Tuning Tips

| Problem | Solution |
|---------|----------|
| ORB score 0.05, threshold 0.08 | Increase `orb_match_threshold` to 0.10 |
| Histogram score 0.30, threshold 0.45 | Decrease `hist_threshold` to 0.30 |
| All scores low (< 0.10) | Increase `orb_nfeatures` to 800-1000 |
| Too many false matches | Decrease thresholds (make stricter) |

## Files
- `app.py` — Main application (edit `MATCHING_CONFIG` here)
- `TUNING_GUIDE.md` — Detailed tuning guide
- `test_debug_match.py` — Debug script to test matching
- `static/uploads/descriptors/` — Stored ORB descriptors (created at registration)

## Common Adjustments

**Start here (more lenient):**
```python
MATCHING_CONFIG = {
    'orb_ratio_test': 0.75,
    'orb_match_threshold': 0.12,      # More lenient
    'hist_threshold': 0.40,            # More lenient
    'orb_nfeatures': 800,              # More features
}
```

**If too many false matches, make stricter:**
```python
MATCHING_CONFIG = {
    'orb_ratio_test': 0.65,
    'orb_match_threshold': 0.05,      # Stricter
    'hist_threshold': 0.50,            # Stricter
    'orb_nfeatures': 500,              # Fewer features
}
```

---
**Questions?** Check `TUNING_GUIDE.md` for detailed explanation.

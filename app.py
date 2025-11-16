from flask import Flask, request, jsonify, render_template, session, send_file
from flask_session import Session
import os
import sqlite3
import base64
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import csv
import random
import string
import cv2
import face_recognition
import numpy as np
from werkzeug.utils import secure_filename
from io import BytesIO, StringIO
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
DATABASE = 'Election.db'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64 MB limit

CANDIDATE_UPLOAD_FOLDER = os.path.join(app.config['UPLOAD_FOLDER'], 'candidates')
if not os.path.exists(CANDIDATE_UPLOAD_FOLDER):
    os.makedirs(CANDIDATE_UPLOAD_FOLDER)

DESCRIPTORS_FOLDER = os.path.join(app.config['UPLOAD_FOLDER'], 'descriptors')
if not os.path.exists(DESCRIPTORS_FOLDER):
    os.makedirs(DESCRIPTORS_FOLDER)

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# ============= ELECTION CONFIGURATION =============
ELECTION_CONFIG = {
    'enable_countdown_timer': True,
    'election_end_time': None,  # Will be fetched from DB or set manually (ISO format string)
    'enable_email_notifications': True,
    'enable_2fa': True,
    'enable_audit_logging': True,
    'enable_export': True,
}

# ============= SMTP CONFIGURATION FOR EMAIL =============
SMTP_CONFIG = {
    'server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
    'port': int(os.getenv('SMTP_PORT', '587')),
    'sender_email': os.getenv('EMAIL_USER', ''),
    'sender_password': os.getenv('EMAIL_PASSWORD', ''),
}

def send_email(recipient, subject, body, is_html=False):
    """Send email notification"""
    if not ELECTION_CONFIG['enable_email_notifications']:
        return False
    
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_CONFIG['sender_email']
        msg['To'] = recipient
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'html' if is_html else 'plain'))
        
        server = smtplib.SMTP(SMTP_CONFIG['server'], SMTP_CONFIG['port'])
        server.starttls()
        server.login(SMTP_CONFIG['sender_email'], SMTP_CONFIG['sender_password'])
        server.send_message(msg)
        server.quit()
        
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False

# ============= TUNABLE MATCHING THRESHOLDS =============
# These can be adjusted to improve accuracy. Lower thresholds = more matches (but more false positives).
# Higher thresholds = stricter (more false negatives but fewer false positives).
MATCHING_CONFIG = {
    'orb_ratio_test': 0.75,           # Slightly more lenient ratio test
    'orb_match_threshold': 0.05,      # More lenient to account for lighting variance
    'hist_threshold': 0.35,            # Slightly more lenient
    'orb_nfeatures': 800,              # Number of ORB features to detect
    'sift_match_threshold': 0.15,      # SIFT match threshold (higher = more lenient)
    'ensemble_threshold': 0.55,        # Ensemble voting threshold (weighted score)
    'blur_threshold': 100.0,           # Laplacian variance threshold for blur detection
    'enable_sift': False,              # Disable SIFT by default (can be noisy in some environments)
    'enable_ensemble': True,           # Enable ensemble voting
    # Stricter acceptance controls to avoid false positives
    'ensemble_strict_threshold': 0.39, # Slightly relaxed to accommodate current environment
    'ensemble_margin': 0.03,           # Slightly smaller margin while still requiring separation
    'min_methods_agree': 2,            # Require at least N methods (of 3) to agree
    'require_single_face': True,       # Reject if multiple faces detected
    'enable_hog': True,                # Include HOG similarity in ensemble
    'hog_threshold': 0.50,             # HOG cosine similarity threshold (0..1)
    'enable_lbph': True,               # Include LBPH similarity if available
    'lbph_scale': 80.0,                # Scale for converting LBPH distance to similarity
}
# ======================================================

# Prepare OpenCV Haar cascade for fallback face detection
try:
    FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
except Exception:
    face_cascade = None

def detect_face_roi_from_np(img_np, size=(128,128)):
    """Detect the largest face in an RGB numpy image and return a resized grayscale ROI or None."""
    if face_cascade is None:
        return None
    try:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    except Exception:
        try:
            gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        except Exception:
            return None
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    # choose largest face
    faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    x,y,w,h = faces[0]
    roi = gray[y:y+h, x:x+w]
    try:
        roi = cv2.resize(roi, size)
    except Exception:
        return None
    return roi

def count_faces(img_np):
    if face_cascade is None:
        return 1
    try:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    except Exception:
        try:
            gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        except Exception:
            return 0
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces)

def is_image_blurry(img_gray, threshold=None):
    """Detect if image is blurry using Laplacian variance.
    Returns True if blurry (variance < threshold), False if sharp.
    """
    if threshold is None:
        threshold = MATCHING_CONFIG.get('blur_threshold', 100.0)
    
    if img_gray is None or img_gray.size == 0:
        return True
    
    try:
        laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        is_blur = laplacian_var < threshold
        return is_blur
    except Exception:
        return False

def enhance_image_contrast(roi):
    """Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Returns enhanced ROI.
    """
    if roi is None or roi.size == 0:
        return roi
    
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(roi)
        return enhanced
    except Exception:
        return roi

def sift_match(img_np, known_image_path, threshold=None):
    """Use SIFT keypoint matching between images. Returns (matched, score).
    SIFT is more robust but slower than ORB.
    """
    if threshold is None:
        threshold = MATCHING_CONFIG.get('sift_match_threshold', 0.15)
    
    try:
        if not cv2.SIFT_create:
            return False, 0.0  # SIFT not available
    except Exception:
        return False, 0.0
    
    try:
        if not os.path.exists(known_image_path):
            return False, 0.0
        
        known_bgr = cv2.imread(known_image_path)
        if known_bgr is None:
            return False, 0.0
        
        known_rgb = cv2.cvtColor(known_bgr, cv2.COLOR_BGR2RGB)
        roi_known = detect_face_roi_from_np(known_rgb)
        roi_unknown = detect_face_roi_from_np(img_np)
        
        if roi_known is None or roi_unknown is None:
            return False, 0.0
        
        # Enhance contrast for better SIFT detection
        roi_known = enhance_image_contrast(roi_known)
        roi_unknown = enhance_image_contrast(roi_unknown)
        
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(roi_known, None)
        kp2, des2 = sift.detectAndCompute(roi_unknown, None)
        
        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            return False, 0.0
        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test
        good = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < threshold * n.distance:
                    good.append(m)
        
        score = len(good) / max(len(des1), len(des2)) if good else 0.0
        matched = score >= 0.10  # SIFT threshold
        return matched, float(score)
    
    except Exception as e:
        return False, 0.0

def hist_similarity(roi1, roi2):
    """Return histogram correlation between two grayscale ROIs (0..1 higher is more similar)."""
    if roi1 is None or roi2 is None:
        return 0.0
    h1 = cv2.calcHist([roi1], [0], None, [256], [0,256])
    h2 = cv2.calcHist([roi2], [0], None, [256], [0,256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    # correlation in [-1,1]
    score = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    # clamp to 0..1
    return max(0.0, float(score))

def hog_similarity(roi1, roi2):
    """Compute cosine similarity between HOG descriptors of two ROIs. Returns 0..1."""
    try:
        if roi1 is None or roi2 is None:
            return 0.0
        # HOG expects specific window size; resize to 64x128 (WxH)
        roi1r = cv2.resize(roi1, (64, 128))
        roi2r = cv2.resize(roi2, (64, 128))

        hog = cv2.HOGDescriptor(_winSize=(64, 128),
                                 _blockSize=(16, 16),
                                 _blockStride=(8, 8),
                                 _cellSize=(8, 8),
                                 _nbins=9)
        d1 = hog.compute(roi1r)
        d2 = hog.compute(roi2r)
        if d1 is None or d2 is None:
            return 0.0
        a = d1.reshape(-1).astype(np.float32)
        b = d2.reshape(-1).astype(np.float32)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        sim = float(np.dot(a, b) / (na * nb))
        return max(0.0, min(1.0, sim))
    except Exception:
        return 0.0

def template_similarity(roi1, roi2):
    """Normalized cross-correlation between two equal-sized ROIs (0..1)."""
    try:
        if roi1 is None or roi2 is None:
            return 0.0
        # Ensure same size
        if roi1.shape != roi2.shape:
            roi2 = cv2.resize(roi2, (roi1.shape[1], roi1.shape[0]))
        # matchTemplate expects uint8
        a = roi1.astype(np.uint8)
        b = roi2.astype(np.uint8)
        res = cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)
        # Single value because images same size
        sim = float(res[0][0]) if res.size == 1 else float(res.max())
        return max(0.0, min(1.0, sim))
    except Exception:
        return 0.0

def opencv_orb_match(img_np, known_image_path, ratio_test=None, match_threshold=None):
    """Use ORB keypoint matching between unknown img_np and known_image_path. Returns (matched, score).
    score is fraction of good matches over max(keypoints_count).
    Uses MATCHING_CONFIG defaults if thresholds not provided.
    """
    if ratio_test is None:
        ratio_test = MATCHING_CONFIG['orb_ratio_test']
    if match_threshold is None:
        match_threshold = MATCHING_CONFIG['orb_match_threshold']
    
    try:
        if not os.path.exists(known_image_path):
            return False, 0.0
        known_bgr = cv2.imread(known_image_path)
        if known_bgr is None:
            return False, 0.0

        # Convert both images to RGB and get face ROIs
        known_rgb = cv2.cvtColor(known_bgr, cv2.COLOR_BGR2RGB)
        roi_known = detect_face_roi_from_np(known_rgb)
        roi_unknown = detect_face_roi_from_np(img_np)
        if roi_known is None or roi_unknown is None:
            return False, 0.0

        # Enhance contrast for robustness
        roi_known = enhance_image_contrast(roi_known)
        roi_unknown = enhance_image_contrast(roi_unknown)

        # ORB works on grayscale images (we already have grayscale ROIs)
        orb = cv2.ORB_create(nfeatures=MATCHING_CONFIG['orb_nfeatures'])
        kp1, des1 = orb.detectAndCompute(roi_known, None)
        kp2, des2 = orb.detectAndCompute(roi_unknown, None)
        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            return False, 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2, k=2)
        # ratio test
        good = []
        for m, n in matches:
            if m.distance < ratio_test * n.distance:
                good.append(m)

        score = len(good) / max(len(des1), len(des2))
        return (score >= match_threshold), float(score)
    except Exception:
        return False, 0.0

def opencv_fallback_match(img_np, known_image_path, thresh=None):
    """Try to match the face in img_np against known_image_path using OpenCV; return (matched, score).
    First try ORB matching (more robust). If ORB can't run or returns low score, fall back to histogram.
    Uses MATCHING_CONFIG defaults if thresh not provided.
    """
    if thresh is None:
        thresh = MATCHING_CONFIG['hist_threshold']
    
    # Try ORB matching first with config thresholds
    matched, score = opencv_orb_match(img_np, known_image_path)
    if matched:
        return True, score

    # If ORB didn't match, try histogram similarity (legacy)
    try:
        if not os.path.exists(known_image_path):
            return False, 0.0
        known_bgr = cv2.imread(known_image_path)
        if known_bgr is None:
            return False, 0.0
        known_rgb = cv2.cvtColor(known_bgr, cv2.COLOR_BGR2RGB)
        roi_known = detect_face_roi_from_np(known_rgb)
        roi_unknown = detect_face_roi_from_np(img_np)
        if roi_known is None or roi_unknown is None:
            return False, 0.0
        # Enhance contrast before histogram
        roi_known = enhance_image_contrast(roi_known)
        roi_unknown = enhance_image_contrast(roi_unknown)
        score = hist_similarity(roi_known, roi_unknown)
        return (score >= thresh), score
    except Exception:
        return False, 0.0

def ensemble_match(img_np, known_image_path):
    """Ensemble voting: combine ORB, SIFT, histogram, and optional HOG matching.
    Returns (matched, ensemble_score, method_details).
    """
    results = {}
    
    # Method 1: ORB matching
    try:
        orb_matched, orb_score = opencv_orb_match(img_np, known_image_path)
        results['orb'] = {'matched': orb_matched, 'score': orb_score, 'weight': 0.30}
    except Exception:
        results['orb'] = {'matched': False, 'score': 0.0, 'weight': 0.30}
    
    # Method 2: SIFT matching (if enabled and available)
    if MATCHING_CONFIG.get('enable_sift', False):
        try:
            sift_matched, sift_score = sift_match(img_np, known_image_path)
            results['sift'] = {'matched': sift_matched, 'score': sift_score, 'weight': 0.10}
        except Exception:
            results['sift'] = {'matched': False, 'score': 0.0, 'weight': 0.10}
    else:
        results['sift'] = {'matched': False, 'score': 0.0, 'weight': 0.0}
    
    # Method 3: Histogram similarity
    try:
        if not os.path.exists(known_image_path):
            hist_score = 0.0
        else:
            known_bgr = cv2.imread(known_image_path)
            known_rgb = cv2.cvtColor(known_bgr, cv2.COLOR_BGR2RGB)
            roi_known = detect_face_roi_from_np(known_rgb)
            roi_unknown = detect_face_roi_from_np(img_np)
            if roi_known is not None and roi_unknown is not None:
                hist_score = hist_similarity(roi_known, roi_unknown)
            else:
                hist_score = 0.0
        results['histogram'] = {'matched': hist_score >= MATCHING_CONFIG['hist_threshold'], 'score': hist_score, 'weight': 0.25}
    except Exception:
        results['histogram'] = {'matched': False, 'score': 0.0, 'weight': 0.25}

    # Method 4: HOG similarity (optional)
    if MATCHING_CONFIG.get('enable_hog', True):
        try:
            if os.path.exists(known_image_path):
                known_bgr = cv2.imread(known_image_path)
                known_rgb = cv2.cvtColor(known_bgr, cv2.COLOR_BGR2RGB)
                roi_known = detect_face_roi_from_np(known_rgb)
                roi_unknown = detect_face_roi_from_np(img_np)
                hog_score = hog_similarity(roi_unknown, roi_known)
            else:
                hog_score = 0.0
            results['hog'] = {'matched': hog_score >= MATCHING_CONFIG.get('hog_threshold', 0.55), 'score': hog_score, 'weight': 0.15}
        except Exception:
            results['hog'] = {'matched': False, 'score': 0.0, 'weight': 0.15}
    else:
        results['hog'] = {'matched': False, 'score': 0.0, 'weight': 0.0}

    # Method 5: LBPH similarity (optional)
    if MATCHING_CONFIG.get('enable_lbph', True):
        try:
            lbph_score = 0.0
            if os.path.exists(known_image_path) and face_cascade is not None and hasattr(cv2, 'face') and hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
                known_bgr = cv2.imread(known_image_path)
                known_rgb = cv2.cvtColor(known_bgr, cv2.COLOR_BGR2RGB)
                roi_known = detect_face_roi_from_np(known_rgb)
                roi_unknown = detect_face_roi_from_np(img_np)
                if roi_known is not None and roi_unknown is not None:
                    # Train LBPH on the single known face and score the unknown
                    model = cv2.face.LBPHFaceRecognizer_create()
                    model.train([roi_known], np.array([0]))
                    label, distance = model.predict(roi_unknown)
                    scale = float(MATCHING_CONFIG.get('lbph_scale', 80.0))
                    lbph_score = float(np.exp(-distance / scale))  # map distance to 0..1
            results['lbph'] = {'matched': lbph_score >= 0.50, 'score': lbph_score, 'weight': 0.10}
        except Exception:
            results['lbph'] = {'matched': False, 'score': 0.0, 'weight': 0.10}
    else:
        results['lbph'] = {'matched': False, 'score': 0.0, 'weight': 0.0}

    # Method 6: Template NCC similarity
    try:
        if os.path.exists(known_image_path):
            known_bgr = cv2.imread(known_image_path)
            known_rgb = cv2.cvtColor(known_bgr, cv2.COLOR_BGR2RGB)
            roi_known = detect_face_roi_from_np(known_rgb)
            roi_unknown = detect_face_roi_from_np(img_np)
            # Enhance contrast slightly for template matching
            if roi_known is not None and roi_unknown is not None:
                roi_known_e = enhance_image_contrast(roi_known)
                roi_unknown_e = enhance_image_contrast(roi_unknown)
                ncc = template_similarity(roi_unknown_e, roi_known_e)
            else:
                ncc = 0.0
        else:
            ncc = 0.0
        results['template'] = {'matched': ncc >= 0.50, 'score': ncc, 'weight': 0.25}
    except Exception:
        results['template'] = {'matched': False, 'score': 0.0, 'weight': 0.25}
    
    # Method 7: Precomputed ORB descriptors (if available)
    try:
        candidate_base = os.path.splitext(os.path.basename(known_image_path))[0]
        descriptor_path = os.path.join(DESCRIPTORS_FOLDER, f"{candidate_base}_orb.npy")
        if os.path.exists(descriptor_path):
            pre_matched, pre_score = match_against_precomputed_descriptors(img_np, descriptor_path)
            results['precomp_orb'] = {'matched': pre_matched, 'score': pre_score, 'weight': 0.25}
        else:
            results['precomp_orb'] = {'matched': False, 'score': 0.0, 'weight': 0.25}
    except Exception:
        results['precomp_orb'] = {'matched': False, 'score': 0.0, 'weight': 0.25}

    # Weighted voting
    total_weight = sum(r['weight'] for r in results.values())
    if total_weight > 0:
        weighted_score = sum(r['score'] * r['weight'] for r in results.values()) / total_weight
    else:
        weighted_score = 0.0
    
    ensemble_threshold = MATCHING_CONFIG.get('ensemble_threshold', 0.55)
    ensemble_matched = weighted_score >= ensemble_threshold
    
    return ensemble_matched, weighted_score, results

def compute_and_save_orb_descriptors(image_path, descriptor_path, size=(128, 128)):
    """Compute ORB descriptors from a face image and save as .npy.
    Returns (keypoints, descriptors) if successful, (None, None) otherwise.
    """
    try:
        if not os.path.exists(image_path):
            return None, None
        bgr = cv2.imread(image_path)
        if bgr is None:
            return None, None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        roi = detect_face_roi_from_np(rgb, size=size)
        if roi is None:
            return None, None
        
        orb = cv2.ORB_create(nfeatures=500)
        kp, des = orb.detectAndCompute(roi, None)
        if des is None:
            return None, None
        
        # Save descriptors to .npy file
        np.save(descriptor_path, des)
        return kp, des
    except Exception as e:
        print(f"Error computing ORB descriptors for {image_path}: {e}")
        return None, None

def match_against_precomputed_descriptors(img_np, descriptor_path, ratio_test=None, match_threshold=None):
    """Match img_np ROI against precomputed descriptors loaded from descriptor_path (.npy).
    Returns (matched, score).
    Uses MATCHING_CONFIG defaults if thresholds not provided.
    """
    if ratio_test is None:
        ratio_test = MATCHING_CONFIG['orb_ratio_test']
    if match_threshold is None:
        match_threshold = MATCHING_CONFIG['orb_match_threshold']
    
    try:
        if not os.path.exists(descriptor_path):
            return False, 0.0
        
        # Load precomputed descriptors
        des_known = np.load(descriptor_path)
        
        # Compute descriptors for the input image
        roi_unknown = detect_face_roi_from_np(img_np)
        if roi_unknown is None:
            return False, 0.0
        
        orb = cv2.ORB_create(nfeatures=MATCHING_CONFIG['orb_nfeatures'])
        kp2, des2 = orb.detectAndCompute(roi_unknown, None)
        if des2 is None or len(des2) < 4:
            return False, 0.0
        
        # Match
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des_known, des2, k=2)
        
        # Ratio test
        good = []
        for m, n in matches:
            if m.distance < ratio_test * n.distance:
                good.append(m)
        
        score = len(good) / max(len(des_known), len(des2))
        return (score >= match_threshold), float(score)
    except Exception as e:
        print(f"Error matching against precomputed descriptors: {e}")
        return False, 0.0

# Initialize database
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Student_details (
            Name TEXT,
            Roll_Number TEXT,
            Class TEXT,
            Email TEXT,
            Username TEXT UNIQUE,
            Password TEXT,
            Photo TEXT
        )
    ''')
    conn.commit()
    conn.close()

def init_result():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS Result (
                Name TEXT,
                Photo TEXT,
                Class TEXT,
                Role TEXT,
                Votes INTEGER
            )
        ''')
    conn.commit()
    conn.close()





def init_candidate_table():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Candidate (
            Name TEXT,
            Class TEXT,
            Photo TEXT,
            Description TEXT,
            Role TEXT
        )
    ''')
    # Ensure Role column exists (migration-safe)
    try:
        cursor.execute("PRAGMA table_info(Candidate)")
        cols = [r[1] for r in cursor.fetchall()]
        if 'Role' not in cols:
            cursor.execute("ALTER TABLE Candidate ADD COLUMN Role TEXT")
    except Exception:
        pass
    conn.commit()
    conn.close()

def init_vote_table():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Vote (
            Name TEXT,
            Username TEXT,
            Email TEXT,
            Roll_Number TEXT,
            Candidate TEXT,
            Role TEXT
        )
    ''')
    # Ensure Role column and unique index on (Username, Role)
    try:
        cursor.execute("PRAGMA table_info(Vote)")
        cols = [r[1] for r in cursor.fetchall()]
        if 'Role' not in cols:
            cursor.execute("ALTER TABLE Vote ADD COLUMN Role TEXT")
        # Create unique index if not exists
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_vote_user_role ON Vote(Username, Role)")
    except Exception:
        pass
    conn.commit()
    conn.close()

def init_audit_log_table():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            action TEXT,
            username TEXT,
            email TEXT,
            ip_address TEXT,
            details TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_audit_action(action, username='', email='', details=''):
    """Log an action to the audit log"""
    if not ELECTION_CONFIG['enable_audit_logging']:
        return
    
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        ip_address = request.remote_addr if request else 'unknown'
        cursor.execute('''
            INSERT INTO Audit_log (timestamp, action, username, email, ip_address, details)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (timestamp, action, username, email, ip_address, details))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Audit log error: {e}")


@app.route('/register', methods=['POST'])
def register():
    try:
        # Ensure DB/tables exist
        init_db()

        full_name = request.form.get('fullName')
        roll_number = request.form.get('rollNumber')
        class_name = request.form.get('class')
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')

        if not username:
            # Fallback: derive a username if not provided
            if full_name and roll_number:
                username = (full_name[:4].lower() + '_' + roll_number)
            else:
                return jsonify({"success": False, "message": "Username is required."})

        # Prefer file upload (binary blob) to avoid huge base64 payloads
        photo_file = request.files.get('photo')
        photo_path = None

        if photo_file and photo_file.filename:
            # Save uploaded file securely
            filename = secure_filename(f"{username}.png")
            photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename).replace('\\', '/')
            photo_file.save(photo_path)
        else:
            # Fallback to base64 in form (older clients)
            photo_data_url = request.form.get('photoData', '')
            if not photo_data_url or 'base64,' not in photo_data_url:
                return jsonify({"success": False, "message": "Photo is required."})
            header, encoded = photo_data_url.split(',', 1)
            try:
                image_data = base64.b64decode(encoded)
            except Exception:
                return jsonify({"success": False, "message": "Invalid photo data."})
            photo_filename = f"{username}.png"
            photo_path = os.path.join(app.config['UPLOAD_FOLDER'], photo_filename).replace('\\', '/')
            with open(photo_path, 'wb') as f:
                f.write(image_data)

        # Compute and save ORB descriptors for faster login matching
        descriptor_filename = f"{username}_orb.npy"
        descriptor_path = os.path.join(DESCRIPTORS_FOLDER, descriptor_filename)
        try:
            kp, des = compute_and_save_orb_descriptors(photo_path, descriptor_path)
            if des is not None:
                print(f"ORB descriptors computed and saved for {username} at {descriptor_path}")
            else:
                print(f"Warning: Could not compute ORB descriptors for {username}")
        except Exception as e:
            print(f"Error computing ORB descriptors during registration: {e}")

        # Store in database
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO Student_details (Name, Roll_Number, Class, Email, Username, Password, Photo)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (full_name, roll_number, class_name, email, username, password, photo_path))

        conn.commit()
        conn.close()

        # Log registration to terminal so it's easy to verify
        try:
            print(f"Registered user: {username}, photo saved to: {photo_path}")
        except Exception:
            pass

        # Log to audit log
        log_audit_action('REGISTER', username=username, email=email, details=f'Registered user {full_name}')

        # Send registration confirmation email
        if email:
            subject = "Registration Successful - Face Voting System"
            body = f"""
Hello {full_name},

Thank you for registering with the Face Voting System!

Your registration details:
- Username: {username}
- Roll Number: {roll_number}
- Class: {class_name}

You can now log in using your face and password.

Best regards,
Election Administration
"""
            send_email(email, subject, body)

        return jsonify({"success": True})


    except sqlite3.IntegrityError:
        return jsonify({"success": False, "message": "Username already exists."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/admin')
def admin():
    return render_template("Admin.html")

@app.route('/student-login')
def student_login():
    return render_template("student_login.html")

@app.route('/student-registration')
def student_registration():
    return render_template("student_registration.html")

# Utility to check if table exists
def check_table_exists():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Student_details'")
    result = cursor.fetchone()
    conn.close()
    return result is not None

@app.route('/face-login', methods=['POST'])
def face_login():
    if 'photo' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image uploaded'}), 400

    file = request.files['photo']
    

    try:
        img_stream = BytesIO(file.read())
        img = Image.open(img_stream)

        # Ensure image is 8-bit RGB (face_recognition requires 8-bit RGB arrays)
        try:
            img = img.convert("RGB")
        except Exception as e:
            return jsonify({'status': 'error', 'message': 'Unsupported image type, must be 8bit gray or RGB image.'})



        # Preprocess: resize to a standard size to improve face_recognition reliability
        try:
            # Resize to 640x480 (may stretch) — keeps processing consistent and limits memory
            img = img.resize((640, 480))
        except Exception:
            # If resize fails, continue with original image
            pass

        # Save a debug copy so you can inspect what the server received
        try:
            debug_path = os.path.join(app.config['UPLOAD_FOLDER'], 'debug_face.png')
            img.save(debug_path)
        except Exception:
            # ignore debug saving issues
            debug_path = None

        # Convert PIL image to numpy (face_recognition expects an array)
        try:
            # Ensure RGB and convert to numpy
            pil_rgb = img.convert('RGB')
            img_np = np.asarray(pil_rgb)
            # If image has more than 3 channels (shouldn't after convert), slice first 3
            if img_np.ndim == 2:
                img_np = np.stack([img_np] * 3, axis=-1)
            if img_np.shape[-1] > 3:
                img_np = img_np[:, :, :3]
            # Ensure contiguous uint8
            img_np = np.ascontiguousarray(img_np, dtype=np.uint8)
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Failed converting image to array: {str(e)}'})

        # Optional: ensure only one face is present to avoid mismatches
        try:
            if MATCHING_CONFIG.get('require_single_face', True):
                faces_cnt = count_faces(img_np)
                if faces_cnt != 1:
                    return jsonify({'status': 'error', 'message': f'Expected exactly 1 face, detected {faces_cnt}. Please retake photo.'})
        except Exception:
            pass

        # Now detect face with clearer error handling so we can see shape/dtype if it fails
        try:
            unknown_face_encodings = face_recognition.face_encodings(img_np)
        except Exception as e:
            # Provide helpful debug info and attempt an OpenCV fallback match
            info = f"face_recognition error: {str(e)}; img_np.shape={getattr(img_np, 'shape', None)}, dtype={getattr(img_np, 'dtype', None)}"
            print(info)

            # Strict ensemble identification with consensus and margin
            try:
                # Ensure single face if configured
                if MATCHING_CONFIG.get('require_single_face', True):
                    faces_cnt = count_faces(img_np)
                    if faces_cnt != 1:
                        return jsonify({'status': 'error', 'message': f'Expected exactly 1 face, detected {faces_cnt}. Please retake photo.'})

                conn2 = sqlite3.connect(DATABASE)
                cur2 = conn2.cursor()
                cur2.execute("SELECT Name, Roll_Number, Class, Email, Username, Photo FROM Student_details")
                rows = cur2.fetchall()
                conn2.close()

                best_match = None
                best_score = 0.0
                second_best_score = 0.0
                best_details = {}
                best_method = ''

                for student in rows:
                    name, roll_number, class_name, email, username, photo_path = student
                    known_image_path = os.path.join(UPLOAD_FOLDER, os.path.basename(photo_path))
                    matched, ensemble_score, method_details = ensemble_match(img_np, known_image_path)
                    print(f'Ensemble (fallback) {username}: matched={matched}, score={ensemble_score}, details={method_details}')
                    if ensemble_score > best_score:
                        second_best_score = best_score
                        best_match = (name, roll_number, class_name, email, username, photo_path)
                        best_score = ensemble_score
                        best_details = method_details
                        best_method = f'ensemble (score={ensemble_score:.3f})'

                if best_match:
                    methods_agree = sum(1 for v in best_details.values() if v.get('matched'))
                    strict_thr = MATCHING_CONFIG.get('ensemble_strict_threshold', 0.65)
                    margin = MATCHING_CONFIG.get('ensemble_margin', 0.10)
                    min_agree = MATCHING_CONFIG.get('min_methods_agree', 2)
                    if (best_score >= strict_thr) and (best_score - second_best_score >= margin) and (methods_agree >= min_agree):
                        name, roll_number, class_name, email, username, photo_path = best_match
                        session['user'] = {
                            'name': name,
                            'email': email,
                            'roll_number': roll_number,
                            'class': class_name,
                            'username': username,
                            'photo': f'/{photo_path}'
                        }
                        # Update auth flags: mark face verified, preserve password flag if same user
                        prev = session.get('auth', {}) or {}
                        session['auth'] = {
                            'username': username,
                            'password': bool(prev.get('password') and prev.get('username') == username),
                            'face': True
                        }
                        return jsonify({'status': 'success', 'data': session['user'], 'method': best_method, 'score': best_score, 'second_best': second_best_score, 'methods_agree': methods_agree})

                return jsonify({'status': 'error', 'message': 'Face not recognized confidently. Please try again.'})
            except Exception as inner_e:
                print('Ensemble strict error:', inner_e)
                return jsonify({'status': 'error', 'message': 'Image processing failed: ' + str(e) + f" (debug image: {debug_path})"})

        if not unknown_face_encodings:
            # No face detected by face_recognition: try precomputed descriptors or ensemble matching
            try:
                conn2 = sqlite3.connect(DATABASE)
                cur2 = conn2.cursor()
                cur2.execute("SELECT Name, Roll_Number, Class, Email, Username, Photo FROM Student_details")
                rows = cur2.fetchall()
                conn2.close()
                
                best_match = None
                best_score = 0.0
                second_best_score = 0.0
                best_details = {}
                best_method = ''
                
                for student in rows:
                    name, roll_number, class_name, email, username, photo_path = student
                    known_image_path = os.path.join(UPLOAD_FOLDER, os.path.basename(photo_path))
                    
                    # Use ensemble matching if enabled, otherwise fallback
                    if MATCHING_CONFIG.get('enable_ensemble', True):
                        matched, ensemble_score, method_details = ensemble_match(img_np, known_image_path)
                        print(f'Ensemble match {username}: matched={matched}, score={ensemble_score:.3f}')
                        print(f'  Details: {method_details}')
                        
                        if ensemble_score > best_score:
                            second_best_score = best_score
                            best_match = (name, roll_number, class_name, email, username, photo_path)
                            best_score = ensemble_score
                            best_details = method_details
                            best_method = f'ensemble (score={ensemble_score:.3f})'
                    else:
                        # Fallback to simple matching
                        matched, score = opencv_fallback_match(img_np, known_image_path)
                        print(f'OpenCV fallback compare {username}: matched={matched}, score={score}')
                        
                        if matched and score > best_score:
                            best_match = (name, roll_number, class_name, email, username, photo_path)
                            best_score = score
                            best_method = f'fallback (score={score:.3f})'
                
                # Apply strict acceptance criteria
                if best_match:
                    methods_agree = sum(1 for v in best_details.values() if v.get('matched'))
                    strict_thr = MATCHING_CONFIG.get('ensemble_strict_threshold', 0.65)
                    margin = MATCHING_CONFIG.get('ensemble_margin', 0.10)
                    if (best_score >= strict_thr) and (best_score - second_best_score >= margin) and (methods_agree >= MATCHING_CONFIG.get('min_methods_agree', 2)):
                        name, roll_number, class_name, email, username, photo_path = best_match
                        session['user'] = {
                            'name': name,
                            'email': email,
                            'roll_number': roll_number,
                            'class': class_name,
                            'username': username,
                            'photo': f'/{photo_path}'
                        }
                        return jsonify({'status': 'success', 'data': session['user'], 'method': best_method, 'score': best_score, 'second_best': second_best_score, 'methods_agree': methods_agree})
            
            except Exception as inner_e:
                print('Ensemble/precomputed fallback error:', inner_e)

            return jsonify({
                'status': 'error',
                'message': 'No face detected in captured image. (debug image: ' + str(debug_path) + ')',
                'debug_path': debug_path,
                'img_shape': getattr(img_np, 'shape', None),
                'img_dtype': str(getattr(img_np, 'dtype', None))
            })

        unknown_encoding = unknown_face_encodings[0]

    except UnidentifiedImageError:
        return jsonify({'status': 'error', 'message': 'Uploaded image format not supported.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Image processing failed: {str(e)}'})

    # Use ensemble selection even when encodings are present, to avoid false positives
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT Name, Roll_Number, Class, Email, Username, Photo FROM Student_details")
    rows = cursor.fetchall()
    conn.close()

    best_match = None
    best_score = 0.0
    second_best_score = 0.0
    best_details = {}
    best_method = ''

    for student in rows:
        name, roll_number, class_name, email, username, photo_path = student
        known_image_path = os.path.join(UPLOAD_FOLDER, os.path.basename(photo_path))
        try:
            matched, ensemble_score, method_details = ensemble_match(img_np, known_image_path)
            if ensemble_score > best_score:
                second_best_score = best_score
                best_match = (name, roll_number, class_name, email, username, photo_path)
                best_score = ensemble_score
                best_details = method_details
                best_method = f'ensemble (score={ensemble_score:.3f})'
        except Exception:
            continue

    if best_match:
        methods_agree = sum(1 for v in best_details.values() if v.get('matched'))
        strict_thr = MATCHING_CONFIG.get('ensemble_strict_threshold', 0.65)
        margin = MATCHING_CONFIG.get('ensemble_margin', 0.10)
        if (best_score >= strict_thr) and (best_score - second_best_score >= margin) and (methods_agree >= MATCHING_CONFIG.get('min_methods_agree', 2)):
            name, roll_number, class_name, email, username, photo_path = best_match
            session['user'] = {
                'name': name,
                'email': email,
                'roll_number': roll_number,
                'class': class_name,
                'username': username,
                'photo': f'/{photo_path}'
            }
            prev = session.get('auth', {}) or {}
            session['auth'] = {
                'username': username,
                'password': bool(prev.get('password') and prev.get('username') == username),
                'face': True
            }
            # Log successful face login
            log_audit_action('LOGIN_FACE_SUCCESS', username=username, email=email, details=f'Score: {best_score:.3f}, Method: {best_method}')
            return jsonify({'status': 'success', 'data': session['user'], 'method': best_method, 'score': best_score, 'second_best': second_best_score, 'methods_agree': methods_agree})

    # Log failed face login
    log_audit_action('LOGIN_FACE_FAILED', details='Face not recognized or match threshold not met')
    return jsonify({'status': 'error', 'message': 'Face not recognized in records.'})


@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT Name, Roll_Number, Class, Email, Photo FROM Student_details
            WHERE Username = ? AND Password = ?
        ''', (username, password))

        result = cursor.fetchone()
        conn.close()

        if result:
            name, roll_number, class_name, email, photo_path = result
            # Set/merge session user and mark password verification
            session['user'] = {
                'name': name,
                'email': email,
                'roll_number': roll_number,
                'class': class_name,
                'username': username,
                'photo': photo_path
            }
            prev = session.get('auth', {}) or {}
            session['auth'] = {
                'username': username,
                'password': True,
                'face': bool(prev.get('face') and prev.get('username') == username)
            }
            
            # Log successful login
            log_audit_action('LOGIN_PASSWORD_SUCCESS', username=username, email=email)
            
            return jsonify({
                'success': True,
                'data': session['user']
            })
        else:
            # Log failed login attempt
            log_audit_action('LOGIN_PASSWORD_FAILED', username=username, details='Invalid credentials')
            return jsonify({'success': False, 'message': 'Profile not found.'})

    except Exception as e:
        log_audit_action('LOGIN_PASSWORD_ERROR', username=data.get('username', ''), details=str(e))
        return jsonify({'success': False, 'message': str(e)})


    

@app.route('/admin-login', methods=['POST'])
def admin_login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        if username == os.getenv('ADMIN_USERNAME') and password == os.getenv('ADMIN_PASSWORD'):
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'Invalid credentials.'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/add-candidate', methods=['POST'])
def add_candidate():
    try:
        init_db()
        init_result()
        init_candidate_table()
        init_vote_table()
        name = request.form['name']
        class_name = request.form['class']
        description = request.form['description']
        role = request.form.get('role', '').strip()
        photo = request.files['photo']

        if not photo:
            return jsonify({'success': False, 'message': 'Photo is required.'})

        # Save photo
        filename = secure_filename(f"{name}.png")
        photo_path = os.path.join(CANDIDATE_UPLOAD_FOLDER, filename).replace("\\", "/")
        photo.save(photo_path)

        # Save candidate details in the database
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO Candidate (Name, Class, Photo, Description, Role)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, class_name, photo_path, description, role))
        conn.commit()
        conn.close()

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/candidates')
def candidates():
    if 'user' not in session:
        return jsonify({'error': 'User not logged in'}), 403
    auth = session.get('auth', {}) or {}
    if not (auth.get('password') and auth.get('face')):
        return jsonify({'error': 'Please complete both password and face login to continue.'}), 403

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT Name, Class, Photo, Description, Role FROM Candidate")
    rows = cursor.fetchall()
    conn.close()

    user = session['user']
    # Group candidates by role
    candidates_by_role = {}
    for name, clazz, photo, desc, role in rows:
        role_key = role or 'General'
        candidates_by_role.setdefault(role_key, []).append({
            'name': name,
            'class': clazz,
            'photo': photo,
            'description': desc,
            'role': role_key,
        })
    return render_template('candidates.html', candidates_by_role=candidates_by_role, user=user)

@app.route('/submit-vote', methods=['POST'])
def submit_vote():
    if 'user' not in session:
        return jsonify({'success': False, 'message': 'User not logged in.'})
    auth = session.get('auth', {}) or {}
    if not (auth.get('password') and auth.get('face')):
        return jsonify({'success': False, 'message': 'Please complete both password and face login before voting.'})

    try:
        data = request.get_json()
        # Expect payload: { selections: { Role1: CandidateName1, Role2: CandidateName2, ... } }
        selections = data.get('selections') or {}

        user = session['user']
        name = user['name']
        username = user['username']
        email = user['email']
        roll_number = user['roll_number']

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        # Ensure the Vote table exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Vote (
                Name TEXT,
                Username TEXT,
                Email TEXT,
                Roll_Number TEXT,
                Candidate TEXT,
                Role TEXT
            )
        ''')

        # Insert votes for each role
        # Validate that all roles present
        # Determine required roles from Candidate table
        cursor.execute("SELECT DISTINCT COALESCE(Role, 'General') FROM Candidate")
        required_roles = [r[0] for r in cursor.fetchall()]
        missing_roles = [r for r in required_roles if r not in selections or not selections[r]]
        if missing_roles:
            return jsonify({'success': False, 'message': f'Missing selection for roles: {", ".join(missing_roles)}'})

        # Enforce unique per role vote (Username, Role)
        for role, candidate_name in selections.items():
            # Validate candidate belongs to the specified role
            cursor.execute("SELECT 1 FROM Candidate WHERE Name = ? AND COALESCE(Role, 'General') = ?", (candidate_name, role))
            if cursor.fetchone() is None:
                return jsonify({'success': False, 'message': f'Invalid candidate "{candidate_name}" for role "{role}".'})

            # Check if this user already voted for this role
            cursor.execute("SELECT 1 FROM Vote WHERE Username = ? AND Role = ?", (username, role))
            if cursor.fetchone() is not None:
                return jsonify({'success': False, 'message': f'You have already voted for role "{role}".'})

            cursor.execute('''
                INSERT INTO Vote (Name, Username, Email, Roll_Number, Candidate, Role)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (name, username, email, roll_number, candidate_name, role))

        conn.commit()
        conn.close()

        # Log vote to audit log with all selected candidates
        vote_details = ', '.join([f'{role}: {candidate_name}' for role, candidate_name in selections.items()])
        log_audit_action('VOTE_SUBMITTED', username=username, email=email, details=vote_details)

        # Send vote confirmation email
        if email:
            subject = "Vote Confirmation - Face Voting System"
            vote_summary = '\n'.join([f'  • {role}: {candidate_name}' for role, candidate_name in selections.items()])
            body = f"""
Hello {name},

Your vote has been successfully recorded!

Your selections:
{vote_summary}

Thank you for participating in the election.

Best regards,
Election Administration
"""
            send_email(email, subject, body)

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/check-vote', methods=['POST'])
def check_vote():
    if 'user' not in session:
        return jsonify({'alreadyVoted': False})

    user = session['user']
    username = user['username']
    email = user['email']

    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        # Count required roles
        cursor.execute("SELECT COUNT(DISTINCT COALESCE(Role, 'General')) FROM Candidate")
        total_roles = cursor.fetchone()[0]

        # Count roles this user has voted for
        cursor.execute("SELECT COUNT(DISTINCT COALESCE(Role, 'General')) FROM Vote WHERE Username = ? AND Email = ?", (username, email))
        voted_roles = cursor.fetchone()[0]
        conn.close()

        if voted_roles >= total_roles and total_roles > 0:
            return jsonify({'alreadyVoted': True})
        else:
            return jsonify({'alreadyVoted': False})
    except Exception as e:
        return jsonify({'alreadyVoted': False, 'error': str(e)})

@app.route('/get-voters', methods=['GET'])
def get_voters():
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        # Fetch all users from Student_details
        cursor.execute("SELECT Name, Username, Email FROM Student_details")
        students = cursor.fetchall()

        # Total roles
        cursor.execute("SELECT COUNT(DISTINCT COALESCE(Role, 'General')) FROM Candidate")
        total_roles = cursor.fetchone()[0]

        # Check voting status for each user (voted all roles?)
        voters = []
        for student in students:
            name, username, email = student
            cursor.execute("SELECT COUNT(DISTINCT COALESCE(Role, 'General')) FROM Vote WHERE Username = ? AND Email = ?", (username, email))
            voted_roles = cursor.fetchone()[0]
            voted = (voted_roles >= total_roles and total_roles > 0)
            voters.append({
                'name': name,
                'username': username,
                'email': email,
                'voted': voted
            })

        conn.close()
        return jsonify(voters)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-stats', methods=['GET'])
def get_stats():
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        # Fetch all candidates including role
        cursor.execute("SELECT Name, Photo, Class, COALESCE(Role, 'General') as Role FROM Candidate")
        candidates = cursor.fetchall()

        stats = []
        for candidate in candidates:
            name, photo, class_name, role = candidate

            # Count votes for each candidate
            cursor.execute("SELECT COUNT(*) FROM Vote WHERE Candidate = ? AND COALESCE(Role, 'General') = ?", (name, role))
            votes = cursor.fetchone()[0]

            stats.append({
                'name': name,
                'photo': photo,
                'class': class_name,
                'role': role,
                'votes': votes
            })

        conn.close()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save-results', methods=['POST'])
def save_results():
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        # Create the Result table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Result (
                Name TEXT,
                Photo TEXT,
                Class TEXT,
                Role TEXT,
                Votes INTEGER
            )
        ''')

        # Fetch all candidates and their votes
        cursor.execute("SELECT Name, Photo, Class, COALESCE(Role, 'General') as Role FROM Candidate")
        candidates = cursor.fetchall()

        # Clear existing data in the Result table
        cursor.execute("DELETE FROM Result")

        for candidate in candidates:
            name, photo, class_name, role = candidate
            cursor.execute("SELECT COUNT(*) FROM Vote WHERE Candidate = ? AND COALESCE(Role, 'General') = ?", (name, role))
            votes = cursor.fetchone()[0]

            # Insert candidate data into the Result table
            cursor.execute('''
                INSERT INTO Result (Name, Photo, Class, Role, Votes)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, photo, class_name, role, votes))

        conn.commit()
        conn.close()

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/export-results', methods=['GET'])
def export_results():
    """Export election results as CSV"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        # Fetch all results from the Result table
        cursor.execute("SELECT Name, Photo, Class, COALESCE(Role, 'General') as Role, Votes FROM Result ORDER BY Role, Votes DESC")
        results = cursor.fetchall()
        conn.close()

        # Create CSV in memory
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Candidate Name', 'Class', 'Role', 'Votes'])
        
        # Write data
        for row in results:
            name, photo, class_name, role, votes = row
            writer.writerow([name, class_name, role, votes])
        
        # Get CSV string
        csv_data = output.getvalue()
        output.close()
        
        # Create response with CSV file
        response = BytesIO(csv_data.encode('utf-8'))
        response.seek(0)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'election_results_{timestamp}.csv'
        
        return send_file(
            response,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


def result():
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        # Fetch all results from the Result table
        cursor.execute("SELECT Name, Photo, Class, COALESCE(Role, 'General') as Role, Votes FROM Result")
        results = cursor.fetchall()
        conn.close()

        # Format results for rendering
        formatted_results = [
            {'name': row[0], 'photo': row[1], 'class': row[2], 'role': row[3], 'votes': row[4]}
            for row in results
        ]

        return render_template('result.html', results=formatted_results)
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/check-result-or-vote', methods=['POST'])
def check_result_or_vote():
    if 'user' not in session:
        return jsonify({'message': 'User not logged in.'}), 403

    user = session['user']
    username = user['username']
    email = user['email']

    # Require both password and face verification before allowing voting
    auth = session.get('auth', {}) or {}
    if not (auth.get('password') and auth.get('face')):
        return jsonify({'message': 'Please complete both password and face login to vote.'})

    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        # Check if the Result table is empty
        cursor.execute("SELECT COUNT(*) FROM Result")
        result_count = cursor.fetchone()[0]

        if result_count > 0:
            conn.close()
            return jsonify({'redirectTo': 'result'})

        # Determine total roles
        cursor.execute("SELECT COUNT(DISTINCT COALESCE(Role, 'General')) FROM Candidate")
        total_roles = cursor.fetchone()[0]
        # Count roles voted by this user
        cursor.execute("SELECT COUNT(DISTINCT COALESCE(Role, 'General')) FROM Vote WHERE Username = ? AND Email = ?", (username, email))
        voted_roles = cursor.fetchone()[0]
        conn.close()

        if voted_roles >= total_roles and total_roles > 0:
            return jsonify({'message': 'You have already voted for all roles. Please wait for the result.'})
        else:
            return jsonify({'redirectTo': 'candidates'})
    except Exception as e:
        return jsonify({'message': f'Error: {str(e)}'})

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return '', 204  # Return a success response with no content

@app.route('/delete-election', methods=['POST'])
def delete_election():
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        # Drop all tables
        tables = ['Student_details', 'Candidate', 'Result', 'Vote']
        for table in tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")

        conn.commit()
        conn.close()

        # Delete all files in the uploads directory
        for folder in [UPLOAD_FOLDER, CANDIDATE_UPLOAD_FOLDER]:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    file_path = os.path.join(folder, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/debug-match', methods=['POST'])
def debug_match():
    """Debug endpoint to test matching between two uploaded images.
    Upload 'image1' and 'image2' to see ORB and histogram matching scores.
    """
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Please upload both image1 and image2'}), 400
    
    try:
        file1 = request.files['image1']
        file2 = request.files['image2']
        
        # Load first image
        img1_stream = BytesIO(file1.read())
        img1_pil = Image.open(img1_stream).convert('RGB')
        img1_np = np.asarray(img1_pil, dtype=np.uint8)
        
        # Load second image (save to file for opencv_orb_match)
        img2_stream = BytesIO(file2.read())
        img2_pil = Image.open(img2_stream).convert('RGB')
        img2_path = os.path.join(UPLOAD_FOLDER, 'debug_match_tmp.png')
        img2_pil.save(img2_path)
        
        # Test ORB matching
        print(f"DEBUG-MATCH: Testing ORB matching...")
        print(f"  MATCHING_CONFIG: {MATCHING_CONFIG}")
        
        orb_matched, orb_score = opencv_orb_match(img1_np, img2_path)
        print(f"  ORB result: matched={orb_matched}, score={orb_score}")
        
        # Test histogram similarity
        roi1 = detect_face_roi_from_np(img1_np)
        img2_np = np.asarray(img2_pil, dtype=np.uint8)
        roi2 = detect_face_roi_from_np(img2_np)
        
        hist_score = 0.0
        if roi1 is not None and roi2 is not None:
            hist_score = hist_similarity(roi1, roi2)
        print(f"  Histogram result: score={hist_score}")
        
        # Test precomputed matching (if image1 looks like a registered user)
        precomp_matched, precomp_score = False, 0.0
        if roi1 is not None:
            # Try to use precomputed descriptors from image1's ROI
            desc_tmp = os.path.join(DESCRIPTORS_FOLDER, 'debug_match_desc.npy')
            try:
                orb_inst = cv2.ORB_create(nfeatures=MATCHING_CONFIG['orb_nfeatures'])
                kp1, des1 = orb_inst.detectAndCompute(roi1, None)
                if des1 is not None:
                    np.save(desc_tmp, des1)
                    precomp_matched, precomp_score = match_against_precomputed_descriptors(img2_np, desc_tmp)
                    os.remove(desc_tmp)
            except Exception as e:
                print(f"  Precomputed test error: {e}")
        
        # Cleanup
        if os.path.exists(img2_path):
            os.remove(img2_path)
        
        # Return detailed report
        result = {
            'config': MATCHING_CONFIG,
            'results': {
                'orb': {
                    'matched': bool(orb_matched),
                    'score': float(orb_score),
                    'threshold': MATCHING_CONFIG['orb_match_threshold']
                },
                'histogram': {
                    'score': float(hist_score),
                    'threshold': MATCHING_CONFIG['hist_threshold']
                },
                'precomputed_orb': {
                    'matched': bool(precomp_matched),
                    'score': float(precomp_score),
                    'threshold': MATCHING_CONFIG['orb_match_threshold']
                }
            },
            'recommendation': 'Adjust MATCHING_CONFIG thresholds based on these scores. Increase thresholds to be more lenient (accept lower scores).'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/audit-log', methods=['GET'])
def get_audit_log():
    """Fetch audit log for admin dashboard"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # Fetch audit log sorted by timestamp (newest first)
        cursor.execute('''
            SELECT timestamp, action, username, email, ip_address, details
            FROM Audit_log
            ORDER BY timestamp DESC
            LIMIT 1000
        ''')
        
        logs = cursor.fetchall()
        conn.close()
        
        # Format as list of dicts
        audit_logs = [
            {
                'timestamp': row[0],
                'action': row[1],
                'username': row[2],
                'email': row[3],
                'ip_address': row[4],
                'details': row[5]
            }
            for row in logs
        ]
        
        return jsonify({'success': True, 'logs': audit_logs})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/admin/election-config', methods=['GET', 'POST'])
def admin_election_config():
    """Get or update election configuration"""
    if request.method == 'GET':
        # Return current election config
        return jsonify({
            'success': True,
            'config': ELECTION_CONFIG
        })
    else:
        # Update election config (POST)
        try:
            data = request.get_json()
            # Update ELECTION_CONFIG with provided fields
            for key in ['enable_countdown_timer', 'election_end_time', 'enable_email_notifications', 'enable_2fa', 'enable_audit_logging', 'enable_export']:
                if key in data:
                    ELECTION_CONFIG[key] = data[key]
            
            log_audit_action('CONFIG_UPDATE', details=f'Election config updated: {data}')
            
            return jsonify({'success': True, 'config': ELECTION_CONFIG})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    # Ensure DB and tables exist before starting
    try:
        init_db()
        init_candidate_table()
        init_vote_table()
        init_result()
        init_audit_log_table()
    except Exception as e:
        print(f"Database initialization failed: {e}")

    app.run(debug=True)


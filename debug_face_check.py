from PIL import Image
import numpy as np
import face_recognition

p = 'static/uploads/debug_face.png'

try:
    img = Image.open(p)
    print('PIL mode:', img.mode, 'size:', img.size)
    arr = np.asarray(img)
    print('np shape:', getattr(arr, 'shape', None), 'dtype:', getattr(arr, 'dtype', None), 'contiguous:', arr.flags['C_CONTIGUOUS'])
    arr2 = np.ascontiguousarray(arr, dtype=np.uint8)
    print('after ascontiguousarray dtype/shape:', arr2.dtype, getattr(arr2, 'shape', None))
    try:
        locs = face_recognition.face_locations(arr2)
        print('face_locations count:', len(locs))
        encs = face_recognition.face_encodings(arr2)
        print('face_encodings count:', len(encs))
    except Exception as e:
        print('face_recognition error:', repr(e))
except FileNotFoundError:
    print('debug image not found at', p)
except Exception as e:
    print('error reading image:', repr(e))

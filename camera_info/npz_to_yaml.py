import numpy as np
import yaml

# --- User Settings ---
npz_path = 'abko_webcam_calibration.npz'   # Calibration file path
output_yaml = 'camera_calibration.yaml'     # Output YAML filename
image_width = 640                           # Webcam resolution (change if needed)
image_height = 480
camera_name = 'abko_webcam'                 # Camera name (arbitrary)

# --- Load Data ---
calib = np.load(npz_path)
K = calib['camera_matrix']
D = calib['dist_coeffs']

# Function to convert NumPy types to Python native types
def convert_to_python_type(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_to_python_type(item) for item in obj]
    else:
        return obj

# --- Construct YAML Data ---
data = {
    'image_width': int(image_width),
    'image_height': int(image_height),
    'camera_name': camera_name,
    'camera_matrix': {
        'rows': 3, 
        'cols': 3, 
        'data': convert_to_python_type(K.flatten())
    },
    'distortion_model': 'plumb_bob',
    'distortion_coefficients': {
        'rows': 1, 
        'cols': len(D.flatten()), 
        'data': convert_to_python_type(D.flatten())
    },
    'rectification_matrix': {
        'rows': 3, 
        'cols': 3, 
        'data': convert_to_python_type(np.eye(3).flatten())
    },
    'projection_matrix': {
        'rows': 3, 
        'cols': 4,
        'data': [
            float(K[0,0]), 0.0, float(K[0,2]), 0.0,
            0.0, float(K[1,1]), float(K[1,2]), 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
    }
}

# --- Save YAML (clean format with default_flow_style=None) ---
with open(output_yaml, 'w') as f:
    yaml.dump(data, f, default_flow_style=None, sort_keys=False)
print(f"✅ Saved as {output_yaml}")

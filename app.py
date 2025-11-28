from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import tensorflow as tf
try:
    from tensorflow import keras
except ImportError:
    import keras
import base64
from io import BytesIO
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/outputs', exist_ok=True)

# Global variables for models
face_to_sketch_model = None
sketch_to_face_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_models():
    """Load the CycleGAN models"""
    global face_to_sketch_model, sketch_to_face_model
    
    loaded_files = []
    
    try:
        # Try to load models - adjust paths as needed
        if os.path.exists('models/face_to_sketch.h5'):
            face_to_sketch_model = keras.models.load_model('models/face_to_sketch.h5', compile=False, safe_mode=False)
            loaded_files.append('face_to_sketch.h5')
            print("Loaded face_to_sketch model")
        
        if os.path.exists('models/sketch_to_face.h5'):
            sketch_to_face_model = keras.models.load_model('models/sketch_to_face.h5', compile=False, safe_mode=False)
            loaded_files.append('sketch_to_face.h5')
            print("Loaded sketch_to_face model")
        
        # Alternative naming conventions
        if face_to_sketch_model is None and os.path.exists('models'):
            for file in os.listdir('models'):
                if file.endswith('.h5') and file not in loaded_files:
                    if ('face' in file.lower() and 'sketch' in file.lower()) or ('photo' in file.lower() and 'sketch' in file.lower()):
                        face_to_sketch_model = keras.models.load_model(f'models/{file}', compile=False, safe_mode=False)
                        loaded_files.append(file)
                        print(f"Loaded model: {file}")
                        break
        
        if sketch_to_face_model is None and os.path.exists('models'):
            for file in os.listdir('models'):
                if file.endswith('.h5') and file not in loaded_files:
                    if (('sketch' in file.lower() and 'face' in file.lower()) or 
                        ('sketch' in file.lower() and 'photo' in file.lower())):
                        sketch_to_face_model = keras.models.load_model(f'models/{file}', compile=False, safe_mode=False)
                        loaded_files.append(file)
                        print(f"Loaded model: {file}")
                        break
        
        # If still None, try loading any .h5 files in models directory
        if os.path.exists('models'):
            h5_files = [f for f in os.listdir('models') if f.endswith('.h5')]
            if len(h5_files) >= 2:
                # Load first file if face_to_sketch not loaded
                if face_to_sketch_model is None and h5_files[0] not in loaded_files:
                    face_to_sketch_model = keras.models.load_model(f'models/{h5_files[0]}', compile=False, safe_mode=False)
                    loaded_files.append(h5_files[0])
                    print(f"Loaded model: {h5_files[0]}")
                # Load second file if sketch_to_face not loaded
                if sketch_to_face_model is None and len(h5_files) > 1:
                    # Find a file different from already loaded ones
                    for h5_file in h5_files:
                        if h5_file not in loaded_files:
                            sketch_to_face_model = keras.models.load_model(f'models/{h5_file}', compile=False, safe_mode=False)
                            loaded_files.append(h5_file)
                            print(f"Loaded model: {h5_file}")
                            break
            elif len(h5_files) == 1:
                print(f"Warning: Only one model file found ({h5_files[0]}). Please ensure both models are in the models directory.")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure model files are in the 'models' directory")

def preprocess_image(image_path):
    """Preprocess image for model input"""
    # Load and resize image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256))  # Adjust size based on your model
    img_array = np.array(img)
    
    # Normalize to [-1, 1] range (typical for CycleGAN)
    img_array = img_array.astype(np.float32) / 127.5 - 1.0
    
    # Expand dimensions for batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def postprocess_image(prediction):
    """Postprocess model output to image"""
    # Remove batch dimension if present
    if len(prediction.shape) == 4:
        prediction = prediction[0]
    
    # Denormalize from [-1, 1] to [0, 255]
    prediction = (prediction + 1.0) * 127.5
    prediction = np.clip(prediction, 0, 255).astype(np.uint8)
    
    return prediction

def detect_image_type(image_path):
    """Simple heuristic to detect if image is a sketch or real face"""
    img = cv2.imread(image_path)
    if img is None:
        return 'unknown'
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Calculate color variance (sketches typically have lower variance)
    color_variance = np.var(img)
    
    # Heuristics: sketches usually have high edge density and lower color variance
    if edge_ratio > 0.15 and color_variance < 1000:
        return 'sketch'
    elif edge_ratio < 0.10:
        return 'real'
    else:
        # If uncertain, check for color saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:, :, 1])
        if saturation < 50:
            return 'sketch'
        else:
            return 'real'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/convert', methods=['POST'])
def convert_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Detect image type
            image_type = detect_image_type(filepath)
            
            # Determine conversion direction
            if image_type == 'sketch':
                model = sketch_to_face_model
                conversion_type = 'Sketch to Face'
            elif image_type == 'real':
                model = face_to_sketch_model
                conversion_type = 'Face to Sketch'
            else:
                # Default: try face to sketch first
                model = face_to_sketch_model
                conversion_type = 'Face to Sketch (Auto-detected)'
            
            if model is None:
                return jsonify({'error': 'Model not loaded. Please ensure model files are in the models directory.'}), 500
            
            # Preprocess image
            input_img = preprocess_image(filepath)
            
            # Run inference
            output = model.predict(input_img, verbose=0)
            
            # Postprocess output
            output_img = postprocess_image(output)
            
            # Convert to PIL Image
            result_image = Image.fromarray(output_img)
            
            # Convert to base64 for sending to frontend
            buffered = BytesIO()
            result_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'image': f'data:image/png;base64,{img_str}',
                'conversion_type': conversion_type,
                'detected_type': image_type
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Conversion failed: {str(e)}'}), 500

@app.route('/api/manual-convert', methods=['POST'])
def manual_convert():
    """Manually specify conversion direction"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        direction = request.form.get('direction', 'face_to_sketch')  # or 'sketch_to_face'
        
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Select model based on direction
            if direction == 'face_to_sketch':
                model = face_to_sketch_model
                conversion_type = 'Face to Sketch'
            else:
                model = sketch_to_face_model
                conversion_type = 'Sketch to Face'
            
            if model is None:
                return jsonify({'error': 'Model not loaded. Please ensure model files are in the models directory.'}), 500
            
            # Preprocess image
            input_img = preprocess_image(filepath)
            
            # Run inference
            output = model.predict(input_img, verbose=0)
            
            # Postprocess output
            output_img = postprocess_image(output)
            
            # Convert to PIL Image
            result_image = Image.fromarray(output_img)
            
            # Convert to base64 for sending to frontend
            buffered = BytesIO()
            result_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'image': f'data:image/png;base64,{img_str}',
                'conversion_type': conversion_type
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Conversion failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("Loading CycleGAN models...")
    load_models()
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)


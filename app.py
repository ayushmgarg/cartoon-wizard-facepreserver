from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from PIL import Image
import base64
import io
from cartoon_engine import enhanced_pipeline, emotion_detector, advanced_viz
app = Flask(__name__)

def decode_image(b64_string):
    """Decode base64 image from browser to numpy RGB array."""
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    img_bytes = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((512, 512), Image.LANCZOS) if max(img.size) > 512 else img
    return np.array(img)

def encode_image(np_img):
    """Encode numpy RGB array back to base64 PNG for the browser."""
    pil_img = Image.fromarray(np_img.astype(np.uint8))
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cartoonize', methods=['POST'])
def cartoonize():
    try:
        data = request.get_json()
        image_b64  = data.get('image')
        show_overlay = data.get('show_overlay', False)

        image = decode_image(image_b64)

        # Run full pipeline
        result = enhanced_pipeline.process_image_full(
            image,
            enable_preprocessing=True,
            denoise_strength='medium',
            auto_enhance=True,
            white_balance=True,
            preserve_identity=True,
            emotion_adaptive=True,
            region_aware=True,
            show_steps=False
        )

        result_img  = result['final']
        metadata    = result['metadata']
        emotion_res = emotion_detector.detect_emotion(image)

        if show_overlay:
            result_img = advanced_viz.draw_face_detection_overlay(
                result_img,
                show_landmarks=True,
                show_emotion=True,
                emotion_result=emotion_res
            )
            identity_score = metadata.get('identity_similarity', 0)
            h, w = result_img.shape[:2]
            cv2.putText(result_img, f"Identity: {identity_score:.2f}",
                        (w - 210, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        return jsonify({
            'success':          True,
            'result_image':     encode_image(result_img),
            'emotion':          emotion_res.get('emotion', 'unknown').upper(),
            'emotion_confidence': round(float(emotion_res.get('confidence', 0)), 1),
            'identity_score':   round(float(metadata.get('identity_similarity', 0)), 3),
            'face_detected':    bool(metadata.get('face_detected', False))
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e),
                        'trace': traceback.format_exc()})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
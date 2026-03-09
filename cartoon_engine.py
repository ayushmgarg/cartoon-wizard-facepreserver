import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
import mediapipe as mp
from scipy.ndimage import gaussian_filter, median_filter
import base64
import io
import os

class CartoonProcessor:
 
    def __init__(self):
        self.methods = ['bilateral', 'adaptive', 'oil_paint', 'pencil_sketch']

    def bilateral_cartoon(self, image, d=9, sigma_color=75, sigma_space=75, edge_threshold=100):
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Step 1: Edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = median_filter(gray, size=3).astype(np.uint8)
        edges = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            blockSize=9,
            C=2
        )

        # Step 2: Bilateral filtering for smoothing while preserving edges
        smooth = image.copy()
        for _ in range(2):  # Apply multiple times for stronger effect
            smooth = cv2.bilateralFilter(smooth, d, sigma_color, sigma_space)

        # Step 3: Color quantization
        smooth = self._quantize_colors(smooth, k=8)

        # Step 4: Combine edges with smoothed image
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        edge_mask = edges_colored.astype(np.float32) / 255.0
        cartoon = (smooth.astype(np.float32) * edge_mask).astype(np.uint8)

        return cartoon

    def adaptive_cartoon(self, image, num_bilateral=7, edge_preserve=0.1):
       
        # Reduce noise while keeping edges sharp
        img = cv2.medianBlur(image, 5)

        # Multiple bilateral filters
        for _ in range(num_bilateral):
            img = cv2.bilateralFilter(img, d=9, sigmaColor=9, sigmaSpace=7)

        # Edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9, 2
        )

        # Color quantization
        img = self._quantize_colors(img, k=9)

        # Combine
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        edge_mask = edges.astype(np.float32) / 255.0
        cartoon = (img.astype(np.float32) * edge_mask).astype(np.uint8)

        return cartoon

    def oil_paint_effect(self, image, size=7, dynRatio=1):
        
        try:
            result = cv2.xphoto.oilPainting(image, size, dynRatio)
        except AttributeError:
            result = cv2.medianBlur(image, size if size % 2 == 1 else size + 1)
            result = cv2.bilateralFilter(result, 9, 75, 75)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.dilate(edges, None)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        result = np.where(edges_colored > 0, [0, 0, 0], result)
        return result

    def pencil_sketch(self, image):
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        inv_gray = 255 - gray
        blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

    def _quantize_colors(self, image, k=8):
        
        data = image.reshape((-1, 3)).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        centers = np.uint8(centers)
        quantized = centers[labels.flatten()]
        quantized = quantized.reshape(image.shape)

        return quantized

    def apply(self, image, method='bilateral', intensity=1.0, **kwargs):
        
        if method == 'bilateral':
            result = self.bilateral_cartoon(image, **kwargs)
        elif method == 'adaptive':
            result = self.adaptive_cartoon(image, **kwargs)
        elif method == 'oil_paint':
            result = self.oil_paint_effect(image, **kwargs)
        elif method == 'pencil_sketch':
            result = self.pencil_sketch(image)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Blend with original based on intensity
        if 0 < intensity < 1:
            result = cv2.addWeighted(image, 1-intensity, result, intensity, 0)

        return result

class FaceSegmenter:
    
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        # Define face regions using landmark indices
        self.regions = {
            'left_eye': list(range(33, 133)) + list(range(145, 154)),
            'right_eye': list(range(362, 385)) + list(range(398, 407)),
            'nose': list(range(1, 5)) + list(range(48, 60)) + list(range(114, 120)),
            'mouth': list(range(0, 17)) + list(range(37, 42)) + list(range(267, 273)),
            'face_oval': list(range(10, 338)),
        }

        print("✅ Face Segmentation initialized with MediaPipe")

    def detect_face_regions(self, image):
        
        h, w = image.shape[:2]
        results = self.face_mesh.process(image)

        if not results.multi_face_landmarks:
            print("⚠️ No face detected!")
            return None

        landmarks = results.multi_face_landmarks[0]

        # Convert landmarks to coordinates
        coords = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            coords.append((x, y))

        # Create masks for each region
        masks = {}
        for region_name, indices in self.regions.items():
            mask = np.zeros((h, w), dtype=np.uint8)

            # Get points for this region
            points = np.array([coords[i] for i in indices if i < len(coords)])

            if len(points) > 0:
                # Create convex hull
                hull = cv2.convexHull(points)
                cv2.fillConvexPoly(mask, hull, 255)

            masks[region_name] = mask

        # Create overall face mask
        face_mask = np.zeros((h, w), dtype=np.uint8)
        face_points = np.array(coords)
        hull = cv2.convexHull(face_points)
        cv2.fillConvexPoly(face_mask, hull, 255)
        masks['face'] = face_mask

        return {
            'masks': masks,
            'landmarks': coords,
            'face_detected': True
        }

    def create_region_importance_map(self, image):
        
        regions = self.detect_face_regions(image)
        if not regions:
            # Return None for both values when no face detected
            return None, None

        h, w = image.shape[:2]
        importance_map = np.zeros((h, w), dtype=np.float32)

        # Assign importance weights
        weights = {
            'left_eye': 0.9,
            'right_eye': 0.9,
            'nose': 0.7,
            'mouth': 0.8,
            'face_oval': 0.5
        }

        for region_name, weight in weights.items():
            if region_name in regions['masks']:
                mask = regions['masks'][region_name].astype(np.float32) / 255.0
                importance_map += mask * weight

        # Normalize
        importance_map = np.clip(importance_map, 0, 1)

        return importance_map, regions

class EmotionDetector:
    

    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        # Emotion-to-parameter mapping
        # Each emotion maps to: [saturation, contrast, smoothing, edge_strength]
        self.emotion_params = {
            'happy': {
                'saturation': 1.3,      # More vibrant
                'contrast': 1.1,
                'smoothing': 0.8,       # Less smoothing
                'edge_strength': 0.9,
                'color_temp': 'warm'    # Warmer tones
            },
            'sad': {
                'saturation': 0.7,      # Desaturated
                'contrast': 0.9,
                'smoothing': 1.2,       # More smoothing
                'edge_strength': 0.7,
                'color_temp': 'cool'    # Cooler tones
            },
            'angry': {
                'saturation': 1.2,
                'contrast': 1.4,        # High contrast
                'smoothing': 0.6,       # Sharp
                'edge_strength': 1.3,   # Strong edges
                'color_temp': 'warm'
            },
            'surprise': {
                'saturation': 1.4,      # Very vibrant
                'contrast': 1.2,
                'smoothing': 0.7,
                'edge_strength': 1.1,
                'color_temp': 'neutral'
            },
            'fear': {
                'saturation': 0.8,
                'contrast': 1.1,
                'smoothing': 1.0,
                'edge_strength': 1.0,
                'color_temp': 'cool'
            },
            'neutral': {
                'saturation': 1.0,
                'contrast': 1.0,
                'smoothing': 1.0,
                'edge_strength': 1.0,
                'color_temp': 'neutral'
            },
            'disgust': {
                'saturation': 0.9,
                'contrast': 1.1,
                'smoothing': 1.0,
                'edge_strength': 1.0,
                'color_temp': 'cool'
            }
        }

        print("✅ Emotion Detector initialized")

    def detect_emotion(self, image):
       
        try:
            # DeepFace expects BGR
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Analyze emotions
            result = DeepFace.analyze(
                img_bgr,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )

            if isinstance(result, list):
                result = result[0]

            emotion = result['dominant_emotion']
            confidence = result['emotion'][emotion]

            return {
                'emotion': emotion,
                'confidence': confidence,
                'all_emotions': result['emotion']
            }

        except Exception as e:
            print(f"⚠️ Emotion detection failed: {e}")
            return {
                'emotion': 'neutral',
                'confidence': 1.0,
                'all_emotions': {'neutral': 1.0}
            }

    def get_parameters_for_emotion(self, emotion):
        
        return self.emotion_params.get(emotion, self.emotion_params['neutral'])

    def apply_emotion_adjustments(self, image, emotion):
        
        params = self.get_parameters_for_emotion(emotion)

        # Convert to float
        img = image.astype(np.float32) / 255.0

        # Adjust saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] *= params['saturation']
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Adjust contrast
        img = np.clip((img - 0.5) * params['contrast'] + 0.5, 0, 1)

        # Apply color temperature
        if params['color_temp'] == 'warm':
            # Increase red, decrease blue
            img[:, :, 0] *= 1.1  # R
            img[:, :, 2] *= 0.9  # B
        elif params['color_temp'] == 'cool':
            # Decrease red, increase blue
            img[:, :, 0] *= 0.9  # R
            img[:, :, 2] *= 1.1  # B

        # Convert back
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        return img

class IdentityPreserver:
    

    def __init__(self, similarity_threshold=0.6):
        self.similarity_threshold = similarity_threshold
        self.model_name = 'Facenet512'  # Options: VGG-Face, Facenet, Facenet512, ArcFace
        print("✅ Identity Preserver initialized (using DeepFace)")
        print(f"   Model: {self.model_name}")
        print(f"   Similarity threshold: {similarity_threshold}")

    def get_face_encoding(self, image):
        
        try:
            # Convert RGB to BGR for DeepFace
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Get embedding
            embedding_objs = DeepFace.represent(
                img_path=img_bgr,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend='opencv'
            )

            if len(embedding_objs) > 0:
                embedding = np.array(embedding_objs[0]['embedding'])
                return embedding
            else:
                return None

        except Exception as e:
            print(f"⚠️ Face encoding failed: {e}")
            return None

    def calculate_similarity(self, encoding1, encoding2):
        
        if encoding1 is None or encoding2 is None:
            return 0.0

        # Cosine similarity
        dot_product = np.dot(encoding1, encoding2)
        norm1 = np.linalg.norm(encoding1)
        norm2 = np.linalg.norm(encoding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        # Convert to 0-1 range (cosine similarity is -1 to 1)
        similarity = (similarity + 1) / 2

        return float(similarity)

    def iterative_refinement(self, original_image, cartoonized_image,
                            cartoonizer, max_iterations=5):
        
        # Get original face encoding
        print("   🔍 Extracting original face encoding...")
        original_encoding = self.get_face_encoding(original_image)

        if original_encoding is None:
            print("   ⚠️ Cannot preserve identity - no face detected in original")
            return cartoonized_image

        best_result = cartoonized_image.copy()
        best_similarity = 0.0
        intensity = 1.0

        print(f"\n   🔄 Starting identity preservation refinement...")

        for iteration in range(max_iterations):
            # Get current encoding
            current_encoding = self.get_face_encoding(cartoonized_image)

            if current_encoding is None:
                print(f"      Iteration {iteration + 1}: No face detected, reducing intensity")
                intensity *= 0.8
                cartoonized_image = cartoonizer.apply(
                    original_image,
                    method='bilateral',
                    intensity=intensity
                )
                continue

            # Calculate similarity
            similarity = self.calculate_similarity(original_encoding, current_encoding)

            print(f"      Iteration {iteration + 1}: Similarity = {similarity:.3f}, Intensity = {intensity:.2f}")

            # Check if we've reached acceptable similarity
            if similarity >= self.similarity_threshold:
                print(f"   ✅ Target similarity reached!")
                return cartoonized_image

            # Track best result
            if similarity > best_similarity:
                best_similarity = similarity
                best_result = cartoonized_image.copy()

            # Adjust intensity based on similarity
            if similarity < 0.3:
                intensity *= 0.7  # Reduce dramatically
            elif similarity < 0.5:
                intensity *= 0.85  # Reduce moderately
            else:
                intensity *= 0.95  # Fine-tune

            # Re-apply cartoonization with adjusted intensity
            cartoonized_image = cartoonizer.apply(
                original_image,
                method='bilateral',
                intensity=intensity
            )

        print(f"   ⚠️ Max iterations reached. Best similarity: {best_similarity:.3f}")
        return best_result

    def blend_with_landmarks(self, original, cartoonized, blend_factor=0.3):
        
        # Simple weighted blend
        blended = cv2.addWeighted(original, blend_factor, cartoonized, 1-blend_factor, 0)
        return blended

class SmartCartoonizationPipeline:
    

    def __init__(self, cartoonizer, face_segmenter, emotion_detector, identity_preserver):
        self.cartoonizer = cartoonizer
        self.face_segmenter = face_segmenter
        self.emotion_detector = emotion_detector
        self.identity_preserver = identity_preserver

        print("✅ Smart Cartoonization Pipeline initialized")

    def process_image(self, image, preserve_identity=True, emotion_adaptive=True,
                     region_aware=True, show_steps=False):
        
        print("\n" + "=" * 60)
        print("PROCESSING IMAGE...")
        print("=" * 60)

        results = {
            'original': image.copy(),
            'steps': {},
            'metadata': {}
        }

        # Step 1: Detect emotion
        if emotion_adaptive:
            print("\n📊 Step 1: Detecting emotion...")
            emotion_result = self.emotion_detector.detect_emotion(image)
            emotion = emotion_result['emotion']
            confidence = emotion_result['confidence']

            print(f"   Detected: {emotion.upper()} (confidence: {confidence:.2f})")
            results['metadata']['emotion'] = emotion
            results['metadata']['emotion_confidence'] = confidence

            # Apply emotion-based adjustments
            image_adjusted = self.emotion_detector.apply_emotion_adjustments(image, emotion)
            results['steps']['emotion_adjusted'] = image_adjusted
        else:
            emotion = 'neutral'
            image_adjusted = image.copy()

        # Step 2: Face segmentation and region detection
        if region_aware:
            print("\n🎭 Step 2: Detecting face regions...")
            importance_map, face_regions = self.face_segmenter.create_region_importance_map(image)

            if importance_map is not None:
                print(f"   Face detected with {len(face_regions['landmarks'])} landmarks")
                results['metadata']['face_detected'] = True
                results['steps']['importance_map'] = importance_map
                results['steps']['face_masks'] = face_regions['masks']
            else:
                print("   ⚠️ No face detected, using global cartoonization")
                importance_map = None
                region_aware = False
                results['metadata']['face_detected'] = False
        else:
            importance_map = None

        # Step 3: Apply cartoonization
        print("\n🎨 Step 3: Applying cartoonization...")

        # Get emotion-specific parameters
        emotion_params = self.emotion_detector.get_parameters_for_emotion(emotion)

        if region_aware and importance_map is not None:
            # Region-aware cartoonization
            print("   Using region-aware processing...")
            cartoonized = self._apply_region_aware_cartoon(
                image_adjusted,
                importance_map,
                emotion_params
            )
        else:
            # Global cartoonization
            print("   Using global processing...")
            cartoonized = self.cartoonizer.apply(
                image_adjusted,
                method='bilateral',
                intensity=1.0
            )

        results['steps']['cartoonized_initial'] = cartoonized

        # Step 4: Identity preservation
        if preserve_identity and results['metadata'].get('face_detected', False):
            print("\n👤 Step 4: Preserving identity...")
            cartoonized = self.identity_preserver.iterative_refinement(
                image,
                cartoonized,
                self.cartoonizer,
                max_iterations=5
            )

            # Calculate final similarity
            original_encoding = self.identity_preserver.get_face_encoding(image)
            final_encoding = self.identity_preserver.get_face_encoding(cartoonized)

            if original_encoding is not None and final_encoding is not None:
                similarity = self.identity_preserver.calculate_similarity(
                    original_encoding, final_encoding
                )
                print(f"   Final identity similarity: {similarity:.3f}")
                results['metadata']['identity_similarity'] = similarity

        results['final'] = cartoonized

        # Visualize steps if requested
        if show_steps:
            self._visualize_steps(results)

        print("\n✅ Processing complete!")
        return results

    def _apply_region_aware_cartoon(self, image, importance_map, emotion_params):
        
        h, w = image.shape[:2]

        # High intensity for background/less important regions
        cartoon_strong = self.cartoonizer.apply(
            image,
            method='bilateral',
            intensity=1.0 * emotion_params['smoothing']
        )

        # Low intensity for important face features
        cartoon_weak = self.cartoonizer.apply(
            image,
            method='bilateral',
            intensity=0.5 * emotion_params['smoothing']
        )

        # Blend based on importance map
        importance_map_3d = np.expand_dims(importance_map, axis=2)
        result = (cartoon_weak * importance_map_3d +
                 cartoon_strong * (1 - importance_map_3d))

        return result.astype(np.uint8)

    def _visualize_steps(self, results):
        
        steps_to_show = []
        titles = []

        if 'original' in results:
            steps_to_show.append(results['original'])
            titles.append('Original')

        if 'emotion_adjusted' in results['steps']:
            steps_to_show.append(results['steps']['emotion_adjusted'])
            emotion = results['metadata'].get('emotion', 'unknown')
            titles.append(f'Emotion Adjusted\n({emotion})')

        if 'importance_map' in results['steps']:
            imap = results['steps']['importance_map']
            imap_colored = cv2.applyColorMap(
                (imap * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            imap_colored = cv2.cvtColor(imap_colored, cv2.COLOR_BGR2RGB)
            steps_to_show.append(imap_colored)
            titles.append('Importance Map')

        if 'cartoonized_initial' in results['steps']:
            steps_to_show.append(results['steps']['cartoonized_initial'])
            titles.append('Initial Cartoon')

        if 'final' in results:
            steps_to_show.append(results['final'])
            similarity = results['metadata'].get('identity_similarity', 0)
            titles.append(f'Final Result\n(Similarity: {similarity:.2f})')

        # Plot
        n = len(steps_to_show)
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4))

        if n == 1:
            axes = [axes]

        for ax, img, title in zip(axes, steps_to_show, titles):
            ax.imshow(img)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.axis('off')

        plt.tight_layout()
        plt.show()

class MultiStyleCartoonizer:
    
    def __init__(self):
        self.styles = ['anime', 'comic', 'watercolor', 'oil_paint', 'pencil_sketch', 'pop_art']
        print("✅ Multi-Style Cartoonizer initialized")

    def anime_style(self, image):
        
        # Super smooth bilateral filtering
        smooth = image.copy()
        for _ in range(3):
            smooth = cv2.bilateralFilter(smooth, 9, 95, 95)

        # Enhance saturation
        hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= 1.5  # Increase saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        smooth = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Color quantization (fewer colors = more anime-like)
        data = smooth.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, 6, None, criteria, 10,
                                       cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()].reshape(smooth.shape)

        # Strong edges — keep thin, clean lines
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 9, 2)
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        # Correct combine: multiply color by edge mask normalized to [0,1]
        # This preserves color where no edge, darkens at edge lines
        edge_mask = edges_3ch.astype(np.float32) / 255.0
        result = (quantized.astype(np.float32) * edge_mask).astype(np.uint8)

        return result

    def comic_style(self, image):
        
        # Strong bilateral filtering
        smooth = cv2.bilateralFilter(image, 9, 75, 75)
        smooth = cv2.bilateralFilter(smooth, 9, 75, 75)

        # Bold edges
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, None, iterations=1)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        edges = 255 - edges

        # High contrast
        smooth = cv2.convertScaleAbs(smooth, alpha=1.3, beta=10)

        # Color quantization (comic book colors)
        data = smooth.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, 10, None, criteria, 10,
                                       cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()].reshape(smooth.shape)

        # Correct combine: draw dark outlines on top of colored fill
        # edges here is already inverted (255 = no edge, 0 = edge)
        edge_mask = edges.astype(np.float32) / 255.0
        result = (quantized.astype(np.float32) * edge_mask).astype(np.uint8)

        return result

    def watercolor_style(self, image):
        
        # Multiple bilateral filters for soft effect
        watercolor = image.copy()
        for _ in range(4):
            watercolor = cv2.bilateralFilter(watercolor, 9, 50, 50)

        # Add slight blur
        watercolor = cv2.GaussianBlur(watercolor, (7, 7), 0)

        # Reduce saturation slightly
        hsv = cv2.cvtColor(watercolor, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= 0.8
        watercolor = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Soft edges
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
        edges = cv2.GaussianBlur(edges, (5, 5), 0)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        # Watercolor needs visible texture — increase edge influence
        # Also add a paper texture by brightening slightly
        edges_norm = edges.astype(np.float32) / 255.0
        edge_darkening = 1.0 - (edges_norm * 0.4)   # darken along edges by up to 40%
        result = (watercolor.astype(np.float32) * edge_darkening).astype(np.uint8)

        # Slight brightness lift to simulate paper
        result = cv2.convertScaleAbs(result, alpha=1.0, beta=15)

        return result

    def oil_paint_style(self, image):
        
        # Check if xphoto module is available
        try:
            result = cv2.xphoto.oilPainting(image, 7, 1)
        except:
            # Fallback: median filter approximation
            result = cv2.medianBlur(image, 7)
            result = cv2.bilateralFilter(result, 9, 75, 75)

        # Enhance colors
        result = cv2.convertScaleAbs(result, alpha=1.2, beta=5)

        return result

    def pencil_sketch_style(self, image):
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Invert
        inv_gray = 255 - gray

        # Gaussian blur
        blur = cv2.GaussianBlur(inv_gray, (21, 21), 0, 0)

        # Dodge blend
        sketch = cv2.divide(gray, 255 - blur, scale=256)

        # Convert to RGB
        sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

        return sketch_rgb

    def pop_art_style(self, image):
        
        # Aggressive color quantization
        data = image.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, 4, None, criteria, 10,
                                       cv2.KMEANS_RANDOM_CENTERS)

        # Make colors very vibrant
        centers = centers * 1.3
        centers = np.clip(centers, 0, 255)
        centers = np.uint8(centers)

        quantized = centers[labels.flatten()].reshape(image.shape)

        # Very bold edges
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.dilate(edges, None, iterations=2)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        # High contrast
        quantized = cv2.convertScaleAbs(quantized, alpha=1.4, beta=10)

        # Correct combine: use edge mask to draw bold outlines
        edge_mask = (255 - edges).astype(np.float32) / 255.0
        result = (quantized.astype(np.float32) * edge_mask).astype(np.uint8)

        # Boost vibrancy further
        result = cv2.convertScaleAbs(result, alpha=1.3, beta=0)

        return result

    def apply_style(self, image, style='anime'):
        
        if style == 'anime':
            return self.anime_style(image)
        elif style == 'comic':
            return self.comic_style(image)
        elif style == 'watercolor':
            return self.watercolor_style(image)
        elif style == 'oil_paint':
            return self.oil_paint_style(image)
        elif style == 'pencil_sketch':
            return self.pencil_sketch_style(image)
        elif style == 'pop_art':
            return self.pop_art_style(image)
        else:
            raise ValueError(f"Unknown style: {style}")

    def create_style_comparison(self, image, save_path=None):
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        # Original
        axes[0].imshow(image)
        axes[0].set_title('Original', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Each style
        for idx, style in enumerate(self.styles, 1):
            print(f"Generating {style} style...")
            result = self.apply_style(image, style)
            axes[idx].imshow(result)
            axes[idx].set_title(style.replace('_', ' ').title(),
                              fontsize=14, fontweight='bold')
            axes[idx].axis('off')

        # Hide last subplot
        axes[-1].axis('off')

        plt.tight_layout()
        plt.suptitle('Multi-Style Cartoonization Comparison',
                    fontsize=18, fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved to: {save_path}")

        return fig

class AdvancedVisualizer:
    

    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        print("✅ Advanced Visualizer initialized")

    def draw_face_detection_overlay(self, image, show_landmarks=True,
                                   show_emotion=True, emotion_result=None):
       
        result_img = image.copy()
        h, w = image.shape[:2]

        # Detect face
        results = self.face_detection.process(image)

        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                box_w = int(bbox.width * w)
                box_h = int(bbox.height * h)

                # Draw bounding box
                cv2.rectangle(result_img, (x, y), (x + box_w, y + box_h),
                            (0, 255, 0), 3)

                # Draw confidence score
                confidence = detection.score[0]
                cv2.putText(result_img, f'Face: {confidence:.2f}',
                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                          0.7, (0, 255, 0), 2)

                # Draw key points
                if show_landmarks:
                    for keypoint in detection.location_data.relative_keypoints:
                        kp_x = int(keypoint.x * w)
                        kp_y = int(keypoint.y * h)
                        cv2.circle(result_img, (kp_x, kp_y), 4, (255, 0, 0), -1)

                # Draw emotion label
                if show_emotion and emotion_result:
                    emotion = emotion_result.get('emotion', 'Unknown')
                    conf = emotion_result.get('confidence', 0)

                    # Create label background
                    label = f'{emotion.upper()}: {conf:.1f}%'
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                0.8, 2)[0]

                    # Draw background rectangle
                    cv2.rectangle(result_img,
                                (x, y + box_h + 5),
                                (x + label_size[0] + 10, y + box_h + label_size[1] + 15),
                                (0, 255, 0), -1)

                    # Draw text
                    cv2.putText(result_img, label,
                              (x + 5, y + box_h + label_size[1] + 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return result_img

    def create_heatmap_overlay(self, image, importance_map):
        
        # Normalize importance map
        heatmap = (importance_map * 255).astype(np.uint8)

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend with original
        overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)

        return overlay

    def create_processing_dashboard(self, original, preprocessed, cartoonized,
                                   emotion_result, metrics):
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # Row 1: Main images
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original)
        ax1.set_title('Original', fontsize=14, fontweight='bold')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(preprocessed)
        ax2.set_title('Preprocessed', fontsize=14, fontweight='bold')
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(cartoonized)
        ax3.set_title('Cartoonized', fontsize=14, fontweight='bold')
        ax3.axis('off')

        # Original with face detection
        ax4 = fig.add_subplot(gs[0, 3])
        detected = self.draw_face_detection_overlay(
            original,
            show_landmarks=True,
            show_emotion=True,
            emotion_result=emotion_result
        )
        ax4.imshow(detected)
        ax4.set_title('Face Detection', fontsize=14, fontweight='bold')
        ax4.axis('off')

        # Row 2: Analysis
        ax5 = fig.add_subplot(gs[1, :2])
        if emotion_result and 'all_emotions' in emotion_result:
            emotions = emotion_result['all_emotions']
            bars = ax5.barh(list(emotions.keys()), list(emotions.values()))
            ax5.set_xlabel('Confidence', fontsize=12)
            ax5.set_title('Emotion Distribution', fontsize=14, fontweight='bold')

            # Color bars
            colors = plt.cm.viridis(np.linspace(0, 1, len(emotions)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)

        # Metrics
        ax6 = fig.add_subplot(gs[1, 2:])
        ax6.axis('off')

        if metrics:
            metrics_text = "Processing Metrics:\n\n"
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metrics_text += f"{key}: {value:.3f}\n"

            ax6.text(0.1, 0.5, metrics_text,
                    fontsize=12, verticalalignment='center',
                    family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Row 3: Side by side comparison
        ax7 = fig.add_subplot(gs[2, :2])
        comparison = np.hstack([original, cartoonized])
        ax7.imshow(comparison)
        ax7.set_title('Before → After Comparison', fontsize=14, fontweight='bold')
        ax7.axis('off')

        # Difference map
        ax8 = fig.add_subplot(gs[2, 2:])
        diff = cv2.absdiff(original, cartoonized)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        ax8.imshow(diff_gray, cmap='hot')
        ax8.set_title('Processing Intensity Map', fontsize=14, fontweight='bold')
        ax8.axis('off')

        plt.suptitle('Advanced Cartoonization Analysis Dashboard',
                    fontsize=18, fontweight='bold', y=0.98)

        return fig

    def create_region_breakdown(self, image, face_regions):
        
        if not face_regions or 'masks' not in face_regions:
            return None

        masks = face_regions['masks']
        regions = ['left_eye', 'right_eye', 'nose', 'mouth', 'face']

        n_regions = len(regions)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        # Original
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontweight='bold')
        axes[0].axis('off')

        # Each region
        for idx, region in enumerate(regions, 1):
            if region in masks:
                mask = masks[region]

                # Apply mask
                masked = image.copy()
                mask_3d = np.stack([mask] * 3, axis=2)
                masked = np.where(mask_3d > 0, masked, 255)

                axes[idx].imshow(masked)
                axes[idx].set_title(f'{region.replace("_", " ").title()}',
                                  fontweight='bold')
                axes[idx].axis('off')

        # Hide unused subplot
        if n_regions + 1 < len(axes):
            axes[-1].axis('off')

        plt.tight_layout()
        return fig

class EnhancedCartoonizationPipeline:
    

    def __init__(self, preprocessor, cartoonizer, face_segmenter,
                 emotion_detector, identity_preserver):
        self.preprocessor = preprocessor
        self.cartoonizer = cartoonizer
        self.face_segmenter = face_segmenter
        self.emotion_detector = emotion_detector
        self.identity_preserver = identity_preserver

        print("✅ Enhanced Cartoonization Pipeline initialized")

    def process_image_full(self, image,
                          # Preprocessing options
                          enable_preprocessing=True,
                          denoise_strength='medium',
                          auto_enhance=True,
                          white_balance=True,
                          remove_shadows=False,
                          # Cartoonization options
                          preserve_identity=True,
                          emotion_adaptive=True,
                          region_aware=True,
                          # Visualization
                          show_steps=False):
        
        print("\n" + "=" * 60)
        print("ENHANCED PIPELINE PROCESSING")
        print("=" * 60)

        results = {
            'original': image.copy(),
            'steps': {},
            'metadata': {}
        }

        # PHASE 1: PREPROCESSING
        if enable_preprocessing:
            print("\n🔧 PHASE 1: PRE-PROCESSING")
            preprocessed = self.preprocessor.preprocess_full(
                image,
                denoise_strength=denoise_strength,
                auto_enhance=auto_enhance,
                white_balance=white_balance,
                remove_shadows=remove_shadows,
                correct_blur=True,
                target_size=1024
            )
            results['steps']['preprocessed'] = preprocessed
            image_to_process = preprocessed
        else:
            image_to_process = image

        # PHASE 2: EMOTION DETECTION
        print("\n😊 PHASE 2: EMOTION DETECTION")
        if emotion_adaptive:
            emotion_result = self.emotion_detector.detect_emotion(image_to_process)
            emotion = emotion_result['emotion']
            confidence = emotion_result['confidence']

            print(f"   Detected: {emotion.upper()} (confidence: {confidence:.2f})")
            results['metadata']['emotion'] = emotion
            results['metadata']['emotion_confidence'] = confidence

            # Apply emotion-based adjustments
            image_adjusted = self.emotion_detector.apply_emotion_adjustments(
                image_to_process, emotion
            )
            results['steps']['emotion_adjusted'] = image_adjusted
        else:
            emotion = 'neutral'
            image_adjusted = image_to_process.copy()

        # PHASE 3: FACE SEGMENTATION
        print("\n🎭 PHASE 3: FACE SEGMENTATION")
        if region_aware:
            result_tuple = self.face_segmenter.create_region_importance_map(image_adjusted)

            # Handle None return (no face detected)
            if result_tuple is None or result_tuple[0] is None:
                print("   ⚠️ No face detected, using global processing")
                importance_map = None
                face_regions = None
                region_aware = False
                results['metadata']['face_detected'] = False
            else:
                importance_map, face_regions = result_tuple
                print(f"   Face detected with {len(face_regions['landmarks'])} landmarks")
                results['metadata']['face_detected'] = True
                results['steps']['importance_map'] = importance_map
                results['steps']['face_masks'] = face_regions['masks']
        else:
            importance_map = None
            face_regions = None

        # PHASE 4: CARTOONIZATION
        print("\n🎨 PHASE 4: CARTOONIZATION")
        emotion_params = self.emotion_detector.get_parameters_for_emotion(emotion)

        if region_aware and importance_map is not None:
            print("   Using region-aware processing...")
            cartoonized = self._apply_region_aware_cartoon(
                image_adjusted,
                importance_map,
                emotion_params
            )
        else:
            print("   Using global processing...")
            cartoonized = self.cartoonizer.apply(
                image_adjusted,
                method='bilateral',
                intensity=1.0
            )

        results['steps']['cartoonized_initial'] = cartoonized

        # PHASE 5: IDENTITY PRESERVATION
        print("\n👤 PHASE 5: IDENTITY PRESERVATION")
        if preserve_identity and results['metadata'].get('face_detected', False):
            cartoonized = self.identity_preserver.iterative_refinement(
                image_to_process,
                cartoonized,
                self.cartoonizer,
                max_iterations=5
            )

            # Calculate final similarity
            original_encoding = self.identity_preserver.get_face_encoding(image_to_process)
            final_encoding = self.identity_preserver.get_face_encoding(cartoonized)

            if original_encoding is not None and final_encoding is not None:
                similarity = self.identity_preserver.calculate_similarity(
                    original_encoding, final_encoding
                )
                print(f"   Final identity similarity: {similarity:.3f}")
                results['metadata']['identity_similarity'] = similarity

        results['final'] = cartoonized

        # Visualize if requested
        if show_steps:
            self._visualize_full_pipeline(results)

        print("\n" + "=" * 60)
        print("✅ COMPLETE PIPELINE FINISHED!")
        print("=" * 60)

        return results

    def _apply_region_aware_cartoon(self, image, importance_map, emotion_params):
        """Apply region-aware cartoonization"""
        cartoon_strong = self.cartoonizer.apply(
            image,
            method='bilateral',
            intensity=1.0 * emotion_params['smoothing']
        )

        cartoon_weak = self.cartoonizer.apply(
            image,
            method='bilateral',
            intensity=0.5 * emotion_params['smoothing']
        )

        importance_map_3d = np.expand_dims(importance_map, axis=2)
        result = (cartoon_weak * importance_map_3d +
                 cartoon_strong * (1 - importance_map_3d))

        return result.astype(np.uint8)

    def _visualize_full_pipeline(self, results):
        import matplotlib.pyplot as plt

        steps_to_show = []
        titles = []

        # Original
        steps_to_show.append(results['original'])
        titles.append('Original')

        # Preprocessed
        if 'preprocessed' in results['steps']:
            steps_to_show.append(results['steps']['preprocessed'])
            titles.append('Preprocessed')

        # Emotion adjusted
        if 'emotion_adjusted' in results['steps']:
            steps_to_show.append(results['steps']['emotion_adjusted'])
            emotion = results['metadata'].get('emotion', 'unknown')
            titles.append(f'Emotion Adjusted\n({emotion})')

        # Importance map
        if 'importance_map' in results['steps']:
            imap = results['steps']['importance_map']
            imap_colored = cv2.applyColorMap(
                (imap * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            imap_colored = cv2.cvtColor(imap_colored, cv2.COLOR_BGR2RGB)
            steps_to_show.append(imap_colored)
            titles.append('Region Importance')

        # Initial cartoon
        if 'cartoonized_initial' in results['steps']:
            steps_to_show.append(results['steps']['cartoonized_initial'])
            titles.append('Initial Cartoon')

        # Final result
        steps_to_show.append(results['final'])
        similarity = results['metadata'].get('identity_similarity', 0)
        titles.append(f'Final Result\n(Similarity: {similarity:.2f})')

        # Plot
        n = len(steps_to_show)
        cols = 3
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes = axes.flatten() if n > 1 else [axes]

        for idx, (img, title) in enumerate(zip(steps_to_show, titles)):
            axes[idx].imshow(img)
            axes[idx].set_title(title, fontsize=11, fontweight='bold')
            axes[idx].axis('off')

        # Hide unused subplots
        for idx in range(len(steps_to_show), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()
    
class ImagePreprocessor:
   
    def __init__(self):
        print("✅ Image Preprocessor initialized")

    def auto_enhance(self, image):
       
        print("   🔧 Applying auto-enhancement...")

        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        return enhanced

    def denoise(self, image, strength='medium'):
        """
        Remove noise from image
        Args:
            strength: 'light', 'medium', 'strong'
        """
        print(f"   🔧 Applying {strength} denoising...")

        strength_map = {
            'light': (3, 3, 7, 21),
            'medium': (5, 5, 7, 21),
            'strong': (7, 7, 7, 21)
        }

        h, hColor, templateWindowSize, searchWindowSize = strength_map.get(
            strength, strength_map['medium']
        )

        # Non-local means denoising
        denoised = cv2.fastNlMeansDenoisingColored(
            image, None, h, hColor,
            templateWindowSize, searchWindowSize
        )

        return denoised

    def correct_white_balance(self, image):
        """
        Automatic white balance correction
        """
        print("   🔧 Correcting white balance...")

        # Simple gray world assumption
        result = image.copy().astype(np.float32)

        for i in range(3):  # RGB channels
            avg = result[:, :, i].mean()
            result[:, :, i] = result[:, :, i] * (128 / avg)

        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    def sharpen(self, image, amount=1.0):
        """
        Sharpen image
        Args:
            amount: Sharpening amount (0.5-2.0)
        """
        print(f"   🔧 Sharpening (amount: {amount})...")

        # Sharpening kernel
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]) * amount

        # Normalize
        kernel = kernel / kernel.sum() if kernel.sum() != 0 else kernel

        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened

    def remove_shadows(self, image):
        """
        Reduce shadow effects
        """
        print("   🔧 Removing shadows...")

        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply morphological operations to detect shadows
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        shadow_mask = cv2.morphologyEx(l, cv2.MORPH_CLOSE, kernel)

        # Normalize
        l_normalized = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)

        # Merge back
        result = cv2.merge([l_normalized, a, b])
        result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

        return result

    def correct_exposure(self, image, target_brightness=128):
        """
        Correct over/under exposure
        """
        print(f"   🔧 Correcting exposure (target: {target_brightness})...")

        # Calculate current brightness
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        current_brightness = gray.mean()

        # Calculate adjustment
        adjustment = target_brightness / current_brightness

        # Apply adjustment
        result = image.astype(np.float32) * adjustment
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def resize_smartly(self, image, target_size=1024):
        """
        Smart resize maintaining aspect ratio
        Args:
            target_size: Maximum dimension size
        """
        h, w = image.shape[:2]
        max_dim = max(h, w)

        if max_dim <= target_size:
            return image

        scale = target_size / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)

        print(f"   🔧 Resizing from {w}x{h} to {new_w}x{new_h}...")

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        return resized

    def detect_and_correct_blur(self, image):
        """
        Detect blur and apply correction if needed
        """
        # Calculate blur metric (Laplacian variance)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur_metric = cv2.Laplacian(gray, cv2.CV_64F).var()

        print(f"   🔍 Blur metric: {blur_metric:.2f}")

        # If image is blurry (low variance), apply sharpening
        if blur_metric < 100:
            print("   ⚠️ Blurry image detected, applying correction...")
            return self.sharpen(image, amount=1.5)

        return image

    def preprocess_full(self, image, denoise_strength='medium',
                       auto_enhance=True, white_balance=True,
                       remove_shadows=False, correct_blur=True,
                       target_size=1024):
        
        print("\n" + "=" * 60)
        print("PREPROCESSING IMAGE...")
        print("=" * 60)

        result = image.copy()

        # Step 1: Resize if needed
        result = self.resize_smartly(result, target_size)

        # Step 2: Denoise
        if denoise_strength:
            result = self.denoise(result, denoise_strength)

        # Step 3: White balance
        if white_balance:
            result = self.correct_white_balance(result)

        # Step 4: Remove shadows
        if remove_shadows:
            result = self.remove_shadows(result)

        # Step 5: Auto enhance
        if auto_enhance:
            result = self.auto_enhance(result)

        # Step 6: Blur correction
        if correct_blur:
            result = self.detect_and_correct_blur(result)

        print("✅ Preprocessing complete!")

        return result

    def visualize_preprocessing(self, original, preprocessed):
        """
        Visualize before/after preprocessing
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(original)
        axes[0].set_title('Original', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(preprocessed)
        axes[1].set_title('Preprocessed', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

# ── Initialize all objects once when the module loads ──
print("Initializing Cartoon Wizard engine...")

preprocessor       = ImagePreprocessor()
cartoonizer        = CartoonProcessor()
face_segmenter     = FaceSegmenter()
emotion_detector   = EmotionDetector()
identity_preserver = IdentityPreserver(similarity_threshold=0.6)
multi_style        = MultiStyleCartoonizer()
advanced_viz       = AdvancedVisualizer()

enhanced_pipeline  = EnhancedCartoonizationPipeline(
    preprocessor=preprocessor,
    cartoonizer=cartoonizer,
    face_segmenter=face_segmenter,
    emotion_detector=emotion_detector,
    identity_preserver=identity_preserver
)

print("Engine ready.")
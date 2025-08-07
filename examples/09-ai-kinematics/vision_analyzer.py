#!/usr/bin/env python3
"""
Vision analyzer for robotic manipulation using Claude vision API
Integrates with RGBD feed to provide AI-powered object detection and pixel coordinate selection
"""

import os
import sys
import json
import base64
import anthropic
import cv2
import numpy as np
from PIL import Image, ImageDraw
from typing import Optional, Tuple, Dict, Any
import tempfile
import time

class VisionAnalyzer:
    """
    Vision analyzer that uses Claude's vision API to identify objects and return pixel coordinates
    """
    
    def __init__(self):
        self.client = None
        self.setup_api_key()
        
    def setup_api_key(self):
        """Set up Anthropic API key"""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        
        if not api_key:
            print("[Vision] Warning: No API key found. AI vision features will not work.")
            print("[Vision] Set ANTHROPIC_API_KEY environment variable to enable AI features.")
            return
            
        # Initialize client
        try:
            self.client = anthropic.Anthropic(api_key=api_key)
            print("[Vision] Claude client initialized successfully")
        except Exception as e:
            print(f"[Vision] Error initializing Claude client: {e}")
            self.client = None
    
    def opencv_to_pil(self, cv_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image (BGR) to PIL Image (RGB)"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)
    
    def save_temp_image(self, cv_image: np.ndarray) -> str:
        """Save OpenCV image to temporary file and return path"""
        pil_image = self.opencv_to_pil(cv_image)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_path = temp_file.name
        temp_file.close()
        
        # Save image
        pil_image.save(temp_path, 'PNG')
        return temp_path
    
    def image_to_base64(self, cv_image: np.ndarray) -> str:
        """Convert OpenCV image to base64 string"""
        pil_image = self.opencv_to_pil(cv_image)
        
        # Convert to base64
        import io
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def refine_pixel_with_depth(self, rgb_x, rgb_y, depth, rgb, max_radius=8):
        """Move (rgb_x,rgb_y) onto nearest pixel on same depth plane for better accuracy."""
        if depth is None:
            return rgb_x, rgb_y
            
        h_r, w_r = rgb.shape[:2]
        h_d, w_d = depth.shape[:2]
        
        # Map RGB coordinates to depth coordinates
        d_x = int(rgb_x * w_d / w_r)
        d_y = int(rgb_y * h_d / h_r)
        
        # Check bounds
        if not (0 <= d_y < h_d and 0 <= d_x < w_d):
            return rgb_x, rgb_y
            
        base_z = depth[d_y, d_x]
        if base_z <= 0:
            return rgb_x, rgb_y
            
        # Search for best pixel on same depth surface
        best = (0, 0, float('inf'))  # dx, dy, score
        
        for r in range(1, max_radius + 1):
            for dy in range(-r, r + 1):
                dx_range = [-r, r] if abs(dy) == r else range(-r, r + 1)
                for dx in dx_range:
                    yy, xx = d_y + dy, d_x + dx
                    if 0 <= yy < h_d and 0 <= xx < w_d:
                        z = depth[yy, xx]
                        if 0 < z <= base_z * 1.05:  # Same surface (within 5%)
                            score = abs(z - base_z)
                            if score < best[2]:
                                best = (dx, dy, score)
        
        # Convert back to RGB coordinates
        dx, dy, _ = best
        refined_x = rgb_x + dx * w_r / w_d
        refined_y = rgb_y + dy * h_r / h_d
        
        return int(refined_x), int(refined_y)

    def analyze_image(self, rgb_image: np.ndarray, target_description: str, depth_frame: np.ndarray = None) -> Optional[Dict[str, Any]]:
        """
        Analyze RGB image to find target object and return pixel coordinates
        
        Args:
            rgb_image: OpenCV RGB image (BGR format)
            target_description: Natural language description of target object
            depth_frame: Optional depth frame for pixel refinement
            
        Returns:
            Dictionary with analysis results or None if failed
        """
        print(f"[DEBUG] Starting image analysis for: '{target_description}'")
        print(f"[DEBUG] Image shape: {rgb_image.shape}")
        print(f"[DEBUG] Client available: {self.client is not None}")
        
        if self.client is None:
            print("[Vision] Error: Claude client not initialized")
            print("[DEBUG] Cannot analyze - no client available")
            return None
        
        try:
            # Get original image dimensions
            img_height, img_width = rgb_image.shape[:2]
            print(f"[Vision] Analyzing image: {img_width}x{img_height}")
            
            # Upsample image for better Claude accuracy if needed
            scale_factor = 1.0
            if max(img_width, img_height) < 1200:
                scale_factor = min(2.0, 1200 / max(img_width, img_height))
                scaled_width = int(img_width * scale_factor)
                scaled_height = int(img_height * scale_factor)
                rgb_for_analysis = cv2.resize(rgb_image, (scaled_width, scaled_height), 
                                            interpolation=cv2.INTER_CUBIC)
                print(f"[Vision] Upsampled to: {scaled_width}x{scaled_height} (scale: {scale_factor:.2f})")
            else:
                rgb_for_analysis = rgb_image
                scaled_width, scaled_height = img_width, img_height
            
            # Convert image to base64
            print("[DEBUG] Converting image to base64...")
            base64_image = self.image_to_base64(rgb_for_analysis)
            print(f"[DEBUG] Base64 image length: {len(base64_image)}")
            
            # Create improved system prompt with precision emphasis
            system_prompt = f"""You are a Vision-Language-Action model for robotic manipulation with RGBD cameras.
            Analyze the provided RGB image to identify the specified object and determine target pixel coordinates.
            
            Image dimensions: {scaled_width}x{scaled_height}
            
            Look carefully at the image to identify: "{target_description}"
            
            CRITICAL: The output pixel must be the EXACT CENTER of the object ±2 pixels.
            Remember the camera image will be rescaled to a smaller depth frame (192x256).
            Large localization errors will break the robot grasping system.
            
            Provide both a bounding box around the object AND the best grasp point within it.
            Consider the object's center or the most stable grasping point.
            
            Return your response in this exact JSON format:
            {{
                "success": true,
                "bbox": [x0, y0, x1, y1],
                "target_pixel": [x, y],
                "confidence": 0.95,
                "object_found": true,
                "object_description": "brief description of what you found",
                "reasoning": "explanation of why you chose these coordinates",
                "approach_suggestion": "suggested approach direction or grip type"
            }}
            
            If you cannot find the object, return:
            {{
                "success": false,
                "bbox": null,
                "target_pixel": null,
                "confidence": 0.0,
                "object_found": false,
                "reasoning": "explanation of why object was not found"
            }}
            
            Ensure pixel coordinates are within bounds: x (0 to {scaled_width-1}), y (0 to {scaled_height-1}).
            The bounding box should tightly enclose the object with minimal background.
            """
            
            # Prepare message
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": f"Find and locate: {target_description}\n\nAnalyze the image and provide PRECISE target pixel coordinates and bounding box for robotic interaction. Focus on the exact center of the object."
                    }
                ]
            }
            
            print(f"[Vision] Sending request to Claude for: '{target_description}'")
            print("[DEBUG] Making API call to Anthropic...")
            print(f"[DEBUG] Using model: claude-sonnet-4-20250514")
            print(f"[DEBUG] Max tokens: 1000")
            print(f"[DEBUG] Client API key check: {self.client.api_key[:10]}...")
            print(f"[DEBUG] Client API key full: {self.client.api_key}")
            
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                system=system_prompt,
                messages=[message]
            )
            
            print("[DEBUG] API call completed successfully")
            
            # Parse response
            response_text = response.content[0].text.strip()
            print(f"[Vision] Claude response received")
            print(f"[DEBUG] Response length: {len(response_text)}")
            
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                result = json.loads(json_text)
                print(f"[DEBUG] JSON parsed successfully: {result.get('success', 'unknown')}")
            else:
                print("[Vision] No valid JSON found in response")
                print(f"[DEBUG] Raw response: {response_text[:200]}...")
                return None
            
            # Validate and process result
            if result.get('success') and result.get('target_pixel'):
                x, y = result['target_pixel']
                
                # Scale coordinates back to original image size
                x = int(x / scale_factor)
                y = int(y / scale_factor)
                
                # Process bounding box if available
                if result.get('bbox'):
                    x0, y0, x1, y1 = result['bbox']
                    # Scale bbox back to original size
                    x0 = int(x0 / scale_factor)
                    y0 = int(y0 / scale_factor)
                    x1 = int(x1 / scale_factor)
                    y1 = int(y1 / scale_factor)
                    result['bbox'] = [x0, y0, x1, y1]
                    
                    # Refine target pixel using bbox center if more accurate
                    bbox_center_x = (x0 + x1) // 2
                    bbox_center_y = (y0 + y1) // 2
                    
                    # Use bbox center if target pixel is far from it (likely less accurate)
                    if abs(x - bbox_center_x) > 20 or abs(y - bbox_center_y) > 20:
                        print(f"[Vision] Using bbox center ({bbox_center_x}, {bbox_center_y}) instead of target pixel ({x}, {y})")
                        x, y = bbox_center_x, bbox_center_y
                
                # Ensure coordinates are within original image bounds
                x = max(0, min(int(x), img_width - 1))
                y = max(0, min(int(y), img_height - 1))
                
                # Apply depth-based refinement if depth frame is available
                if depth_frame is not None:
                    print(f"[Vision] Original pixel: ({x}, {y})")
                    refined_x, refined_y = self.refine_pixel_with_depth(x, y, depth_frame, rgb_image)
                    if refined_x != x or refined_y != y:
                        print(f"[Vision] Depth-refined pixel: ({refined_x}, {refined_y})")
                        x, y = refined_x, refined_y
                
                result['target_pixel'] = [x, y]
                
                print(f"[Vision] Final object location: pixel ({x}, {y}) with confidence {result.get('confidence', 'N/A')}")
                return result
            else:
                print(f"[Vision] Object not found: {result.get('reasoning', 'No reason provided')}")
                return result
                
        except json.JSONDecodeError as e:
            print(f"[Vision] JSON decode error: {e}")
            print(f"[Vision] Raw response: {response_text}")
            return None
        except Exception as e:
            print(f"[Vision] Analysis error: {e}")
            print(f"[DEBUG] Exception during analysis: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_visualization(self, rgb_image: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """
        Create visualization overlay on the RGB image showing the target location
        
        Args:
            rgb_image: Original RGB image
            result: Analysis result dictionary
            
        Returns:
            Image with visualization overlay
        """
        vis_image = rgb_image.copy()
        
        if not result.get('success') or not result.get('target_pixel'):
            # Show "not found" message
            cv2.putText(vis_image, "Object not found", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return vis_image
        
        x, y = result['target_pixel']
        confidence = result.get('confidence', 0.0)
        
        # Draw bounding box if available
        if result.get('bbox'):
            x0, y0, x1, y1 = result['bbox']
            # Draw bounding box rectangle
            cv2.rectangle(vis_image, (x0, y0), (x1, y1), (255, 0, 0), 2)
            # Add bbox label
            cv2.putText(vis_image, "BBox", (x0, y0 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw crosshair
        cross_size = 20
        thickness = 2
        
        # Draw cross lines
        cv2.line(vis_image, (x - cross_size, y), (x + cross_size, y), (0, 255, 0), thickness)
        cv2.line(vis_image, (x, y - cross_size), (x, y + cross_size), (0, 255, 0), thickness)
        
        # Draw circle around target
        cv2.circle(vis_image, (x, y), cross_size, (0, 255, 0), thickness)
        
        # Add text info
        text = f"Target: ({x}, {y})"
        cv2.putText(vis_image, text, (x + cross_size + 5, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        confidence_text = f"Confidence: {confidence:.2f}"
        cv2.putText(vis_image, confidence_text, (x + cross_size + 5, y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add object description if available
        if result.get('object_description'):
            desc_text = result['object_description'][:30] + "..." if len(result['object_description']) > 30 else result['object_description']
            cv2.putText(vis_image, desc_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_image
    
    def is_available(self) -> bool:
        """Check if vision analyzer is available (API key set)"""
        return self.client is not None

# Test function
def test_vision_analyzer():
    """Test the vision analyzer with a sample prompt"""
    print("Testing Vision Analyzer")
    print("=" * 50)
    
    analyzer = VisionAnalyzer()
    
    if not analyzer.is_available():
        print("Vision analyzer not available (no API key)")
        return
    
    # Create a test image (colorful pattern)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some colored shapes for testing
    cv2.rectangle(test_image, (100, 100), (200, 200), (0, 255, 255), -1)  # Yellow rectangle
    cv2.circle(test_image, (400, 300), 50, (0, 0, 255), -1)  # Red circle
    cv2.rectangle(test_image, (300, 50), (500, 150), (255, 0, 0), -1)  # Blue rectangle
    
    # Add text labels
    cv2.putText(test_image, "Yellow Box", (110, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(test_image, "Red Circle", (350, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(test_image, "Blue Box", (320, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Test analysis
    target_description = "yellow box"
    print(f"Analyzing for: {target_description}")
    
    result = analyzer.analyze_image(test_image, target_description)
    
    if result:
        print("\nAnalysis Results:")
        print(json.dumps(result, indent=2))
        
        # Create visualization
        vis_image = analyzer.create_visualization(test_image, result)
        
        # Show result
        cv2.imshow('Test Image', test_image)
        cv2.imshow('Vision Analysis Result', vis_image)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Analysis failed")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_vision_analyzer()
    else:
        print("Vision Analyzer Module")
        print("Use 'python vision_analyzer.py test' to run test") 
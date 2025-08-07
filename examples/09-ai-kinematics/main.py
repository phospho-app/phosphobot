import numpy as np
import cv2
from rgbd_feed import DemoApp
from vision_analyzer import VisionAnalyzer
import requests
import os
import sys

class KinematicsApp(DemoApp):
    def __init__(self):
        super().__init__()
        self.intrinsic_mat = None
        self.depth_intrinsic_mat = None
        self.rgb_intrinsic_mat = None
        
        # Interactive configuration
        self.setup_configuration()
        
        # Initialize vision analyzer after API key is set
        self.vision_analyzer = VisionAnalyzer()
            
        self.vision_result = None
        
        self.current_rgb = None
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 10
        params.adaptiveThreshConstant = 7
        params.minMarkerPerimeterRate = 0.1
        params.maxMarkerPerimeterRate = 0.8
        params.polygonalApproxAccuracyRate = 0.05
        params.minCornerDistanceRate = 0.1
        params.minDistanceToBorder = 3
        params.minOtsuStdDev = 5.0
        params.perspectiveRemovePixelPerCell = 4
        params.perspectiveRemoveIgnoredMarginPerCell = 0.13
        params.maxErroneousBitsInBorderRate = 0.2
        params.minMarkerDistanceRate = 0.05
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 30
        params.cornerRefinementMinAccuracy = 0.1
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, params)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        self.aruco_marker_id = 0
        self.aruco_marker_size = 0.04
        self.current_pose_T_cam_marker = None
        
        self.clicked_pixel = None
        self.clicked_marker = None
        
        self.position_tolerance = 0.03
        self.orientation_tolerance = 0.2

    def setup_configuration(self):
        """Interactive setup for robot endpoint, API key, and mode"""
        print("AI Kinematics Configuration")
        print("=" * 40)
        
        # Robot endpoint
        default_endpoint = "http://192.168.178.190/move/absolute?robot_id=0"
        endpoint_input = input(f"Robot endpoint (press Enter for default: {default_endpoint}): ").strip()
        self.robot_endpoint = endpoint_input if endpoint_input else default_endpoint
        print(f"Using robot endpoint: {self.robot_endpoint}")
        
        # API key
        api_key_input = input("Anthropic API key (press Enter to skip AI features): ").strip()
        if api_key_input:
            os.environ['ANTHROPIC_API_KEY'] = api_key_input
            print("API key set successfully")
        else:
            print("No API key provided - AI features will be disabled")
        
        # Mode selection
        print("\nMode selection:")
        print("1. Manual mode - click to select targets")
        print("2. AI mode - automatic object detection")
        mode_input = input("Select mode (1 or 2): ").strip()
        
        if mode_input == "2":
            self.vision_mode = True
            target_input = input("Enter target object description (e.g., 'red cup'): ").strip()
            self.vision_target_description = target_input if target_input else "red cup"
            print(f"AI mode enabled - looking for: {self.vision_target_description}")
        else:
            self.vision_mode = False
            print("Manual mode enabled")
        
        print("Configuration complete")
        print("=" * 40)

    def estimate_pose_from_corners(self, corners_2d):
        obj_pts = np.array([[0, 0, 0],
                            [self.aruco_marker_size, 0, 0],
                            [self.aruco_marker_size, self.aruco_marker_size, 0],
                            [0, self.aruco_marker_size, 0]], dtype=np.float32)
        success, rvec, tvec = cv2.solvePnP(obj_pts, corners_2d, self.rgb_intrinsic_mat, self.dist_coeffs)
        if not success:
            return None
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return T

    def detect_aruco_marker(self):
        if self.current_rgb is None or self.rgb_intrinsic_mat is None:
            return False
            
        gray = cv2.cvtColor(self.current_rgb, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        
        if ids is not None and self.aruco_marker_id in ids.flatten():
            idx = list(ids.flatten()).index(self.aruco_marker_id)
            self.current_pose_T_cam_marker = self.estimate_pose_from_corners(corners[idx][0])
            return self.current_pose_T_cam_marker is not None
        else:
            self.current_pose_T_cam_marker = None
            return False

    def pixel_to_marker_coordinates(self, rgb_x, rgb_y):
        if self.current_pose_T_cam_marker is None:
            print("Cannot compute - marker pose not available")
            return None
            
        depth_val = self.get_depth_at_rgb_coordinate(rgb_x, rgb_y)
        if depth_val is None or depth_val <= 0:
            print("Invalid depth at pixel")
            return None

        fx, fy = self.rgb_intrinsic_mat[0, 0], self.rgb_intrinsic_mat[1, 1]
        cx, cy = self.rgb_intrinsic_mat[0, 2], self.rgb_intrinsic_mat[1, 2]
        X = (rgb_x - cx) * depth_val / fx
        Y = (rgb_y - cy) * depth_val / fy
        Z = depth_val
        pt_cam = np.array([X, Y, Z, 1.0])
        
        pt_marker = np.linalg.inv(self.current_pose_T_cam_marker) @ pt_cam
        marker_coords = pt_marker[:3]
        
        print(f"RGB pixel ({rgb_x},{rgb_y}) depth {depth_val:.3f} m -> Camera ({X:.3f},{Y:.3f},{Z:.3f}) -> Marker ({marker_coords[0]:.3f},{marker_coords[1]:.3f},{marker_coords[2]:.3f}) m")
        
        return marker_coords

    def convert_to_robot_frame(self, aruco_coords_m):
        ax, ay, az = aruco_coords_m
        
        robot_x_m = -ay
        robot_y_m = -ax
        robot_z_m = -az - 0.13
        
        rx_cm = robot_x_m * 100.0
        ry_cm = robot_y_m * 100.0
        rz_cm = robot_z_m * 100.0
        
        return rx_cm, ry_cm, rz_cm

    def prompt_and_move_robot(self, rel_coords):
        rx_cm, ry_cm, rz_cm = self.convert_to_robot_frame(rel_coords)
        print(f"Debug (ArUco frame m): X={rel_coords[0]:.3f}  Y={rel_coords[1]:.3f}  Z={rel_coords[2]:.3f}")
        print(f"Converted to robot frame cm: X={rx_cm:.1f}  Y={ry_cm:.1f}  Z={rz_cm:.1f}")
        
        try:
            payload = {
                "x": rx_cm,
                "y": ry_cm,
                "z": rz_cm,
                "open": 0,
                "max_trials": 100,
                "position_tolerance": self.position_tolerance,
                "orientation_tolerance": self.orientation_tolerance
            }
            resp = requests.post(self.robot_endpoint, json=payload, timeout=10)
            print(f"HTTP status: {resp.status_code}")
            if resp.ok:
                print("Robot move command sent successfully")
            else:
                print(f"Robot move failed: {resp.text}")
        except Exception as e:
            print(f"Error sending move command: {e}")
    
    def pixel_to_3d_position(self, u, v, depth_value, camera_matrix=None):
        K = camera_matrix if camera_matrix is not None else self.depth_intrinsic_mat
        
        if K is None:
            print("Warning: No camera intrinsic matrix available yet")
            return None
            
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        
        X = (u - cx) * depth_value / fx
        Y = (v - cy) * depth_value / fy
        Z = depth_value
        
        return (X, Y, Z)
    
    def scale_coordinates_rgb_to_depth(self, rgb_x, rgb_y):
        if self.current_rgb is None or self.depth is None:
            return rgb_x, rgb_y
            
        rgb_height, rgb_width = self.current_rgb.shape[:2]
        depth_height, depth_width = self.depth.shape[:2]
        
        if rgb_width != depth_width or rgb_height != depth_height:
            scale_x = depth_width / rgb_width
            scale_y = depth_height / rgb_height
            depth_x = int(rgb_x * scale_x)
            depth_y = int(rgb_y * scale_y)
            return depth_x, depth_y
        else:
            return rgb_x, rgb_y
    
    def get_depth_at_rgb_coordinate(self, rgb_x, rgb_y):
        if self.depth is None:
            return None
            
        depth_x, depth_y = self.scale_coordinates_rgb_to_depth(rgb_x, rgb_y)
        
        if 0 <= depth_y < self.depth.shape[0] and 0 <= depth_x < self.depth.shape[1]:
            return float(self.depth[depth_y, depth_x])
        else:
            return None
    
    def on_mouse(self, event, x, y, flags, param):
        if self.depth is not None and event == cv2.EVENT_MOUSEMOVE:
            if 0 <= y < self.depth.shape[0] and 0 <= x < self.depth.shape[1]:
                depth_value = float(self.depth[y, x])
                
                if self.current_rgb is not None:
                    rgb_height, rgb_width = self.current_rgb.shape[:2]
                    depth_height, depth_width = self.depth.shape[:2]
                    
                    if rgb_width != depth_width or rgb_height != depth_height:
                        scale_x = rgb_width / depth_width
                        scale_y = rgb_height / depth_height
                        rgb_x = int(x * scale_x)
                        rgb_y = int(y * scale_y)
                    else:
                        rgb_x, rgb_y = x, y
                else:
                    rgb_x, rgb_y = x, y
                
                self.last_mouse_pos = (rgb_x, rgb_y)
                self.depth_value = depth_value
                
                if not self.vision_mode:
                    if self.rgb_intrinsic_mat is not None:
                        pos_3d = self.pixel_to_3d_position(rgb_x, rgb_y, depth_value, self.rgb_intrinsic_mat)
                        if pos_3d:
                            X, Y, Z = pos_3d
                            print(f"Manual D({x},{y})->RGB({rgb_x},{rgb_y}) -> Depth: {depth_value:.3f}m, 3D: ({X:.3f},{Y:.3f},{Z:.3f})")
                    else:
                        print(f"Depth at D({x},{y}): {depth_value:.3f} meters")
        
        elif self.vision_mode and event == cv2.EVENT_LBUTTONDOWN:
            if self.depth is not None and 0 <= y < self.depth.shape[0] and 0 <= x < self.depth.shape[1]:
                depth_value = float(self.depth[y, x])
                
                if self.current_rgb is not None:
                    rgb_height, rgb_width = self.current_rgb.shape[:2]
                    depth_height, depth_width = self.depth.shape[:2]
                    
                    if rgb_width != depth_width or rgb_height != depth_height:
                        scale_x = rgb_width / depth_width
                        scale_y = rgb_height / depth_height
                        rgb_x = int(x * scale_x)
                        rgb_y = int(y * scale_y)
                    else:
                        rgb_x, rgb_y = x, y
                else:
                    rgb_x, rgb_y = x, y
                
                self.last_mouse_pos = (rgb_x, rgb_y)
                self.depth_value = depth_value
                
                if self.rgb_intrinsic_mat is not None:
                    pos_3d = self.pixel_to_3d_position(rgb_x, rgb_y, depth_value, self.rgb_intrinsic_mat)
                    if pos_3d:
                        X, Y, Z = pos_3d
                        print(f"Vision Click D({x},{y})->RGB({rgb_x},{rgb_y}) -> Depth: {depth_value:.3f}m, 3D: ({X:.3f},{Y:.3f},{Z:.3f})")
    
        if not self.vision_mode and event == cv2.EVENT_LBUTTONDOWN and self.depth is not None:
            if 0 <= y < self.depth.shape[0] and 0 <= x < self.depth.shape[1]:
                depth_value = float(self.depth[y, x])
                if depth_value <= 0:
                    print("Invalid depth at selected pixel")
                    return
                    
                if self.current_rgb is not None:
                    rgb_height, rgb_width = self.current_rgb.shape[:2]
                    depth_height, depth_width = self.depth.shape[:2]
                    if rgb_width != depth_width or rgb_height != depth_height:
                        scale_x = rgb_width / depth_width
                        scale_y = rgb_height / depth_height
                        rgb_x = int(x * scale_x)
                        rgb_y = int(y * scale_y)
                    else:
                        rgb_x, rgb_y = x, y
                else:
                    rgb_x, rgb_y = x, y
                    
                self.clicked_pixel = (rgb_x, rgb_y)
                self.clicked_marker = None
                
                if self.current_pose_T_cam_marker is not None:
                    marker_coords = self.pixel_to_marker_coordinates(rgb_x, rgb_y)
                    if marker_coords is not None:
                        self.clicked_marker = marker_coords
                        self.prompt_and_move_robot(marker_coords)
                    else:
                        print("Could not convert to marker coordinates")
                else:
                    print("No marker detected - cannot transform coordinates")

    def start_processing_stream(self, mode):
        print("="*60)
        print("STARTING RGBD STREAM")
        print("="*60)
        
        if mode == 'manual':
            print("MANUAL MODE ACTIVE")
            print("Instructions:")
            print("  • Move mouse over depth window to see 3D positions")
            print("  • Press 'q' to quit")
            
        elif mode == 'ai':
            print("AI VISION MODE ACTIVE")
            print(f"Task: {self.vision_target_description}")
            print("Instructions:")
            print("  • AI will analyze the first frame automatically")
            print("  • Green crosshair shows AI target")
            print("  • Press 'q' to quit")
        
        print("Waiting for RGBD stream...")
        
        first_frame_processed = False
        
        while True:
            self.event.wait()

            depth = self.session.get_depth_frame()
            rgb = self.session.get_rgb_frame()
            confidence = self.session.get_confidence_frame()
            
            depth_intrinsic_coeffs = self.session.get_intrinsic_mat()
            
            depth_intrinsic_mat = self.get_intrinsic_mat_from_coeffs(depth_intrinsic_coeffs)
            rgb_intrinsic_mat = depth_intrinsic_mat.copy()
            
            camera_pose = self.session.get_camera_pose()

            self.intrinsic_mat = depth_intrinsic_mat
            self.depth_intrinsic_mat = depth_intrinsic_mat
            self.rgb_intrinsic_mat = rgb_intrinsic_mat
            
            if not hasattr(self, '_intrinsics_printed'):
                print(f"Using shared intrinsics - fx:{depth_intrinsic_mat[0,0]:.1f} fy:{depth_intrinsic_mat[1,1]:.1f} cx:{depth_intrinsic_mat[0,2]:.1f} cy:{depth_intrinsic_mat[1,2]:.1f}")
                print("iPhone RGB and depth cameras are co-located with very similar intrinsics")
                self._intrinsics_printed = True

            if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                depth = cv2.flip(depth, 1)
                rgb = cv2.flip(rgb, 1)

            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            self.depth = depth
            self.current_rgb = rgb.copy()
            
            if self.current_rgb is not None and self.rgb_intrinsic_mat is not None:
                was_detected = self.current_pose_T_cam_marker is not None
                self.detect_aruco_marker()
                if self.current_pose_T_cam_marker is not None and not was_detected:
                    print(f"Marker {self.aruco_marker_id} detected and pose estimated")
                elif self.current_pose_T_cam_marker is None and was_detected:
                    print(f"Marker {self.aruco_marker_id} lost")
            
            if mode == 'ai' and not first_frame_processed and self.vision_target_description and self.rgb_intrinsic_mat is not None:
                print("Analyzing first frame with AI...")
                
                result = self.vision_analyzer.analyze_image(self.current_rgb, self.vision_target_description, self.depth)
                
                if result and result.get('success'):
                    self.vision_result = result
                    x, y = result['target_pixel']
                    
                    rgb_height, rgb_width = self.current_rgb.shape[:2]
                    depth_height, depth_width = self.depth.shape[:2]
                    
                    if rgb_width != depth_width or rgb_height != depth_height:
                        scale_x = depth_width / rgb_width
                        scale_y = depth_height / rgb_height
                        depth_x = int(x * scale_x)
                        depth_y = int(y * scale_y)
                        print(f"Scaling coordinates: RGB({x},{y}) -> Depth({depth_x},{depth_y})")
                        print(f"RGB: {rgb_width}x{rgb_height}, Depth: {depth_width}x{depth_height}")
                    else:
                        depth_x, depth_y = x, y
                    
                    if 0 <= depth_y < self.depth.shape[0] and 0 <= depth_x < self.depth.shape[1]:
                        depth_value = float(self.depth[depth_y, depth_x])
                        
                        if depth_value > 0:
                            pos_3d = self.pixel_to_3d_position(x, y, depth_value, self.rgb_intrinsic_mat)
                            if pos_3d:
                                X, Y, Z = pos_3d
                                print("AI FOUND TARGET!")
                                print(f"Object: {result.get('object_description', 'Found')}")
                                print(f"RGB Pixel: ({x}, {y})")
                                print(f"Depth Pixel: ({depth_x}, {depth_y})")
                                print(f"Depth: {depth_value:.3f}m")
                                print(f"3D Position: ({X:.3f}, {Y:.3f}, {Z:.3f})m")
                                if self.current_pose_T_cam_marker is not None:
                                    marker_coords = self.pixel_to_marker_coordinates(x, y)
                                    if marker_coords is not None:
                                        print(f"Relative to marker: ({marker_coords[0]:.3f},{marker_coords[1]:.3f},{marker_coords[2]:.3f})")
                                        
                                        self.prompt_and_move_robot(marker_coords)
                                    else:
                                        print("Could not convert to marker coordinates")
                                else:
                                    print("No marker detected - cannot transform coordinates")
                                print(f"Confidence: {result.get('confidence', 'N/A')}")
                                
                                self.last_mouse_pos = (x, y)
                                self.depth_value = depth_value
                            else:
                                print("Could not calculate 3D position")
                        else:
                            print(f"Invalid depth at target: {depth_value}")
                    else:
                        print(f"Target coordinates out of bounds: RGB({x},{y}) -> Depth({depth_x},{depth_y})")
                        print(f"Depth frame size: {depth_width}x{depth_height}")
                else:
                    print("AI could not find the target object")
                    if result:
                        print(f"Reason: {result.get('reasoning', 'Unknown')}")
                    else:
                        print("No result returned from vision analyzer")
                
                first_frame_processed = True

            if self.vision_mode and self.vision_result:
                rgb = self.vision_analyzer.create_visualization(rgb, self.vision_result)
            
            if self.current_pose_T_cam_marker is not None:
                gray = cv2.cvtColor(self.current_rgb, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = self.aruco_detector.detectMarkers(gray)
                
                if ids is not None and self.aruco_marker_id in ids.flatten():
                    idx = list(ids.flatten()).index(self.aruco_marker_id)
                    cv2.aruco.drawDetectedMarkers(rgb, [corners[idx]])
                    cv2.drawFrameAxes(rgb, self.rgb_intrinsic_mat, self.dist_coeffs, 
                                    cv2.Rodrigues(self.current_pose_T_cam_marker[:3,:3])[0], 
                                    self.current_pose_T_cam_marker[:3,3], self.aruco_marker_size*0.5)
                    
                    if self.clicked_pixel and self.clicked_marker is not None:
                        click_u, click_v = self.clicked_pixel
                        cv2.circle(rgb, (click_u, click_v), 6, (0, 0, 255), 2)
                        label = f"({self.clicked_marker[0]:.2f},{self.clicked_marker[1]:.2f},{self.clicked_marker[2]:.2f})m"
                        cv2.putText(rgb, label, (click_u + 8, click_v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            
            mode_text = f"Mode: {'AI VISION' if self.vision_mode else 'MANUAL'}"
            cv2.putText(rgb, mode_text, (10, rgb.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if self.vision_mode and self.vision_target_description:
                target_text = f"Task: {self.vision_target_description[:40]}..."
                cv2.putText(rgb, target_text, (10, rgb.shape[0] - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if self.last_mouse_pos and self.depth_value is not None:
                rgb_x, rgb_y = self.last_mouse_pos
                
                # Add crosshair to RGB view
                color_rgb = (0, 255, 0) if self.vision_mode else (255, 255, 255)
                if self.vision_mode:
                    cv2.line(rgb, (rgb_x - 15, rgb_y), (rgb_x + 15, rgb_y), color_rgb, 2)
                    cv2.line(rgb, (rgb_x, rgb_y - 15), (rgb_x, rgb_y + 15), color_rgb, 2)
                    cv2.circle(rgb, (rgb_x, rgb_y), 10, color_rgb, 2)
                else:
                    cv2.circle(rgb, (rgb_x, rgb_y), 5, color_rgb, -1)

            cv2.imshow('RGB', rgb)

            depth_vis = depth.copy()
            
            if self.last_mouse_pos and self.depth_value is not None:
                rgb_x, rgb_y = self.last_mouse_pos
                
                depth_x, depth_y = self.scale_coordinates_rgb_to_depth(rgb_x, rgb_y)
                
                color = (0, 255, 0) if self.vision_mode else (255, 255, 255)
                
                overlay = depth_vis.copy()
                if self.vision_mode:
                    cv2.line(overlay, (depth_x - 15, depth_y), (depth_x + 15, depth_y), color, 2)
                    cv2.line(overlay, (depth_x, depth_y - 15), (depth_x, depth_y + 15), color, 2)
                    cv2.circle(overlay, (depth_x, depth_y), 10, color, 2)
                else:
                    cv2.circle(overlay, (depth_x, depth_y), 5, color, -1)
                
                alpha = 0.7 if self.vision_mode else 0.5
                depth_vis = cv2.addWeighted(overlay, alpha, depth_vis, 1 - alpha, 0)
                
                if self.rgb_intrinsic_mat is not None:
                    pos_3d = self.pixel_to_3d_position(rgb_x, rgb_y, self.depth_value, self.rgb_intrinsic_mat)
                    if pos_3d:
                        X, Y, Z = pos_3d
                        if self.vision_mode:
                            text = f"AI: {self.depth_value:.3f}m ({X:.2f},{Y:.2f},{Z:.2f})"
                        else:
                            text = f"{self.depth_value:.3f}m ({X:.2f},{Y:.2f},{Z:.2f})"
                    else:
                        text = f"{self.depth_value:.3f} m"
                else:
                    text = f"{self.depth_value:.3f} m"
                    
                cv2.putText(depth_vis, text, (depth_x + 20, depth_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if mode == 'ai':
                mode_indicator = "AI VISION"
                color = (0, 255, 0)
            else:
                mode_indicator = "MANUAL"
                color = (255, 255, 255)
            
            cv2.putText(depth_vis, mode_indicator, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imshow(self.depth_window_name, depth_vis)

            if confidence.shape[0] > 0 and confidence.shape[1] > 0:
                cv2.imshow('Confidence', confidence * 100)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break

            self.event.clear()

def main():
    print("AI Kinematics - RGBD 3D Position Calculator")
    print("=" * 50)
    
    app = KinematicsApp()
    
    try:
        print("Attempting to connect to Record3D device...")
        app.connect_to_device(dev_idx=0)
        print("Connected to Record3D device successfully")

        # Determine mode based on vision_mode setting
        mode = 'ai' if app.vision_mode else 'manual'
        app.start_processing_stream(mode)
    except RuntimeError as e:
        print(f"Device connection error: {e}")
        print("Make sure:")
        print("   • Record3D app is running on your iPhone")
        print("   • iPhone is connected to the same network")
        print("   • No firewall is blocking the connection")
        return
    except ImportError as e:
        print(f"Import error: {e}")
        print("Install missing dependencies:")
        print("   pip install -r requirements.txt")
        return
    except KeyboardInterrupt:
        print("Application stopped by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            cv2.destroyAllWindows()
        except:
            pass

if __name__ == '__main__':
    main() 
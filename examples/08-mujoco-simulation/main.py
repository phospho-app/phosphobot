#!/usr/bin/env python3
"""
SO-100 Integrated Control System
Combines real-time simulation control with physical robot mirroring
"""
import time
import numpy as np
import requests
import math
import threading
from so100_simulation import SO100Simulation

class SO100IntegratedControl:
    def __init__(self):
        print("=" * 70)
        print("SO-100 INTEGRATED CONTROL SYSTEM")
        print("Real-time Simulation + Robot Mirroring")
        print("=" * 70)
        
        # Interactive configuration
        self.setup_configuration()
        
        print("Initializing simulation with MuJoCo viewer...")
        self.sim = SO100Simulation()
        
        self.mirror_frequency = 0.05
        self.mirroring_active = False
        self.mirror_thread = None
        
        self.joint_calibration = {
            "offsets": [0.0, math.radians(-90.0), math.radians(90.0), 0.0, math.radians(-180.0), 0.0],
            "directions": [1, 1, 1, 1, 1, 1],
            "mapping": [0, 1, 2, 3, 4, 5]
        }
        
        print("Using hardcoded joint calibration:")
        print("  Joint 1 (Pitch): -90° offset")
        print("  Joint 2 (Elbow): +90° offset") 
        print("  Joint 4 (Wrist_Roll): -180° offset")
        
        print("System ready!")
        print("You can interact with the MuJoCo viewer (rotate, zoom, etc.)")
        
        time.sleep(2)
        
        num_joints = len(self.sim.joint_names)
        print(f"Robot has {num_joints} joints:")
        for i, name in enumerate(self.sim.joint_names):
            print(f"  {i}: {name}")

    def setup_configuration(self):
        """Interactive setup for robot endpoint"""
        print("SO-100 Configuration")
        print("=" * 40)
        
        # Robot endpoint
        default_endpoint = "http://192.168.178.190:80"
        endpoint_input = input(f"Robot endpoint (press Enter for default: {default_endpoint}): ").strip()
        self.phosphobot_ip = "192.168.178.190"
        self.phosphobot_port = 80
        
        if endpoint_input:
            # Parse custom endpoint
            if endpoint_input.startswith("http://"):
                endpoint_input = endpoint_input[7:]
            if ":" in endpoint_input:
                self.phosphobot_ip = endpoint_input.split(":")[0]
                self.phosphobot_port = int(endpoint_input.split(":")[1])
            else:
                self.phosphobot_ip = endpoint_input
                self.phosphobot_port = 80
        
        self.joints_read_url = f"http://{self.phosphobot_ip}:{self.phosphobot_port}/joints/read"
        print(f"Using robot endpoint: {self.joints_read_url}")
        
        print("Configuration complete")
        print("=" * 40)

    def apply_calibration(self, raw_angles):
        if len(raw_angles) != 6:
            return raw_angles
            
        calibrated_angles = [0.0] * 6
        
        for sim_joint_idx in range(6):
            phys_joint_idx = self.joint_calibration["mapping"][sim_joint_idx]
            if phys_joint_idx < len(raw_angles):
                angle = raw_angles[phys_joint_idx]
                angle *= self.joint_calibration["directions"][phys_joint_idx]
                angle += self.joint_calibration["offsets"][phys_joint_idx]
                calibrated_angles[sim_joint_idx] = angle
        
        return calibrated_angles

    def read_physical_robot_angles(self):
        try:
            response = requests.post(self.joints_read_url, timeout=2)
            response.raise_for_status()
            data = response.json()
            
            if "angles" in data and data["angles"]:
                return data["angles"]
            return None
        except Exception as e:
            if hasattr(self, '_last_error_time'):
                if time.time() - self._last_error_time > 5:
                    print(f"Error reading robot angles: {e}")
                    self._last_error_time = time.time()
            else:
                self._last_error_time = time.time()
            return None

    def mirror_robot_worker(self):
        print(f"Starting robot mirroring at {1/self.mirror_frequency:.1f} Hz")
        print("Physical robot movements will be mirrored in simulation")
        print("Press 's' + Enter to stop mirroring")
        
        consecutive_errors = 0
        max_consecutive_errors = 20
        threading_errors = 0
        
        time.sleep(2.0)
        
        while self.mirroring_active:
            try:
                raw_angles = self.read_physical_robot_angles()
                
                if raw_angles is not None and len(raw_angles) == 6:
                    calibrated_angles = self.apply_calibration(raw_angles)
                    
                    try:
                        self.sim.set_joint_angles(calibrated_angles)
                        consecutive_errors = 0
                        
                    except Exception as e:
                        if "copyDataVisual" in str(e) or "stack is in use" in str(e):
                            threading_errors += 1
                            if threading_errors % 10 == 0:
                                print(f"Skipped {threading_errors} updates due to MuJoCo threading (normal)")
                        else:
                            consecutive_errors += 1
                            if consecutive_errors % 5 == 0:
                                print(f"Simulation update error ({consecutive_errors}): {e}")
                else:
                    consecutive_errors += 1
                    
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors % 5 == 0:
                    print(f"Mirror error ({consecutive_errors}): {e}")
            
            if consecutive_errors >= max_consecutive_errors:
                print(f"Stopping mirroring due to {consecutive_errors} consecutive errors")
                self.mirroring_active = False
                break
                
            time.sleep(self.mirror_frequency)
        
        if threading_errors > 0:
            print(f"Total threading conflicts handled: {threading_errors}")
        print("Robot mirroring stopped")

    def start_mirroring(self):
        if self.mirroring_active:
            print("Mirroring is already active")
            return
            
        print("Testing connection to physical robot...")
        test_angles = self.read_physical_robot_angles()
        if test_angles is None:
            print("Cannot connect to physical robot")
            print(f"Make sure robot is accessible at {self.joints_read_url}")
            return
            
        print("Connection successful!")
        
        self.mirroring_active = True
        self.mirror_thread = threading.Thread(target=self.mirror_robot_worker, daemon=True)
        self.mirror_thread.start()

    def stop_mirroring(self):
        if not self.mirroring_active:
            print("Mirroring is not active")
            return
            
        self.mirroring_active = False
        if self.mirror_thread:
            self.mirror_thread.join(timeout=2)
        print("Mirroring stopped")

    def show_workspace_info(self):
        print("\n" + "="*50)
        print("WORKSPACE LIMITS:")
        print("  X (left/right): -0.338 to +0.338 meters (±34cm)")
        print("  Y (forward):    -0.383 to -0.045 meters (4.5-38cm forward)")
        print("  Z (up/down):     0.096 to  0.315 meters (10-32cm high)")
        print("\nREMEMBER: Y must be NEGATIVE (negative = forward from base)")
        print("="*50)
    
    def show_commands(self):
        print("\n" + "="*50)
        print("AVAILABLE COMMANDS:")
        print("\nROBOT MIRRORING:")
        print("  'mirror' - Start real-time mirroring from physical robot")
        print("  'stop' or 's' - Stop mirroring")
        print("\nINFORMATION:")
        print("  'current' - Show current pose")
        print("  'workspace' - Show workspace limits")
        print("  'commands' - Show this help")
        print("\nEXIT:")
        print("  'q' - Quit")
        print("="*50)
    
    def get_current_pose(self):
        try:
            pos = self.sim.get_end_effector_position()
            angles = self.sim.get_current_joint_angles()
            
            print(f"\nCURRENT STATE:")
            print(f"  End effector position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m")
            print(f"  Joint angles (degrees): {', '.join([f'{np.degrees(a):.1f}' for a in angles])}")
            
            if self.mirroring_active:
                print(f"  Robot mirroring: ACTIVE")
            else:
                print(f"  Robot mirroring: INACTIVE")
                
        except Exception as e:
            print(f"  Could not get current state: {e}")
    
    def run(self):
        self.show_workspace_info()
        self.get_current_pose()
        self.show_commands()

        print("\nAuto-starting robot mirroring...")
        self.start_mirroring()

        try:
            while self.sim.viewer.is_running():
                print("\n" + "-"*40)
                user_input = input("Enter command: ").strip()
                
                if user_input.lower() == 'q':
                    break
                elif user_input.lower() == 'current':
                    self.get_current_pose()
                elif user_input.lower() == 'workspace':
                    self.show_workspace_info()
                elif user_input.lower() == 'commands':
                    self.show_commands()
                elif user_input.lower() == 'mirror':
                    self.start_mirroring()
                elif user_input.lower() in ['stop', 's']:
                    self.stop_mirroring()
                
        except KeyboardInterrupt:
            print("\n\nShutting down system...")
        finally:
            if self.mirroring_active:
                self.stop_mirroring()
            print("Closing simulation...")
            self.sim.close()
            print("System shutdown complete.")

def main():
    try:
        control_system = SO100IntegratedControl()
        control_system.run()
        
    except Exception as e:
        print(f"Failed to initialize system: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure so100_arm.xml exists")
        print("2. Check that MuJoCo viewer can launch")
        print("3. Verify all dependencies are installed")

if __name__ == "__main__":
    main() 
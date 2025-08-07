import mujoco as mj
import mujoco.viewer
import numpy as np
import time
import threading
from typing import List, Optional, Tuple
import queue

class SO100Simulation:
    def __init__(self):
        """Initialize the SO-100 MuJoCo simulation with GUI."""
        print("Initializing SO-100 simulation...")
        
        # Load the SO-100 model
        self.model = self._load_so100_model()
        self.data = mj.MjData(self.model)

        # Threading control with proper synchronization
        self.data_lock = threading.RLock()
        self.viewer_lock = threading.RLock()
        
        # Simulation control
        self.running = True
        self.paused = False
        
        # Command queue for thread-safe communication
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Initialize viewer with better error handling
        self._init_viewer()

        # Get actuator names and joint information
        actuator_names = [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]

        # Get important body and joint indices
        self.end_effector_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'Fixed_Jaw')
        if self.end_effector_id == -1:
            possible_names = ['hand', 'gripper', 'tool', 'tcp', 'wrist_3_link']
            for name in possible_names:
                self.end_effector_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
                if self.end_effector_id != -1:
                    break
        
        # Get joint information
        self.joint_names = []
        self.joint_ids = []
        for i in range(self.model.njnt):
            joint_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, i)
            if joint_name in actuator_names:
                self.joint_names.append(joint_name)
                self.joint_ids.append(i)

        self.dof_ids = []
        for joint_id in self.joint_ids:
            self.dof_ids.append(self.model.jnt_dofadr[joint_id])
        
        print(f"Found {len(self.joint_names)} joints: {self.joint_names}")
        print(f"End effector body ID: {self.end_effector_id}")
        
        # Start simulation thread
        self.sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.sim_thread.start()
        
        print("SO-100 simulation initialized successfully!")

    def _init_viewer(self):
        """Initialize viewer with better error handling."""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self._viewer_context = mj.viewer.launch_passive(self.model, self.data)
                self.viewer = self._viewer_context.__enter__()
                print(f"Viewer initialized successfully (attempt {attempt + 1})")
                return
            except Exception as e:
                print(f"Viewer initialization attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    raise
                time.sleep(0.5)

    def pause_simulation(self):
        """Pause the simulation loop."""
        self.paused = True
        print("Simulation paused")

    def resume_simulation(self):
        """Resume the simulation loop."""
        self.paused = False
        print("Simulation resumed")

    def _load_so100_model(self):
        """Load the SO-100 model from the XML file."""
        try:
            return mj.MjModel.from_xml_path("so100_arm.xml")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _simulation_loop(self):
        """Enhanced simulation loop with better error handling."""
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while self.running and self.viewer.is_running():
            try:
                # Check if paused
                if self.paused:
                    time.sleep(0.01)
                    continue
                
                # Process commands from command queue
                self._process_commands()
                
                # Step simulation with proper locking
                with self.data_lock:
                    mj.mj_step(self.model, self.data)
                
                # Viewer sync with enhanced error handling
                self._safe_viewer_sync()
                
                # Reset error counter on success
                consecutive_errors = 0
                
            except Exception as e:
                consecutive_errors += 1
                
                # Handle specific known errors
                if self._is_known_threading_error(e):
                    # Skip this frame for known threading errors
                    time.sleep(0.001)
                    continue
                else:
                    print(f"Simulation loop error ({consecutive_errors}/{max_consecutive_errors}): {e}")
                    
                    # If too many consecutive errors, break
                    if consecutive_errors >= max_consecutive_errors:
                        print("Too many consecutive errors, stopping simulation")
                        break
                    
                    time.sleep(0.01)
            
            time.sleep(0.002)  # 500 Hz simulation
    
    def _process_commands(self):
        """Process commands from the command queue."""
        try:
            while not self.command_queue.empty():
                try:
                    command = self.command_queue.get_nowait()
                    self._handle_command(command)
                except queue.Empty:
                    break
        except Exception as e:
            print(f"Command processing error: {e}")
    
    def _handle_command(self, command):
        """Handle a specific command."""
        if command['type'] == 'set_joint_angles':
            with self.data_lock:
                joint_angles = command['angles']
                self.data.ctrl[:] = joint_angles
                
                # Teleport joints for immediate response
                for i, angle in enumerate(joint_angles):
                    if i < len(self.joint_ids):
                        joint_id = self.joint_ids[i]
                        qpos_adr = self.model.jnt_qposadr[joint_id]
                        self.data.qpos[qpos_adr] = angle
                
                self.data.qvel[:] = 0
                mj.mj_forward(self.model, self.data)
            
            # Send response
            try:
                self.response_queue.put({'success': True}, block=False)
            except queue.Full:
                pass
    
    def _safe_viewer_sync(self):
        """Safely sync viewer with enhanced error handling."""
        max_sync_attempts = 3
        
        for attempt in range(max_sync_attempts):
            try:
                with self.viewer_lock:
                    if self.viewer and hasattr(self.viewer, 'sync'):
                        self.viewer.sync()
                return  # Success
                
            except Exception as e:
                if self._is_known_threading_error(e):
                    if attempt < max_sync_attempts - 1:
                        time.sleep(0.001)  # Brief pause before retry
                        continue
                    else:
                        # Skip this frame after all attempts
                        return
                else:
                    # Unknown error, re-raise
                    raise
    
    def _is_known_threading_error(self, error):
        """Check if this is a known threading error we can safely ignore."""
        error_str = str(error).lower()
        known_errors = [
            'copydatavisual',
            'stack is in use',
            'attempting to copy mjdata while stack is in use',
            'viewer sync error',
            'opengl context',
            'glfw'
        ]
        return any(known_error in error_str for known_error in known_errors)
    
    def get_end_effector_position(self, data_override=None) -> np.ndarray:
        """Get current end effector position with thread safety."""
        d = data_override if data_override is not None else self.data
        
        with self.data_lock:
            if self.end_effector_id != -1:
                return d.xpos[self.end_effector_id].copy()
            else:
                # Use TCP site if available
                tcp_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, 'tcp')
                if tcp_id != -1:
                    return d.site_xpos[tcp_id].copy()
                else:
                    return d.xpos[-1].copy()
    
    def get_current_joint_angles(self) -> np.ndarray:
        """Get current joint angles with thread safety."""
        with self.data_lock:
            joint_angles = []
            for joint_id in self.joint_ids:
                joint_angles.append(self.data.qpos[self.model.jnt_qposadr[joint_id]])
        return np.array(joint_angles)
    
    def set_joint_angles(self, joint_angles: List[float]):
        """Set joint angles using command queue for thread safety."""
        if len(joint_angles) != len(self.joint_ids):
            print(f"Error: Incorrect number of joint angles. Expected {len(self.joint_ids)}, got {len(joint_angles)}.")
            return

        # Send command via queue
        command = {
            'type': 'set_joint_angles',
            'angles': joint_angles
        }
        
        try:
            self.command_queue.put(command, timeout=0.1)
            
            # Wait for response
            try:
                response = self.response_queue.get(timeout=0.1)
                return response.get('success', False)
            except queue.Empty:
                return True  # Assume success if no response
                
        except queue.Full:
            print("Warning: Command queue full, joint angles may not be set")
            return False

    def close(self):
        """Close the simulation with proper cleanup."""
        print("Closing simulation...")
        self.running = False
        
        # Wait for simulation thread to finish
        if self.sim_thread.is_alive():
            self.sim_thread.join(timeout=2)
        
        # Close viewer
        if hasattr(self, '_viewer_context'):
            try:
                self._viewer_context.__exit__(None, None, None)
            except:
                pass
        
        print("Simulation closed cleanly")


def main():
    """Main function to demonstrate the SO-100 simulation."""
    print("Starting SO-100 MuJoCo simulation...")
    
    # Create simulation
    sim = SO100Simulation()
    
    print("\nSimulation started! MuJoCo viewer should be open.")
    print("You can interact with the viewer and call simulation methods.")
    
    # Wait a moment for initialization
    time.sleep(2)
    
    print("\n" + "="*50)
    print("Simulation is running. The viewer will stay open.")
    print("You can call sim.set_joint_angles() or sim.get_current_joint_angles() interactively.")
    print("Press Ctrl+C to exit.")
    print("="*50)
    
    try:
        # Keep main thread alive
        while sim.viewer.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nClosing simulation...")
    finally:
        sim.close()

if __name__ == "__main__":
    main() 
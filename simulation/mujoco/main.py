import time
import sys
import mujoco
import numpy as np

print(
    f"Starting MuJoCo simulation in GUI mode. Python version: {sys.version} (requires: Python 3.8+)"
)

# a simple scene for testing
xml_content = """
<mujoco model="test_scene">
    <compiler angle="radian"/>
    <default>
        <joint damping="0.2" frictionloss="0.1"/>
        <geom friction="1.0 0.005 0.0001"/>
    </default>
    <worldbody>
        <geom pos="0 0 0" size="0 0 .125" type="plane" rgba="0.5 0.5 0.5 1" name="floor"/>
        <light pos="0 0 3" dir="0 0 -1" directional="false"/>
        <body pos="0 0 1">
            <freejoint/>
            <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
        </body>
    </worldbody>
</mujoco>
"""

try:
    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)
    
    print("MuJoCo model loaded successfully")
    print(f"Model has {model.njnt} joints and {model.nbody} bodies")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("MuJoCo viewer launched successfully")
        print("Press ESC to exit or close the viewer window")
        
        while viewer.is_running():
            step_start = time.time()
            
            mujoco.mj_step(model, data)
            
            viewer.sync()
            
            # simple time keeping (target: 60 FPS)
            time_until_next_step = 1.0/60.0 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                
        print("MuJoCo simulation ended")
        
except Exception as e:
    print(f"Error running MuJoCo simulation: {e}")
    print("Make sure MuJoCo is properly installed: pip install mujoco")
    sys.exit(1) 
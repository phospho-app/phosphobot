from phosphobot.camera import AllCameras
from phosphobot.ai import ai
from phosphobot.api.client import PhosphoApi
import time

# from act import process_image

client = PhosphoApi(base_url="http://localhost:80")

# print(client.status_status_get())

# Get a camera frame
allcameras = AllCameras()
cameras_satus = allcameras.status()

time.sleep(0.2)
observation = {}

# Altenatively, add a check here to filter which cameras you want to use
rgb_frames = allcameras.get_rgb_frames_for_all_cameras()
print(rgb_frames.keys())


### END INIT ###


# Get the frames
frame = allcameras.get_rgb_frame(camera_id=0)
print(frame.shape)
frame = allcameras.get_rgb_frame(camera_id=0)

stereo_camera = allcameras.get_camera_by_id(id=0)
print(f"{stereo_camera.camera_name}: {stereo_camera.camera_type}")

# Format required camera frames

# get the robot state
state = client.control.read_joints()
print(state)


# At this stage, we have our formated x element
class MyActModel(ai.Model):
    def __init__(self, pretrained_policy_name_or_path, revision: str | None = None):
        super().__init__()
        self.policy = ai.ACT(pretrained_policy_name_or_path, revision=revision)

    def forward(self, wrist_camera, main_camera):
        if self.step % 100:
            macro_prediciton = self.pi0(
                wrist_camera=wrist_camera, main_camera=main_camera
            )
        else:
            macro_prediciton = None
        micro_prediction = self.act(
            main_camera, self.latest_predict["macro_prediction"]
        )
        return macro_prediciton, micro_prediction

    def move(self, joints):
        self.robot.move_joints(joints)


m = MyActModel("PLB/pi0-so100-orangelegobrick-wristcam")

# Do the forward and get the actions chunck

# Loop through the action chunck on the robot

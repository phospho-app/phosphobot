export type Temperature = {
  current: number | null;
  max: number | null;
};

export type RobotConfigStatus = {
  name: string;
  device_name?: string;
  robot_type: "manipulator" | "mobile" | "other";
  temperature?: Temperature[];
};
export type SingleCameraStatus = {
  camera_id: number;
  is_active: boolean;
  camera_type:
    | "classic"
    | "stereo"
    | "realsense"
    | "dummy"
    | "dummy_stereo"
    | "zmq"
    | "unknown";
  width: number;
  height: number;
  fps: number;
};

export type DatasetInfoResponse = {
  status: "ok" | "error";
  robot_type?: string;
  robot_dof?: number;
  number_of_episodes?: number;
  image_keys?: string[];
  image_frames?: Record<string, string>;
};

export type AllCamerasStatus = {
  cameras_status: SingleCameraStatus[];
  is_stereo_camera_available: boolean;
  realsense_available: boolean;
  video_cameras_ids: number[];
};

export type ServerStatus = {
  status: "ok" | "error";
  name: string;
  robots: string[];
  robot_status: RobotConfigStatus[];
  cameras: AllCamerasStatus;
  version_id: string;
  is_recording: boolean;
  ai_running_status: "stopped" | "running" | "paused" | "waiting";
  leader_follower_status: boolean;
  server_ip: string;
  server_port: number;
};

export type AdminTokenSettings = {
  huggingface: boolean;
  wandb: boolean;
};

export type DatasetListResponse = {
  pushed_datasets: string[];
  local_datasets: string[];
};

export type SupabaseTrainingModel = {
  id: number;
  status: "succeeded" | "failed" | "running" | "canceled";
  user_id: string;
  dataset_name: string;
  model_name: string;
  requested_at: string;
  terminated_at: string | null;
  used_wandb: boolean | null;
  model_type: string;
  training_params: null | Record<string, string | number | null>;
  session_count: number;
  success_rate: number | null;
};

export type TrainingsList = {
  models: SupabaseTrainingModel[];
};

export type Session = {
  user_id: string;
  user_email: string;
  email_confirmed: boolean;
  access_token: string;
  refresh_token: string;
  expires_in: number;
};

export type TorqueStatus = {
  current_torque: number[];
};

export type TrainingParamsACT = {
  batch_size?: number;
  steps?: number;
};

export type TrainingParamsGR00T = {
  batch_size?: number;
  learning_rate?: number;
  epochs?: number;
  validation_dataset_name?: string;
};

export type TrainingParamsSmolVLA = {
  batch_size?: number;
  steps?: number;
};

export type TrainingRequest = {
  model_type: "gr00t" | "ACT" | "smolvla" | "custom";
  dataset_name: string;
  model_name: string;
  wandb_api_key?: string;
  training_params?: TrainingParamsACT | TrainingParamsGR00T | TrainingParamsSmolVLA;
};

export type AdminSettings = {
  dataset_name: string;
  freq: number;
  episode_format: "lerobot_v2.1" | "lerobot_v2" | "json";
  video_codec: string;
  video_size: [number, number];
  task_instruction: string;
  cameras_to_record: number[] | null;
  hf_private_mode: boolean;
};

export type AIStatusResponse = {
  id?: string;
  status: "waiting" | "running" | "stopped" | "paused";
};

export type TeleopSettings = {
  vr_scaling: number;
};

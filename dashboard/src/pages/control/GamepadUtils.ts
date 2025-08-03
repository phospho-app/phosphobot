// GamepadUtils.ts - Types, constants, and utility functions

import type { ServerStatus } from "@/types";

// ==================== TYPES ====================

export interface ControllerArmPair {
  controller_index: number | null;
  robot_name: string | null;
  controller_name: string;
}

export interface MultiArmGroup {
  id: string;
  name: string;
  controller_index: number | null;
  controller_name: string;
  robot_names: string[];
  control_mode: 'synchronized' | 'sequential';
  active_robot_index: number; // For sequential mode
}

export interface GamepadState {
  connected: boolean;
  buttons: boolean[];
  buttonValues: number[];
  axes: number[];
}

export interface RobotMovement {
  x: number;
  y: number;
  z: number;
  rz: number;
  rx: number;
  ry: number;
  toggleOpen?: boolean;
}

export interface ControllerState {
  buttonsPressed: Set<string>;
  lastExecutionTime: number;
  openState: number;
  lastButtonStates: boolean[];
  lastTriggerValue: number;
  triggerControlActive: boolean;
  resetSent: boolean;
}

export interface AnalogValues {
  leftTrigger: number;
  rightTrigger: number;
  leftStickX: number;
  leftStickY: number;
  rightStickX: number;
  rightStickY: number;
}

export interface GamepadInfo {
  index: number;
  id: string;
  name: string;
}

export type ControlType = "analog-vertical" | "analog-horizontal" | "digital" | "trigger";
export type ConfigMode = "individual" | "multi-arm";

export interface Control {
  key: string;
  label: string;
  buttons: string[];
  description: string;
  icon: React.ReactNode;
  type: ControlType;
}

// ==================== CONSTANTS ====================

// Configuration constants
export const BASE_URL = `http://${window.location.hostname}:${window.location.port}/`;
export const STEP_SIZE = 1; // in centimeters
export const LOOP_INTERVAL = 10; // ms (~100 Hz)
export const INSTRUCTIONS_PER_SECOND = 30;
export const DEBOUNCE_INTERVAL = 1000 / INSTRUCTIONS_PER_SECOND;
export const AXIS_DEADZONE = 0.15; // Deadzone for analog sticks
export const AXIS_SCALE = 2; // Scale factor for analog stick movement

// Gamepad button mappings (standard gamepad layout)
export const BUTTON_MAPPINGS: Record<number, RobotMovement> = {
  12: { x: 0, y: 0, z: 0, rz: 0, rx: STEP_SIZE * 3.14, ry: 0 }, // D-pad up - wrist pitch up
  13: { x: 0, y: 0, z: 0, rz: 0, rx: -STEP_SIZE * 3.14, ry: 0 }, // D-pad down - wrist pitch down
  14: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: -STEP_SIZE * 3.14 }, // D-pad left - wrist roll counter-clockwise
  15: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: STEP_SIZE * 3.14 }, // D-pad right - wrist roll clockwise
  4: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: 0, toggleOpen: true }, // L1/LB - toggle gripper
  5: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: 0, toggleOpen: true }, // R1/RB - toggle gripper
  0: { x: 0, y: 0, z: 0, rz: 0, rx: -STEP_SIZE * 3.14, ry: 0 }, // A/X button - wrist pitch down
  1: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: STEP_SIZE * 3.14 }, // B/Circle - wrist roll clockwise
  2: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: -STEP_SIZE * 3.14 }, // X/Square - wrist roll counter-clockwise
  3: { x: 0, y: 0, z: 0, rz: 0, rx: STEP_SIZE * 3.14, ry: 0 }, // Y/Triangle - wrist pitch up
  9: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: 0 }, // Start/Menu - move to sleep position
  10: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: 0 }, // Start/Menu (alternate index) - move to sleep position
};

// Special button mappings for multi-arm control
export const MULTI_ARM_SPECIAL_BUTTONS: Record<number, string> = {
  8: 'switch_robot', // Select button (8) or Back button for switching active robot in sequential mode
  10: 'mode_switch', // L3/R3 (stick clicks) for mode switching
  11: 'mode_switch',
};

// Button names for display
export const BUTTON_NAMES = [
  "A/X",
  "B/Circle", 
  "X/Square",
  "Y/Triangle",
  "L1/LB",
  "R1/RB",
  "L2/LT",
  "R2/RT",
  "Select/Back",
  "Start/Menu",
  "L3",
  "R3",
  "D-Pad Up",
  "D-Pad Down",
  "D-Pad Left",
  "D-Pad Right",
  "Home/Guide",
];

// ==================== UTILITY FUNCTIONS ====================

export const robotIDFromName = (name?: string | null, serverStatus?: ServerStatus): number => {
  if (name === undefined || name === null || !serverStatus?.robot_status) {
    return 0;
  }
  const index = serverStatus.robot_status.findIndex(
    (robot) => robot.device_name === name,
  );
  return index === -1 ? 0 : index;
};

export const postData = async (
  url: string, 
  data: Record<string, unknown>, 
  queryParam?: Record<string, string | number>
) => {
  try {
    let newUrl = url;
    if (queryParam) {
      const urlParams = new URLSearchParams();
      Object.entries(queryParam).forEach(([key, value]) => {
        urlParams.append(key, value.toString());
      });
      if (urlParams.toString()) {
        newUrl += "?" + urlParams.toString();
      }
    }

    const response = await fetch(newUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });
    if (!response.ok) {
      throw new Error(`Network response was not ok: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error posting data:", error);
  }
};

export const processAnalogSticks = (
  gamepad: Gamepad,
  lastTriggerValue: number = 0,
): RobotMovement & { gripperValue?: number } => {
  const movement: RobotMovement & { gripperValue?: number } = {
    x: 0,
    y: 0,
    z: 0,
    rz: 0,
    rx: 0,
    ry: 0,
  };

  // Left stick - Rotation (X) and Forward/Backward (Y)
  const leftX = Math.abs(gamepad.axes[0]) > AXIS_DEADZONE ? gamepad.axes[0] : 0;
  const leftY = Math.abs(gamepad.axes[1]) > AXIS_DEADZONE ? gamepad.axes[1] : 0;

  // Right stick - Left/Right strafe (X) and Up/Down (Y)
  const rightX = Math.abs(gamepad.axes[2]) > AXIS_DEADZONE ? gamepad.axes[2] : 0;
  const rightY = Math.abs(gamepad.axes[3]) > AXIS_DEADZONE ? gamepad.axes[3] : 0;

  // Map to robot movement
  movement.rz = leftX * STEP_SIZE * 3.14 * AXIS_SCALE; // Rotation (from left stick X)
  movement.z = -leftY * STEP_SIZE * AXIS_SCALE; // Up/down (from left stick Y)
  movement.y = -rightX * STEP_SIZE * AXIS_SCALE; // Left/right strafe (from right stick X)
  movement.x = -rightY * STEP_SIZE * AXIS_SCALE; // Forward/backward (from right stick Y)

  // Triggers - check both axes and buttons
  let leftTrigger = 0;
  let rightTrigger = 0;

  // First try to get triggers from axes (common for most gamepads)
  if (gamepad.axes.length >= 6) {
    leftTrigger = gamepad.axes[6] > 0.1 ? gamepad.axes[6] : 0;
    rightTrigger = gamepad.axes[7] > -0.9 ? (gamepad.axes[7] + 1) / 2 : 0; // Convert from [-1, 1] to [0, 1]
  }

  // If triggers aren't in axes or are zero, check buttons 6 and 7
  if (leftTrigger === 0 && gamepad.buttons.length > 6 && gamepad.buttons[6]) {
    leftTrigger = gamepad.buttons[6].value || (gamepad.buttons[6].pressed ? 1 : 0);
  }
  if (rightTrigger === 0 && gamepad.buttons.length > 7 && gamepad.buttons[7]) {
    rightTrigger = gamepad.buttons[7].value || (gamepad.buttons[7].pressed ? 1 : 0);
  }

  // Both triggers control gripper - use whichever has higher value
  const triggerValue = Math.max(leftTrigger, rightTrigger);

  // Always return the current trigger value
  if (triggerValue > 0 || lastTriggerValue > 0) {
    movement.gripperValue = triggerValue;
  }

  return movement;
};

export const getControlName = (index: number): string => {
  if (index === 0) return "wrist-pitch-down";
  else if (index === 1) return "wrist-roll-right";
  else if (index === 2) return "wrist-roll-left";
  else if (index === 3) return "wrist-pitch-up";
  else if (index === 4 || index === 5) return "gripper-toggle";
  else if (index === 9 || index === 10) return "sleep";
  else if (index === 12) return "wrist-pitch-up";
  else if (index === 13) return "wrist-pitch-down";
  else if (index === 14) return "wrist-roll-left";
  else if (index === 15) return "wrist-roll-right";
  
  return BUTTON_NAMES[index] || `Button ${index}`;
};

export const getRobotsToControl = (config: any, configMode: ConfigMode): string[] => {
  if (configMode === "individual") {
    return [config.robot_name];
  } else {
    // Multi-arm group
    if (config.control_mode === 'sequential') {
      return [config.robot_names[config.active_robot_index]];
    } else {
      return config.robot_names;
    }
  }
};

export const applyControlMode = (data: any, controlMode: string): any => {
  switch (controlMode) {
    case 'synchronized':
      return data; // All robots move identically
    case 'sequential':
      return data; // Only active robot moves (handled in getRobotsToControl)
    default:
      return data;
  }
};

export const initRobot = async (robotName: string, serverStatus?: ServerStatus) => {
  try {
    await postData(
      BASE_URL + "move/init",
      {},
      {
        robot_id: robotIDFromName(robotName, serverStatus),
      },
    );
    await new Promise((resolve) => setTimeout(resolve, 2000));
    const initData = {
      x: 0,
      y: 0,
      z: 0,
      rx: 0,
      ry: 0,
      rz: 0,
      open: 1,
    };
    await postData(BASE_URL + "move/absolute", initData, {
      robot_id: robotIDFromName(robotName, serverStatus),
    });
  } catch (error) {
    console.error("Error during init:", error);
  }
};

export const getAvailableGamepads = (): GamepadInfo[] => {
  const gamepads = navigator.getGamepads();
  const available: GamepadInfo[] = [];
  
  for (let i = 0; i < gamepads.length; i++) {
    const gamepad = gamepads[i];
    if (gamepad) {
      const fullName = gamepad.id;
      const shortName = fullName.split(' ').slice(0, 3).join(' ');
      
      available.push({
        index: i,
        id: fullName,
        name: shortName || `Controller ${i + 1}`
      });
    }
  }
  
  return available;
};

export const getAvailableControllers = (
  currentPairIndex: number, 
  controllerArmPairs: ControllerArmPair[], 
  availableGamepads: GamepadInfo[]
): GamepadInfo[] => {
  const usedControllerIds = new Set<number>();

  controllerArmPairs.forEach((pair, index) => {
    if (index !== currentPairIndex && pair.controller_index !== null) {
      usedControllerIds.add(pair.controller_index);
    }
  });

  return availableGamepads.filter((gamepad) => !usedControllerIds.has(gamepad.index));
};

export const getAvailableRobots = (
  currentPairIndex: number, 
  controllerArmPairs: ControllerArmPair[], 
  serverStatus?: ServerStatus
) => {
  const usedRobotNames = new Set<string>();

  controllerArmPairs.forEach((pair, index) => {
    if (index !== currentPairIndex && pair.robot_name !== null) {
      usedRobotNames.add(pair.robot_name);
    }
  });

  return serverStatus?.robot_status?.filter(
    (robot) => robot.device_name && !usedRobotNames.has(robot.device_name)
  ) || [];
};

export const getAvailableControllersForGroup = (availableGamepads: GamepadInfo[]): GamepadInfo[] => {
  // Allow any controller to be used - don't exclude based on other groups
  // This enables one controller to control multiple groups
  return availableGamepads;
};

export const getAvailableRobotsForGroup = (
  currentGroupId: string, 
  multiArmGroups: MultiArmGroup[], 
  serverStatus?: ServerStatus
) => {
  const usedRobotNames = new Set<string>();

  multiArmGroups.forEach((group) => {
    if (group.id !== currentGroupId) {
      group.robot_names.forEach(name => usedRobotNames.add(name));
    }
  });

  return serverStatus?.robot_status?.filter(
    (robot) => robot.device_name && !usedRobotNames.has(robot.device_name)
  ) || [];
};

export const extractAnalogValues = (gamepad: Gamepad): AnalogValues => {
  // Update analog trigger values
  let leftTriggerVal = 0;
  let rightTriggerVal = 0;

  // Check axes first
  if (gamepad.axes.length >= 6) {
    leftTriggerVal = gamepad.axes[6] > 0.1 ? gamepad.axes[6] : 0;
    rightTriggerVal = gamepad.axes[7] > -0.9 ? (gamepad.axes[7] + 1) / 2 : 0;
  }

  // Check buttons if no axis values
  if (leftTriggerVal === 0 && gamepad.buttons.length > 6 && gamepad.buttons[6]) {
    leftTriggerVal = gamepad.buttons[6].value || 0;
  }
  if (rightTriggerVal === 0 && gamepad.buttons.length > 7 && gamepad.buttons[7]) {
    rightTriggerVal = gamepad.buttons[7].value || 0;
  }

  // Get analog stick values
  const leftStickX = Math.abs(gamepad.axes[0]) > AXIS_DEADZONE ? gamepad.axes[0] : 0;
  const leftStickY = Math.abs(gamepad.axes[1]) > AXIS_DEADZONE ? gamepad.axes[1] : 0;
  const rightStickX = Math.abs(gamepad.axes[2]) > AXIS_DEADZONE ? gamepad.axes[2] : 0;
  const rightStickY = Math.abs(gamepad.axes[3]) > AXIS_DEADZONE ? gamepad.axes[3] : 0;

  return {
    leftTrigger: leftTriggerVal,
    rightTrigger: rightTriggerVal,
    leftStickX: leftStickX,
    leftStickY: leftStickY,
    rightStickX: rightStickX,
    rightStickY: rightStickY,
  };
};
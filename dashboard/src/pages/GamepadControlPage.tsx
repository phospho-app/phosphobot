// import controlschema from "@/assets/ControlSchema.png";
import { LoadingPage } from "@/components/common/loading";
import { SpeedSelect } from "@/components/common/speed-select";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { fetcher } from "@/lib/utils";
import { ServerStatus } from "@/types";
import {
  ArrowDownFromLine,
  ArrowUpFromLine,
  Gamepad2,
  Play,
  RotateCcw,
  RotateCw,
  Space,
  Square,
  Home,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import useSWR from "swr";

// GamepadVisualizer component
interface GamepadState {
  connected: boolean;
  buttons: boolean[];
  buttonValues: number[];
  axes: number[];
}

function GamepadVisualizer({ gamepadIndex }: { gamepadIndex: number | null }) {
  const [gamepadState, setGamepadState] = useState<GamepadState>({
    connected: false,
    buttons: [],
    buttonValues: [],
    axes: [],
  });

  useEffect(() => {
    if (gamepadIndex === null) {
      setGamepadState({ connected: false, buttons: [], buttonValues: [], axes: [] });
      return;
    }

    const updateGamepadState = () => {
      const gamepads = navigator.getGamepads();
      const gamepad = gamepads[gamepadIndex];
      
      if (gamepad) {
        setGamepadState({
          connected: true,
          buttons: Array.from(gamepad.buttons).map(b => b.pressed),
          buttonValues: Array.from(gamepad.buttons).map(b => b.value),
          axes: Array.from(gamepad.axes),
        });
      }
    };

    const interval = setInterval(updateGamepadState, 50); // 20Hz update
    return () => clearInterval(interval);
  }, [gamepadIndex]);

  if (!gamepadState.connected) {
    return null;
  }

  const buttonNames = [
    "A/X", "B/Circle", "X/Square", "Y/Triangle",
    "L1/LB", "R1/RB", "L2/LT", "R2/RT",
    "Select/Back", "Start/Menu", "L3", "R3",
    "D-Pad Up", "D-Pad Down", "D-Pad Left", "D-Pad Right",
    "Home/Guide"
  ];

  // Get trigger values from either axes or buttons
  let leftTriggerValue = 0;
  let rightTriggerValue = 0;
  
  // First check axes
  if (gamepadState.axes.length > 6) {
    leftTriggerValue = gamepadState.axes[6] || 0;
    rightTriggerValue = gamepadState.axes[7] || 0;
  }
  
  // If no trigger values from axes, check buttons 6 and 7
  if (leftTriggerValue === 0 && gamepadState.buttonValues.length > 6) {
    leftTriggerValue = gamepadState.buttonValues[6] || 0;
  }
  if (rightTriggerValue === 0 && gamepadState.buttonValues.length > 7) {
    rightTriggerValue = gamepadState.buttonValues[7] || 0;
  }

  return (
    <Card className="mt-4">
      <CardHeader>
        <CardTitle className="text-sm">Gamepad State</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <h4 className="text-sm font-medium mb-2">Buttons</h4>
          <div className="grid grid-cols-4 gap-2">
            {gamepadState.buttons.map((pressed, index) => (
              <div
                key={index}
                className={`text-xs p-2 rounded text-center ${
                  pressed ? "bg-primary text-primary-foreground" : "bg-muted"
                }`}
              >
                {buttonNames[index] || `Button ${index}`}
                {/* Show analog value for L2/R2 if they're analog buttons */}
                {(index === 6 || index === 7) && gamepadState.buttonValues[index] > 0 && gamepadState.buttonValues[index] < 1 && (
                  <div className="text-[10px] mt-1">
                    {(gamepadState.buttonValues[index] * 100).toFixed(0)}%
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
        
        <div>
          <h4 className="text-sm font-medium mb-2">Analog Sticks & Triggers</h4>
          <div className="space-y-2">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs mb-1">Left Stick X: {gamepadState.axes[0]?.toFixed(2) || "0.00"}</p>
                <Progress value={(gamepadState.axes[0] + 1) * 50} className="h-2" />
              </div>
              <div>
                <p className="text-xs mb-1">Left Stick Y: {gamepadState.axes[1]?.toFixed(2) || "0.00"}</p>
                <Progress value={(gamepadState.axes[1] + 1) * 50} className="h-2" />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs mb-1">Right Stick X: {gamepadState.axes[2]?.toFixed(2) || "0.00"}</p>
                <Progress value={(gamepadState.axes[2] + 1) * 50} className="h-2" />
              </div>
              <div>
                <p className="text-xs mb-1">Right Stick Y: {gamepadState.axes[3]?.toFixed(2) || "0.00"}</p>
                <Progress value={(gamepadState.axes[3] + 1) * 50} className="h-2" />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs mb-1">Left Trigger: {leftTriggerValue.toFixed(2)}</p>
                <Progress value={leftTriggerValue * 100} className="h-2" />
              </div>
              <div>
                <p className="text-xs mb-1">Right Trigger: {rightTriggerValue.toFixed(2)}</p>
                <Progress value={rightTriggerValue * 100} className="h-2" />
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Component for analog trigger buttons with gradient fill
function TriggerButton({ label, buttons, value, icon, onClick }: { 
  label: string; 
  buttons: string[]; 
  value: number; 
  icon: React.ReactNode;
  onClick?: () => void;
}) {
  return (
    <Card 
      className="relative flex flex-col items-center justify-center p-4 overflow-hidden h-full cursor-pointer hover:bg-accent transition-colors"
      onClick={onClick}
    >
      {/* Gradient fill from bottom to top based on value */}
      <div 
        className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-primary/50 to-primary/30 transition-all duration-100"
        style={{ height: `${value * 100}%` }}
      />
      <div className="relative z-10 flex flex-col items-center">
        {icon}
        <span className="mt-2 font-bold text-xs text-center block">
          {label}
        </span>
        <span className="text-[10px] text-muted-foreground text-center mt-1">
          {buttons.join(", ")}
        </span>
        {value > 0 && (
          <span className="text-[10px] text-center block mt-1">
            {Math.round(value * 100)}%
          </span>
        )}
      </div>
    </Card>
  );
}

// Component for control buttons
function ControlButton({ control, isActive, analogValue, onClick }: { 
  control: any; 
  isActive: boolean; 
  analogValue?: number;
  onClick?: () => void;
}) {
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  
  const handleMouseDown = () => {
    if (!onClick) return;
    
    // For analog controls, start continuous movement
    if (control.type.startsWith('analog')) {
      onClick(); // Initial click
      intervalRef.current = setInterval(() => {
        onClick();
      }, 100); // Send command every 100ms while held
    } else {
      // For digital controls, just single click
      onClick();
    }
  };
  
  const handleMouseUp = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };
  
  const handleMouseLeave = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };
  
  // Only show gradient if the analog value is in the correct direction
  const showGradient = control.type.startsWith('analog') && analogValue !== undefined && analogValue > 0;
  
  return (
    <Card 
      className={`relative flex flex-col items-center justify-center p-4 cursor-pointer transition-colors overflow-hidden h-full ${
        isActive ? "bg-primary/20 dark:bg-primary/30" : "bg-card hover:bg-accent"
      }`}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseLeave}
      onTouchStart={handleMouseDown}
      onTouchEnd={handleMouseUp}
    >
      {showGradient && (
        <div 
          className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-primary/50 to-primary/30 transition-all duration-100"
          style={{ 
            height: control.type === 'analog-vertical' 
              ? `${analogValue * 100}%` 
              : '100%',
            width: control.type === 'analog-horizontal' 
              ? `${analogValue * 100}%`
              : '100%',
            left: control.type === 'analog-horizontal' ? '0' : '0',
            right: control.type === 'analog-horizontal' ? 'auto' : '0',
          }}
        />
      )}
      <div className="relative z-10 flex flex-col items-center">
        {control.icon}
        <span className="mt-2 font-bold text-xs text-center">
          {control.label}
        </span>
        <span className="text-[10px] text-muted-foreground text-center mt-1">
          {control.buttons.join(", ")}
        </span>
      </div>
    </Card>
  );
}

export default function GamepadControlPage() {
  const { data: serverStatus, error: serverError } = useSWR<ServerStatus>(
    ["/status"],
    fetcher,
    {
      refreshInterval: 5000,
    },
  );

  const [isMoving, setIsMoving] = useState(false);
  const [activeButtons, setActiveButtons] = useState<Set<string>>(new Set());
  const [analogValues, setAnalogValues] = useState<{ 
    leftTrigger: number; 
    rightTrigger: number;
    leftStickX: number;
    leftStickY: number;
    rightStickX: number;
    rightStickY: number;
  }>({
    leftTrigger: 0,
    rightTrigger: 0,
    leftStickX: 0,
    leftStickY: 0,
    rightStickX: 0,
    rightStickY: 0,
  });
  const [selectedRobotName, setSelectedRobotName] = useState<string | null>(
    null,
  );
  const [selectedSpeed, setSelectedSpeed] = useState<number>(0.5);
  const [gamepadConnected, setGamepadConnected] = useState(false);
  const [gamepadIndex, setGamepadIndex] = useState<number | null>(null);
  const [autoStartTriggered, setAutoStartTriggered] = useState(false);

  // Refs to manage our control loop and state
  const buttonsPressed = useRef(new Set<string>());
  const intervalIdRef = useRef<NodeJS.Timeout | null>(null);
  const lastExecutionTimeRef = useRef(0);
  const openStateRef = useRef(1);
  const lastButtonStates = useRef<boolean[]>([]);
  const lastTriggerValue = useRef<number>(0);
  const triggerControlActive = useRef(false);
  const resetSent = useRef(false);

  // Configuration constants
  const BASE_URL = `http://${window.location.hostname}:${window.location.port}/`;
  const STEP_SIZE = 1; // in centimeters
  const LOOP_INTERVAL = 10; // ms (~100 Hz)
  const INSTRUCTIONS_PER_SECOND = 30;
  const DEBOUNCE_INTERVAL = 1000 / INSTRUCTIONS_PER_SECOND;
  const AXIS_DEADZONE = 0.15; // Deadzone for analog sticks
  const AXIS_SCALE = 2; // Scale factor for analog stick movement

  interface RobotMovement {
    x: number;
    y: number;
    z: number;
    rz: number;
    rx: number;
    ry: number;
    toggleOpen?: boolean;
  }

  // Gamepad button mappings (standard gamepad layout)
  const BUTTON_MAPPINGS: Record<number, RobotMovement> = {
    12: { x: 0, y: 0, z: 0, rz: 0, rx: STEP_SIZE * 3.14, ry: 0 }, // D-pad up - wrist pitch up
    13: { x: 0, y: 0, z: 0, rz: 0, rx: -STEP_SIZE * 3.14, ry: 0 }, // D-pad down - wrist pitch down
    14: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: -STEP_SIZE * 3.14 }, // D-pad left - wrist roll counter-clockwise
    15: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: STEP_SIZE * 3.14 }, // D-pad right - wrist roll clockwise
    4: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: 0, toggleOpen: true }, // L1/LB - toggle gripper
    5: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: 0, toggleOpen: true }, // R1/RB - toggle gripper
    0: { x: 0, y: 0, z: 0, rz: 0, rx: -STEP_SIZE * 3.14, ry: 0 }, // A/X button - wrist pitch down (same as D-pad down)
    1: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: STEP_SIZE * 3.14 }, // B/Circle - wrist roll clockwise (same as D-pad right)
    2: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: -STEP_SIZE * 3.14 }, // X/Square - wrist roll counter-clockwise (same as D-pad left)
    3: { x: 0, y: 0, z: 0, rz: 0, rx: STEP_SIZE * 3.14, ry: 0 }, // Y/Triangle - wrist pitch up (same as D-pad up)
    9: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: 0 }, // Start/Menu - move to sleep position (handled specially)
    10: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: 0 }, // Start/Menu (alternate index) - move to sleep position (handled specially)
  };

  // Axis mappings for analog sticks
  // Left stick: Rotation (X-axis) and Forward/Backward (Y-axis)
  // Right stick: Left/Right strafe (X-axis) and Up/Down (Y-axis)
  const processAnalogSticks = (gamepad: Gamepad): RobotMovement & { gripperValue?: number } => {
    const movement: RobotMovement & { gripperValue?: number } = { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: 0 };
    
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
    if (triggerValue > 0 || lastTriggerValue.current > 0) {
      movement.gripperValue = triggerValue;
    }
    
    return movement;
  };

  const robotIDFromName = (name?: string | null) => {
    if (name === undefined || name === null || !serverStatus?.robot_status) {
      return 0;
    }
    const index = serverStatus.robot_status.findIndex(
      (robot) => robot.device_name === name,
    );
    return index === -1 ? 0 : index;
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const postData = async (url: string, data: any, queryParam?: any) => {
    try {
      let newUrl = url;
      if (queryParam) {
        const urlParams = new URLSearchParams(queryParam);
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

  // Gamepad connection handlers
  useEffect(() => {
    const handleGamepadConnected = (e: GamepadEvent) => {
      console.log("Gamepad connected:", e.gamepad);
      setGamepadConnected(true);
      setGamepadIndex(e.gamepad.index);
    };

    const handleGamepadDisconnected = (e: GamepadEvent) => {
      console.log("Gamepad disconnected:", e.gamepad);
      if (e.gamepad.index === gamepadIndex) {
        setGamepadConnected(false);
        setGamepadIndex(null);
        buttonsPressed.current.clear();
      }
    };

    window.addEventListener("gamepadconnected", handleGamepadConnected);
    window.addEventListener("gamepaddisconnected", handleGamepadDisconnected);

    // Check for already connected gamepads
    const gamepads = navigator.getGamepads();
    for (let i = 0; i < gamepads.length; i++) {
      if (gamepads[i]) {
        setGamepadConnected(true);
        setGamepadIndex(i);
        break;
      }
    }

    return () => {
      window.removeEventListener("gamepadconnected", handleGamepadConnected);
      window.removeEventListener("gamepaddisconnected", handleGamepadDisconnected);
    };
  }, [gamepadIndex]);

  useEffect(() => {
    if (
      !selectedRobotName &&
      serverStatus?.robot_status &&
      serverStatus.robot_status.length > 0 &&
      serverStatus.robot_status[0].device_name
    ) {
      setSelectedRobotName(serverStatus.robot_status[0].device_name);
    }
  }, [serverStatus, selectedRobotName]);

  // Auto-start when both gamepad is connected and robot is selected
  useEffect(() => {
    if (gamepadConnected && selectedRobotName && !isMoving && !autoStartTriggered) {
      setAutoStartTriggered(true);
      startMoving();
    }
  }, [gamepadConnected, selectedRobotName, isMoving, autoStartTriggered]);

  // Main control loop
  useEffect(() => {
    if (isMoving && gamepadConnected && gamepadIndex !== null) {
      const controlRobot = () => {
        const gamepads = navigator.getGamepads();
        const gamepad = gamepads[gamepadIndex];
        
        if (!gamepad) return;

        const currentTime = Date.now();
        if (currentTime - lastExecutionTimeRef.current >= DEBOUNCE_INTERVAL) {
          let deltaX = 0,
            deltaY = 0,
            deltaZ = 0,
            deltaRZ = 0,
            deltaRX = 0,
            deltaRY = 0;
          let didToggleOpen = false;

          // Process button inputs
          gamepad.buttons.forEach((button, index) => {
            const wasPressed = lastButtonStates.current[index] || false;
            const isPressed = button.pressed;

            if (isPressed && !wasPressed) {
              // Button just pressed
              if (BUTTON_MAPPINGS[index]) {
                // Set active button for visual feedback
                const buttonNames = ["A/X", "B/Circle", "X/Square", "Y/Triangle", "L1/LB", "R1/RB", "L2/LT", "R2/RT", 
                                   "Select/Back", "Start/Menu", "L3", "R3", "D-Pad Up", "D-Pad Down", "D-Pad Left", "D-Pad Right"];
                const buttonName = buttonNames[index] || `Button ${index}`;
                
                // Map to combined control names
                let controlName = buttonName;
                if (index === 0) controlName = "wrist-pitch-down"; // A
                else if (index === 1) controlName = "wrist-roll-right"; // B
                else if (index === 2) controlName = "wrist-roll-left"; // X
                else if (index === 3) controlName = "wrist-pitch-up"; // Y
                else if (index === 4 || index === 5) controlName = "gripper-toggle"; // L1 or R1
                else if (index === 9 || index === 10) controlName = "sleep"; // Start
                else if (index === 12) controlName = "wrist-pitch-up"; // D-pad Up
                else if (index === 13) controlName = "wrist-pitch-down"; // D-pad Down
                else if (index === 14) controlName = "wrist-roll-left"; // D-pad Left
                else if (index === 15) controlName = "wrist-roll-right"; // D-pad Right
                
                setActiveButtons(prev => new Set(prev).add(controlName));
                
                if ((index === 9 || index === 10) && !resetSent.current) {
                  // Start button - move to sleep position
                  postData(BASE_URL + "move/sleep", {}, {
                    robot_id: robotIDFromName(selectedRobotName),
                  });
                  resetSent.current = true;
                } else if (BUTTON_MAPPINGS[index].toggleOpen) {
                  didToggleOpen = true;
                } else {
                  buttonsPressed.current.add(index.toString());
                }
              }
            } else if (!isPressed && wasPressed) {
              // Button just released
              if (index === 9 || index === 10) {
                resetSent.current = false; // Allow reset to be sent again
              }
              buttonsPressed.current.delete(index.toString());
              // Clear active button when released
              const buttonNames = ["A/X", "B/Circle", "X/Square", "Y/Triangle", "L1/LB", "R1/RB", "L2/LT", "R2/RT", 
                                 "Select/Back", "Start/Menu", "L3", "R3", "D-Pad Up", "D-Pad Down", "D-Pad Left", "D-Pad Right"];
              const buttonName = buttonNames[index] || `Button ${index}`;
              
              // Map to combined control names
              let controlName = buttonName;
              if (index === 0) controlName = "wrist-pitch-down"; // A
              else if (index === 1) controlName = "wrist-roll-right"; // B
              else if (index === 2) controlName = "wrist-roll-left"; // X
              else if (index === 3) controlName = "wrist-pitch-up"; // Y
              else if (index === 4 || index === 5) controlName = "gripper-toggle"; // L1 or R1
              else if (index === 9 || index === 10) controlName = "sleep"; // Start
              else if (index === 12) controlName = "wrist-pitch-up"; // D-pad Up
              else if (index === 13) controlName = "wrist-pitch-down"; // D-pad Down
              else if (index === 14) controlName = "wrist-roll-left"; // D-pad Left
              else if (index === 15) controlName = "wrist-roll-right"; // D-pad Right
              
              setActiveButtons(prev => {
                const newSet = new Set(prev);
                newSet.delete(controlName);
                return newSet;
              });
            }

            lastButtonStates.current[index] = isPressed;
          });

          // Accumulate button movements
          buttonsPressed.current.forEach((buttonStr) => {
            const buttonIndex = parseInt(buttonStr);
            if (BUTTON_MAPPINGS[buttonIndex]) {
              deltaX += BUTTON_MAPPINGS[buttonIndex].x;
              deltaY += BUTTON_MAPPINGS[buttonIndex].y;
              deltaZ += BUTTON_MAPPINGS[buttonIndex].z;
              deltaRZ += BUTTON_MAPPINGS[buttonIndex].rz;
              deltaRX += BUTTON_MAPPINGS[buttonIndex].rx;
              deltaRY += BUTTON_MAPPINGS[buttonIndex].ry;
            }
          });

          // Process analog stick inputs
          const analogMovement = processAnalogSticks(gamepad);
          deltaX += analogMovement.x;
          deltaY += analogMovement.y;
          deltaZ += analogMovement.z;
          deltaRZ += analogMovement.rz;
          deltaRX += analogMovement.rx;
          deltaRY += analogMovement.ry;
          
          // Update analog trigger values for visual feedback
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
          
          setAnalogValues({ 
            leftTrigger: leftTriggerVal, 
            rightTrigger: rightTriggerVal,
            leftStickX: leftStickX,
            leftStickY: leftStickY,
            rightStickX: rightStickX,
            rightStickY: rightStickY,
          });

          // Handle gripper control
          let gripperValue = openStateRef.current;
          
          // Check if trigger value has changed significantly (this can reactivate trigger control)
          if (analogMovement.gripperValue !== undefined && Math.abs(analogMovement.gripperValue - lastTriggerValue.current) > 0.05) {
            // Trigger value changed - it can take control
            triggerControlActive.current = true;
            lastTriggerValue.current = analogMovement.gripperValue;
          }
          
          if (didToggleOpen) {
            // A button was pressed - always toggle and disable trigger control
            // const previousState = openStateRef.current;
            openStateRef.current = openStateRef.current > 0.5 ? 0 : 1;
            gripperValue = openStateRef.current;
            triggerControlActive.current = false;
          } else if (analogMovement.gripperValue !== undefined && triggerControlActive.current) {
            // Use trigger value for gripper only if trigger control is active
            gripperValue = analogMovement.gripperValue;
            openStateRef.current = gripperValue;
          }

          // Apply speed scaling for mobile robots
          const currentRobot = serverStatus?.robot_status.find(
            (r) => r.device_name === selectedRobotName,
          );
          const isMobile = currentRobot?.robot_type === "mobile";

          if (isMobile) {
            deltaX *= selectedSpeed;
            deltaY *= selectedSpeed;
            deltaRZ *= selectedSpeed;
          }

          if (
            deltaX !== 0 ||
            deltaY !== 0 ||
            deltaZ !== 0 ||
            deltaRZ !== 0 ||
            deltaRX !== 0 ||
            deltaRY !== 0 ||
            didToggleOpen ||
            (analogMovement.gripperValue !== undefined && triggerControlActive.current)
          ) {
            const data = {
              x: deltaX,
              y: deltaY,
              z: deltaZ,
              rx: deltaRX,
              ry: deltaRY,
              rz: deltaRZ,
              open: gripperValue,
            };
            postData(BASE_URL + "move/relative", data, {
              robot_id: robotIDFromName(selectedRobotName),
            });
          }
          lastExecutionTimeRef.current = currentTime;
        }
      };

      const intervalId = setInterval(controlRobot, LOOP_INTERVAL);
      intervalIdRef.current = intervalId;
      return () => {
        if (intervalIdRef.current) {
          clearInterval(intervalIdRef.current);
        }
      };
    }
  }, [isMoving, gamepadConnected, gamepadIndex, selectedSpeed, serverStatus, selectedRobotName]);

  const initRobot = async () => {
    try {
      await postData(
        BASE_URL + "move/init",
        {},
        {
          robot_id: robotIDFromName(selectedRobotName),
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
        robot_id: robotIDFromName(selectedRobotName),
      });
    } catch (error) {
      console.error("Error during init:", error);
    }
  };

  const startMoving = async () => {
    await initRobot();
    setIsMoving(true);
  };

  const stopMoving = async () => {
    setIsMoving(false);
    buttonsPressed.current.clear();
  };

  const controls = [
    // Movement controls
    {
      key: "move-forward",
      label: "Forward",
      buttons: ["Right Stick ↑"],
      description: "Move forward",
      icon: <ArrowUpFromLine className="size-6" />,
      type: "analog-vertical" as const,
    },
    {
      key: "move-backward", 
      label: "Backward",
      buttons: ["Right Stick ↓"],
      description: "Move backward",
      icon: <ArrowDownFromLine className="size-6" />,
      type: "analog-vertical" as const,
    },
    {
      key: "move-left",
      label: "Strafe Left",
      buttons: ["Right Stick ←"],
      description: "Move left",
      icon: <RotateCcw className="size-6" />,
      type: "analog-horizontal" as const,
    },
    {
      key: "move-right",
      label: "Strafe Right", 
      buttons: ["Right Stick →"],
      description: "Move right",
      icon: <RotateCw className="size-6" />,
      type: "analog-horizontal" as const,
    },
    {
      key: "move-up",
      label: "Up",
      buttons: ["Left Stick ↑"],
      description: "Move up",
      icon: <ArrowUpFromLine className="size-6" />,
      type: "analog-vertical" as const,
    },
    {
      key: "move-down",
      label: "Down",
      buttons: ["Left Stick ↓"],
      description: "Move down", 
      icon: <ArrowDownFromLine className="size-6" />,
      type: "analog-vertical" as const,
    },
    {
      key: "rotate-left",
      label: "Rotate Left",
      buttons: ["Left Stick ←"],
      description: "Rotate counter-clockwise",
      icon: <RotateCcw className="size-6" />,
      type: "analog-horizontal" as const,
    },
    {
      key: "rotate-right",
      label: "Rotate Right",
      buttons: ["Left Stick →"],
      description: "Rotate clockwise",
      icon: <RotateCw className="size-6" />,
      type: "analog-horizontal" as const,
    },
    // Wrist controls
    {
      key: "wrist-pitch-up",
      label: "Wrist Up",
      buttons: ["D-Pad Up", "Y/Triangle"],
      description: "Wrist pitch up",
      icon: <ArrowUpFromLine className="size-6" />,
      type: "digital" as const,
    },
    {
      key: "wrist-pitch-down",
      label: "Wrist Down",
      buttons: ["D-Pad Down", "A/X"],
      description: "Wrist pitch down",
      icon: <ArrowDownFromLine className="size-6" />,
      type: "digital" as const,
    },
    {
      key: "wrist-roll-left",
      label: "Wrist Roll CCW",
      buttons: ["D-Pad Left", "X/Square"],
      description: "Wrist roll counter-clockwise",
      icon: <RotateCcw className="size-6" />,
      type: "digital" as const,
    },
    {
      key: "wrist-roll-right",
      label: "Wrist Roll CW",
      buttons: ["D-Pad Right", "B/Circle"],
      description: "Wrist roll clockwise",
      icon: <RotateCw className="size-6" />,
      type: "digital" as const,
    },
    // Gripper controls
    {
      key: "gripper-toggle",
      label: "Toggle Gripper",
      buttons: ["L1/LB", "R1/RB"],
      description: "Toggle gripper open/close",
      icon: <Space className="size-6" />,
      type: "digital" as const,
    },
    {
      key: "gripper-analog",
      label: "Gripper Control",
      buttons: ["L2/LT", "R2/RT"],
      description: "Analog gripper control (0-100%)",
      icon: <Space className="size-6" />,
      type: "trigger" as const,
    },
    // Special functions
    {
      key: "sleep",
      label: "Sleep",
      buttons: ["Start/Menu"],
      description: "Move to sleep position",
      icon: <Home className="size-6" />,
      type: "digital" as const,
    },
  ];

  if (serverError) return <div>Failed to load server status.</div>;
  if (!serverStatus) return <LoadingPage />;

  const selectedRobot = serverStatus.robot_status.find(
    (robot) => robot.device_name === selectedRobotName,
  );
  const isMobileRobot = selectedRobot?.robot_type === "mobile";

  return (
    <div className="container mx-auto px-4 py-6 space-y-8">
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-col items-center space-y-4">
            <div className="flex items-center gap-2">
              <Gamepad2 className={`size-8 ${gamepadConnected ? 'text-green-500' : 'text-gray-400'}`} />
              <span className="text-lg font-semibold">
                {gamepadConnected ? 'Gamepad Connected' : 'No Gamepad Detected'}
              </span>
            </div>
            {!gamepadConnected && (
              <div className="text-center space-y-2">
                <p className="text-sm text-muted-foreground">
                  Connect a game controller to your computer
                </p>
                <p className="text-lg font-medium text-primary animate-pulse">
                  Press any button on your controller to activate
                </p>
                <p className="text-xs text-muted-foreground">
                  (Browser security requires a button press to detect gamepads)
                </p>
                {selectedRobotName && (
                  <p className="text-sm text-green-600 dark:text-green-400 mt-2">
                    Robot control will start automatically when gamepad is detected
                  </p>
                )}
              </div>
            )}
          </div>
          
          <div className="flex items-center justify-center mt-6 gap-x-2 flex-wrap">
            <Select
              value={selectedRobotName || ""}
              onValueChange={(value) => setSelectedRobotName(value)}
              disabled={isMoving}
            >
              <SelectTrigger id="follower-robot" className="min-w-[200px]">
                <SelectValue placeholder="Select robot to move" />
              </SelectTrigger>
              <SelectContent>
                {serverStatus.robot_status.map((robot) => (
                  <SelectItem
                    key={robot.device_name}
                    value={robot.device_name || "Undefined port"}
                  >
                    {robot.name} ({robot.device_name})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {isMoving ? (
              <Button variant="destructive" onClick={stopMoving}>
                <Square className="mr-2 h-4 w-4" />
                Stop the Robot
              </Button>
            ) : (
              <Button
                variant="default"
                onClick={startMoving}
                disabled={!selectedRobotName || !gamepadConnected}
              >
                <Play className="mr-2 h-4 w-4" />
                Start Moving Robot
              </Button>
            )}
            {isMobileRobot && (
              <SpeedSelect
                defaultValue={selectedSpeed}
                onChange={(newSpeed) => setSelectedSpeed(newSpeed)}
                title="Movement speed"
                minSpeed={0.1}
                maxSpeed={1.0}
                step={0.1}
              />
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="pt-6">
          <h3 className="text-lg font-semibold mb-4">Gamepad Controls</h3>
          <div className="grid grid-cols-2 gap-4 mb-6">
            <div>
              <h4 className="font-medium mb-2">Movement & Rotation</h4>
              <ul className="text-sm space-y-1 text-muted-foreground">
                <li>• <span className="font-medium">Left Stick</span>: Rotate (X) / Move up-down (Y)</li>
                <li>• <span className="font-medium">Right Stick</span>: Strafe left-right (X) / Move forward-back (Y)</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-2">Gripper Control</h4>
              <ul className="text-sm space-y-1 text-muted-foreground">
                <li>• <span className="font-medium">L1/R1 (Bumpers)</span>: Toggle open/close</li>
                <li>• <span className="font-medium">L2/R2 (Triggers)</span>: Analog control (0-100%)</li>
              </ul>
            </div>
          </div>
          
          <div className="mb-4">
            <h4 className="font-medium mb-2">Wrist Control</h4>
            <p className="text-sm text-muted-foreground mb-2">
              Use either D-Pad or face buttons (ABXY) for wrist movements:
            </p>
            <div className="grid grid-cols-2 gap-2 text-sm text-muted-foreground">
              <div>• <span className="font-medium">Up (D-Pad/Y)</span>: Pitch up</div>
              <div>• <span className="font-medium">Down (D-Pad/A)</span>: Pitch down</div>
              <div>• <span className="font-medium">Left (D-Pad/X)</span>: Roll counter-clockwise</div>
              <div>• <span className="font-medium">Right (D-Pad/B)</span>: Roll clockwise</div>
            </div>
          </div>
          
          <div className="mb-6">
            <h4 className="font-medium mb-2">Special Functions</h4>
            <ul className="text-sm space-y-1 text-muted-foreground">
              <li>• <span className="font-medium">Start/Menu</span>: Move arm to sleep position</li>
            </ul>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {controls.map((control) => (
              <TooltipProvider key={control.key}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    {control.type === "trigger" ? (
                      <div>
                        <TriggerButton 
                          label={control.label}
                          buttons={control.buttons}
                          value={Math.max(analogValues.leftTrigger, analogValues.rightTrigger)}
                          icon={control.icon}
                          onClick={() => {
                            // Only allow clicks when robot is moving
                            if (!isMoving) return;
                            
                            // Toggle gripper for trigger button click
                            openStateRef.current = openStateRef.current > 0.5 ? 0 : 1;
                            const data = {
                              x: 0,
                              y: 0,
                              z: 0,
                              rx: 0,
                              ry: 0,
                              rz: 0,
                              open: openStateRef.current,
                            };
                            postData(BASE_URL + "move/relative", data, {
                              robot_id: robotIDFromName(selectedRobotName),
                            });
                          }}
                        />
                      </div>
                    ) : (
                      <div>
                        <ControlButton
                          control={control}
                          isActive={activeButtons.has(control.key)}
                          analogValue={
                            control.key === "move-forward" ? (analogValues.rightStickY < 0 ? -analogValues.rightStickY : 0) :
                            control.key === "move-backward" ? (analogValues.rightStickY > 0 ? analogValues.rightStickY : 0) :
                            control.key === "move-left" ? (analogValues.rightStickX < 0 ? -analogValues.rightStickX : 0) :
                            control.key === "move-right" ? (analogValues.rightStickX > 0 ? analogValues.rightStickX : 0) :
                            control.key === "move-up" ? (analogValues.leftStickY < 0 ? -analogValues.leftStickY : 0) :
                            control.key === "move-down" ? (analogValues.leftStickY > 0 ? analogValues.leftStickY : 0) :
                            control.key === "rotate-left" ? (analogValues.leftStickX < 0 ? -analogValues.leftStickX : 0) :
                            control.key === "rotate-right" ? (analogValues.leftStickX > 0 ? analogValues.leftStickX : 0) :
                            undefined
                          }
                          onClick={() => {
                            // Only allow clicks when robot is moving
                            if (!isMoving) return;
                            
                            // Handle click based on control type
                            if (control.type === "digital") {
                              // Simulate the button press for digital controls
                              const data = {
                                x: 0,
                                y: 0,
                                z: 0,
                                rx: 0,
                                ry: 0,
                                rz: 0,
                                open: openStateRef.current,
                              };
                              
                              // Apply the control action
                              if (control.key === "wrist-pitch-up") {
                                data.rx = STEP_SIZE * 3.14;
                              } else if (control.key === "wrist-pitch-down") {
                                data.rx = -STEP_SIZE * 3.14;
                              } else if (control.key === "wrist-roll-left") {
                                data.ry = -STEP_SIZE * 3.14;
                              } else if (control.key === "wrist-roll-right") {
                                data.ry = STEP_SIZE * 3.14;
                              } else if (control.key === "gripper-toggle") {
                                openStateRef.current = openStateRef.current > 0.5 ? 0 : 1;
                                data.open = openStateRef.current;
                              } else if (control.key === "sleep") {
                                postData(BASE_URL + "move/sleep", {}, {
                                  robot_id: robotIDFromName(selectedRobotName),
                                });
                                return;
                              }
                              
                              postData(BASE_URL + "move/relative", data, {
                                robot_id: robotIDFromName(selectedRobotName),
                              });
                            } else if (control.type.startsWith("analog")) {
                              // Handle analog controls
                              const data = {
                                x: 0,
                                y: 0,
                                z: 0,
                                rx: 0,
                                ry: 0,
                                rz: 0,
                                open: openStateRef.current,
                              };
                              
                              // Apply movement based on control
                              const moveAmount = STEP_SIZE * 2; // Slightly larger for click control
                              
                              if (control.key === "move-forward") {
                                data.x = moveAmount;
                              } else if (control.key === "move-backward") {
                                data.x = -moveAmount;
                              } else if (control.key === "move-left") {
                                data.y = moveAmount;
                              } else if (control.key === "move-right") {
                                data.y = -moveAmount;
                              } else if (control.key === "move-up") {
                                data.z = moveAmount;
                              } else if (control.key === "move-down") {
                                data.z = -moveAmount;
                              } else if (control.key === "rotate-left") {
                                data.rz = -moveAmount * 3.14;
                              } else if (control.key === "rotate-right") {
                                data.rz = moveAmount * 3.14;
                              }
                              
                              // Apply speed scaling for mobile robots
                              const currentRobot = serverStatus?.robot_status.find(
                                (r) => r.device_name === selectedRobotName,
                              );
                              const isMobile = currentRobot?.robot_type === "mobile";
                              
                              if (isMobile) {
                                data.x *= selectedSpeed;
                                data.y *= selectedSpeed;
                                data.rz *= selectedSpeed;
                              }
                              
                              postData(BASE_URL + "move/relative", data, {
                                robot_id: robotIDFromName(selectedRobotName),
                              });
                            }
                          }}
                        />
                      </div>
                    )}
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>{control.description}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            ))}
          </div>
        </CardContent>
      </Card>

      {gamepadConnected && (
        <GamepadVisualizer gamepadIndex={gamepadIndex} />
      )}
    </div>
  );
} 
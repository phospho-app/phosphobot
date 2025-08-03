// GamepadComponents.tsx - UI Components and Custom Hooks

import React, { useEffect, useState, useRef } from "react";
import { toast } from "sonner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import type { ServerStatus } from "@/types";
import {
  GamepadState,
  Control,
  GamepadInfo,
  ControllerArmPair,
  MultiArmGroup,
  ConfigMode,
  ControllerState,
  AnalogValues,
  BUTTON_NAMES,
  BASE_URL,
  BUTTON_MAPPINGS,
  MULTI_ARM_SPECIAL_BUTTONS,
  DEBOUNCE_INTERVAL,
  getAvailableGamepads,
  processAnalogSticks,
  getControlName,
  getRobotsToControl,
  applyControlMode,
  postData,
  robotIDFromName,
  extractAnalogValues,
} from './GamepadUtils';

// ==================== CUSTOM HOOKS ====================

// Hook for gamepad detection and management
export function useGamepadDetection(
  configMode: ConfigMode,
  controllerArmPairs: ControllerArmPair[],
  multiArmGroups: MultiArmGroup[],
  hasUserMadeSelection: boolean,
  setControllerArmPairs: React.Dispatch<React.SetStateAction<ControllerArmPair[]>>,
  setMultiArmGroups: React.Dispatch<React.SetStateAction<MultiArmGroup[]>>,
  setHasUserMadeSelection: React.Dispatch<React.SetStateAction<boolean>>
) {
  const [gamepadConnected, setGamepadConnected] = useState(false);
  const [availableGamepads, setAvailableGamepads] = useState<GamepadInfo[]>([]);

  useEffect(() => {
    const updateGamepadList = () => {
      const available = getAvailableGamepads();
      setAvailableGamepads(available);
      setGamepadConnected(available.length > 0);
      
      // Auto-select logic
      const hasAnyControllerAssigned = controllerArmPairs.some(pair => pair.controller_index !== null) ||
                                      multiArmGroups.some(group => group.controller_index !== null);
      
      if (available.length > 0 && !hasAnyControllerAssigned && !hasUserMadeSelection) {
        if (configMode === "individual") {
          setControllerArmPairs(prev => {
            const newPairs = [...prev];
            newPairs[0] = {
              ...newPairs[0],
              controller_index: available[0].index,
              controller_name: available[0].name,
            };
            return newPairs;
          });
        } else {
          setMultiArmGroups(prev => {
            const newGroups = [...prev];
            if (newGroups.length > 0) {
              newGroups[0] = {
                ...newGroups[0],
                controller_index: available[0].index,
                controller_name: available[0].name,
              };
            }
            return newGroups;
          });
        }
      }
      
      // Handle complete disconnection
      if (available.length === 0 && hasAnyControllerAssigned) {
        if (configMode === "individual") {
          setControllerArmPairs(prev => 
            prev.map(pair => ({
              ...pair,
              controller_index: null,
              controller_name: "",
            }))
          );
        } else {
          setMultiArmGroups(prev => 
            prev.map(group => ({
              ...group,
              controller_index: null,
              controller_name: "",
            }))
          );
        }
        setHasUserMadeSelection(false);
      }
    };

    const handleGamepadConnected = (e: GamepadEvent) => {
      console.log("Gamepad connected:", e.gamepad);
      updateGamepadList();
    };

    const handleGamepadDisconnected = (e: GamepadEvent) => {
      console.log("Gamepad disconnected:", e.gamepad);
      updateGamepadList();
    };

    updateGamepadList();
    const intervalId = setInterval(updateGamepadList, 1000);

    window.addEventListener("gamepadconnected", handleGamepadConnected);
    window.addEventListener("gamepaddisconnected", handleGamepadDisconnected);

    return () => {
      clearInterval(intervalId);
      window.removeEventListener("gamepadconnected", handleGamepadConnected);
      window.removeEventListener("gamepaddisconnected", handleGamepadDisconnected);
    };
  }, [hasUserMadeSelection, configMode, controllerArmPairs, multiArmGroups, setControllerArmPairs, setMultiArmGroups, setHasUserMadeSelection]);

  return { gamepadConnected, availableGamepads };
}

// Hook for main control loop
export function useGamepadControl(
  isMoving: boolean,
  configMode: ConfigMode,
  controllerArmPairs: ControllerArmPair[],
  multiArmGroups: MultiArmGroup[],
  selectedSpeed: number,
  serverStatus: ServerStatus | undefined,
  setActiveButtonsPerPair: React.Dispatch<React.SetStateAction<Map<number, Set<string>>>>,
  setAnalogValuesPerController: React.Dispatch<React.SetStateAction<Map<number, AnalogValues>>>,
  setAnalogValues: React.Dispatch<React.SetStateAction<AnalogValues>>,
  setMultiArmGroups: React.Dispatch<React.SetStateAction<MultiArmGroup[]>>
) {
  const controlStates = useRef<Map<string, ControllerState>>(new Map());

  useEffect(() => {
    if (!isMoving) return;

    let activeConfigs: any[] = [];
    
    if (configMode === "individual") {
      activeConfigs = controllerArmPairs.filter(
        pair => pair.controller_index !== null && pair.robot_name !== null
      );
    } else {
      activeConfigs = multiArmGroups.filter(
        group => group.controller_index !== null && group.robot_names.length > 0
      );
    }

    if (activeConfigs.length === 0) return;

    const controlAllRobots = () => {
      const gamepads = navigator.getGamepads();

      activeConfigs.forEach((config, configIndex) => {
        const gamepad = gamepads[config.controller_index!];
        if (!gamepad) return;

        const configKey = configMode === "individual" ? `individual-${configIndex}` : `group-${config.id}`;
        
        // Get or create control state
        if (!controlStates.current.has(configKey)) {
          controlStates.current.set(configKey, {
            buttonsPressed: new Set<string>(),
            lastExecutionTime: 0,
            openState: 1,
            lastButtonStates: [],
            lastTriggerValue: 0,
            triggerControlActive: false,
            resetSent: false,
          });
        }

        const state = controlStates.current.get(configKey)!;
        const currentTime = Date.now();

        if (currentTime - state.lastExecutionTime >= DEBOUNCE_INTERVAL) {
          let deltaX = 0,
            deltaY = 0,
            deltaZ = 0,
            deltaRZ = 0,
            deltaRX = 0,
            deltaRY = 0;
          let didToggleOpen = false;

          // Process button inputs
          gamepad.buttons.forEach((button, index) => {
            const wasPressed = state.lastButtonStates[index] || false;
            const isPressed = button.pressed;

            if (isPressed && !wasPressed) {
              // Handle multi-arm special buttons first
              if (configMode === "multi-arm" && MULTI_ARM_SPECIAL_BUTTONS[index]) {
                const specialAction = MULTI_ARM_SPECIAL_BUTTONS[index];
                
                if (specialAction === 'switch_robot' && 'active_robot_index' in config) {
                  // Switch active robot in sequential mode
                  if (config.control_mode === 'sequential') {
                    setMultiArmGroups(prev => prev.map(group => 
                      group.id === config.id 
                        ? { ...group, active_robot_index: (group.active_robot_index + 1) % group.robot_names.length }
                        : group
                    ));
                    toast.info(`Switched to ${config.robot_names[(config.active_robot_index + 1) % config.robot_names.length]}`);
                  }
                }
                
                return; // Don't process as regular movement
              }

              // Regular button processing
              if (BUTTON_MAPPINGS[index]) {
                const controlName = getControlName(index);

                // Update visual feedback for active buttons
                setActiveButtonsPerPair((prev) => {
                  const newMap = new Map(prev);
                  const currentSet = newMap.get(configIndex) || new Set();
                  newMap.set(configIndex, new Set(currentSet).add(controlName));
                  return newMap;
                });

                if ((index === 9 || index === 10) && !state.resetSent) {
                  // Sleep command - apply to relevant robots
                  const robotsToControl = getRobotsToControl(config, configMode);
                  robotsToControl.forEach(robotName => {
                    postData(
                      BASE_URL + "move/sleep",
                      {},
                      {
                        robot_id: robotIDFromName(robotName, serverStatus),
                      },
                    );
                  });
                  state.resetSent = true;
                } else if (BUTTON_MAPPINGS[index].toggleOpen) {
                  didToggleOpen = true;
                } else {
                  state.buttonsPressed.add(index.toString());
                }
              }
            } else if (!isPressed && wasPressed) {
              // Button just released
              if (index === 9 || index === 10) {
                state.resetSent = false;
              }
              state.buttonsPressed.delete(index.toString());
              
              // Clear active button when released for visual feedback
              if (BUTTON_MAPPINGS[index]) {
                const controlName = getControlName(index);

                setActiveButtonsPerPair((prev) => {
                  const newMap = new Map(prev);
                  const currentSet = newMap.get(configIndex) || new Set();
                  const newSet = new Set(currentSet);
                  newSet.delete(controlName);
                  newMap.set(configIndex, newSet);
                  return newMap;
                });
              }
            }

            state.lastButtonStates[index] = isPressed;
          });

          // Accumulate button movements
          state.buttonsPressed.forEach((buttonStr) => {
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
          const analogMovement = processAnalogSticks(gamepad, state.lastTriggerValue);
          deltaX += analogMovement.x;
          deltaY += analogMovement.y;
          deltaZ += analogMovement.z;
          deltaRZ += analogMovement.rz;
          deltaRX += analogMovement.rx;
          deltaRY += analogMovement.ry;

          // Handle gripper control
          let gripperValue = state.openState;

          // Check if trigger value has changed significantly
          if (
            analogMovement.gripperValue !== undefined &&
            Math.abs(analogMovement.gripperValue - state.lastTriggerValue) > 0.05
          ) {
            // Trigger value changed - it can take control
            state.triggerControlActive = true;
            state.lastTriggerValue = analogMovement.gripperValue;
          }

          if (didToggleOpen) {
            // A button was pressed - always toggle and disable trigger control
            state.openState = state.openState > 0.5 ? 0 : 1;
            gripperValue = state.openState;
            state.triggerControlActive = false;
          } else if (
            analogMovement.gripperValue !== undefined &&
            state.triggerControlActive
          ) {
            // Use trigger value for gripper only if trigger control is active
            gripperValue = analogMovement.gripperValue;
            state.openState = gripperValue;
          }

          // Apply speed scaling to all robot types
          deltaX *= selectedSpeed;
          deltaY *= selectedSpeed;
          deltaZ *= selectedSpeed;
          deltaRX *= selectedSpeed;
          deltaRY *= selectedSpeed;
          deltaRZ *= selectedSpeed;

          if (
            deltaX !== 0 ||
            deltaY !== 0 ||
            deltaZ !== 0 ||
            deltaRZ !== 0 ||
            deltaRX !== 0 ||
            deltaRY !== 0 ||
            didToggleOpen ||
            (analogMovement.gripperValue !== undefined &&
               state.triggerControlActive)
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

            // Send commands to appropriate robots based on config mode
            const robotsToControl = getRobotsToControl(config, configMode);
            robotsToControl.forEach((robotName, _) => {
              let finalData = { ...data };
              
              // Apply control mode modifications
              if (configMode === "multi-arm" && 'control_mode' in config) {
                finalData = applyControlMode(data, config.control_mode);
              }
              
              postData(BASE_URL + "move/relative", finalData, {
                robot_id: robotIDFromName(robotName, serverStatus),
              });
            });
          }
          state.lastExecutionTime = currentTime;
        }
      });

      // Update visual feedback for all active controllers
      if (activeConfigs.length > 0) {
        const allActiveControllerIndices = new Set<number>();
        
        activeConfigs.forEach(config => {
          if (config.controller_index !== null) {
            allActiveControllerIndices.add(config.controller_index);
          }
        });

        // Update analog values for each unique controller
        const newAnalogValuesPerController = new Map<number, AnalogValues>();
        
        allActiveControllerIndices.forEach(controllerIndex => {
          const gamepad = gamepads[controllerIndex];
          if (gamepad) {
            const analogValues = extractAnalogValues(gamepad);
            newAnalogValuesPerController.set(controllerIndex, analogValues);
          }
        });

        setAnalogValuesPerController(newAnalogValuesPerController);

        // Also update the legacy analogValues for the first controller (backward compatibility)
        const firstControllerIndex = Array.from(allActiveControllerIndices)[0];
        if (firstControllerIndex !== undefined && newAnalogValuesPerController.has(firstControllerIndex)) {
          setAnalogValues(newAnalogValuesPerController.get(firstControllerIndex)!);
        }
      }
    };

    const intervalId = setInterval(controlAllRobots, 10); // LOOP_INTERVAL = 10ms
    
    return () => {
      clearInterval(intervalId);
    };
  }, [isMoving, controllerArmPairs, multiArmGroups, selectedSpeed, serverStatus, configMode, setActiveButtonsPerPair, setAnalogValuesPerController, setAnalogValues, setMultiArmGroups]);

  return { controlStates };
}

// ==================== UI COMPONENTS ====================

// GamepadVisualizer component
export function GamepadVisualizer({ gamepadIndex }: { gamepadIndex: number | null }) {
  const [gamepadState, setGamepadState] = useState<GamepadState>({
    connected: false,
    buttons: [],
    buttonValues: [],
    axes: [],
  });

  useEffect(() => {
    if (gamepadIndex === null) {
      setGamepadState({
        connected: false,
        buttons: [],
        buttonValues: [],
        axes: [],
      });
      return;
    }

    const updateGamepadState = () => {
      const gamepads = navigator.getGamepads();
      const gamepad = gamepads[gamepadIndex];

      if (gamepad) {
        setGamepadState({
          connected: true,
          buttons: Array.from(gamepad.buttons).map((b) => b.pressed),
          buttonValues: Array.from(gamepad.buttons).map((b) => b.value),
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
        <CardTitle className="text-sm">Controller {(gamepadIndex ?? 0) + 1} State</CardTitle>
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
                {BUTTON_NAMES[index] || `Button ${index}`}
                {/* Show analog value for L2/R2 if they're analog buttons */}
                {(index === 6 || index === 7) &&
                  gamepadState.buttonValues[index] > 0 &&
                  gamepadState.buttonValues[index] < 1 && (
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
                <p className="text-xs mb-1">
                  Left Stick X: {gamepadState.axes[0]?.toFixed(2) || "0.00"}
                </p>
                <Progress
                  value={(gamepadState.axes[0] + 1) * 50}
                  className="h-2"
                />
              </div>
              <div>
                <p className="text-xs mb-1">
                  Left Stick Y: {gamepadState.axes[1]?.toFixed(2) || "0.00"}
                </p>
                <Progress
                  value={(gamepadState.axes[1] + 1) * 50}
                  className="h-2"
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs mb-1">
                  Right Stick X: {gamepadState.axes[2]?.toFixed(2) || "0.00"}
                </p>
                <Progress
                  value={(gamepadState.axes[2] + 1) * 50}
                  className="h-2"
                />
              </div>
              <div>
                <p className="text-xs mb-1">
                  Right Stick Y: {gamepadState.axes[3]?.toFixed(2) || "0.00"}
                </p>
                <Progress
                  value={(gamepadState.axes[3] + 1) * 50}
                  className="h-2"
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs mb-1">
                  Left Trigger: {leftTriggerValue.toFixed(2)}
                </p>
                <Progress value={leftTriggerValue * 100} className="h-2" />
              </div>
              <div>
                <p className="text-xs mb-1">
                  Right Trigger: {rightTriggerValue.toFixed(2)}
                </p>
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
export function TriggerButton({
  label,
  buttons,
  value,
  icon,
  onClick,
}: {
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
export function ControlButton({
  control,
  isActive,
  analogValue,
  onClick,
}: {
  control: Control;
  isActive: boolean;
  analogValue?: number;
  onClick?: () => void;
}) {
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const handleMouseDown = () => {
    if (!onClick) return;

    // For analog controls, start continuous movement
    if (control.type.startsWith("analog")) {
      onClick(); // Initial click
      intervalRef.current = setInterval(() => {
        onClick();
      }, 100); // Send command every 100ms while held
    } else {
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
  const showGradient =
    control.type.startsWith("analog") &&
    analogValue !== undefined &&
    analogValue > 0;

  return (
    <Card
      className={`relative flex flex-col items-center justify-center p-4 cursor-pointer transition-colors overflow-hidden h-full ${
        isActive
          ? "bg-primary/20 dark:bg-primary/30"
          : "bg-card hover:bg-accent"
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
            height:
              control.type === "analog-vertical"
                ? `${analogValue * 100}%`
                : "100%",
            width:
              control.type === "analog-horizontal"
                ? `${analogValue * 100}%`
                : "100%",
            left: control.type === "analog-horizontal" ? "0" : "0",
            right: control.type === "analog-horizontal" ? "auto" : "0",
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
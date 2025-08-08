import controlschema from "@/assets/ControlSchema.png";
import { LoadingPage } from "@/components/common/loading";
import { SpeedSelect } from "@/components/common/speed-select";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
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
  ArrowDown,
  ArrowDownFromLine,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  ArrowUpFromLine,
  ChevronDown,
  ChevronUp,
  Play,
  RotateCcw,
  RotateCw,
  Space,
  Square,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import useSWR from "swr";

export function KeyboardControl() {
  const { data: serverStatus, error: serverError } = useSWR<ServerStatus>(
    ["/status"],
    fetcher,
    {
      refreshInterval: 5000,
    },
  );

  const [isMoving, setIsMoving] = useState(false);
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [selectedRobotName, setSelectedRobotName] = useState<string | null>(
    null,
  );
  const [selectedSpeed, setSelectedSpeed] = useState<number>(0.8); // State for speed

  // Refs to manage our control loop and state
  const keysPressedRef = useRef(new Set<string>());
  const intervalIdRef = useRef<NodeJS.Timeout | null>(null);
  const lastExecutionTimeRef = useRef(0);
  // open state is continuous between 0 (fully closed) and 1 (fully open)
  const openStateRef = useRef(1);

  // NEW PARAMETER: full close time in milliseconds at speed = 1.
  // Change this value to adjust how long it takes to fully close the gripper.
  const FULL_CLOSE_MS = 400; // <-- fully close after ~400ms hold at speed=1

  // Derived parameter: how long a "brief press" should be (as a fraction of full close)
  const BRIEF_PRESS_MS = Math.max(20, Math.round(FULL_CLOSE_MS * 0.25));

  // Refs for space (gripper hold) behavior
  const spacePressStartRef = useRef<number | null>(null);
  // Interval used when the user holds the UI button but `isMoving` is false
  const uiHoldIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Configuration constants (from control.js)
  const BASE_URL = `http://${window.location.hostname}:${window.location.port}/`; // Use template literal for clarity
  const STEP_SIZE = 1; // in centimeters
  const LOOP_INTERVAL = 10; // ms (~50 Hz)
  const INSTRUCTIONS_PER_SECOND = 30;
  const DEBOUNCE_INTERVAL = 1000 / INSTRUCTIONS_PER_SECOND;

  interface RobotMovement {
    x: number;
    y: number;
    z: number;
    rz: number;
    rx: number;
    ry: number;
    toggleOpen?: boolean;
  }

  // Mappings for keys (from control.js)
  const KEY_MAPPINGS: Record<string, RobotMovement> = {
    f: { x: 0, y: 0, z: STEP_SIZE, rz: 0, rx: 0, ry: 0 },
    v: { x: 0, y: 0, z: -STEP_SIZE, rz: 0, rx: 0, ry: 0 },
    ArrowUp: { x: STEP_SIZE, y: 0, z: 0, rz: 0, rx: 0, ry: 0 },
    ArrowDown: { x: -STEP_SIZE, y: 0, z: 0, rz: 0, rx: 0, ry: 0 },
    ArrowRight: { x: 0, y: 0, z: 0, rz: -STEP_SIZE * 3.14, rx: 0, ry: 0 },
    ArrowLeft: { x: 0, y: 0, z: 0, rz: STEP_SIZE * 3.14, rx: 0, ry: 0 },
    d: { x: 0, y: 0, z: 0, rz: 0, rx: STEP_SIZE * 3.14, ry: 0 },
    g: { x: 0, y: 0, z: 0, rz: 0, rx: -STEP_SIZE * 3.14, ry: 0 },
    b: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: STEP_SIZE * 3.14 },
    c: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: -STEP_SIZE * 3.14 },
    // space no longer toggles; we handle it with hold logic below
    " ": { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: 0 },
  };

  const robotIDFromName = (name?: string | null) => {
    if (name === undefined || name === null || !serverStatus?.robot_status) {
      return 0; // Default to the first robot
    }
    const index = serverStatus.robot_status.findIndex(
      (robot) => robot.device_name === name,
    );
    return index === -1 ? 0 : index; // Return 0 if not found or first one
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
      console.error("Error posting data:", error); // Enhanced logging
    }
  };

  // compute target open value (0..1) from a hold duration in milliseconds
  const computeTargetOpenFromHoldMs = (holdMs: number) => {
    const holdSec = Math.max(0, holdMs) / 1000;
    // Effective hold is scaled by selectedSpeed so higher speed -> faster closing
    const speedFactor = Math.max(0.01, selectedSpeed);
    const effectiveHold = holdSec * speedFactor; // seconds in "effective" units

    // Full-close parameter in effective seconds (derived from FULL_CLOSE_MS)
    const fullCloseEffective = Math.max(0.001, FULL_CLOSE_MS / 1000);

    const denom = Math.log(1 + fullCloseEffective);
    const numer = Math.log(1 + effectiveHold);
    const norm = denom > 0 ? Math.min(1, numer / denom) : 1;

    // Target open state decreases from 1 down to 0 as norm goes 0->1
    return Math.max(0, 1 - norm);
  };

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.repeat) return;
      setActiveKey(event.key);

      const keyLower = event.key.toLowerCase();

      // Special handling for space (gripper)
      const isSpace = event.key === " " || keyLower === " ";
      if (isSpace) {
        // If gripper is not fully open, a single press always opens it immediately
        if (openStateRef.current < 0.99) {
          openStateRef.current = 1;
          // Send immediate open command
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
          // Don't add to keysPressedRef: this press is consumed as "open" action
          return;
        }

        // If gripper is (fully) open, start the hold-closure behavior
        keysPressedRef.current.add(" ");
        spacePressStartRef.current = Date.now();
        return;
      }

      if (KEY_MAPPINGS[keyLower]) {
        keysPressedRef.current.add(keyLower);
      } else if (KEY_MAPPINGS[event.key]) {
        keysPressedRef.current.add(event.key);
      }
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      setActiveKey(null);
      const keyLower = event.key.toLowerCase();

      const isSpace = event.key === " " || keyLower === " ";
      if (isSpace) {
        keysPressedRef.current.delete(" ");
        spacePressStartRef.current = null;
        return;
      }

      if (KEY_MAPPINGS[keyLower]) {
        keysPressedRef.current.delete(keyLower);
      } else if (KEY_MAPPINGS[event.key]) {
        keysPressedRef.current.delete(event.key);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [selectedRobotName, serverStatus, selectedSpeed]);

  useEffect(() => {
    if (
      !selectedRobotName &&
      serverStatus?.robot_status &&
      serverStatus.robot_status.length > 0 &&
      serverStatus.robot_status[0].device_name
    ) {
      setSelectedRobotName(serverStatus.robot_status[0].device_name);
    }
  }, [serverStatus, selectedRobotName]); // Simplified dependency array

  // Start the control loop only when the robot is moving.
  useEffect(() => {
    if (isMoving) {
      const controlRobot = () => {
        const currentTime = Date.now();
        if (currentTime - lastExecutionTimeRef.current >= DEBOUNCE_INTERVAL) {
          let deltaX = 0,
            deltaY = 0,
            deltaZ = 0,
            deltaRZ = 0,
            deltaRX = 0,
            deltaRY = 0;

          // Note: we no longer use toggleOpen from KEY_MAPPINGS. The space key is
          // handled via spacePressStartRef and keysPressedRef contains " " while held.
          keysPressedRef.current.forEach((key) => {
            if (KEY_MAPPINGS[key]) {
              deltaX += KEY_MAPPINGS[key].x;
              deltaY += KEY_MAPPINGS[key].y;
              deltaZ += KEY_MAPPINGS[key].z;
              deltaRZ += KEY_MAPPINGS[key].rz;
              deltaRX += KEY_MAPPINGS[key].rx;
              deltaRY += KEY_MAPPINGS[key].ry;
            }
          });

          // Apply speed scaling to all robot types
          deltaX *= selectedSpeed;
          deltaY *= selectedSpeed;
          deltaZ *= selectedSpeed;
          deltaRX *= selectedSpeed;
          deltaRY *= selectedSpeed;
          deltaRZ *= selectedSpeed;

          // Handle gripper hold-to-close behavior when space is held
          if (keysPressedRef.current.has(" ") && spacePressStartRef.current) {
            const holdMs = Math.max(0, Date.now() - spacePressStartRef.current);

            const targetOpen = computeTargetOpenFromHoldMs(holdMs);

            // Update openStateRef directly to the computed target (continuous)
            openStateRef.current = Math.max(0, Math.min(1, targetOpen));
          }

          // Build and send command if any movement or open state changed since last send
          if (
            deltaX !== 0 ||
            deltaY !== 0 ||
            deltaZ !== 0 ||
            deltaRZ !== 0 ||
            deltaRX !== 0 ||
            deltaRY !== 0 ||
            // Always include open state so gripper updates are sent while holding
            keysPressedRef.current.has(" ")
          ) {
            const data = {
              x: deltaX,
              y: deltaY,
              z: deltaZ,
              rx: deltaRX,
              ry: deltaRY,
              rz: deltaRZ,
              open: openStateRef.current,
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
  }, [isMoving, selectedSpeed, serverStatus, selectedRobotName]); // Added dependencies

  // UI-controlled hold start (mouse / touch)
  const startSpacePressFromUI = (
    e?: MouseEvent | TouchEvent | React.MouseEvent | React.TouchEvent,
  ) => {
    // Prevent focus/drag behavior on touch
    if (e && e.preventDefault) e.preventDefault();

    // If gripper is not fully open, open immediately on a single press
    if (openStateRef.current < 0.99) {
      openStateRef.current = 1;
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
      return;
    }

    // Begin holding
    keysPressedRef.current.add(" ");
    spacePressStartRef.current = Date.now();
    setActiveKey(" ");

    // If the main keyboard-controlled loop isn't running, we need to send
    // updates ourselves while the UI button is held. Start a temporary interval.
    if (!isMoving) {
      if (uiHoldIntervalRef.current) {
        clearInterval(uiHoldIntervalRef.current);
        uiHoldIntervalRef.current = null;
      }
      uiHoldIntervalRef.current = setInterval(() => {
        if (!spacePressStartRef.current) return;
        const holdMs = Date.now() - spacePressStartRef.current;
        const targetOpen = computeTargetOpenFromHoldMs(holdMs);
        openStateRef.current = Math.max(0, Math.min(1, targetOpen));
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
      }, LOOP_INTERVAL);
    }
  };

  // UI-controlled hold end (mouse up / touch end / leave)
  const endSpacePressFromUI = () => {
    const start = spacePressStartRef.current;
    // Clear UI interval if present
    if (uiHoldIntervalRef.current) {
      clearInterval(uiHoldIntervalRef.current);
      uiHoldIntervalRef.current = null;
    }

    // If there was no real start, nothing to do
    if (!start) {
      setActiveKey(null);
      keysPressedRef.current.delete(" ");
      spacePressStartRef.current = null;
      return;
    }

    const holdMs = Math.max(0, Date.now() - start);

    // Delete the pressed flag and clear timing
    keysPressedRef.current.delete(" ");
    spacePressStartRef.current = null;
    setActiveKey(null);

    // If the press was very brief, treat as brief press (BRIEF_PRESS_MS)
    const usedHoldMs = holdMs < BRIEF_PRESS_MS ? BRIEF_PRESS_MS : holdMs;

    const targetOpen = computeTargetOpenFromHoldMs(usedHoldMs);
    openStateRef.current = Math.max(0, Math.min(1, targetOpen));

    // Send final update so UI click produces immediate effect (even if isMoving is false)
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
  };

  useEffect(() => {
    // Ensure we clean up UI interval when unmounting
    return () => {
      if (uiHoldIntervalRef.current) {
        clearInterval(uiHoldIntervalRef.current);
        uiHoldIntervalRef.current = null;
      }
    };
  }, []);

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
      // Ensure our local state tracks the robot
      openStateRef.current = 1;
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
    // Optionally, send a stop command or zero movement command here if needed
  };

  const controls = [
    {
      key: "ArrowUp",
      description: "Move in the positive X direction",
      icon: <ArrowUp className="size-6" />,
    },
    {
      key: "ArrowDown",
      description: "Move in the negative X direction",
      icon: <ArrowDown className="size-6" />,
    },
    {
      key: "ArrowLeft",
      description: "Rotate Z counter-clockwise (yaw)",
      icon: <ArrowLeft className="size-6" />,
    },
    {
      key: "ArrowRight",
      description: "Rotate Z clockwise (yaw)",
      icon: <ArrowRight className="size-6" />,
    },
    {
      key: "F",
      description: "Increase Z (move up)",
      icon: <ChevronUp className="size-6" />,
    },
    {
      key: "V",
      description: "Decrease Z (move down)",
      icon: <ChevronDown className="size-6" />,
    },
    {
      key: "D",
      description: "Wrist pitch up",
      icon: <ArrowUpFromLine className="size-6" />,
    },
    {
      key: "G",
      description: "Wrist pitch down",
      icon: <ArrowDownFromLine className="size-6" />,
    },
    {
      key: "B",
      description: "Wrist roll clockwise",
      icon: <RotateCw className="size-6" />,
    },
    {
      key: "C",
      description: "Wrist roll counter-clockwise",
      icon: <RotateCcw className="size-6" />,
    },
    {
      key: " ",
      description: "Hold Space to close the gripper. Press once to open.",
      icon: <Space className="size-6" />,
    },
  ];

  if (serverError) return <div>Failed to load server status.</div>; // Handle error case
  if (!serverStatus) return <LoadingPage />;

  return (
    <div className="container mx-auto px-4 py-6 space-y-8">
      <Card>
        <CardContent className="pt-6">
          {" "}
          {/* Added pt-6 for padding consistency */}
          <figure className="flex flex-col items-center">
            <img
              src={controlschema}
              alt="Robot Control Schema"
              className="w-full max-w-md rounded-md shadow"
            />
            <figcaption className="text-sm text-muted-foreground text-center mt-2">
              Press the key to move the robot in the corresponding direction
            </figcaption>
          </figure>
          <div className="flex items-center justify-center mt-6 gap-x-2 flex-wrap">
            {" "}
            {/* Added flex-wrap for smaller screens */}
            <Select
              value={selectedRobotName || ""}
              onValueChange={(value) => setSelectedRobotName(value)}
              disabled={isMoving}
            >
              <SelectTrigger id="follower-robot" className="min-w-[200px]">
                {" "}
                {/* Added min-width */}
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
                disabled={!selectedRobotName} // Disable if no robot is selected
              >
                <Play className="mr-2 h-4 w-4" />
                Start Moving Robot
              </Button>
            )}
            <SpeedSelect
              defaultValue={selectedSpeed}
              onChange={(newSpeed) => setSelectedSpeed(newSpeed)}
              title="Movement speed"
              minSpeed={0.1}
              maxSpeed={2.0} // Allow faster speeds for all robot types
              step={0.1}
            />
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
        {" "}
        {/* Adjusted grid for responsiveness */}
        {controls.map((control) => (
          <TooltipProvider key={control.key}>
            <Tooltip>
              <TooltipTrigger asChild>
                <Card
                  className={`flex flex-col items-center justify-center p-4 cursor-pointer hover:bg-accent transition-colors ${
                    activeKey === control.key
                      ? "bg-primary text-primary-foreground"
                      : "bg-card" // Ensure default background for card
                  }`}
                  // For the space control we use press (mouse/touch) handlers so long presses work
                  onMouseDown={
                    control.key === " "
                      ? (e) => startSpacePressFromUI(e)
                      : undefined
                  }
                  onTouchStart={
                    control.key === " "
                      ? (e) => startSpacePressFromUI(e)
                      : undefined
                  }
                  onMouseUp={
                    control.key === " "
                      ? () => endSpacePressFromUI()
                      : undefined
                  }
                  onMouseLeave={
                    control.key === " "
                      ? () => endSpacePressFromUI()
                      : undefined
                  }
                  onTouchEnd={
                    control.key === " "
                      ? () => endSpacePressFromUI()
                      : undefined
                  }
                  onTouchCancel={
                    control.key === " "
                      ? () => endSpacePressFromUI()
                      : undefined
                  }
                  onClick={() => {
                    if (!isMoving && control.key !== " ") {
                      // For non-spacebar clicks, simulate a quick key press only if not already moving via keyboard
                      const move =
                        KEY_MAPPINGS[control.key.toLowerCase()] ||
                        KEY_MAPPINGS[control.key];
                      if (move) {
                        const data = {
                          x: move.x * selectedSpeed,
                          y: move.y * selectedSpeed,
                          z: move.z * selectedSpeed,
                          rx: move.rx * selectedSpeed,
                          ry: move.ry * selectedSpeed,
                          rz: move.rz * selectedSpeed,
                          open: openStateRef.current,
                        };

                        postData(BASE_URL + "move/relative", data, {
                          robot_id: robotIDFromName(selectedRobotName),
                        });
                      }
                      setActiveKey(control.key);
                      setTimeout(() => setActiveKey(null), 200); // Briefly highlight
                      return; // Prevent falling through to old logic for these buttons if not moving
                    }

                    // New behavior for spacebar click: handled by press handlers above
                    if (control.key === " ") {
                      // no-op here because we handle open/close in the press handlers
                      return;
                    } else if (isMoving) {
                      const K = KEY_MAPPINGS[control.key.toLowerCase()]
                        ? control.key.toLowerCase()
                        : control.key;
                      keysPressedRef.current.add(K);
                      setActiveKey(control.key);
                      setTimeout(() => {
                        keysPressedRef.current.delete(K);
                        setActiveKey(null);
                      }, 300); // Duration of simulated press
                    }
                  }}
                >
                  {control.icon}
                  <span className="mt-2 font-bold">
                    {control.key === " " ? "SPACE" : control.key.toUpperCase()}
                  </span>
                </Card>
              </TooltipTrigger>
              <TooltipContent>
                <p>{control.description}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        ))}
      </div>
    </div>
  );
}

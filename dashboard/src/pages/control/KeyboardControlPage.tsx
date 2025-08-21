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
  const [selectedSpeed, setSelectedSpeed] = useState<number>(0.8);

  // Refs to manage our control loop and state
  const keysPressedRef = useRef(new Set<string>());
  const intervalIdRef = useRef<NodeJS.Timeout | null>(null);
  const lastExecutionTimeRef = useRef(0);
  // open state is continuous between 0 (fully closed) and 1 (fully open)
  const openStateRef = useRef(1);

  // Time in milliseconds to go from fully open to fully closed at speed = 1.
  const FULL_TRANSITION_MS = 500;

  // Configuration constants
  const BASE_URL = `http://${window.location.hostname}:${window.location.port}/`;
  const STEP_SIZE = 1;
  const LOOP_INTERVAL = 10; // ms, target loop frequency
  const INSTRUCTIONS_PER_SECOND = 30; // Max command send rate
  const DEBOUNCE_INTERVAL = 1000 / INSTRUCTIONS_PER_SECOND;

  interface RobotMovement {
    x: number;
    y: number;
    z: number;
    rz: number;
    rx: number;
    ry: number;
  }

  // Mappings for keys
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
    " ": { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: 0 }, // Placeholder, logic handled separately
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

  // Effect for handling keyboard inputs
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.repeat) return;
      setActiveKey(event.key);
      const keyLower = event.key.toLowerCase();

      const isSpace = event.key === " ";
      if (isSpace) {
        // UX Feature: If gripper is not fully open, a single press will fully open it.
        if (openStateRef.current < 0.99) {
          openStateRef.current = 1;
          // Send an immediate command to ensure responsiveness
          const data = { x: 0, y: 0, z: 0, rx: 0, ry: 0, rz: 0, open: 1 };
          postData(BASE_URL + "move/relative", data, {
            robot_id: robotIDFromName(selectedRobotName),
          });
          // Consume this keydown; don't add " " to keysPressedRef.
          return;
        }
        keysPressedRef.current.add(" ");
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

      if (event.key === " ") {
        keysPressedRef.current.delete(" ");
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
  }, [selectedRobotName, serverStatus]);

  // Effect for setting the default selected robot
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

  // The main, always-on control loop
  useEffect(() => {
    const controlRobot = () => {
      const currentTime = Date.now();
      const deltaT_ms = currentTime - lastExecutionTimeRef.current;

      if (deltaT_ms < DEBOUNCE_INTERVAL) {
        return; // Throttle command sending to avoid overwhelming the server
      }

      const lastOpenState = openStateRef.current;
      let openStateChanged = false;

      // --- 1. Gripper State Calculation (Always Active) ---
      const gripperChangePerMs = (1.0 / FULL_TRANSITION_MS) * selectedSpeed;
      const gripperChangeAmount = gripperChangePerMs * deltaT_ms;

      if (keysPressedRef.current.has(" ")) {
        openStateRef.current -= gripperChangeAmount; // Close gripper
      } else {
        openStateRef.current += gripperChangeAmount; // Open gripper
      }
      openStateRef.current = Math.max(0, Math.min(1, openStateRef.current)); // Clamp to [0, 1]

      if (Math.abs(openStateRef.current - lastOpenState) > 0.001) {
        openStateChanged = true;
      }

      // --- 2. Robot Movement Calculation (Active only if `isMoving`) ---
      let deltaX = 0,
        deltaY = 0,
        deltaZ = 0,
        deltaRZ = 0,
        deltaRX = 0,
        deltaRY = 0;
      let hasMovement = false;

      if (isMoving) {
        keysPressedRef.current.forEach((key) => {
          const move = KEY_MAPPINGS[key];
          if (move && key !== " ") {
            deltaX += move.x;
            deltaY += move.y;
            deltaZ += move.z;
            deltaRZ += move.rz;
            deltaRX += move.rx;
            deltaRY += move.ry;
          }
        });

        if (deltaX || deltaY || deltaZ || deltaRZ || deltaRX || deltaRY) {
          hasMovement = true;
          deltaX *= selectedSpeed;
          deltaY *= selectedSpeed;
          deltaZ *= selectedSpeed;
          deltaRX *= selectedSpeed;
          deltaRY *= selectedSpeed;
          deltaRZ *= selectedSpeed;
        }
      }

      // --- 3. Send Command if Needed ---
      if (hasMovement || openStateChanged) {
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
    };

    intervalIdRef.current = setInterval(controlRobot, LOOP_INTERVAL);

    return () => {
      if (intervalIdRef.current) {
        clearInterval(intervalIdRef.current);
      }
    };
  }, [isMoving, selectedSpeed, serverStatus, selectedRobotName]);

  // UI button handlers for space (mouse/touch)
  const startSpacePressFromUI = (e?: React.MouseEvent | React.TouchEvent) => {
    e?.preventDefault();
    // Same UX as keyboard: single press on a non-open gripper opens it.
    if (openStateRef.current < 0.99) {
      openStateRef.current = 1;
      const data = { x: 0, y: 0, z: 0, rx: 0, ry: 0, rz: 0, open: 1 };
      postData(BASE_URL + "move/relative", data, {
        robot_id: robotIDFromName(selectedRobotName),
      });
      return;
    }
    keysPressedRef.current.add(" ");
    setActiveKey(" ");
  };

  const endSpacePressFromUI = () => {
    keysPressedRef.current.delete(" ");
    setActiveKey(null);
  };

  const initRobot = async () => {
    try {
      await postData(
        BASE_URL + "move/init",
        {},
        { robot_id: robotIDFromName(selectedRobotName) },
      );
      await new Promise((resolve) => setTimeout(resolve, 2000));
      const initData = { x: 0, y: 0, z: 0, rx: 0, ry: 0, rz: 0, open: 1 };
      await postData(BASE_URL + "move/absolute", initData, {
        robot_id: robotIDFromName(selectedRobotName),
      });
      openStateRef.current = 1; // Sync local state
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
      description:
        "Hold to close gripper, release to open. Press once to fully open.",
      icon: <Space className="size-6" />,
    },
  ];

  if (serverError) return <div>Failed to load server status.</div>;
  if (!serverStatus) return <LoadingPage />;

  return (
    <div className="container mx-auto px-4 py-6 space-y-8">
      <Card>
        <CardContent className="pt-6">
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
                disabled={!selectedRobotName}
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
              maxSpeed={2.0}
              step={0.1}
            />
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
        {controls.map((control) => (
          <TooltipProvider key={control.key}>
            <Tooltip>
              <TooltipTrigger asChild>
                <Card
                  className={`flex flex-col items-center justify-center p-4 cursor-pointer hover:bg-accent transition-colors ${
                    activeKey === control.key
                      ? "bg-primary text-primary-foreground"
                      : "bg-card"
                  }`}
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
                    control.key === " " ? endSpacePressFromUI : undefined
                  }
                  onMouseLeave={
                    control.key === " " ? endSpacePressFromUI : undefined
                  }
                  onTouchEnd={
                    control.key === " " ? endSpacePressFromUI : undefined
                  }
                  onTouchCancel={
                    control.key === " " ? endSpacePressFromUI : undefined
                  }
                  onClick={() => {
                    // Clicks on spacebar are handled by press/release events above
                    if (control.key === " ") return;

                    if (!isMoving) {
                      // For non-spacebar clicks, simulate a quick key press
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
                      setTimeout(() => setActiveKey(null), 200);
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

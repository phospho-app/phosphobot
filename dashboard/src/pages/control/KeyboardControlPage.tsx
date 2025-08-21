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
  const lastExecutionTimeRef = useRef(0);
  // open state is continuous between 0 (fully closed) and 1 (fully open)
  const openStateRef = useRef(1);
  const lastSentOpenStateRef = useRef(1);

  // How long it takes to fully close the gripper at speed = 1.
  const FULL_CLOSE_MS = 400;

  // Configuration constants
  const BASE_URL = `http://${window.location.hostname}:${window.location.port}/`;
  const STEP_SIZE = 1; // in centimeters
  const LOOP_INTERVAL = 15; // ms, a bit slower is fine and reduces load
  const INSTRUCTIONS_PER_SECOND = 30;
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
    " ": { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: 0 },
  };

  const robotIDFromName = (name?: string | null) => {
    if (name === undefined || name === null || !serverStatus?.robot_status) {
      return 0; // Default to the first robot
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

      await fetch(newUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });
    } catch (error) {
      console.error("Error posting data:", error);
    }
  };

  // Keyboard event listeners
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.repeat) return;
      setActiveKey(event.key);

      const key = event.key === " " ? " " : event.key.toLowerCase();
      if (KEY_MAPPINGS[key] || KEY_MAPPINGS[event.key]) {
        keysPressedRef.current.add(KEY_MAPPINGS[key] ? key : event.key);
      }
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      setActiveKey(null);
      const key = event.key === " " ? " " : event.key.toLowerCase();
      keysPressedRef.current.delete(key);
      keysPressedRef.current.delete(event.key);
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, []); // Empty dependency array means this runs only once

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

  // Main control loop - runs continuously
  useEffect(() => {
    const controlRobot = () => {
      const currentTime = Date.now();
      if (currentTime - lastExecutionTimeRef.current < DEBOUNCE_INTERVAL) {
        return; // Debounce
      }

      // --- 1. HANDLE GRIPPER ---
      const gripperChangePerSecond = 1 / (FULL_CLOSE_MS / 1000);
      const timeSinceLastFrame =
        (currentTime - lastExecutionTimeRef.current) / 1000;
      const deltaOpen =
        gripperChangePerSecond * selectedSpeed * timeSinceLastFrame;

      if (keysPressedRef.current.has(" ")) {
        // Space is held -> close gripper
        openStateRef.current = Math.max(0, openStateRef.current - deltaOpen);
      } else {
        // Space is not held -> open gripper
        openStateRef.current = Math.min(1, openStateRef.current + deltaOpen);
      }

      // --- 2. HANDLE ARM MOVEMENT (only if isMoving) ---
      let deltaX = 0,
        deltaY = 0,
        deltaZ = 0,
        deltaRZ = 0,
        deltaRX = 0,
        deltaRY = 0;

      if (isMoving) {
        keysPressedRef.current.forEach((key) => {
          if (key !== " " && KEY_MAPPINGS[key]) {
            deltaX += KEY_MAPPINGS[key].x;
            deltaY += KEY_MAPPINGS[key].y;
            deltaZ += KEY_MAPPINGS[key].z;
            deltaRZ += KEY_MAPPINGS[key].rz;
            deltaRX += KEY_MAPPINGS[key].rx;
            deltaRY += KEY_MAPPINGS[key].ry;
          }
        });
        // Apply speed scaling
        deltaX *= selectedSpeed;
        deltaY *= selectedSpeed;
        deltaZ *= selectedSpeed;
        deltaRX *= selectedSpeed;
        deltaRY *= selectedSpeed;
        deltaRZ *= selectedSpeed;
      }

      // --- 3. SEND COMMAND IF NEEDED ---
      const armIsMoving =
        deltaX !== 0 ||
        deltaY !== 0 ||
        deltaZ !== 0 ||
        deltaRZ !== 0 ||
        deltaRX !== 0 ||
        deltaRY !== 0;

      // Use an epsilon to avoid floating point issues
      const gripperStateChanged =
        Math.abs(openStateRef.current - lastSentOpenStateRef.current) > 0.001;

      if (armIsMoving || gripperStateChanged) {
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
        lastSentOpenStateRef.current = openStateRef.current;
      }

      lastExecutionTimeRef.current = currentTime;
    };

    const intervalId = setInterval(controlRobot, LOOP_INTERVAL);
    return () => clearInterval(intervalId); // Cleanup on unmount
  }, [isMoving, selectedSpeed, serverStatus, selectedRobotName]); // Rerun setup if these change

  // UI-controlled hold start (mouse / touch)
  const startSpacePressFromUI = (e?: React.MouseEvent | React.TouchEvent) => {
    e?.preventDefault();
    keysPressedRef.current.add(" ");
    setActiveKey(" ");
  };

  // UI-controlled hold end (mouse up / touch end / leave)
  const endSpacePressFromUI = () => {
    keysPressedRef.current.delete(" ");
    setActiveKey(null);
  };

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
      const initData = { x: 0, y: 0, z: 0, rx: 0, ry: 0, rz: 0, open: 1 };
      await postData(BASE_URL + "move/absolute", initData, {
        robot_id: robotIDFromName(selectedRobotName),
      });
      openStateRef.current = 1;
      lastSentOpenStateRef.current = 1;
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
      description: "Move forward",
      icon: <ArrowUp className="size-6" />,
    },
    {
      key: "ArrowDown",
      description: "Move backward",
      icon: <ArrowDown className="size-6" />,
    },
    {
      key: "ArrowLeft",
      description: "Yaw left",
      icon: <ArrowLeft className="size-6" />,
    },
    {
      key: "ArrowRight",
      description: "Yaw right",
      icon: <ArrowRight className="size-6" />,
    },
    {
      key: "F",
      description: "Move up",
      icon: <ChevronUp className="size-6" />,
    },
    {
      key: "V",
      description: "Move down",
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
      description: "Hold to close gripper, release to open",
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
                    // Clicks on movement keys will simulate a short keypress
                    if (isMoving && control.key !== " ") {
                      const K = KEY_MAPPINGS[control.key.toLowerCase()]
                        ? control.key.toLowerCase()
                        : control.key;
                      keysPressedRef.current.add(K);
                      setActiveKey(control.key);
                      setTimeout(() => {
                        keysPressedRef.current.delete(K);
                        setActiveKey(null);
                      }, 200); // Duration of simulated press
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

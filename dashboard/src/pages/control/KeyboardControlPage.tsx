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
  const [selectedSpeed, setSelectedSpeed] = useState<number>(0.5); // State for speed

  // Refs to manage our control loop and state
  const keysPressedRef = useRef(new Set<string>());
  const intervalIdRef = useRef<NodeJS.Timeout | null>(null);
  const lastExecutionTimeRef = useRef(0);
  const openStateRef = useRef(1);

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
    " ": { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: 0, toggleOpen: true },
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

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.repeat) return;
      setActiveKey(event.key);

      const keyLower = event.key.toLowerCase();
      if (KEY_MAPPINGS[keyLower]) {
        keysPressedRef.current.add(keyLower);
      } else if (KEY_MAPPINGS[event.key]) {
        // For keys like ArrowUp, Space
        keysPressedRef.current.add(event.key);
      }
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      setActiveKey(null);
      const keyLower = event.key.toLowerCase();
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
  }, []);

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
          let didToggleOpen = false;

          keysPressedRef.current.forEach((key) => {
            if (KEY_MAPPINGS[key]) {
              deltaX += KEY_MAPPINGS[key].x;
              deltaY += KEY_MAPPINGS[key].y;
              deltaZ += KEY_MAPPINGS[key].z;
              deltaRZ += KEY_MAPPINGS[key].rz;
              deltaRX += KEY_MAPPINGS[key].rx;
              deltaRY += KEY_MAPPINGS[key].ry;
              if (KEY_MAPPINGS[key].toggleOpen) {
                didToggleOpen = true;
              }
            }
          });

          // Apply speed scaling for mobile robots
          const currentRobot = serverStatus?.robot_status.find(
            (r) => r.device_name === selectedRobotName,
          );
          const isMobile = currentRobot?.robot_type === "mobile";

          if (isMobile) {
            deltaX *= selectedSpeed;
            deltaY *= selectedSpeed; // Assuming Y might be used for strafing
            deltaRZ *= selectedSpeed; // Scale turning speed
            // deltaZ, deltaRX, deltaRY are typically not scaled by mobile base speed
          }

          if (didToggleOpen) {
            openStateRef.current = openStateRef.current > 0.99 ? 0 : 1;
            keysPressedRef.current.delete(" ");
          }

          if (
            deltaX !== 0 ||
            deltaY !== 0 ||
            deltaZ !== 0 ||
            deltaRZ !== 0 ||
            deltaRX !== 0 ||
            deltaRY !== 0 ||
            didToggleOpen
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
      description: "Toggle the open state",
      icon: <Space className="size-6" />,
    },
  ];

  if (serverError) return <div>Failed to load server status.</div>; // Handle error case
  if (!serverStatus) return <LoadingPage />;

  // Determine current robot type for conditional rendering
  const selectedRobot = serverStatus.robot_status.find(
    (robot) => robot.device_name === selectedRobotName,
  );
  const isMobileRobot = selectedRobot?.robot_type === "mobile";

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
            {isMobileRobot && (
              <SpeedSelect
                defaultValue={selectedSpeed} // Current speed state from RobotControllerPage
                onChange={(newSpeed) => setSelectedSpeed(newSpeed)}
                title="Movement speed"
                minSpeed={0.1} // Specific min for mobile robots
                maxSpeed={1.0} // Specific max for mobile robots
                step={0.1} // Specific step for mobile robots
              />
            )}
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
                  onClick={() => {
                    if (!isMoving && control.key !== " ") {
                      // For non-spacebar clicks, simulate a quick key press only if not already moving via keyboard
                      // This is for single-step movements via UI click
                      // This part might need more fleshed out logic if complex interactions are desired
                      console.warn(
                        "UI button clicks for movement are illustrative and might need specific handling if robot is not 'moving' via keyboard.",
                      );
                      // A simple way: send one command
                      const move =
                        KEY_MAPPINGS[control.key.toLowerCase()] ||
                        KEY_MAPPINGS[control.key];
                      if (move) {
                        const data = {
                          x: move.x,
                          y: move.y,
                          z: move.z,
                          rx: move.rx,
                          ry: move.ry,
                          rz: move.rz,
                          open: openStateRef.current,
                        };
                        // Apply speed for mobile robots here too if single step UI clicks should be speed-sensitive
                        if (isMobileRobot) {
                          data.x = selectedSpeed;
                          data.y = selectedSpeed;
                          // rz = 0.6 is a 90° degree rotation
                          data.rz = selectedSpeed * 0.5;
                        }
                        postData(BASE_URL + "move/relative", data, {
                          robot_id: robotIDFromName(selectedRobotName),
                        });
                      }
                      setActiveKey(control.key);
                      setTimeout(() => setActiveKey(null), 200); // Briefly highlight
                      return; // Prevent falling through to old logic for these buttons if not moving
                    }

                    // Original logic for space bar and when isMoving
                    if (control.key === " ") {
                      openStateRef.current = openStateRef.current === 1 ? 0 : 1;
                      const data = {
                        x: 0,
                        y: 0,
                        z: 0,
                        rx: 0,
                        ry: 0,
                        rz: 0,
                        open: openStateRef.current,
                      };
                      // Send command even if not "moving" via keyboard loop
                      postData(BASE_URL + "move/relative", data, {
                        robot_id: robotIDFromName(selectedRobotName),
                      });
                      // No setActiveKey or keysPressedRef manipulation for space, it's a toggle
                    } else if (isMoving) {
                      // Only manipulate keysPressedRef if already in keyboard move mode
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

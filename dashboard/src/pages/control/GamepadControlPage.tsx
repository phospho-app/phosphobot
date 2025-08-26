import { LoadingPage } from "@/components/common/loading";
import { SpeedSelect } from "@/components/common/speed-select";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { fetcher } from "@/lib/utils";
import type { ServerStatus } from "@/types";
import {
  AlertCircle,
  ArrowDownFromLine,
  ArrowUpFromLine,
  Gamepad2,
  Home,
  Minus,
  Play,
  Plus,
  RotateCcw,
  RotateCw,
  Space,
  Square,
  Target,
  Users,
} from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import useSWR from "swr";

import {
  ControlButton,
  GamepadVisualizer,
  TriggerButton,
  useGamepadControl,
  useGamepadDetection,
} from "./GamepadComponents";
import {
  AnalogValues,
  ConfigMode,
  Control,
  ControllerArmPair,
  MultiArmGroup,
  getAvailableControllers,
  getAvailableControllersForGroup,
  getAvailableRobots,
  getAvailableRobotsForGroup,
  initRobot,
  robotIDFromName,
} from "./GamepadUtils";

export function GamepadControl() {
  const { data: serverStatus, error: serverError } = useSWR<ServerStatus>(
    ["/status"],
    fetcher,
    {
      refreshInterval: 5000,
    },
  );

  const [isMoving, setIsMoving] = useState(false);
  const [selectedSpeed, setSelectedSpeed] = useState<number>(0.8);
  const [configMode, setConfigMode] = useState<ConfigMode>("individual");

  // Multi-gamepad and multi-arm support states
  const [controllerArmPairs, setControllerArmPairs] = useState<
    ControllerArmPair[]
  >([{ controller_index: null, robot_name: null, controller_name: "" }]);

  // Multi-arm group states
  const [multiArmGroups, setMultiArmGroups] = useState<MultiArmGroup[]>([]);

  const [hasUserMadeSelection, setHasUserMadeSelection] = useState(false);

  // Active states for visual feedback - one per pair/group
  const [activeButtonsPerPair, setActiveButtonsPerPair] = useState<
    Map<number, Set<string>>
  >(new Map());
  const [analogValuesPerController, setAnalogValuesPerController] = useState<
    Map<number, AnalogValues>
  >(new Map());
  const [_, setAnalogValues] = useState<AnalogValues>({
    leftTrigger: 0,
    rightTrigger: 0,
    leftStickX: 0,
    leftStickY: 0,
    rightStickX: 0,
    rightStickY: 0,
  });

  // Use our custom hooks
  const { gamepadConnected, availableGamepads } = useGamepadDetection(
    configMode,
    controllerArmPairs,
    multiArmGroups,
    hasUserMadeSelection,
    setControllerArmPairs,
    setMultiArmGroups,
    setHasUserMadeSelection,
  );

  const { controlStates } = useGamepadControl(
    isMoving,
    configMode,
    controllerArmPairs,
    multiArmGroups,
    selectedSpeed,
    serverStatus,
    setActiveButtonsPerPair,
    setAnalogValuesPerController,
    setAnalogValues,
    setMultiArmGroups,
  );

  // Auto-select robot when server status loads
  useEffect(() => {
    if (
      !controllerArmPairs[0].robot_name &&
      serverStatus?.robot_status &&
      serverStatus.robot_status.length > 0 &&
      serverStatus.robot_status[0].device_name &&
      configMode === "individual"
    ) {
      setControllerArmPairs((prev) => {
        const newPairs = [...prev];
        newPairs[0] = {
          ...newPairs[0],
          robot_name: serverStatus.robot_status[0].device_name || null,
        };
        return newPairs;
      });
    }
  }, [serverStatus, configMode]);

  // Initialize multi-arm group when switching modes
  useEffect(() => {
    if (
      configMode === "multi-arm" &&
      multiArmGroups.length === 0 &&
      serverStatus?.robot_status
    ) {
      const availableRobots = serverStatus.robot_status
        .filter((robot) => robot.device_name)
        .map((robot) => robot.device_name!);

      if (availableRobots.length >= 2) {
        // Create initial group with first 2-3 robots, but don't assign all robots automatically
        const initialRobotCount = Math.min(
          3,
          Math.ceil(availableRobots.length / 2),
        );
        setMultiArmGroups([
          {
            id: "group-1",
            name: "Multi-Arm Group 1",
            controller_index: null,
            controller_name: "",
            robot_names: availableRobots.slice(0, initialRobotCount),
            control_mode: "synchronized",
            active_robot_index: 0,
          },
        ]);
      }
    }
  }, [configMode, serverStatus, multiArmGroups.length]);

  const startMoving = async () => {
    let robotsToInit: string[] = [];

    if (configMode === "individual") {
      const validPairs = controllerArmPairs.filter(
        (pair) => pair.controller_index !== null && pair.robot_name !== null,
      );
      robotsToInit = validPairs.map((pair) => pair.robot_name!);
    } else {
      const validGroups = multiArmGroups.filter(
        (group) =>
          group.controller_index !== null && group.robot_names.length > 0,
      );
      robotsToInit = validGroups.flatMap((group) => group.robot_names);
    }

    // Initialize all robots
    for (const robotName of robotsToInit) {
      await initRobot(robotName, serverStatus);
    }

    setIsMoving(true);
  };

  const stopMoving = async () => {
    setIsMoving(false);
    controlStates.current.clear();
    setActiveButtonsPerPair(new Map());
  };

  // Configuration mode switching
  const switchConfigMode = (newMode: ConfigMode) => {
    if (isMoving) {
      return;
    }

    setConfigMode(newMode);
    setHasUserMadeSelection(false);
    controlStates.current.clear();
    setActiveButtonsPerPair(new Map());

    if (newMode === "multi-arm" && multiArmGroups.length === 0) {
      // Initialize first multi-arm group
      const availableRobots =
        serverStatus?.robot_status
          ?.filter((robot) => robot.device_name)
          ?.map((robot) => robot.device_name!) || [];

      if (availableRobots.length >= 2) {
        setMultiArmGroups([
          {
            id: "group-1",
            name: "Multi-Arm Group 1",
            controller_index: null,
            controller_name: "",
            robot_names: availableRobots.slice(0, 2),
            control_mode: "synchronized",
            active_robot_index: 0,
          },
        ]);
      }
    }
  };

  // Multi-arm group management functions
  const addMultiArmGroup = () => {
    // Always allow adding groups - user can configure them after creation
    setHasUserMadeSelection(true);

    setMultiArmGroups([
      ...multiArmGroups,
      {
        id: `group-${Date.now()}`,
        name: `Multi-Arm Group ${multiArmGroups.length + 1}`,
        controller_index: null, // Let user select
        controller_name: "",
        robot_names: [], // Let user select robots
        control_mode: "synchronized",
        active_robot_index: 0,
      },
    ]);
  };

  const removeMultiArmGroup = (groupId: string) => {
    if (multiArmGroups.length <= 1) {
      return;
    }

    setMultiArmGroups((prev) => prev.filter((group) => group.id !== groupId));
    controlStates.current.delete(`group-${groupId}`);
  };

  const updateMultiArmGroup = (
    groupId: string,
    updates: Partial<MultiArmGroup>,
  ) => {
    if (updates.controller_index !== undefined) {
      setHasUserMadeSelection(true);
    }

    setMultiArmGroups((prev) =>
      prev.map((group) =>
        group.id === groupId ? { ...group, ...updates } : group,
      ),
    );
  };

  // Individual pair management functions
  const addControllerArmPair = () => {
    const usedControllerIds = new Set<number>();
    const usedRobotNames = new Set<string>();

    controllerArmPairs.forEach((pair) => {
      if (pair.controller_index !== null)
        usedControllerIds.add(pair.controller_index);
      if (pair.robot_name !== null) usedRobotNames.add(pair.robot_name);
    });

    const availableControllers = availableGamepads.filter(
      (gamepad) => !usedControllerIds.has(gamepad.index),
    );
    const availableRobots =
      serverStatus?.robot_status?.filter(
        (robot) => robot.device_name && !usedRobotNames.has(robot.device_name),
      ) || [];

    if (availableControllers.length >= 1 && availableRobots.length >= 1) {
      const newController = availableControllers[0];
      const newRobot = availableRobots[0];

      setHasUserMadeSelection(true);

      setControllerArmPairs([
        ...controllerArmPairs,
        {
          controller_index: newController.index,
          robot_name: newRobot.device_name || null,
          controller_name: newController.name,
        },
      ]);
    } else {
      toast.error("Connect more controllers or robots to create a new pair");
    }
  };

  const removeControllerArmPair = (pairIndex: number) => {
    if (controllerArmPairs.length <= 1) {
      return;
    }

    setControllerArmPairs((prevPairs) =>
      prevPairs.filter((_, index) => index !== pairIndex),
    );

    controlStates.current.delete(`individual-${pairIndex}`);
    setActiveButtonsPerPair((prev) => {
      const newMap = new Map(prev);
      newMap.delete(pairIndex);
      return newMap;
    });
  };

  const updateControllerArmPair = (
    pairIndex: number,
    field: keyof ControllerArmPair,
    value: string | number,
  ) => {
    if (field === "controller_index") {
      setHasUserMadeSelection(true);
    }

    setControllerArmPairs((prevPairs) => {
      const newPairs = [...prevPairs];

      if (field === "controller_index") {
        const selectedController = availableGamepads.find(
          (gamepad) => gamepad.index === parseInt(value.toString()),
        );
        if (selectedController) {
          newPairs[pairIndex] = {
            ...newPairs[pairIndex],
            controller_index: selectedController.index,
            controller_name: selectedController.name,
          };
        }
      } else if (field === "robot_name") {
        newPairs[pairIndex] = {
          ...newPairs[pairIndex],
          robot_name: value.toString(),
        };
      } else if (field === "controller_name") {
        newPairs[pairIndex] = {
          ...newPairs[pairIndex],
          controller_name: value.toString(),
        };
      }

      return newPairs;
    });
  };

  // Check if configuration is valid
  const isConfigValid = () => {
    if (configMode === "individual") {
      return controllerArmPairs.every(
        (pair) => pair.controller_index !== null && pair.robot_name !== null,
      );
    } else {
      return multiArmGroups.every(
        (group) =>
          group.controller_index !== null && group.robot_names.length > 0,
      );
    }
  };

  const hasAvailableGamepads = availableGamepads.length > 0;
  const hasAvailableRobots = (serverStatus?.robot_status?.length || 0) > 0;

  const controls: Control[] = [
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

  return (
    <div className="container mx-auto px-4 py-6 space-y-8">
      <Card>
        <CardHeader className="flex flex-col gap-y-2">
          <div className="flex items-center justify-between">
            <CardDescription>
              Control robot arms with game controllers - individual or multi-arm
              modes
            </CardDescription>
            <div className="flex items-center space-x-4">
              <Label htmlFor="config-mode">Control Mode:</Label>
              <div className="flex items-center space-x-2">
                <Target className="h-4 w-4" />
                <Switch
                  id="config-mode"
                  checked={configMode === "multi-arm"}
                  onCheckedChange={(checked) =>
                    switchConfigMode(checked ? "multi-arm" : "individual")
                  }
                  disabled={isMoving}
                />
                <Users className="h-4 w-4" />
              </div>
              <Badge
                variant={configMode === "individual" ? "secondary" : "default"}
              >
                {configMode === "individual"
                  ? "Individual Control"
                  : "Multi-Arm Control"}
              </Badge>
            </div>
          </div>

          <Accordion type="single" collapsible>
            <AccordionItem value="item-1">
              <AccordionTrigger>How to setup?</AccordionTrigger>
              <AccordionContent>
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium mb-2">General Setup:</h4>
                    <ul className="list-disc pl-5 space-y-1">
                      <li>
                        Connect game controllers to your computer via USB or
                        Bluetooth.
                      </li>
                      <li>
                        Press any button on each controller to activate it.
                      </li>
                      <li>
                        Ensure robot arms are connected via USB and powered on.
                      </li>
                      <li>
                        Calibrate all robot arms by going to /calibration.
                      </li>
                    </ul>
                  </div>

                  <div>
                    <h4 className="font-medium mb-2">
                      Individual Control Mode:
                    </h4>
                    <ul className="list-disc pl-5 space-y-1">
                      <li>
                        Each robot arm is controlled by one dedicated
                        controller.
                      </li>
                      <li>
                        Create multiple robot-controller pairs for independent
                        control.
                      </li>
                    </ul>
                  </div>

                  <div>
                    <h4 className="font-medium mb-2">
                      Multi-Arm Control Mode:
                    </h4>
                    <ul className="list-disc pl-5 space-y-1">
                      <li>
                        <strong>Synchronized:</strong> All arms move identically
                        together.
                      </li>
                      <li>
                        <strong>Sequential:</strong> Use Select/Back button to
                        switch between arms.
                      </li>
                      <li>
                        One controller can control multiple robot arms with
                        different modes.
                      </li>
                    </ul>
                  </div>

                  <div className="bg-green-50 dark:bg-green-950 p-3 rounded-lg">
                    <h4 className="font-medium mb-2">Example Scenarios:</h4>
                    <div className="space-y-2 text-sm">
                      <div>
                        <strong>6 Robots + 2 Controllers (Separate):</strong>
                        <ul className="list-disc pl-5 mt-1">
                          <li>Group 1: Controller 1 → Robots A, B, C</li>
                          <li>Group 2: Controller 2 → Robots D, E, F</li>
                        </ul>
                      </div>
                      <div>
                        <strong>
                          6 Robots + 1 Controller (Multiple Groups):
                        </strong>
                        <ul className="list-disc pl-5 mt-1">
                          <li>
                            Group 1: Controller 1 → Robots A, B, C
                            (Synchronized)
                          </li>
                          <li>
                            Group 2: Controller 1 → Robots D, E, F (Sequential)
                          </li>
                          <li>
                            Use different special buttons or modes to switch
                            between groups
                          </li>
                        </ul>
                      </div>
                    </div>
                  </div>

                  {configMode === "multi-arm" && (
                    <div className="bg-blue-50 dark:bg-blue-950 p-3 rounded-lg">
                      <h4 className="font-medium mb-2">
                        Multi-Arm Special Controls:
                      </h4>
                      <ul className="list-disc pl-5 space-y-1 text-sm">
                        <li>
                          <strong>Select/Back Button:</strong> Switch active
                          robot in Sequential mode
                        </li>
                        <li>
                          <strong>L3/R3 (Stick Click):</strong> Reserved for
                          future mode switching
                        </li>
                      </ul>
                    </div>
                  )}
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </CardHeader>
        <CardContent className="flex flex-col gap-y-6">
          {/* Gamepad Status Section */}
          <div className="flex flex-col items-center space-y-4">
            <div className="flex items-center gap-2">
              <Gamepad2
                className={`size-8 ${gamepadConnected ? "text-green-500" : "text-gray-400"}`}
              />
              <span className="text-lg font-semibold">
                {gamepadConnected
                  ? `${availableGamepads.length} ${availableGamepads.length === 1 ? "Gamepad" : "Gamepads"} Connected`
                  : "No Gamepads Detected"}
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
              </div>
            )}
          </div>

          <div className="flex flex-col md:flex-row justify-center gap-4">
            <SpeedSelect
              defaultValue={selectedSpeed}
              onChange={(newSpeed) => setSelectedSpeed(newSpeed)}
              title="Movement speed"
              minSpeed={0.1}
              maxSpeed={2.0}
              step={0.1}
            />
            <Button
              onClick={startMoving}
              disabled={
                isMoving ||
                !isConfigValid() ||
                !hasAvailableGamepads ||
                !hasAvailableRobots
              }
              variant={isMoving ? "outline" : "default"}
              size="lg"
            >
              {!isMoving && <Play className="mr-2 h-4 w-4" />}
              {isMoving ? "Control Running" : "Start Control"}
            </Button>
            <Button
              onClick={stopMoving}
              disabled={!isMoving}
              variant="destructive"
              size="lg"
            >
              <Square className="mr-2 h-4 w-4" />
              Stop Control
            </Button>
          </div>

          {/* Alert Messages */}
          {!hasAvailableGamepads && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>No controllers detected</AlertTitle>
              <AlertDescription>
                Please connect a game controller and press any button to
                activate it.
              </AlertDescription>
            </Alert>
          )}

          {!hasAvailableRobots && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>No robots detected</AlertTitle>
              <AlertDescription>
                Please make sure your robots are connected and powered on.
              </AlertDescription>
            </Alert>
          )}

          {!isConfigValid() && hasAvailableGamepads && hasAvailableRobots && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Incomplete configuration</AlertTitle>
              <AlertDescription>
                Please complete the configuration below to start control.
              </AlertDescription>
            </Alert>
          )}

          {/* Configuration Section */}
          {configMode === "individual" ? (
            <div>
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-medium">Robot-Controller Pairs</h3>
                <div className="flex gap-2">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={addControllerArmPair}
                          disabled={
                            isMoving ||
                            availableGamepads.length <=
                              controllerArmPairs.length ||
                            (serverStatus?.robot_status?.length || 0) <=
                              controllerArmPairs.length
                          }
                        >
                          <Plus className="h-4 w-4" />
                          <span className="sr-only">
                            Add robot-controller pair
                          </span>
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Add a new robot-controller pair</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
              </div>

              {controllerArmPairs.map((pair, index) => (
                <Card key={`controller-arm-pair-${index}`} className="p-4 mb-4">
                  <div className="flex justify-between items-center mb-4">
                    <h4 className="font-medium">Pair {index + 1}</h4>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => removeControllerArmPair(index)}
                            disabled={
                              isMoving || controllerArmPairs.length <= 1
                            }
                          >
                            <Minus className="h-4 w-4" />
                            <span className="sr-only">
                              Remove robot-controller pair
                            </span>
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p>Remove this robot-controller pair</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>

                  <div className="flex flex-col md:flex-row items-center gap-4">
                    <div className="space-y-2 w-full">
                      <Label htmlFor={`robot-arm-${index}`}>Robot Arm</Label>
                      <Select
                        value={pair.robot_name || ""}
                        onValueChange={(value) => {
                          updateControllerArmPair(index, "robot_name", value);
                        }}
                        disabled={isMoving}
                      >
                        <SelectTrigger id={`robot-arm-${index}`}>
                          <SelectValue placeholder="Select robot arm" />
                        </SelectTrigger>
                        <SelectContent>
                          {getAvailableRobots(
                            index,
                            controllerArmPairs,
                            serverStatus,
                          ).map((robot, key) => (
                            <SelectItem
                              key={`select-robot-${index}-${key}`}
                              value={robot.device_name || "Undefined port"}
                            >
                              {robot.name} ({robot.device_name})
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2 w-full">
                      <Label htmlFor={`controller-${index}`}>
                        Game Controller
                      </Label>
                      <Select
                        value={pair.controller_index?.toString() || ""}
                        onValueChange={(value) => {
                          updateControllerArmPair(
                            index,
                            "controller_index",
                            parseInt(value),
                          );
                        }}
                        disabled={isMoving}
                      >
                        <SelectTrigger id={`controller-${index}`}>
                          <SelectValue placeholder="Select controller" />
                        </SelectTrigger>
                        <SelectContent>
                          {getAvailableControllers(
                            index,
                            controllerArmPairs,
                            availableGamepads,
                          ).map((controller) => (
                            <SelectItem
                              key={`select-controller-${index}-${controller.index}`}
                              value={controller.index.toString()}
                            >
                              Controller {controller.index + 1}:{" "}
                              {controller.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          ) : (
            <div>
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-medium">Multi-Arm Groups</h3>
                <div className="flex gap-2">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={addMultiArmGroup}
                          disabled={
                            isMoving ||
                            (serverStatus?.robot_status?.length || 0) ===
                              multiArmGroups.reduce(
                                (total, group) =>
                                  total + group.robot_names.length,
                                0,
                              )
                          }
                        >
                          <Plus className="h-4 w-4" />
                          <span className="sr-only">Add multi-arm group</span>
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Add a new multi-arm group</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
              </div>

              {multiArmGroups.map((group) => (
                <Card key={`multi-arm-group-${group.id}`} className="p-4 mb-4">
                  <div className="flex justify-between items-center mb-4">
                    <h4 className="font-medium">{group.name}</h4>
                    <div className="flex items-center gap-2">
                      {group.control_mode === "sequential" && (
                        <Badge variant="outline" className="text-xs">
                          Active:{" "}
                          {group.robot_names[group.active_robot_index] ||
                            "None"}
                        </Badge>
                      )}
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => removeMultiArmGroup(group.id)}
                              disabled={isMoving || multiArmGroups.length <= 1}
                            >
                              <Minus className="h-4 w-4" />
                              <span className="sr-only">
                                Remove multi-arm group
                              </span>
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>Remove this multi-arm group</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="flex flex-col md:flex-row gap-4">
                      <div className="space-y-2 flex-1">
                        <Label htmlFor={`group-controller-${group.id}`}>
                          Game Controller
                        </Label>
                        <Select
                          value={group.controller_index?.toString() || ""}
                          onValueChange={(value) => {
                            const selectedController = availableGamepads.find(
                              (gamepad) => gamepad.index === parseInt(value),
                            );
                            if (selectedController) {
                              updateMultiArmGroup(group.id, {
                                controller_index: selectedController.index,
                                controller_name: selectedController.name,
                              });
                            }
                          }}
                          disabled={isMoving}
                        >
                          <SelectTrigger id={`group-controller-${group.id}`}>
                            <SelectValue placeholder="Select controller" />
                          </SelectTrigger>
                          <SelectContent>
                            {getAvailableControllersForGroup(
                              availableGamepads,
                            ).map((controller) => (
                              <SelectItem
                                key={`select-group-controller-${group.id}-${controller.index}`}
                                value={controller.index.toString()}
                              >
                                Controller {controller.index + 1}:{" "}
                                {controller.name}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2 flex-1">
                        <Label htmlFor={`group-mode-${group.id}`}>
                          Control Mode
                        </Label>
                        <Select
                          value={group.control_mode}
                          onValueChange={(
                            value: "synchronized" | "sequential",
                          ) => {
                            updateMultiArmGroup(group.id, {
                              control_mode: value,
                            });
                          }}
                          disabled={isMoving}
                        >
                          <SelectTrigger id={`group-mode-${group.id}`}>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="synchronized">
                              <div className="flex items-center gap-2">
                                <Users className="h-4 w-4" />
                                Synchronized - All move together
                              </div>
                            </SelectItem>
                            <SelectItem value="sequential">
                              <div className="flex items-center gap-2">
                                <Target className="h-4 w-4" />
                                Sequential - Switch with Select button
                              </div>
                            </SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <Label>
                          Robot Arms ({group.robot_names.length} selected)
                        </Label>
                        <div className="flex gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => {
                              // Quick select all available robots
                              const availableRobots =
                                getAvailableRobotsForGroup(
                                  group.id,
                                  multiArmGroups,
                                  serverStatus,
                                );
                              const allAvailable = [
                                ...group.robot_names,
                                ...availableRobots.map((r) => r.device_name!),
                              ];
                              updateMultiArmGroup(group.id, {
                                robot_names: allAvailable,
                              });
                            }}
                            disabled={
                              isMoving ||
                              getAvailableRobotsForGroup(
                                group.id,
                                multiArmGroups,
                                serverStatus,
                              ).length === 0
                            }
                            className="text-xs"
                          >
                            Select All Available
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => {
                              updateMultiArmGroup(group.id, {
                                robot_names: [],
                              });
                            }}
                            disabled={isMoving}
                            className="text-xs"
                          >
                            Clear All
                          </Button>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-2 max-h-32 overflow-y-auto">
                        {serverStatus?.robot_status?.map((robot) => {
                          const isUsedByOtherGroup = multiArmGroups.some(
                            (g) =>
                              g.id !== group.id &&
                              g.robot_names.includes(robot.device_name || ""),
                          );
                          return (
                            <div
                              key={robot.device_name}
                              className="flex items-center space-x-2"
                            >
                              <input
                                type="checkbox"
                                id={`robot-${group.id}-${robot.device_name}`}
                                checked={group.robot_names.includes(
                                  robot.device_name || "",
                                )}
                                onChange={(e) => {
                                  const robotName = robot.device_name || "";
                                  if (e.target.checked) {
                                    updateMultiArmGroup(group.id, {
                                      robot_names: [
                                        ...group.robot_names,
                                        robotName,
                                      ],
                                    });
                                  } else {
                                    updateMultiArmGroup(group.id, {
                                      robot_names: group.robot_names.filter(
                                        (name) => name !== robotName,
                                      ),
                                    });
                                  }
                                }}
                                disabled={isMoving || isUsedByOtherGroup}
                                className="rounded"
                              />
                              <Label
                                htmlFor={`robot-${group.id}-${robot.device_name}`}
                                className={`text-sm font-normal cursor-pointer ${
                                  isUsedByOtherGroup
                                    ? "text-muted-foreground line-through"
                                    : ""
                                }`}
                              >
                                {robot.name} ({robot.device_name})
                                {isUsedByOtherGroup && (
                                  <span className="text-xs text-orange-600 ml-1">
                                    (Used)
                                  </span>
                                )}
                              </Label>
                            </div>
                          );
                        })}
                      </div>
                      {group.robot_names.length === 0 && (
                        <p className="text-sm text-red-600 dark:text-red-400">
                          Select at least 1 robot arm for this group
                        </p>
                      )}
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-x-2">
            <Gamepad2 className="size-4" />
            Gamepad Controls Reference
            {configMode === "multi-arm" && (
              <Badge variant="secondary" className="ml-2">
                Multi-Arm Mode
              </Badge>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4 mb-6">
            <div>
              <h4 className="font-medium mb-2">Movement & Rotation</h4>
              <ul className="text-sm space-y-1 text-muted-foreground">
                <li>
                  • <span className="font-medium">Left Stick</span>: Rotate (X)
                  / Move up-down (Y)
                </li>
                <li>
                  • <span className="font-medium">Right Stick</span>: Strafe
                  left-right (X) / Move forward-back (Y)
                </li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-2">Gripper Control</h4>
              <ul className="text-sm space-y-1 text-muted-foreground">
                <li>
                  • <span className="font-medium">L1/R1 (Bumpers)</span>: Toggle
                  open/close
                </li>
                <li>
                  • <span className="font-medium">L2/R2 (Triggers)</span>:
                  Analog control (0-100%)
                </li>
              </ul>
            </div>
          </div>

          <div className="mb-4">
            <h4 className="font-medium mb-2">Wrist Control</h4>
            <p className="text-sm text-muted-foreground mb-2">
              Use either D-Pad or face buttons (ABXY) for wrist movements:
            </p>
            <div className="grid grid-cols-2 gap-2 text-sm text-muted-foreground">
              <div>
                • <span className="font-medium">Up (D-Pad/Y)</span>: Pitch up
              </div>
              <div>
                • <span className="font-medium">Down (D-Pad/A)</span>: Pitch
                down
              </div>
              <div>
                • <span className="font-medium">Left (D-Pad/X)</span>: Roll
                counter-clockwise
              </div>
              <div>
                • <span className="font-medium">Right (D-Pad/B)</span>: Roll
                clockwise
              </div>
            </div>
          </div>

          <div className="mb-6">
            <h4 className="font-medium mb-2">Special Functions</h4>
            <ul className="text-sm space-y-1 text-muted-foreground">
              <li>
                • <span className="font-medium">Start/Menu</span>: Move arm to
                sleep position
              </li>
              {configMode === "multi-arm" && (
                <li>
                  • <span className="font-medium">Select/Back</span>: Switch
                  active robot (Sequential mode)
                </li>
              )}
            </ul>
          </div>

          {configMode === "multi-arm" && (
            <div className="mb-6 p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
              <h4 className="font-medium mb-2 flex items-center gap-2">
                <Users className="h-4 w-4" />
                Multi-Arm Control Modes
              </h4>
              <div className="space-y-2 text-sm text-muted-foreground">
                <div>
                  • <span className="font-medium">Synchronized</span>: All
                  selected arms move identically
                </div>
                <div>
                  • <span className="font-medium">Sequential</span>: Control one
                  arm at a time, switch with Select button
                </div>
              </div>
            </div>
          )}

          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {controls.map((control) => {
              // Calculate combined analog values from all active controllers
              const getCombinedAnalogValue = (controlKey: string): number => {
                let maxValue = 0;

                analogValuesPerController.forEach((values) => {
                  let currentValue = 0;

                  switch (controlKey) {
                    case "move-forward":
                      currentValue =
                        values.rightStickY < 0 ? -values.rightStickY : 0;
                      break;
                    case "move-backward":
                      currentValue =
                        values.rightStickY > 0 ? values.rightStickY : 0;
                      break;
                    case "move-left":
                      currentValue =
                        values.rightStickX < 0 ? -values.rightStickX : 0;
                      break;
                    case "move-right":
                      currentValue =
                        values.rightStickX > 0 ? values.rightStickX : 0;
                      break;
                    case "move-up":
                      currentValue =
                        values.leftStickY < 0 ? -values.leftStickY : 0;
                      break;
                    case "move-down":
                      currentValue =
                        values.leftStickY > 0 ? values.leftStickY : 0;
                      break;
                    case "rotate-left":
                      currentValue =
                        values.leftStickX < 0 ? -values.leftStickX : 0;
                      break;
                    case "rotate-right":
                      currentValue =
                        values.leftStickX > 0 ? values.leftStickX : 0;
                      break;
                  }

                  maxValue = Math.max(maxValue, currentValue);
                });

                return maxValue;
              };

              // Calculate combined trigger values
              const getCombinedTriggerValue = (): number => {
                let maxValue = 0;

                analogValuesPerController.forEach((values) => {
                  const triggerValue = Math.max(
                    values.leftTrigger,
                    values.rightTrigger,
                  );
                  maxValue = Math.max(maxValue, triggerValue);
                });

                return maxValue;
              };

              return (
                <TooltipProvider key={control.key}>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      {control.type === "trigger" ? (
                        <TriggerButton
                          label={control.label}
                          buttons={control.buttons}
                          value={getCombinedTriggerValue()}
                          icon={control.icon}
                        />
                      ) : (
                        <ControlButton
                          control={control}
                          isActive={Array.from(
                            activeButtonsPerPair.values(),
                          ).some((buttonSet) => buttonSet.has(control.key))}
                          analogValue={
                            control.type.startsWith("analog")
                              ? getCombinedAnalogValue(control.key)
                              : undefined
                          }
                        />
                      )}
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>{control.description}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Show gamepad visualizers for active configurations when moving */}
      {isMoving && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-x-2">
              <Gamepad2 className="size-4" />
              Active Controller States
            </CardTitle>
            <CardDescription>
              Real-time controller input visualization for active configurations
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Accordion type="multiple" className="w-full">
              {configMode === "individual"
                ? controllerArmPairs
                    .filter(
                      (pair) =>
                        pair.controller_index !== null &&
                        pair.robot_name !== null,
                    )
                    .map((pair, index) => (
                      <AccordionItem
                        key={`visualizer-individual-${pair.controller_index}`}
                        value={`individual-${index}`}
                      >
                        <AccordionTrigger className="hover:no-underline">
                          <div className="flex items-center gap-3">
                            <div className="flex items-center gap-2">
                              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                              <span className="font-medium">
                                {pair.robot_name} (ID:{" "}
                                {robotIDFromName(pair.robot_name, serverStatus)}
                                )
                              </span>
                            </div>
                            <div className="text-sm text-muted-foreground">
                              → Controller {(pair.controller_index ?? 0) + 1}
                            </div>
                          </div>
                        </AccordionTrigger>
                        <AccordionContent>
                          <GamepadVisualizer
                            gamepadIndex={pair.controller_index}
                          />
                        </AccordionContent>
                      </AccordionItem>
                    ))
                : multiArmGroups
                    .filter(
                      (group) =>
                        group.controller_index !== null &&
                        group.robot_names.length > 0,
                    )
                    .map((group) => (
                      <AccordionItem
                        key={`visualizer-group-${group.id}`}
                        value={`group-${group.id}`}
                      >
                        <AccordionTrigger className="hover:no-underline">
                          <div className="flex items-center gap-3">
                            <div className="flex items-center gap-2">
                              <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                              <span className="font-medium">{group.name}</span>
                            </div>
                            <div className="flex items-center gap-2 text-sm text-muted-foreground">
                              <span>
                                → Controller {(group.controller_index ?? 0) + 1}
                              </span>
                              <Badge variant="secondary" className="text-xs">
                                {group.control_mode}
                              </Badge>
                              {group.control_mode === "sequential" && (
                                <Badge variant="outline" className="text-xs">
                                  Active:{" "}
                                  {group.robot_names[group.active_robot_index]}
                                </Badge>
                              )}
                            </div>
                          </div>
                        </AccordionTrigger>
                        <AccordionContent>
                          <div className="space-y-3">
                            <div className="grid grid-cols-2 gap-4 text-sm">
                              <div>
                                <span className="font-medium">
                                  Control Mode:
                                </span>
                                <span className="ml-2 capitalize">
                                  {group.control_mode}
                                </span>
                              </div>
                              <div>
                                <span className="font-medium">
                                  Robot Count:
                                </span>
                                <span className="ml-2">
                                  {group.robot_names.length}
                                </span>
                              </div>
                            </div>

                            {group.control_mode === "sequential" && (
                              <div className="p-2 bg-blue-50 dark:bg-blue-950 rounded text-sm">
                                <span className="font-medium">
                                  Currently Active:
                                </span>
                                <span className="ml-2">
                                  {group.robot_names[group.active_robot_index]}
                                </span>
                                <span className="ml-2 text-muted-foreground">
                                  (Press Select/Back to switch)
                                </span>
                              </div>
                            )}

                            <div>
                              <span className="font-medium text-sm">
                                Controlled Robots:
                              </span>
                              <div className="mt-1 flex flex-wrap gap-2">
                                {group.robot_names.map(
                                  (robotName, robotIndex) => (
                                    <Badge
                                      key={robotName}
                                      variant={
                                        group.control_mode === "sequential" &&
                                        robotIndex === group.active_robot_index
                                          ? "default"
                                          : "secondary"
                                      }
                                      className="text-xs"
                                    >
                                      {robotName} (ID:{" "}
                                      {robotIDFromName(robotName, serverStatus)}
                                      )
                                    </Badge>
                                  ),
                                )}
                              </div>
                            </div>

                            <GamepadVisualizer
                              gamepadIndex={group.controller_index}
                            />
                          </div>
                        </AccordionContent>
                      </AccordionItem>
                    ))}
            </Accordion>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

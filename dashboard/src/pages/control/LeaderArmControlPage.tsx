import { LoadingPage } from "@/components/common/loading";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
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
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useGlobalStore } from "@/lib/hooks";
import { fetchWithBaseUrl, fetcher } from "@/lib/utils";
import type { ServerStatus } from "@/types";
import {
  AlertCircle,
  ArrowRightLeft,
  Minus,
  Play,
  Plus,
  Settings,
  Square,
} from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import useSWR from "swr";

// Define a type for robot pair
interface RobotPair {
  leader_id: number | null;
  follower_id: number | null;
  leader_serial_id: string;
}

export function LeaderArmControl() {
  // const leaderArmSerialIds = useGlobalStore(
  //   (state) => state.leaderArmSerialIds,
  // );
  const setLeaderArmSerialIds = useGlobalStore(
    (state) => state.setLeaderArmSerialIds,
  );

  const [invertControls, setInvertControls] = useState(false);
  const [enableGravityControl, setEnableGravityControl] = useState(false);

  // State for multiple robot pairs
  const [robotPairs, setRobotPairs] = useState<RobotPair[]>([
    { leader_id: null, follower_id: null, leader_serial_id: "" },
  ]);

  const [gravityCompensationValues, setGravityCompensationValues] = useState({
    shoulder: 100,
    elbow: 100,
    wrist: 100,
  });

  const {
    data: serverStatus,
    error: serverError,
    mutate,
  } = useSWR<ServerStatus>(["/status"], fetcher, {
    refreshInterval: 5000,
    revalidateOnFocus: true,
  });

  // Set connected robots from server status
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const connectedRobots = serverStatus?.robot_status || [];

  // Set default leader and follower when robots are connected
  useEffect(() => {
    if (
      connectedRobots.length >= 2 &&
      robotPairs[0].leader_id === null &&
      robotPairs[0].follower_id === null
    ) {
      // Initialize with first pair
      const initialPairs: RobotPair[] = [
        {
          leader_id: 0,
          follower_id: 1,
          leader_serial_id: connectedRobots[0].device_name || "",
        },
      ];

      setRobotPairs(initialPairs);

      // Update global state with leader serial IDs
      const leaderSerialIds = initialPairs
        .map((pair) => pair.leader_serial_id)
        .filter((id) => id !== "");

      setLeaderArmSerialIds(leaderSerialIds);
    } else if (connectedRobots.length === 1) {
      // Only one robot connected - can't have a pair
      setRobotPairs([
        { leader_id: null, follower_id: 0, leader_serial_id: "" },
      ]);
      setLeaderArmSerialIds([]);
    } else if (connectedRobots.length === 0) {
      // No robots connected
      setRobotPairs([
        { leader_id: null, follower_id: null, leader_serial_id: "" },
      ]);
      setLeaderArmSerialIds([]);
    }
  }, [connectedRobots, connectedRobots.length, setLeaderArmSerialIds]);

  // Update global state when robot pairs change
  useEffect(() => {
    const serialIds = robotPairs
      .map((pair) => pair.leader_serial_id)
      .filter((id) => id !== "");

    setLeaderArmSerialIds(serialIds);
  }, [robotPairs, setLeaderArmSerialIds]);

  const handleMoveStart = async () => {
    // Validate all pairs have both leader and follower
    const invalidPairs = robotPairs.filter(
      (pair) => pair.leader_id === null || pair.follower_id === null,
    );

    if (invalidPairs.length > 0) {
      toast.error(
        "Please select both leader and follower robots for all pairs",
      );
      return;
    }

    fetchWithBaseUrl(`/move/leader/start`, "POST", {
      robot_pairs: robotPairs,
      invert_controls: invertControls,
      enable_gravity_compensation: enableGravityControl,
      gravity_compensation_values: !enableGravityControl
        ? gravityCompensationValues
        : null,
    }).then(() => {
      mutate();
    });
  };

  const handleMoveStop = async () => {
    fetchWithBaseUrl(`/move/leader/stop`, "POST").then(() => {
      mutate();
    });
  };

  const handleSwapRobots = (pairIndex: number) => {
    setRobotPairs((prevPairs) => {
      const newPairs = [...prevPairs];
      const pair = newPairs[pairIndex];

      // Swap leader and follower
      const tempLeader = pair.leader_id;
      const newLeaderSerialId =
        pair.follower_id !== null
          ? connectedRobots[pair.follower_id]?.device_name || ""
          : "";

      newPairs[pairIndex] = {
        leader_id: pair.follower_id,
        follower_id: tempLeader,
        leader_serial_id: newLeaderSerialId,
      };

      return newPairs;
    });
  };

  const addRobotPair = () => {
    // Find available robots that aren't already used
    const usedRobotIds = new Set<number>();

    robotPairs.forEach((pair) => {
      if (pair.leader_id !== null) usedRobotIds.add(pair.leader_id);
      if (pair.follower_id !== null) usedRobotIds.add(pair.follower_id);
    });

    const availableRobots = connectedRobots
      .map((_, index) => index)
      .filter((id) => !usedRobotIds.has(id));

    // Add a new pair with default values if possible
    if (availableRobots.length >= 2) {
      const newLeaderId = availableRobots[0];
      const newFollowerId = availableRobots[1];

      setRobotPairs([
        ...robotPairs,
        {
          leader_id: newLeaderId,
          follower_id: newFollowerId,
          leader_serial_id: connectedRobots[newLeaderId].device_name || "",
        },
      ]);
    } else if (availableRobots.length === 1) {
      // Only one robot available
      setRobotPairs([
        ...robotPairs,
        {
          leader_id: availableRobots[0],
          follower_id: null,
          leader_serial_id:
            connectedRobots[availableRobots[0]].device_name || "",
        },
      ]);
    } else {
      // No robots available
      toast.error("Connect more robots to create a new pair");
    }
  };

  const removeRobotPair = (pairIndex: number) => {
    if (robotPairs.length <= 1) {
      toast.error("At least one robot pair is required");
      return;
    }

    setRobotPairs((prevPairs) =>
      prevPairs.filter((_, index) => index !== pairIndex),
    );
  };

  const updateRobotPair = (
    pairIndex: number,
    field: keyof RobotPair,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    value: any,
  ) => {
    setRobotPairs((prevPairs) => {
      const newPairs = [...prevPairs];

      if (field === "leader_serial_id") {
        // When updating leader serial ID, also update the leader ID
        const selectedRobot = connectedRobots.find(
          (robot) => robot.device_name === value,
        );

        if (selectedRobot) {
          const robotIndex = connectedRobots.indexOf(selectedRobot);
          newPairs[pairIndex] = {
            ...newPairs[pairIndex],
            leader_id: robotIndex,
            leader_serial_id: value,
          };
        }
      } else if (field === "leader_id") {
        // When updating leader ID, also update the serial ID
        const serialId =
          value !== null ? connectedRobots[value]?.device_name || "" : "";
        newPairs[pairIndex] = {
          ...newPairs[pairIndex],
          [field]: value,
          leader_serial_id: serialId,
        };
      } else {
        // Just update the field
        newPairs[pairIndex] = {
          ...newPairs[pairIndex],
          [field]: value,
        };
      }

      return newPairs;
    });
  };

  // Get available follower robots (excluding leaders from all pairs)
  const getAvailableFollowerRobots = (currentPairIndex: number) => {
    const usedLeaderIds = new Set<number>();

    robotPairs.forEach((pair, index) => {
      // Don't exclude the current pair's follower
      if (index !== currentPairIndex && pair.leader_id !== null) {
        usedLeaderIds.add(pair.leader_id);
      }
    });

    return connectedRobots.filter((_, index) => !usedLeaderIds.has(index));
  };

  // Get available leader robots (excluding followers from all pairs)
  const getAvailableLeaderRobots = (currentPairIndex: number) => {
    const usedFollowerIds = new Set<number>();

    robotPairs.forEach((pair, index) => {
      // Don't exclude the current pair's leader
      if (index !== currentPairIndex && pair.follower_id !== null) {
        usedFollowerIds.add(pair.follower_id);
      }
    });

    return connectedRobots.filter((_, index) => !usedFollowerIds.has(index));
  };

  // Loading state
  if (!serverStatus && !serverError) {
    return <LoadingPage />;
  }

  // Check if configuration is valid
  const isConfigValid = robotPairs.every(
    (pair) => pair.leader_id !== null && pair.follower_id !== null,
  );

  return (
    <div className="container mx-auto px-4 py-6 space-y-8">
      <Card>
        <CardHeader className="flex flex-col gap-y-2">
          <CardDescription>
            Control robot arms (followers) using other arms as leaders
          </CardDescription>
          <Accordion type="single" collapsible>
            <AccordionItem value="item-1">
              <AccordionTrigger>How to setup?</AccordionTrigger>
              <AccordionContent>
                <ul className="list-disc pl-5 space-y-1 mt-2">
                  <li>The follower arm should be a regular so-100.</li>
                  <li>
                    if you want to use gravity compensation, the leader arm
                    should be a regular so-100 with gears
                  </li>
                  <li>
                    Both arms need to be connected via USB and powered on.
                  </li>
                  <li>
                    Calibrate both follower and leader arms by going to
                    /calibration.
                  </li>
                </ul>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </CardHeader>
        <CardContent className="flex flex-col gap-y-6">
          <div className="flex flex-col md:flex-row justify-center gap-4">
            <Button
              onClick={handleMoveStart}
              disabled={
                serverStatus?.leader_follower_status ||
                !isConfigValid ||
                connectedRobots.length < 2
              }
              variant={
                serverStatus?.leader_follower_status ? "outline" : "default"
              }
              size="lg"
            >
              {!serverStatus?.leader_follower_status && (
                <Play className="mr-2 h-4 w-4" />
              )}
              {serverStatus?.leader_follower_status
                ? "Control Running"
                : "Start Control"}
            </Button>
            <Button
              onClick={handleMoveStop}
              disabled={!serverStatus?.leader_follower_status}
              variant="destructive"
              size="lg"
            >
              <Square className="mr-2 h-4 w-4" />
              Stop Control
            </Button>
          </div>

          {/* Robot Pairs Management */}
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-medium">Robot Arm Pairs</h3>
            <div className="flex gap-2">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={addRobotPair}
                      disabled={
                        serverStatus?.leader_follower_status ||
                        connectedRobots.length <= robotPairs.length * 2 - 1
                      }
                    >
                      <Plus className="h-4 w-4" />
                      <span className="sr-only">Add robot pair</span>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Add a new leader-follower robot pair</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          </div>

          {connectedRobots.length === 0 && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>No robots detected</AlertTitle>
              <AlertDescription>
                Please make sure your robots are connected and powered on.
              </AlertDescription>
            </Alert>
          )}

          {!isConfigValid && connectedRobots.length > 0 && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Incomplete configuration</AlertTitle>
              <AlertDescription>
                Please select both leader and follower robots for all pairs to
                start control.
              </AlertDescription>
            </Alert>
          )}

          {robotPairs.map((pair, index) => (
            <Card key={`robot-pair-${index}`} className="p-4">
              <div className="flex justify-between items-center mb-4">
                <h4 className="font-medium">Pair {index + 1}</h4>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => removeRobotPair(index)}
                        disabled={
                          serverStatus?.leader_follower_status ||
                          robotPairs.length <= 1
                        }
                      >
                        <Minus className="h-4 w-4" />
                        <span className="sr-only">Remove robot pair</span>
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Remove this robot pair</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>

              <div className="flex flex-col md:flex-row items-center gap-4">
                {/* Leader Robot Selection */}
                <div className="space-y-2 w-full">
                  <Label htmlFor={`leader-robot-${index}`}>
                    Leader Robot (controlling arm)
                  </Label>
                  <Select
                    value={pair.leader_serial_id || ""}
                    onValueChange={(value) => {
                      updateRobotPair(index, "leader_serial_id", value);
                    }}
                    disabled={serverStatus?.leader_follower_status}
                  >
                    <SelectTrigger id={`leader-robot-${index}`}>
                      <SelectValue placeholder="Select leader robot" />
                    </SelectTrigger>
                    <SelectContent>
                      {getAvailableLeaderRobots(index).map((robot, key) => (
                        <SelectItem
                          key={`select-leader-${index}-${key}`}
                          value={robot.device_name || "Undefined port"}
                        >
                          {robot.name} ({robot.device_name})
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {pair.leader_id !== null && pair.follower_id !== null && (
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant="secondary"
                          onClick={() => handleSwapRobots(index)}
                          className="mt-2 max-w-[4rem]"
                          disabled={serverStatus?.leader_follower_status}
                        >
                          <ArrowRightLeft className="h-4 w-4" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">
                          Swap the leader and follower robots
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                )}

                {/* Follower Robot Selection */}
                <div className="space-y-2 w-full">
                  <Label htmlFor={`follower-robot-${index}`}>
                    Follower Robot (will be controlled)
                  </Label>
                  <Select
                    value={
                      pair.follower_id !== null
                        ? connectedRobots[pair.follower_id]?.device_name || ""
                        : ""
                    }
                    onValueChange={(value) => {
                      // Find the robot index by usb_port
                      const selectedRobot = connectedRobots.find(
                        (robot) => robot.device_name === value,
                      );
                      if (selectedRobot) {
                        updateRobotPair(
                          index,
                          "follower_id",
                          connectedRobots.indexOf(selectedRobot),
                        );
                      }
                    }}
                    disabled={serverStatus?.leader_follower_status}
                  >
                    <SelectTrigger id={`follower-robot-${index}`}>
                      <SelectValue placeholder="Select follower robot" />
                    </SelectTrigger>
                    <SelectContent>
                      {getAvailableFollowerRobots(index).map((robot, key) => (
                        <SelectItem
                          key={`select-follower-${index}-${key}`}
                          value={robot.device_name || "Undefined port"}
                        >
                          {robot.name} ({robot.device_name})
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </Card>
          ))}
        </CardContent>
      </Card>

      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-x-2">
            <Settings className="size-4" />
            Leader arm control settings
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="mb-6">
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <div className="flex items-center space-x-3">
                        <Switch
                          id="invert-controls"
                          checked={invertControls}
                          onCheckedChange={setInvertControls}
                          disabled={serverStatus?.leader_follower_status}
                        />
                        <Label className="text-sm font-medium">
                          Mirror Controls
                        </Label>
                      </div>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="max-w-xs">
                        Mirror movements between arms (e.g., leader arm moving
                        left causes follower arm to move right)
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>

                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <div className="flex items-center space-x-3">
                        <Switch
                          id="toggle-gravity-control"
                          checked={enableGravityControl}
                          onCheckedChange={setEnableGravityControl}
                          disabled={serverStatus?.leader_follower_status}
                        />
                        <Label className="text-sm font-medium">
                          Gravity control
                        </Label>
                      </div>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="max-w-xs">
                        When enabled, the leader arm compensates for gravity to
                        make movements smoother. To use this feature, make sure
                        to use 2 regular so100 follower arms with gears.
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>

              {enableGravityControl && (
                <Card className="mb-6">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">
                      Gravity Compensation Settings
                    </CardTitle>
                    <CardDescription>
                      Adjust gravity compensation values for each joint
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <Label htmlFor="shoulder-gravity">
                            Shoulder Joint
                          </Label>
                          <span className="text-xs">
                            {gravityCompensationValues.shoulder}%
                          </span>
                        </div>
                        <Slider
                          id="shoulder-gravity"
                          min={0}
                          max={200}
                          step={1}
                          value={[gravityCompensationValues.shoulder]}
                          onValueChange={(values) =>
                            setGravityCompensationValues({
                              ...gravityCompensationValues,
                              shoulder: values[0],
                            })
                          }
                          disabled={serverStatus?.leader_follower_status}
                        />
                      </div>

                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <Label htmlFor="elbow-gravity">Elbow Joint</Label>
                          <span className="text-xs">
                            {gravityCompensationValues.elbow}%
                          </span>
                        </div>
                        <Slider
                          id="elbow-gravity"
                          min={0}
                          max={200}
                          step={1}
                          value={[gravityCompensationValues.elbow]}
                          onValueChange={(values) =>
                            setGravityCompensationValues({
                              ...gravityCompensationValues,
                              elbow: values[0],
                            })
                          }
                          disabled={serverStatus?.leader_follower_status}
                        />
                      </div>

                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <Label htmlFor="wrist-gravity">Wrist Joint</Label>
                          <span className="text-xs">
                            {gravityCompensationValues.wrist}%
                          </span>
                        </div>
                        <Slider
                          id="wrist-gravity"
                          min={0}
                          max={200}
                          step={1}
                          value={[gravityCompensationValues.wrist]}
                          onValueChange={(values) =>
                            setGravityCompensationValues({
                              ...gravityCompensationValues,
                              wrist: values[0],
                            })
                          }
                          disabled={serverStatus?.leader_follower_status}
                        />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

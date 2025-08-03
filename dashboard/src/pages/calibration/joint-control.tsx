import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ChartContainer } from "@/components/ui/chart";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { fetchWithBaseUrl, fetcher } from "@/lib/utils";
import type { ServerStatus } from "@/types";
import { Activity, Settings, Sliders } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Line, LineChart, ResponsiveContainer, YAxis } from "recharts";
import useSWR from "swr";

type PositionDataPoint = { time: number; value: number; goal: number };
type TorqueDataPoint = { time: number; value: number };

// Physical limits for joints (motor units)
const POSITION_LIMIT = 4095;
// Example torque limits (adjust as needed)
const TORQUE_LIMIT = 200; // replace with actual joint torque limit

export function JointControl() {
  const NUM_JOINTS = 6;
  const NUM_POINTS = 30;

  const [goalAngles, setGoalAngles] = useState<number[]>(
    Array(NUM_JOINTS).fill(0),
  );
  const [updateInterval, setUpdateInterval] = useState(0.1);
  const [plotOption, setPlotOption] = useState<string>("Position");
  const [error, setError] = useState("");
  const [isInitialized, setIsInitialized] = useState(false);

  const [positionBuffers, setPositionBuffers] = useState<PositionDataPoint[][]>(
    Array(NUM_JOINTS)
      .fill(null)
      .map(() =>
        Array.from({ length: NUM_POINTS }, (_, i) => ({
          time: i,
          value: 0,
          goal: 0,
        })),
      ),
  );

  const [torqueBuffers, setTorqueBuffers] = useState<TorqueDataPoint[][]>(
    Array(NUM_JOINTS)
      .fill(null)
      .map(() =>
        Array.from({ length: NUM_POINTS }, (_, i) => ({ time: i, value: 0 })),
      ),
  );

  const [jointPositions, setJointPositions] = useState<number[]>(
    Array(NUM_JOINTS).fill(0),
  );
  const [jointTorques, setJointTorques] = useState<number[]>(
    Array(NUM_JOINTS).fill(0),
  );

  const { data: serverStatus } = useSWR<ServerStatus>(["/status"], fetcher, {
    refreshInterval: 5000,
  });

  const [selectedRobotName, setSelectedRobotName] = useState<string | null>(
    null,
  );

  const intervalRef = useRef<number | null>(null);

  const robotIDFromName = (name?: string | null) => {
    if (name === undefined || name === null || !serverStatus?.robot_status) {
      return 0; // Default to the first robot
    }
    const index = serverStatus.robot_status.findIndex(
      (robot) => robot.device_name === name,
    );
    return index === -1 ? 0 : index; // Return 0 if not found or first one
  };

  const fetchJointPositions = async (): Promise<number[]> => {
    const robotId = robotIDFromName(selectedRobotName);
    const data = await fetchWithBaseUrl(
      `/joints/read?robot_id=${robotId}`,
      "POST",
      {
        unit: "motor_units",
        joints_ids: null,
      },
    );
    return Array.isArray(data.angles) ? data.angles : jointPositions;
  };

  const fetchJointTorques = async (): Promise<number[]> => {
    const robotId = robotIDFromName(selectedRobotName);
    const data = await fetchWithBaseUrl(
      `/torque/read?robot_id=${robotId}`,
      "POST",
    );
    return Array.isArray(data) ? data : jointTorques;
  };

  const sendJointCommands = async () => {
    const robotId = robotIDFromName(selectedRobotName);
    await fetchWithBaseUrl(`/joints/write?robot_id=${robotId}`, "POST", {
      angles: goalAngles,
      unit: "motor_units",
    });
  };

  // Initialize joints from API
  const initializeJoints = async () => {
    try {
      const positions = await fetchJointPositions();
      setJointPositions(positions);
      setGoalAngles(positions);

      // Initialize position buffers with current positions
      setPositionBuffers((prev) =>
        prev.map((buf, idx) =>
          buf.map((pt) => ({
            ...pt,
            value: positions[idx],
            goal: positions[idx],
          })),
        ),
      );

      // Reset torque buffers
      setTorqueBuffers(
        Array(NUM_JOINTS)
          .fill(null)
          .map(() =>
            Array.from({ length: NUM_POINTS }, (_, i) => ({
              time: i,
              value: 0,
            })),
          ),
      );

      setIsInitialized(true);
      setError(""); // Clear any previous errors
    } catch (err) {
      setError(`Failed to initialize joint positions: ${err}`);
    }
  };

  const updateJointGoalAngle = async (jointIndex: number, value: number) => {
    const newGoals = [...goalAngles];
    newGoals[jointIndex] = value;
    setGoalAngles(newGoals);

    // Immediately send command for real-time control
    await sendJointCommands();

    setPositionBuffers((prev) =>
      prev.map((buf, idx) =>
        buf.map((pt) => (idx === jointIndex ? { ...pt, goal: value } : pt)),
      ),
    );
  };

  useEffect(() => {
    if (!serverStatus || !serverStatus.robot_status) return;
    const initialize = async () => {
      if (!isInitialized) {
        await initializeJoints();
      }
    };
    initialize();
  }, [isInitialized, serverStatus]);

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

  // Handle robot switching - reinitialize when robot changes
  useEffect(() => {
    if (selectedRobotName && isInitialized) {
      // Reset initialization state to force reload
      setIsInitialized(false);
      setError("");

      // Reinitialize with new robot
      const reinitialize = async () => {
        await initializeJoints();
      };
      reinitialize();
    }
  }, [selectedRobotName]);

  useEffect(() => {
    if (!isInitialized) return;

    const updateData = async () => {
      try {
        const [positions, torques] = await Promise.all([
          fetchJointPositions(),
          fetchJointTorques(),
        ]);

        setJointPositions(positions);
        setJointTorques(torques);

        setPositionBuffers((prev) =>
          prev.map((buf, idx) => {
            const next = buf.slice(1);
            next.push({
              time: buf[buf.length - 1].time + 1,
              value: positions[idx],
              goal: goalAngles[idx],
            });
            return next;
          }),
        );

        setTorqueBuffers((prev) =>
          prev.map((buf, idx) => {
            const next = buf.slice(1);
            next.push({
              time: buf[buf.length - 1].time + 1,
              value: torques[idx],
            });
            return next;
          }),
        );
      } catch {
        setError("Failed to fetch data");
      }
    };

    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = window.setInterval(async () => {
      await updateData();
    }, updateInterval * 1000);

    (async () => {
      await updateData();
    })();

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [updateInterval, goalAngles, isInitialized, selectedRobotName]);

  return (
    <div className="container mx-auto p-4 max-w-7xl">
      {error && (
        <Alert variant="destructive" className="mb-6">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <Card className="h-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sliders className="h-5 w-5" /> Joint Controls
              </CardTitle>
              <CardDescription>
                Adjust joint positions (motor units)
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="mb-6">
                <Label htmlFor="robot-select" className="text-sm font-medium">
                  Select Robot
                </Label>
                <Select
                  value={selectedRobotName || ""}
                  onValueChange={(value) => setSelectedRobotName(value)}
                >
                  <SelectTrigger id="robot-select" className="mt-2">
                    <SelectValue placeholder="Select robot to control" />
                  </SelectTrigger>
                  <SelectContent>
                    {serverStatus &&
                      serverStatus.robot_status.map((robot) => (
                        <SelectItem
                          key={robot.device_name}
                          value={robot.device_name || "Undefined port"}
                        >
                          {robot.name} ({robot.device_name})
                        </SelectItem>
                      ))}
                  </SelectContent>
                </Select>
              </div>

              {goalAngles.map((angle, i) => (
                <div key={i} className="space-y-3">
                  <div className="flex justify-between items-center">
                    <Label htmlFor={`joint-${i}`}>Joint {i + 1}</Label>
                    <span>{Math.round(angle)} units</span>
                  </div>
                  <Slider
                    id={`joint-${i}`}
                    min={0}
                    max={POSITION_LIMIT}
                    step={1}
                    value={[angle]}
                    onValueChange={(vals) => updateJointGoalAngle(i, vals[0])}
                  />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>0</span>
                    <span>2048</span>
                    <span>4095</span>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-2 space-y-6">
          {/* Display Settings */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" /> Display Settings
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label>Display Options</Label>
                  <RadioGroup
                    value={plotOption}
                    onValueChange={setPlotOption}
                    className="flex flex-col space-y-1 mt-2"
                  >
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="Position" id="position" />
                      <Label htmlFor="position" className="cursor-pointer">
                        Position
                      </Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="Torque" id="torque" />
                      <Label htmlFor="torque" className="cursor-pointer">
                        Torque
                      </Label>
                    </div>
                  </RadioGroup>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="update-interval">
                    Update Interval (seconds)
                  </Label>
                  <Input
                    id="update-interval"
                    type="number"
                    min={0.01}
                    max={1.0}
                    step={0.01}
                    value={updateInterval}
                    onChange={(e) =>
                      setUpdateInterval(Number.parseFloat(e.target.value))
                    }
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Graphs Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />{" "}
                {plotOption === "Position"
                  ? "Joint Positions"
                  : "Joint Torques"}
              </CardTitle>
              <CardDescription>Real-time data for all joints</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Array.from({ length: NUM_JOINTS }, (_, i) => (
                  <div key={i} className="border rounded-lg p-4">
                    <h3 className="text-sm font-medium mb-2">Joint {i + 1}</h3>
                    <ChartContainer
                      config={
                        plotOption === "Position"
                          ? {
                              value: { label: "Position" },
                              goal: { label: "Goal", color: "red" },
                            }
                          : { value: { label: "Torque" } }
                      }
                      className="h-[180px]"
                    >
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart
                          data={
                            plotOption === "Position"
                              ? positionBuffers[i]
                              : torqueBuffers[i]
                          }
                        >
                          <YAxis
                            domain={
                              plotOption === "Position"
                                ? [0, POSITION_LIMIT]
                                : [-TORQUE_LIMIT, TORQUE_LIMIT]
                            }
                            tickFormatter={(value) =>
                              plotOption === "Position"
                                ? Math.round(value).toString()
                                : value.toFixed(1)
                            }
                          />
                          <Line
                            type="monotone"
                            dataKey="value"
                            strokeWidth={2}
                            dot={false}
                          />
                          {plotOption === "Position" && (
                            <Line
                              type="monotone"
                              dataKey="goal"
                              strokeWidth={1}
                              strokeDasharray="4 4"
                              dot={false}
                            />
                          )}
                        </LineChart>
                      </ResponsiveContainer>
                    </ChartContainer>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

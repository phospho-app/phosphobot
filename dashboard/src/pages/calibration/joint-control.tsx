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
import { Slider } from "@/components/ui/slider";
import { fetchWithBaseUrl } from "@/lib/utils";
import { Activity, Settings, Sliders } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Line, LineChart, ResponsiveContainer, YAxis } from "recharts";

type PositionDataPoint = { time: number; value: number; goal: number };
type TorqueDataPoint = { time: number; value: number };

// Physical limits for joints (radians)
const POSITION_LIMIT = Math.PI;
// Example torque limits (adjust as needed)
const TORQUE_LIMIT = 500; // replace with actual joint torque limit

export function JointControl() {
  const NUM_JOINTS = 6;
  const NUM_POINTS = 30;

  const [goalAngles, setGoalAngles] = useState<number[]>(
    Array(NUM_JOINTS).fill(0),
  );
  const [updateInterval, setUpdateInterval] = useState(0.1);
  const [plotOption, setPlotOption] = useState<string>("Position");
  const [error, setError] = useState("");

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
  const [isConnected, setIsConnected] = useState<boolean>(false);

  const intervalRef = useRef<number | null>(null);

  // Check connection status
  const checkConnection = async () => {
    try {
      const status = await fetchWithBaseUrl("/status", "GET");
      setIsConnected(
        Array.isArray(status.robot_status) && status.robot_status.length > 0,
      );
    } catch {
      setIsConnected(false);
    }
  };

  const fetchJointPositions = async (): Promise<number[]> => {
    const data = await fetchWithBaseUrl(`/joints/read`, "POST", {
      unit: "rad",
      joints_ids: null,
    });
    return Array.isArray(data.angles) ? data.angles : jointPositions;
  };

  const fetchJointTorques = async (): Promise<number[]> => {
    const data = await fetchWithBaseUrl(`/torque/read`, "POST");
    return Array.isArray(data) ? data : jointTorques;
  };

  const sendJointCommands = async () => {
    if (!isConnected) return;
    await fetchWithBaseUrl(`/joints/write`, "POST", {
      angles: goalAngles,
      unit: "rad",
    });
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
    checkConnection();

    const updateData = async () => {
      if (!isConnected) return;
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
      await checkConnection();
      await updateData();
    }, updateInterval * 1000);

    (async () => {
      await checkConnection();
      await updateData();
    })();

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [updateInterval, goalAngles, isConnected]);

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
              <CardDescription>Adjust joint angles</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {goalAngles.map((angle, i) => (
                <div key={i} className="space-y-3">
                  <div className="flex justify-between items-center">
                    <Label htmlFor={`joint-${i}`}>Joint {i + 1}</Label>
                    <span>{angle.toFixed(2)} rad</span>
                  </div>
                  <Slider
                    id={`joint-${i}`}
                    min={-Math.PI}
                    max={Math.PI}
                    step={0.01}
                    value={[angle]}
                    onValueChange={(vals) => updateJointGoalAngle(i, vals[0])}
                  />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>-π</span>
                    <span>0</span>
                    <span>π</span>
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
                                ? [-POSITION_LIMIT, POSITION_LIMIT]
                                : [-TORQUE_LIMIT, TORQUE_LIMIT]
                            }
                            tickFormatter={(value) => value.toFixed(3)}
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

import connectionImage from "@/assets/ConnectBot.jpg";
import { RobotConfigModal } from "@/components/common/add-robot-connection";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useGlobalStore } from "@/lib/hooks";
import { fetchWithBaseUrl, fetcher } from "@/lib/utils";
import type { RobotConfigStatus, ServerStatus, TorqueStatus } from "@/types";
import {
  AlertTriangle,
  Bot,
  Crown,
  Hand,
  Link,
  Link2Off,
  LoaderCircle,
  Moon,
  PlusCircle,
  Unplug,
  X,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { toast } from "sonner";
import useSWR, { mutate } from "swr";

import {
  EditTemperatureDialog,
  TemperatureSubmenu,
  getTemperatureInfo,
} from "./temperature";

function RobotStatusMenuItem({
  robotId,
  robotUsbPort,
  robot,
}: {
  robotId: number;
  robotUsbPort: string;
  robot: RobotConfigStatus;
}) {
  const [temperatureDialogOpen, setTemperatureDialogOpen] = useState(false);

  const { data: robotTorqueStatus, mutate: mutateTorque } =
    useSWR<TorqueStatus>([`/torque/read?robot_id=${robotId}`], ([url]) =>
      fetcher(url, "POST"),
    );

  const { data: serverStatus } = useSWR<ServerStatus>(["/status"], ([url]) =>
    fetcher(url),
  );

  const leaderArmSerialIds = useGlobalStore(
    (state) => state.leaderArmSerialIds,
  );
  const addLeaderArmSerialId = useGlobalStore(
    (state) => state.addLeaderArmSerialId,
  );
  const removeLeaderArmSerialId = useGlobalStore(
    (state) => state.removeLeaderArmSerialId,
  );

  const torqueStatus = robotTorqueStatus?.current_torque as
    | number[]
    | undefined;
  const isTorqueEnabled = torqueStatus?.some((status) => status >= 1);

  const temperatureInfo = getTemperatureInfo(robot);

  // Sends a shutdown command to a specific robot and then refreshes its status.
  const moveToSleep = async () => {
    if (!serverStatus) {
      console.error("Server not available");
      return;
    }
    await fetchWithBaseUrl(`/move/sleep?robot_id=${robotId}`, "POST").then(
      mutateTorque,
    );
  };

  const jiggleGripper = async () => {
    if (!serverStatus) {
      console.error("Server not available");
      return;
    }
    await fetchWithBaseUrl(`/move/hello?robot_id=${robotId}`, "POST").then(
      mutateTorque,
    );
  };

  const disableTorque = async () => {
    await fetchWithBaseUrl(`/torque/toggle?robot_id=${robotId}`, "POST", {
      torque_status: false,
    }).then(async () => {
      await new Promise((resolve) => setTimeout(resolve, 100));
      await mutateTorque();
    });
  };

  const enableTorque = async () => {
    if (!serverStatus) {
      console.error("Server not available");
      return;
    }
    await fetchWithBaseUrl(`/torque/toggle?robot_id=${robotId}`, "POST", {
      torque_status: true,
    }).then(async () => {
      await new Promise((resolve) => setTimeout(resolve, 100));
      await mutateTorque();
    });
  };

  const disconnectRobot = async () => {
    if (!serverStatus) {
      console.error("Server not available");
      return;
    }
    const data = await fetchWithBaseUrl(
      `/robot/disconnect?robot_id=${robotId}`,
      "POST",
    );
    if (data) {
      mutate("/status");
    }
  };

  return (
    <Dialog
      open={temperatureDialogOpen}
      onOpenChange={setTemperatureDialogOpen}
    >
      <DropdownMenuSub>
        <DropdownMenuSubTrigger>
          <div className="p-2 text-sm flex justify-between items-center gap-x-2">
            <div className="relative">
              <Bot
                className={`size-5 ${
                  temperatureInfo?.hasOverheat
                    ? "text-red-500"
                    : temperatureInfo?.hasWarning
                      ? "text-orange-500"
                      : ""
                }`}
              />
              {leaderArmSerialIds.includes(robotUsbPort) && (
                <Crown className="absolute -top-2 -right-2 size-3 text-green-500" />
              )}
            </div>
            <div className="flex flex-col items-start mr-1">
              <div className="font-medium">
                #{robotId}: {robot.name}
              </div>
              {robot.device_name && (
                <div className="text-xs text-muted-foreground">
                  {robot.device_name}
                </div>
              )}
              {temperatureInfo && temperatureInfo.hasAnyData && (
                <div
                  className={`text-xs flex items-center gap-1 ${
                    temperatureInfo.hasOverheat
                      ? "text-red-500 animate-pulse"
                      : temperatureInfo.hasWarning
                        ? "text-orange-500"
                        : "text-muted-foreground"
                  }`}
                >
                  <span>
                    Max temperature:{" "}
                    {temperatureInfo.maxTemp !== null
                      ? `${temperatureInfo.maxTemp.toFixed(1)}°C`
                      : "N/A"}
                  </span>
                  {temperatureInfo.hasOverheat && (
                    <AlertTriangle className="size-3 text-red-500" />
                  )}
                  {temperatureInfo.hasWarning &&
                    !temperatureInfo.hasOverheat && (
                      <AlertTriangle className="size-3 text-orange-500" />
                    )}
                </div>
              )}
            </div>
          </div>
        </DropdownMenuSubTrigger>
        <DropdownMenuSubContent>
          <DropdownMenuItem
            onClick={() => jiggleGripper()}
            className="flex items-center gap-2"
          >
            <Hand className="size-4" />
            <span>Jiggle gripper</span>
          </DropdownMenuItem>
          {isTorqueEnabled && (
            <DropdownMenuItem
              onClick={() => disableTorque()}
              className="flex items-center gap-2"
            >
              <Link2Off className="size-4" />
              <span>Unlock position</span>
            </DropdownMenuItem>
          )}
          {!isTorqueEnabled && (
            <DropdownMenuItem
              onClick={() => enableTorque()}
              className="flex items-center gap-2"
            >
              <Link className="size-4" />
              <span>Lock position</span>
            </DropdownMenuItem>
          )}
          {leaderArmSerialIds.includes(robotUsbPort) ? (
            <DropdownMenuItem
              onClick={() => removeLeaderArmSerialId(robotUsbPort)}
              className="flex items-center gap-2"
            >
              <div className="relative">
                <Crown className="size-4" />
                <X className="absolute -top-2 -right-2 size-3 stroke-[3]" />
              </div>
              <span>Remove leader arm mark</span>
            </DropdownMenuItem>
          ) : (
            <DropdownMenuItem
              onClick={() => addLeaderArmSerialId(robotUsbPort)}
              className="flex items-center gap-2"
            >
              <Crown className="size-4" />
              <span>Mark as leader arm</span>
            </DropdownMenuItem>
          )}
          <DropdownMenuItem
            onClick={() => moveToSleep()}
            className="flex items-center gap-2"
          >
            <Moon className="size-4" />
            <span>Move to sleep</span>
          </DropdownMenuItem>
          <TemperatureSubmenu
            robot={robot}
            setTemperatureDialogOpen={setTemperatureDialogOpen}
          />
          <DropdownMenuSeparator />
          <DropdownMenuItem
            onClick={() => disconnectRobot()}
            className="flex items-center gap-2 hover:text-destructive focus:text-destructive"
          >
            <Unplug className="size-4" />
            <span>Disconnect robot</span>
          </DropdownMenuItem>
        </DropdownMenuSubContent>
      </DropdownMenuSub>
      <EditTemperatureDialog
        robotId={robotId}
        robot={robot}
        temperatureDialogOpen={temperatureDialogOpen}
        setTemperatureDialogOpen={setTemperatureDialogOpen}
      />
    </Dialog>
  );
}

function FullscreenImage() {
  const [isFullscreen, setIsFullscreen] = useState(false);

  const openFullscreen = () => {
    setIsFullscreen(true);
    document.body.style.overflow = "hidden";
  };

  const closeFullscreen = () => {
    setIsFullscreen(false);
    document.body.style.overflow = "";
  };

  return (
    <>
      <div className="cursor-pointer" onClick={openFullscreen}>
        <img
          src={connectionImage || "/placeholder.svg"}
          alt="ConnectBot"
          width={100}
          height={100}
          className="rounded-md aspect-square object-cover w-[100px] h-[100px]"
        />
      </div>
      {isFullscreen &&
        createPortal(
          <div
            className="fixed inset-0 bg-black/90 z-[9999] flex items-center justify-center"
            onClick={closeFullscreen}
            style={{ position: "fixed", top: 0, left: 0, right: 0, bottom: 0 }}
          >
            <div className="relative max-w-[90vw] max-h-[100vh]">
              <img
                src={connectionImage || "/placeholder.svg"}
                alt="ConnectBot"
                className="object-contain max-w-full max-h-full"
              />
              <button
                className="absolute top-2 right-2 bg-black/50 rounded-full p-1"
                onClick={(e) => {
                  e.stopPropagation();
                  closeFullscreen();
                }}
              >
                <X className="size-5 text-white" />
              </button>
            </div>
          </div>,
          document.body,
        )}
    </>
  );
}

export function RobotStatusDropdown() {
  const [isConfigModalOpen, setIsConfigModalOpen] = useState(false);

  const { data: serverStatus } = useSWR<ServerStatus>(["/status"], ([url]) =>
    fetcher(url),
  );

  const prevRef = useRef<RobotConfigStatus[]>([]);
  const prevTemperatureStatusRef = useRef<{
    [key: string]: { hasOverheating: boolean; hasWarning: boolean };
  }>({});

  useEffect(() => {
    if (!serverStatus) return;

    const prev = prevRef.current;
    const current = serverStatus.robot_status ?? [];

    // Check for disconnected robots
    const disconnected = prev.filter(
      (r) => !current.some((c) => c.device_name === r.device_name),
    );
    if (disconnected.length > 0) {
      const ids = disconnected.map((r) => r.device_name ?? "").join(", ");
      toast(`Robot${disconnected.length > 1 ? "s" : ""} ${ids} disconnected`);
    }

    // Check for temperature warnings and overheating
    const currentTemperatureStatus: {
      [key: string]: { hasOverheating: boolean; hasWarning: boolean };
    } = {};

    current.forEach((robot, index) => {
      const robotKey = robot.device_name ?? `robot_${index}`;
      let hasOverheating = false;
      let hasWarning = false;

      if (robot.temperature) {
        robot.temperature.forEach((temperature) => {
          if (temperature.current === null || temperature.max === null) return;

          if (temperature.current >= temperature.max - 5) {
            hasOverheating = true;
          } else if (temperature.current >= temperature.max - 15) {
            hasWarning = true;
          }
        });
      }

      currentTemperatureStatus[robotKey] = { hasOverheating, hasWarning };

      // Check if temperature status changed
      const prevStatus = prevTemperatureStatusRef.current[robotKey] || {
        hasOverheating: false,
        hasWarning: false,
      };

      // Toast for new overheating condition
      if (hasOverheating && !prevStatus.hasOverheating) {
        toast.error(
          `Critical Temperature Alert: Robot ${robot.name || `#${index}`} motors have exceeded safe operating temperature. Please move robot to sleep position and disconnect power immediately to prevent hardware damage.`,
          {
            duration: 10000, // Show for 10 seconds
          },
        );
      }

      // Toast for new warning condition (only if not overheating)
      if (
        hasWarning &&
        !hasOverheating &&
        !prevStatus.hasWarning &&
        !prevStatus.hasOverheating
      ) {
        toast.warning(
          `Temperature Warning: Robot ${robot.name || `#${index}`} motor temperature is approaching thermal limits. Consider moving robot to sleep position to allow cooling.`,
          {
            duration: 5000,
          },
        );
      }

      // Toast for temperature returning to normal
      if (
        !hasOverheating &&
        !hasWarning &&
        (prevStatus.hasOverheating || prevStatus.hasWarning)
      ) {
        toast.success(
          `✅ Robot ${robot.name || `#${index}`} temperature normalized.`,
          {
            duration: 3000,
          },
        );
      }
    });

    prevRef.current = current;
    prevTemperatureStatusRef.current = currentTemperatureStatus;
  }, [serverStatus]);

  const leaderArmSerialIds = useGlobalStore(
    (state) => state.leaderArmSerialIds,
  );

  // Check temperature status for all robots
  const checkTemperatureStatus = () => {
    if (!serverStatus?.robot_status)
      return { hasOverheating: false, hasWarning: false };

    let hasOverheating = false;
    let hasWarning = false;

    serverStatus.robot_status.forEach((robot) => {
      if (!robot.temperature) return;

      robot.temperature.forEach((temperature) => {
        if (temperature.current === null || temperature.max === null) return;

        if (temperature.current >= temperature.max - 5) {
          hasOverheating = true;
        } else if (temperature.current >= temperature.max - 15) {
          hasWarning = true;
        }
      });
    });

    return { hasOverheating, hasWarning };
  };

  const { hasOverheating, hasWarning } = checkTemperatureStatus();

  if (!serverStatus) {
    return (
      <Button
        variant="outline"
        className="flex items-center gap-2 relative bg-transparent"
      >
        <LoaderCircle className="animate-spin size-5" />
      </Button>
    );
  }

  const robotConnected =
    serverStatus.robot_status && serverStatus.robot_status.length > 0;

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="outline"
            className={`flex items-center gap-2 relative cursor-pointer ${
              hasOverheating
                ? "border-red-500 bg-red-50 dark:bg-red-900/20 animate-pulse"
                : hasWarning
                  ? "border-orange-500 bg-orange-50 dark:bg-orange-900/20"
                  : ""
            }`}
          >
            {robotConnected ? (
              <span
                className={`size-2 rounded-full ${
                  hasOverheating
                    ? "bg-red-500 animate-pulse"
                    : hasWarning
                      ? "bg-orange-500"
                      : "bg-green-500"
                }`}
              />
            ) : (
              <span className="size-2 rounded-full bg-destructive" />
            )}
            {robotConnected && (
              <>
                {serverStatus.robot_status.map((robot, index) => {
                  // Calculate temperature status for this specific robot
                  const robotTemperatureInfo = (() => {
                    if (!robot.temperature || robot.temperature.length === 0)
                      return null;

                    const hasOverheat = robot.temperature.some(
                      (t) =>
                        t.current !== null &&
                        t.max !== null &&
                        t.current >= t.max - 5,
                    );
                    const hasWarning = robot.temperature.some(
                      (t) =>
                        t.current !== null &&
                        t.max !== null &&
                        t.current >= t.max - 15 &&
                        t.current < t.max - 5,
                    );

                    return { hasOverheat, hasWarning };
                  })();

                  return (
                    <div key={index} className="relative">
                      <Bot
                        className={`size-5 ${
                          robotTemperatureInfo?.hasOverheat
                            ? "text-red-500"
                            : robotTemperatureInfo?.hasWarning
                              ? "text-orange-500"
                              : ""
                        }`}
                      />
                      {robot.device_name &&
                        leaderArmSerialIds.includes(robot.device_name) && (
                          <Crown className="absolute -top-2 -right-2 size-3 text-green-500" />
                        )}
                    </div>
                  );
                })}
                {hasOverheating && (
                  <AlertTriangle className="size-4 text-red-500 animate-pulse" />
                )}
                {hasWarning && !hasOverheating && (
                  <AlertTriangle className="size-4 text-orange-500" />
                )}
              </>
            )}
            {!robotConnected && (
              <>
                <Bot className="size-5" />
                <div className="flex items-center gap-1">
                  <div className="flex items-end text-muted-foreground font-semibold leading-none">
                    <span className="text-[8px] translate-y-[-1px]">Z</span>
                    <span className="text-xs">Z</span>
                    <span className="text-[8px] translate-y-[1px]">Z</span>
                  </div>
                </div>
              </>
            )}
            <span className="sr-only">
              {robotConnected ? "Robot Connected" : "Robot Disconnected"}
            </span>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuLabel className="text-xs text-muted-foreground">
            {robotConnected
              ? `${serverStatus.robot_status.length} Robot${serverStatus.robot_status.length > 1 ? "s" : ""} Connected`
              : "Robot is disconnected"}
          </DropdownMenuLabel>
          {robotConnected && serverStatus.robot_status && (
            <>
              <DropdownMenuSeparator />
              {serverStatus.robot_status.map((robot, index) => {
                return (
                  <RobotStatusMenuItem
                    key={index}
                    robotId={index}
                    robotUsbPort={robot.device_name ?? "unknown"}
                    robot={robot}
                  />
                );
              })}
            </>
          )}
          {!robotConnected && (
            <>
              <div className="flex flex-col items-center p-2">
                <FullscreenImage />
                <span className="text-xs text-muted-foreground mt-2">
                  Check USB-C and Power cable connections
                </span>
              </div>
            </>
          )}
          <DropdownMenuSeparator />
          <DropdownMenuItem
            onClick={() => setIsConfigModalOpen(true)}
            className="flex items-center gap-2"
          >
            <PlusCircle className="size-4" />
            <span>Connect to another robot</span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <RobotConfigModal
        open={isConfigModalOpen}
        onOpenChange={setIsConfigModalOpen}
      />
    </>
  );
}

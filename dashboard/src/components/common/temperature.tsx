import { Alert } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  DialogContent,
  DialogDescription,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { DialogFooter, DialogHeader } from "@/components/ui/dialog";
import {
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { fetchWithBaseUrl } from "@/lib/utils";
import { RobotConfigStatus } from "@/types";
import { AlertTriangle, Settings2, Thermometer } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Label } from "recharts";
import { toast } from "sonner";
import { mutate } from "swr";

export function getTemperatureInfo(robot: RobotConfigStatus) {
  if (!robot.temperature || robot.temperature.length === 0) return null;

  const validTemps = robot.temperature.filter(
    (temp) => temp.current !== null && temp.max !== null,
  );

  let maxTemp = null;
  let hasOverheat = false;
  let hasWarning = false;

  if (validTemps.length > 0) {
    maxTemp = Math.max(...validTemps.map((t) => t.current!));
    hasOverheat = validTemps.some((t) => t.current! >= t.max! - 5);
    hasWarning = validTemps.some(
      (t) => t.current! >= t.max! - 15 && t.current! < t.max! - 5,
    );
  }

  return {
    maxTemp,
    hasOverheat,
    hasWarning,
    hasAnyData: robot.temperature.length > 0,
  };
}

export function TemperatureSubmenu({
  robot,
  setTemperatureDialogOpen,
}: {
  robot: RobotConfigStatus;
  setTemperatureDialogOpen: (open: boolean) => void;
}) {
  const temperatureInfo = getTemperatureInfo(robot);

  return (
    <>
      {robot.temperature && robot.temperature.length > 0 && (
        <>
          <DropdownMenuSeparator />
          <DropdownMenuSub>
            <DropdownMenuSubTrigger>
              <div className="flex items-center gap-2">
                <Thermometer
                  className={`size-4 ${
                    temperatureInfo?.hasOverheat
                      ? "text-red-500"
                      : temperatureInfo?.hasWarning
                        ? "text-orange-500"
                        : ""
                  }`}
                />
                <span
                  className={`${
                    temperatureInfo?.hasOverheat
                      ? "text-red-500"
                      : temperatureInfo?.hasWarning
                        ? "text-orange-500"
                        : ""
                  }`}
                >
                  Motor temperatures
                </span>
                {temperatureInfo?.hasOverheat && (
                  <AlertTriangle className="size-3 text-red-500 ml-auto" />
                )}
                {temperatureInfo?.hasWarning &&
                  !temperatureInfo?.hasOverheat && (
                    <AlertTriangle className="size-3 text-orange-500 ml-auto" />
                  )}
              </div>
            </DropdownMenuSubTrigger>
            <DropdownMenuSubContent>
              {robot.temperature.map((temperature, index) => {
                const isOverheating =
                  temperature.current !== null &&
                  temperature.max !== null &&
                  temperature.current >= temperature.max - 5;
                const isWarning =
                  temperature.current !== null &&
                  temperature.max !== null &&
                  temperature.current >= temperature.max - 15 &&
                  temperature.current < temperature.max - 5;

                return (
                  <DropdownMenuItem
                    key={index}
                    className={`cursor-default hover:bg-transparent focus:bg-transparent ${
                      isOverheating
                        ? "bg-red-100 dark:bg-red-900/30"
                        : isWarning
                          ? "bg-orange-100 dark:bg-orange-900/30"
                          : ""
                    }`}
                    onClick={(e) => e.preventDefault()}
                  >
                    <span className="text-sm">
                      Motor {index + 1}:{" "}
                      {temperature.current !== null
                        ? `${temperature.current.toFixed(1)}°C`
                        : "N/A"}
                      {temperature.max !== null &&
                        ` / ${temperature.max.toFixed(1)}°C`}
                      {isOverheating && (
                        <AlertTriangle className="inline size-3 ml-1 text-red-500" />
                      )}
                      {isWarning && !isOverheating && (
                        <AlertTriangle className="inline size-3 ml-1 text-orange-500" />
                      )}
                    </span>
                  </DropdownMenuItem>
                );
              })}
              <DropdownMenuSeparator />
              <DialogTrigger asChild>
                <DropdownMenuItem
                  className="flex items-center gap-1"
                  onSelect={(e) => {
                    e.preventDefault();
                    setTemperatureDialogOpen(true);
                  }}
                >
                  <Settings2 className="size-4" />
                  <span>Set maximum temperatures</span>
                </DropdownMenuItem>
              </DialogTrigger>
            </DropdownMenuSubContent>
          </DropdownMenuSub>
        </>
      )}
    </>
  );
}

export function EditTemperatureDialog({
  robotId,
  robot,
  temperatureDialogOpen,
  setTemperatureDialogOpen,
}: {
  robotId: number;
  robot: RobotConfigStatus;
  temperatureDialogOpen: boolean;
  setTemperatureDialogOpen: (open: boolean) => void;
}) {
  const [temperatureValues, setTemperatureValues] = useState<number[]>([]);
  const hasInitialized = useRef(false);

  const RECOMMENDED_MAX_TEMP = 70;
  const MAX_TEMP_LIMIT = 100;

  // Initializes temperature values
  useEffect(() => {
    if (temperatureDialogOpen && robot.temperature && !hasInitialized.current) {
      // Initialize with current max values instead of empty strings
      const initialValues = robot.temperature.map((temp) =>
        temp.max !== null ? Math.round(temp.max) : RECOMMENDED_MAX_TEMP,
      );
      setTemperatureValues(initialValues);
      hasInitialized.current = true;
    }

    // Reset initialization flag when dialog closes
    if (!temperatureDialogOpen) {
      hasInitialized.current = false;
    }
  }, [temperatureDialogOpen, robot.temperature]);

  const handleTemperatureUpdate = async () => {
    console.log("Updating temperature limits:", temperatureValues);
    await fetchWithBaseUrl(`/temperature/write?robot_id=${robotId}`, "POST", {
      maximum_temperature: temperatureValues,
    });

    // Refresh the server status to get updated temperature data
    mutate("/status");

    toast.success(`Temperature limits updated for Robot ${robot.name}`);
    setTemperatureDialogOpen(false);
  };

  const handleTemperatureChange = (index: number, value: string) => {
    // Remove any non-digit characters (including decimal points)
    const numbersOnly = value.replace(/[^0-9]/g, "");

    // Limit the value to maximum MAX_TEMP_LIMIT
    const numericValue = parseInt(numbersOnly);
    const limitedValue = Math.min(numericValue, MAX_TEMP_LIMIT);
    // Round to nearest integer
    const roundedValue = Math.round(limitedValue);

    const newValues = [...temperatureValues];
    newValues[index] = roundedValue;

    setTemperatureValues(newValues);
  };

  // Check if any temperature value is above 70
  const hasHighTemperature = temperatureValues.some((val) => {
    return val > RECOMMENDED_MAX_TEMP;
  });

  return (
    <DialogContent className="max-w-2xl">
      <DialogHeader>
        <DialogTitle>
          Set #{robotId}: {robot.name} motor temperature limits
        </DialogTitle>
        <DialogDescription>
          The motors will automatically shut down if the measured temperature
          exceeds the set maximum.
        </DialogDescription>
      </DialogHeader>

      <div className="space-y-4">
        {robot.temperature?.map((_, index) => {
          const inputValue = temperatureValues[index];
          const isAboveRecommendation = inputValue > RECOMMENDED_MAX_TEMP;

          return (
            <div key={index} className="space-y-2">
              <div className="flex items-center space-x-2">
                <Label id={`motor-${index}`} className="w-20">
                  Motor {index + 1}:
                </Label>
                <Input
                  id={`motor-${index}`}
                  type="number"
                  step={1}
                  min={0}
                  max={MAX_TEMP_LIMIT}
                  value={inputValue}
                  onChange={(e) =>
                    handleTemperatureChange(index, e.target.value)
                  }
                  className={`flex-1 ${isAboveRecommendation ? "border-orange-500 focus:border-orange-500 focus:ring-orange-500" : ""}`}
                />
                <span className="text-sm text-muted-foreground">°C</span>
              </div>
            </div>
          );
        })}
      </div>

      {hasHighTemperature && (
        <Alert variant="destructive" className="mt-4">
          <AlertTriangle className="size-4 flex-shrink-0" />
          <span>
            Temperatures above {RECOMMENDED_MAX_TEMP}°C risk damaging your
            motors.
          </span>
        </Alert>
      )}

      <DialogFooter>
        <Button
          variant="outline"
          onClick={() => setTemperatureDialogOpen(false)}
        >
          Cancel
        </Button>
        <Button onClick={handleTemperatureUpdate}>Update</Button>
      </DialogFooter>
    </DialogContent>
  );
}

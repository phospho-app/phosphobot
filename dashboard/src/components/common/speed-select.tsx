import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Slider } from "@/components/ui/slider";
import { Gauge } from "lucide-react";
import { useEffect, useRef, useState } from "react";

// Helper function to clamp a value between a minimum and maximum
const clamp = (value: number, min: number, max: number) =>
  Math.max(min, Math.min(value, max));

interface SpeedSelectProps {
  onChange?: (speed: number) => void;
  defaultValue?: number;
  disabled?: boolean;
  title?: string;
  minSpeed?: number; // New prop for minimum speed
  maxSpeed?: number; // New prop for maximum speed
  step?: number; // New prop for step increment
}

export function SpeedSelect({
  onChange,
  defaultValue = 1.0, // Default for the speed value itself
  disabled = false,
  title = "Playback Speed",
  minSpeed = 0.0, // Default min for the range as per new request
  maxSpeed = 2.0, // Default max for the range as per new request
  step = 0.1, // Default step
}: SpeedSelectProps) {
  // Internal state for the speed, initialized with defaultValue clamped to initial min/max.
  const [internalSpeed, setInternalSpeed] = useState(() =>
    clamp(defaultValue, minSpeed, maxSpeed),
  );

  // Ref to track if the component has mounted, to distinguish initial defaultValue application
  const isMounted = useRef(false);

  // Effect to handle changes in `defaultValue` prop from the parent.
  // This allows the parent to programmatically set/reset the speed.
  useEffect(() => {
    if (isMounted.current) {
      // Only run on subsequent renders, not on initial mount
      const newClampedValue = clamp(defaultValue, minSpeed, maxSpeed);
      if (newClampedValue !== internalSpeed) {
        setInternalSpeed(newClampedValue);
        if (onChange) {
          onChange(newClampedValue);
        }
      }
    } else {
      // On initial mount, internalSpeed is already set by useState initializer.
      // If initial defaultValue (clamped) is different from raw defaultValue,
      // and parent expects onChange for this initial clamping, it could be called here.
      // However, standard behavior for defaultValue is not to call onChange on mount.
      isMounted.current = true;
    }
  }, [defaultValue, minSpeed, maxSpeed, onChange]); // internalSpeed is not in deps to avoid loops if parent syncs defaultValue from onChange

  // Effect to handle changes in `minSpeed` or `maxSpeed` props.
  // This clamps the current internalSpeed (which might have been set by the user)
  // to the new boundaries.
  useEffect(() => {
    const reClampedSpeed = clamp(internalSpeed, minSpeed, maxSpeed);
    if (reClampedSpeed !== internalSpeed) {
      setInternalSpeed(reClampedSpeed);
      if (onChange) {
        onChange(reClampedSpeed);
      }
    }
  }, [minSpeed, maxSpeed, internalSpeed, onChange]); // internalSpeed is a dependency here

  const handleSpeedSliderChange = (value: number[]) => {
    // Slider value is an array, take the first element.
    // Clamp it defensively, though slider should respect its own min/max/step.
    const newSpeed = clamp(value[0], minSpeed, maxSpeed);
    if (newSpeed !== internalSpeed) {
      setInternalSpeed(newSpeed);
      if (onChange) {
        onChange(newSpeed);
      }
    }
  };

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          className="flex items-center gap-2"
          disabled={disabled}
        >
          <Gauge className="h-4 w-4" />
          {/* Ensure speed display respects precision, e.g., 1 decimal place like step */}
          <span>
            {internalSpeed.toFixed(step.toString().split(".")[1]?.length || 1)}x
          </span>
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80">
        <div className="space-y-4">
          <h4 className="font-medium leading-none">{title}</h4>
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">
              {minSpeed.toFixed(1)}x
            </span>
            <span className="text-sm font-medium">
              {internalSpeed.toFixed(
                step.toString().split(".")[1]?.length || 1,
              )}
              x
            </span>
            <span className="text-sm text-muted-foreground">
              {maxSpeed.toFixed(1)}x
            </span>
          </div>
          <Slider
            value={[internalSpeed]} // Slider is driven by the internalSpeed state
            min={minSpeed}
            max={maxSpeed}
            step={step}
            onValueChange={handleSpeedSliderChange} // Updates internalSpeed and calls parent's onChange
            disabled={disabled}
          />
        </div>
      </PopoverContent>
    </Popover>
  );
}

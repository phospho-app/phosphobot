import { PhosphoVRCallout } from "@/components/callout/phospho-vr";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useAuth } from "@/context/AuthContext";
import { useLocalStorageState } from "@/lib/hooks";
import { fetchWithBaseUrl, fetcher } from "@/lib/utils";
import { TeleopSettings } from "@/types";
import { useCallback, useRef } from "react";
import useSWR from "swr";

export function VRControl() {
  const { proUser } = useAuth();
  const debounceTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const [accordionOpen, setAccordionOpen] = useLocalStorageState(
    "vr-how-to-connect-accordion",
    "",
  );

  // Logarithmic scaling functions for the slider
  // Map linear slider position (0-100) to logarithmic scaling value (0.1-3.0)
  const sliderToScaling = (sliderValue: number): number => {
    // Map 0-50 to 0.1-1.0 and 50-100 to 1.0-3.0 logarithmically
    if (sliderValue <= 50) {
      // Lower half: 0.1 to 1.0
      const normalizedValue = sliderValue / 50; // 0 to 1
      return 0.1 * Math.pow(10, normalizedValue); // 0.1 to 1.0 logarithmically
    } else {
      // Upper half: 1.0 to 3.0
      const normalizedValue = (sliderValue - 50) / 50; // 0 to 1
      return 1.0 * Math.pow(3, normalizedValue); // 1.0 to 3.0 logarithmically
    }
  };

  // Map scaling value (0.1-3.0) to linear slider position (0-100)
  const scalingToSlider = (scalingValue: number): number => {
    if (scalingValue <= 1.0) {
      // Lower half: map 0.1-1.0 to 0-50
      const logValue = Math.log10(scalingValue / 0.1); // 0 to 1
      return logValue * 50; // 0 to 50
    } else {
      // Upper half: map 1.0-3.0 to 50-100
      const logValue = Math.log(scalingValue / 1.0) / Math.log(3); // 0 to 1
      return 50 + logValue * 50; // 50 to 100
    }
  };

  const { data: settings, mutate: mutateSettings } = useSWR<TeleopSettings>(
    ["/teleop/settings/read"],
    ([url]) => fetcher(url, "POST"),
    {
      fallbackData: { vr_scaling: 1.0 },
      revalidateOnFocus: false,
    },
  );

  const updateTeleopSetting = useCallback(
    async <K extends keyof TeleopSettings>(
      key: K,
      value: TeleopSettings[K],
    ) => {
      if (!settings) return;

      const updatedSettings = { ...settings, [key]: value };

      // Optimistic update
      await mutateSettings(updatedSettings, false);

      // Debounced server sync
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current);
      }

      debounceTimeoutRef.current = setTimeout(async () => {
        const updatePayload = { [key]: value } as Partial<TeleopSettings>;
        await fetchWithBaseUrl("/teleop/settings", "POST", updatePayload);
        // Revalidate to ensure sync with server
        mutateSettings();
      }, 150);
    },
    [settings, mutateSettings],
  );

  const handleScalingChange = (value: number[]) => {
    const sliderValue = value[0];
    const scalingValue = sliderToScaling(sliderValue);
    updateTeleopSetting("vr_scaling", scalingValue);
  };

  return (
    <div className="space-y-6">
      {!proUser && <PhosphoVRCallout />}

      <Card>
        <CardHeader>
          <CardTitle>VR Control</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* How to Connect Accordion */}
          <Accordion
            type="single"
            collapsible
            value={accordionOpen}
            onValueChange={setAccordionOpen}
          >
            <AccordionItem value="how-to-connect">
              <AccordionTrigger>
                How to connect to your robot in VR?
              </AccordionTrigger>
              <AccordionContent>
                <div className="space-y-4">
                  <p className="text-muted-foreground">
                    Control your robot in virtual reality using a Meta Quest 2,
                    Meta Quest Pro, Meta Quest 3, or Meta Quest 3S. Watch the
                    video to learn how to connect your robot in VR.
                  </p>

                  <div className="aspect-video max-w-2xl">
                    <iframe
                      width="100%"
                      height="100%"
                      src="https://www.youtube.com/embed/AQ-xgCTdj_w?si=tUw1JIWwm75gd5_9"
                      title="Phospho VR Control Demo"
                      frameBorder="0"
                      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                      referrerPolicy="strict-origin-when-cross-origin"
                      className="rounded-lg"
                    ></iframe>
                  </div>

                  <div className="flex flex-wrap gap-3">
                    <Button asChild variant="outline">
                      <a
                        href="https://docs.phospho.ai/examples/teleop"
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        Read the Docs
                      </a>
                    </Button>
                    <Button asChild variant="outline">
                      <a
                        href="https://discord.gg/cbkggY6NSK"
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        Get Help on Discord
                      </a>
                    </Button>
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>

          {/* Settings Section */}
          <div className="space-y-4">
            <h3 className="font-semibold text-base">Settings</h3>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Label htmlFor="scaling-slider">
                  Sensitivity: {settings?.vr_scaling.toFixed(1) ?? "1.0"}
                </Label>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <span className="text-muted-foreground cursor-help">
                        ⓘ
                      </span>
                    </TooltipTrigger>
                    <TooltipContent className="max-w-xs">
                      <p>
                        <strong>Default</strong>: 1.0 (real-life scale)
                        <br />
                        <strong>Low values</strong>: Robot barely moves when you
                        move a lot
                        <br />
                        <strong>High values</strong>: Robot moves a lot when you
                        move a little
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <Slider
                id="scaling-slider"
                min={0}
                max={100}
                step={1}
                value={[scalingToSlider(settings?.vr_scaling ?? 1.0)]}
                onValueChange={handleScalingChange}
                className="w-full"
              />
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>0.1</span>
                <span className="font-medium">1.0</span>
                <span>3.0</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

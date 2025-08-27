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
    const newValue = value[0];
    updateTeleopSetting("vr_scaling", newValue);
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
              <Label htmlFor="scaling-slider">
                Scaling: {settings?.vr_scaling.toFixed(1) ?? "1.0"}
              </Label>
              <Slider
                id="scaling-slider"
                min={0.1}
                max={3.0}
                step={0.1}
                value={[settings?.vr_scaling ?? 1.0]}
                onValueChange={handleScalingChange}
                className="w-full"
              />
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>0.1</span>
                <span>3.0</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

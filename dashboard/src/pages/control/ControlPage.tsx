import { PhosphoVRCallout } from "@/components/callout/phospho-vr";
import { Recorder } from "@/components/common/recorder";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useAuth } from "@/context/AuthContext";
import { useGlobalStore } from "@/lib/hooks";
import { ViewVideoPage } from "@/pages/ViewVideoPage";
import {
  BicepsFlexed,
  Gamepad2,
  Hand,
  Keyboard,
  RectangleGoggles,
} from "lucide-react";
import { useState } from "react";

import { GamepadControl } from "./GamepadControlPage";
import { KeyboardControl } from "./KeyboardControlPage";
import { LeaderArmControl } from "./LeaderArmControlPage";
import { SingleArmReplay } from "./SingleArmReplayPage";

export function ControlPage() {
  const showCamera = useGlobalStore((state) => state.showCamera);
  const setShowCamera = useGlobalStore((state) => state.setShowCamera);
  const [activeTab, setActiveTab] = useState("keyboard");
  const { proUser } = useAuth();

  const controlOptions = [
    {
      value: "keyboard",
      icon: Keyboard,
      label: "Keyboard",
      tooltip: "Keyboard control",
    },
    {
      value: "gamepad",
      icon: Gamepad2,
      label: "Gamepad",
      tooltip: "Gamepad control",
    },
    {
      value: "leader",
      icon: BicepsFlexed,
      label: "Leader arm",
      tooltip: "Leader arm control",
    },
    {
      value: "single",
      icon: Hand,
      label: "By hand",
      tooltip: "Move with your hands",
    },
    { value: "VR", icon: RectangleGoggles, label: "VR", tooltip: "VR control" },
  ];

  return (
    <div>
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <div className="flex flex-col md:flex-row justify-between gap-2">
          <div className="flex items-center gap-2 bg-muted rounded-lg p-1 border-1">
            <span className="text-sm font-medium text-muted-foreground px-2">
              Control
            </span>
            <TabsList className="flex gap-1 border-0 bg-transparent p-0">
              <TooltipProvider delayDuration={300}>
                {controlOptions.map((option) => {
                  const Icon = option.icon;
                  const isActive = activeTab === option.value;
                  return (
                    <Tooltip key={option.value}>
                      <TooltipTrigger asChild>
                        <TabsTrigger
                          value={option.value}
                          className={`cursor-pointer transition-all ${
                            isActive
                              ? "px-3 py-1.5 bg-background shadow-sm"
                              : "px-2 py-1.5 hover:bg-background/50"
                          }`}
                        >
                          <Icon className="size-4" />
                          {isActive && (
                            <span className="ml-2">{option.label}</span>
                          )}
                        </TabsTrigger>
                      </TooltipTrigger>
                      {!isActive && (
                        <TooltipContent>
                          <p>{option.tooltip}</p>
                        </TooltipContent>
                      )}
                    </Tooltip>
                  );
                })}
              </TooltipProvider>
            </TabsList>
          </div>
          <Recorder showCamera={showCamera} setShowCamera={setShowCamera} />
        </div>
        {showCamera && <ViewVideoPage />}
        <TabsContent value="keyboard">
          <KeyboardControl />
        </TabsContent>
        <TabsContent value="gamepad">
          <GamepadControl />
        </TabsContent>
        <TabsContent value="leader">
          <LeaderArmControl />
        </TabsContent>
        <TabsContent value="single">
          <SingleArmReplay />
        </TabsContent>
        <TabsContent value="VR">
          <div className="space-y-6">
            {!proUser && <PhosphoVRCallout />}

            <div className="flex flex-col gap-4 p-6 bg-background rounded-2xl border">
              <div className="space-y-4">
                <h4 className="font-semibold text-lg">
                  How to connect to your robot in VR?
                </h4>
                <p className="text-muted-foreground">
                  Control your robot in virtual reality using a Meta Quest 2,
                  Meta Quest Pro, Meta Quest 3, or Meta Quest 3S. Watch the
                  video to learn how to connect your robot in VR.
                </p>
              </div>

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
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}

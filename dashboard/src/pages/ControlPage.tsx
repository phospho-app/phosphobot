import { Recorder } from "@/components/common/recorder";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useGlobalStore } from "@/lib/hooks";
import { BicepsFlexed, Gamepad2, RectangleGoggles, Keyboard, Hand } from "lucide-react";
import { useState } from "react";

import GamepadControlPage from "./GamepadControlPage";
import KeyboardControlPage from "./KeyboardControlPage";
import LeaderArmPage from "./LeaderArmControlPage";
import ReplayPage from "./SingleArmReplayPage";
import ViewVideo from "./ViewVideoPage";

export default function ControlPage() {
  const showCamera = useGlobalStore((state) => state.showCamera);
  const setShowCamera = useGlobalStore((state) => state.setShowCamera);
  const [activeTab, setActiveTab] = useState("keyboard");

  const controlOptions = [
    { value: "keyboard", icon: Keyboard, label: "Keyboard", tooltip: "Keyboard control" },
    { value: "gamepad", icon: Gamepad2, label: "Gamepad", tooltip: "Gamepad control" },
    { value: "leader", icon: BicepsFlexed, label: "Leader arm", tooltip: "Leader arm control" },
    { value: "single", icon: Hand, label: "By hand", tooltip: "Move with your hands" },
    { value: "VR", icon: RectangleGoggles, label: "VR", tooltip: "VR control" },
  ];

  return (
    <div>
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <div className="flex flex-col md:flex-row justify-between gap-2">
          <div className="flex items-center gap-2 bg-muted rounded-lg p-1 border-1">
            <span className="text-sm font-medium text-muted-foreground px-2">Control</span>
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
                          {isActive && <span className="ml-2">{option.label}</span>}
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
        {showCamera && <ViewVideo />}
        <TabsContent value="keyboard">
          <KeyboardControlPage />
        </TabsContent>
        <TabsContent value="gamepad">
          <GamepadControlPage />
        </TabsContent>
        <TabsContent value="leader">
          <LeaderArmPage />
        </TabsContent>
        <TabsContent value="single">
          <ReplayPage />
        </TabsContent>
        <TabsContent value="VR">
          <div className="flex flex-col gap-4 p-4 bg-background rounded-2xl">
            <div className="text-sm">
              Use the Meta Quest app to control the robot in VR with a Meta
              Quest 2, Meta Quest Pro, Meta Quest 3 or Meta Quest 3s.
            </div>
            <div className="text-sm">
              If you bought a{" "}
              <a
                href="https://robots.phospho.ai/"
                target="_blank"
                className="underline"
              >
                phospho starter pack
              </a>
              , reach out to us on Discord to get access to the app.
            </div>
            <iframe
              width="560"
              height="315"
              src="https://www.youtube.com/embed/AQ-xgCTdj_w?si=tUw1JIWwm75gd5_9"
              title="YouTube video player"
              frame-border="0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
              referrerPolicy="strict-origin-when-cross-origin"
            ></iframe>
            <div className="text-sm"></div>
            <div className="flex gap-2">
              <Button asChild className="min-w-32" variant="secondary">
                <a href="https://docs.phospho.ai/examples/teleop">Learn more</a>
              </Button>
              <Button asChild className="min-w-32" variant={"secondary"}>
                <a href="https://discord.gg/cbkggY6NSK">Reach out on Discord</a>
              </Button>
              <Button asChild className="min-w-32">
                <a href="https://www.meta.com/en-gb/experiences/phospho-teleoperation/8873978782723478/">
                  Get on the Meta Store
                </a>
              </Button>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}

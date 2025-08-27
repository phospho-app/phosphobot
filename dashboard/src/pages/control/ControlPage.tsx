import { Recorder } from "@/components/common/recorder";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useGlobalStore, useLocalStorageState } from "@/lib/hooks";
import { ViewVideoPage } from "@/pages/ViewVideoPage";
import {
  BicepsFlexed,
  Gamepad2,
  Hand,
  Keyboard,
  RectangleGoggles,
} from "lucide-react";
import { useEffect } from "react";

import { GamepadControl } from "./GamepadControlPage";
import { KeyboardControl } from "./KeyboardControlPage";
import { LeaderArmControl } from "./LeaderArmControlPage";
import { SingleArmReplay } from "./SingleArmReplayPage";
import { VRControl } from "./VRControlPage";

export function ControlPage() {
  const showCamera = useGlobalStore((state) => state.showCamera);
  const setShowCamera = useGlobalStore((state) => state.setShowCamera);
  const [activeTab, setActiveTab] = useLocalStorageState("control-active-tab", "keyboard");

  // Read tab from URL on mount (URL takes priority over localStorage)
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const tabParam = urlParams.get('tab');
    if (tabParam) {
      // URL parameter overrides localStorage default
      setActiveTab(tabParam);
    }
    // If no URL param, activeTab already initialized from localStorage
  }, [setActiveTab]);

  // Update URL when tab changes
  const handleTabChange = (newTab: string) => {
    setActiveTab(newTab);
    const url = new URL(window.location.href);
    url.searchParams.set('tab', newTab);
    window.history.pushState({}, '', url.toString());
  };

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
      <Tabs value={activeTab} onValueChange={handleTabChange}>
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
          <VRControl />
        </TabsContent>
      </Tabs>
    </div>
  );
}

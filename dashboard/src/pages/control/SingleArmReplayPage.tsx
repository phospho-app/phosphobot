import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { fetcher } from "@/lib/utils";
import { fetchWithBaseUrl } from "@/lib/utils";
import { ServerStatus } from "@/types";
import { useState } from "react";
import useSWR from "swr";

export function SingleArmReplay() {
  const [gravityControl, enableGravityControl] = useState(false);

  const { data: serverStatus } = useSWR<ServerStatus>(["/status"], ([url]) =>
    fetcher(url),
  );

  const handleInvertGravityControl = async () => {
    if (gravityControl) {
      await fetchWithBaseUrl("/gravity/stop", "POST");
      enableGravityControl(false);
    } else {
      await fetchWithBaseUrl("/gravity/start", "POST");
      enableGravityControl(true);
    }
  };

  const handleDisableTorque = async () => {
    fetchWithBaseUrl("/torque/toggle", "POST", {
      torque_status: false,
    });
  };

  return (
    <div className="flex flex-col space-y-4 items-center justify-center space-x-6">
      <Button variant="outline" className="w-60" onClick={handleDisableTorque}>
        Disable Torque on all robots
      </Button>
      <div className="flex items-center space-x-3">
        <Switch
          id="invert-gravity-control"
          checked={gravityControl}
          onCheckedChange={handleInvertGravityControl}
          disabled={serverStatus?.is_recording}
        />
        <Label className="text-sm font-medium">Gravity Compensation</Label>
        <span
          className={`text-sm font-medium ${gravityControl ? "text-green-500" : "text-red-500"}`}
        >
          {gravityControl ? "On" : "Off"}
        </span>
      </div>
    </div>
  );
}

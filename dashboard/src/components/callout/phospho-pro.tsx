import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { TestTubeDiagonal } from "lucide-react";

const PHOSPHO_PRO_SUBSCRIBE_URL = "https://phospho.ai/pro";

export function PhosphoProCallout({ className }: { className?: string }) {
  return (
    <Card className={cn("border-green-400 py-2 px-4", className)}>
      <CardContent className="flex items-center p-2">
        <div className="flex flex-row justify-between items-center gap-4 w-full">
          <div>
            <TestTubeDiagonal className="text-green-500 size-10" />
          </div>
          <div className="flex-1">
            <div className="font-semibold text-lg mb-1.5">
              Boost your phospho experience with{" "}
              <span className="text-green-500">phospho pro</span>
            </div>
            <div className="mb-3 text-muted-foreground">
              Access the phospho teleoperation app, AI training, exclusive
              Discord channels and more...
            </div>
          </div>
          <div className="flex-shrink-0">
            <Button asChild>
              <a
                href={`${PHOSPHO_PRO_SUBSCRIBE_URL}?utm_source=phosphobot_app`}
                target="_blank"
                rel="noopener noreferrer"
              >
                Learn More
              </a>
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

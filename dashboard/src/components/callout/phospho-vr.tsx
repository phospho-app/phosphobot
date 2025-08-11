import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { RectangleGoggles, TestTubeDiagonal } from "lucide-react";

const PHOSPHO_PRO_SUBSCRIBE_URL = "https://phospho.ai/pro";

export function PhosphoVRCallout({ className }: { className?: string }) {
  return (
    <Card className={cn("border-green-500 border-2", className)}>
      <CardContent className="p-6">
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0">
            <RectangleGoggles className="text-green-500 size-8" />
          </div>
          <div className="flex-1 space-y-4">
            <div>
              <h3 className="font-semibold text-xl mb-2">
                Pick how to unlock VR control
              </h3>
              <p className="text-muted-foreground">
                You can unlock VR control in two ways: subscribe to Phospho Pro
                for full access, or purchase the Meta Store app for basic VR
                control. Choose the option that suits you best.
              </p>
            </div>

            <div className="grid md:grid-cols-2 gap-4">
              {/* Primary option - Subscription */}
              <div className="relative p-4 border-2 border-green-500 rounded-lg bg-green-50/50 dark:bg-green-950/20">
                {/* <div className="absolute -top-2 left-3 bg-green-500 text-white px-2 py-0.5 rounded text-xs font-medium">
                  RECOMMENDED
                </div> */}
                <div className="flex items-center gap-2 mb-2">
                  <TestTubeDiagonal className="size-5 text-green-600" />
                  <span className="font-semibold">Unlock with Phospho Pro</span>
                </div>
                <p className="text-sm text-muted-foreground mb-3">
                  Full access to VR control, advanced AI training, and all
                  premium features
                </p>
                <Button asChild className="w-full">
                  <a
                    href={`${PHOSPHO_PRO_SUBSCRIBE_URL}?utm_source=phosphobot_app`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Subscribe to Pro
                  </a>
                </Button>
              </div>

              {/* Alternative option - Meta Store */}
              <div className="p-4 border rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <RectangleGoggles className="size-5 text-muted-foreground" />
                  <span className="font-semibold">
                    Alternative: Buy the phospho teleoperation app on the Meta
                    Store
                  </span>
                </div>
                <p className="text-sm text-muted-foreground mb-3">
                  This only unlocks only VR control (no additional features)
                </p>
                <Button
                  asChild
                  variant="outline"
                  className="w-full bg-transparent"
                >
                  <a
                    href="https://www.meta.com/en-gb/experiences/phospho-teleoperation/8873978782723478/"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Free trial on the Meta Store
                  </a>
                </Button>
              </div>
            </div>

            <div className="p-4 bg-blue-50 dark:bg-blue-950/20 rounded-lg border border-blue-200 dark:border-blue-800 w-full">
              <p className="text-sm">
                <strong>Have a phospho starter pack?</strong> Contact us to get
                access to the VR app.
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

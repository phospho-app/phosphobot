"use client";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { fetchWithBaseUrl } from "@/lib/utils";
import { Loader2, Video } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { mutate } from "swr";

interface AddZMQCameraModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function AddZMQCameraModal({
  open,
  onOpenChange,
}: AddZMQCameraModalProps) {
  const [tcpAddress, setTcpAddress] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    
    if (isSubmitting) return; // Prevent double submission
    
    if (!tcpAddress.trim()) {
      toast.error("Please enter a TCP address");
      return;
    }

    // Basic validation for TCP address format
    if (!tcpAddress.startsWith("tcp://")) {
      toast.error("TCP address must start with 'tcp://'");
      return;
    }

    setIsSubmitting(true);

    const payload = {
      tcp_address: tcpAddress.trim(),
    };

    const response = await fetchWithBaseUrl(
      "/cameras/add-zmq",
      "POST",
      payload,
    );

    if (response) {
      toast.success("ZMQ camera has been added successfully.");

      // Close modal on success
      onOpenChange(false);

      // Reset form
      setTcpAddress("");

      // Refresh status and settings
      mutate("/status");
      mutate("/admin/settings");
    }

    setIsSubmitting(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Video className="h-5 w-5" />
            Add ZMQ Camera
          </DialogTitle>
          <DialogDescription>
            Add a new ZMQ camera feed by entering the TCP address of the ZMQ
            publisher.
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit}>
          <div className="grid gap-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="tcp-address">TCP Address</Label>
              <Input
                id="tcp-address"
                type="text"
                placeholder="tcp://localhost:5555"
                value={tcpAddress}
                onChange={(e) => setTcpAddress(e.target.value)}
                disabled={isSubmitting}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !isSubmitting) {
                    handleSubmit(e);
                  }
                }}
              />
              <p className="text-sm text-muted-foreground">
                Format: tcp://&lt;host&gt;:&lt;port&gt; (e.g.,
                tcp://localhost:5555)
              </p>
            </div>
          </div>
        </form>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isSubmitting}
          >
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={isSubmitting || !tcpAddress.trim()}
            className="min-w-[120px]"
          >
            {isSubmitting ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Adding...
              </>
            ) : (
              "Add Camera"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

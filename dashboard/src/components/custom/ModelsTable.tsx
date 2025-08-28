import { CopyButton } from "@/components/common/copy-button";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "@/components/ui/pagination";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { fetchWithBaseUrl, fetcher } from "@/lib/utils";
import { cn } from "@/lib/utils";
import type { SupabaseTrainingModel, TrainingsList } from "@/types";
import {
  ArrowUpDown,
  Ban,
  Check,
  ExternalLink,
  Loader2,
  ThumbsUp,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type React from "react";
import { toast } from "sonner";
import useSWR from "swr";

type ModelStatus = "succeeded" | "failed" | "running" | "canceled" | null;
type SortOption = "recent" | "success_rate" | "sessions_count";

interface ModelStatusFilterProps {
  onStatusChange: (status: ModelStatus) => void;
  className?: string;
}

export function ModelStatusFilter({
  onStatusChange,
  className,
}: ModelStatusFilterProps) {
  const [selectedStatus, setSelectedStatus] = useState<ModelStatus>(null);

  const handleStatusChange = (value: string) => {
    // If clicking the already selected status, clear the selection
    const newStatus = value === selectedStatus ? null : (value as ModelStatus);
    setSelectedStatus(newStatus);
    onStatusChange(newStatus);
  };

  return (
    <div className={cn("gap-y-2", className)}>
      <ToggleGroup
        type="single"
        value={selectedStatus || ""}
        onValueChange={handleStatusChange}
      >
        <ToggleGroupItem
          value="succeeded"
          aria-label="Filter by succeeded status"
          className={cn(
            "text-xs flex items-center gap-1.5",
            selectedStatus === "succeeded" &&
              "text-green-600 dark:text-green-500",
          )}
        >
          <Check className="h-4 w-4 text-green-600 dark:text-green-500" />
          <span>Succeeded</span>
        </ToggleGroupItem>
        <ToggleGroupItem
          value="failed"
          aria-label="Filter by failed status"
          className={cn(
            "text-xs flex items-center gap-1.5",
            selectedStatus === "failed" && "text-red-600 dark:text-red-500",
          )}
        >
          <X className="h-4 w-4 text-red-600 dark:text-red-500" />
          <span>Failed</span>
        </ToggleGroupItem>
        <ToggleGroupItem
          value="running"
          aria-label="Filter by running status"
          className={cn(
            "text-xs flex items-center gap-1.5",
            selectedStatus === "running" && "text-blue-600 dark:text-blue-500",
          )}
        >
          <Loader2 className="h-4 w-4 text-blue-600 dark:text-blue-500" />
          <span>Running</span>
        </ToggleGroupItem>
        <ToggleGroupItem
          value="canceled"
          aria-label="Filter by canceled status"
          className={cn(
            "flex items-center gap-1.5",
            selectedStatus === "canceled" && "text-gray-600 dark:text-gray-400",
          )}
        >
          <Ban className="h-4 w-4 text-gray-600 dark:text-gray-400" />
          <span>Canceled</span>
        </ToggleGroupItem>
      </ToggleGroup>
    </div>
  );
}

const ValueWithTooltip = ({ value }: { value: string }) => {
  // Don't use tooltip if value is short
  if (!value || value.length < 30) {
    return <span>{value}</span>;
  }

  // Show a preview of the first 20 characters after the last "/"
  const preview = value.split("/").pop()?.slice(0, 25) + "..." || value;

  return (
    <TooltipProvider>
      <Tooltip delayDuration={100}>
        <TooltipTrigger asChild>
          <span className="border-b border-dotted border-muted-foreground">
            {preview}
          </span>
        </TooltipTrigger>
        <TooltipContent className="max-w-md break-words">
          {value}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

// ModelRow component
const ModelRow: React.FC<{ model: SupabaseTrainingModel }> = ({ model }) => {
  const [isCanceling, setIsCanceling] = useState(false);

  // Set uppercase status for display
  const status = model.status.charAt(0).toUpperCase() + model.status.slice(1);

  // Define model url
  const url = "https://huggingface.co/" + model.model_name;

  const handleCancel = async () => {
    if (
      !confirm(
        "Are you sure you want to cancel this training? This action cannot be undone.",
      )
    ) {
      return;
    }

    setIsCanceling(true);
    const status_response = await fetchWithBaseUrl("/training/cancel", "POST", {
      training_id: model.id,
    });

    if (status_response?.status === "ok") {
      toast.success(status_response?.message);
    }
  };

  return (
    <>
      <TableRow>
        <TableCell className="w-20">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger>
                {status === "Succeeded" && (
                  <Check className="h-4 w-4 inline mr-1 text-green-500" />
                )}
                {status === "Running" && (
                  <Loader2 className="h-4 w-4 animate-spin inline mr-1" />
                )}
                {status === "Failed" && (
                  <X className="h-4 w-4 inline mr-1 text-red-500" />
                )}
                {status === "Canceled" && (
                  <Ban className="h-4 w-4 inline mr-1 text-gray-500" />
                )}
                <span className="text-sm">{status}</span>
              </TooltipTrigger>
              <TooltipContent>
                {status === "Succeeded" && (
                  <>Go to the AI Control page to try out this model</>
                )}
                {status === "Running" && (
                  <>Training is currently in progress.</>
                )}
                {status === "Failed" && (
                  <>Check the error log on the Hugging Face model page.</>
                )}
                {status === "Canceled" && <>Training was canceled.</>}
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </TableCell>
        <TableCell className="min-w-0 w-48">
          <div className="flex items-center flex-row justify-between">
            <div className="min-w-0 flex-1 mr-2">
              {ValueWithTooltip({ value: model.model_name })}
            </div>
            <div className="flex items-center flex-shrink-0">
              <CopyButton text={model.model_name} hint={"Copy model name"} />
              {/* Button to open model page */}
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      onClick={() => window.open(url, "_blank")}
                      title="Go to model"
                      aria-label="Go to model"
                      className="text-blue-500 hover:bg-blue-50 cursor-pointer"
                      variant="ghost"
                      size="icon"
                    >
                      <ExternalLink className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    Go to model page on Hugging Face
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              {/* Cancel button - only show for running models */}
              {model.status === "running" && (
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        onClick={handleCancel}
                        disabled={isCanceling}
                        variant="ghost"
                        size="icon"
                        className="text-orange-600 hover:bg-orange-50 hover:text-orange-700 dark:text-orange-400 dark:hover:bg-orange-950"
                      >
                        {isCanceling ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Ban className="h-4 w-4" />
                        )}
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      {isCanceling ? "Canceling..." : "Cancel training"}
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              )}
            </div>
          </div>
        </TableCell>
        <TableCell className="w-24">
          <span className="text-sm">{model.model_type}</span>
        </TableCell>

        <TableCell className="w-20 text-center">
          <span className="text-sm">{model.session_count}</span>
        </TableCell>
        <TableCell className="w-24 text-center">
          {model.success_rate !== null ? (
            <span className="text-sm">{model.success_rate.toFixed(2)}%</span>
          ) : (
            <span className="text-gray-500 text-sm">N/A</span>
          )}
        </TableCell>
        <TableCell className="w-56">
          <div className="text-sm space-y-1">
            {model.training_params &&
              Object.entries(model.training_params).map(
                ([key, value]) =>
                  value !== null && (
                    <div key={key} className="break-words">
                      <span className="font-medium">{key}:</span>{" "}
                      <span className="break-all">{String(value)}</span>
                    </div>
                  ),
              )}
          </div>
        </TableCell>
        <TableCell className="min-w-0 w-40">
          <div className="min-w-0">
            {ValueWithTooltip({ value: model.dataset_name })}
            <CopyButton text={model.dataset_name} hint={"Copy dataset name"} />
          </div>
        </TableCell>
        <TableCell className="w-36">
          <span className="text-sm">
            {new Date(model.requested_at).toLocaleString()}
          </span>
        </TableCell>
        <TableCell className="w-20 text-center">
          {model.used_wandb ? (
            <Check className="h-4 w-4 inline mr-1 text-green-500" />
          ) : (
            <X className="h-4 w-4 inline mr-1 text-red-500" />
          )}
          <span className="text-sm">{model.used_wandb ? "Yes" : "No"}</span>
        </TableCell>
      </TableRow>
    </>
  );
};

export const ModelsCard: React.FC = () => {
  const {
    data: modelsData,
    isLoading,
    error,
  } = useSWR<TrainingsList>(
    ["/training/models/read"],
    ([endpoint]) => fetcher(endpoint, "POST"),
    {
      refreshInterval: 5000,
    },
  );

  const models = modelsData?.models || [];
  const [statusFilter, setStatusFilter] = useState<ModelStatus>(null);
  const [sortBy, setSortBy] = useState<SortOption>("recent");

  // Filter and sort models
  const filteredAndSortedModels = useMemo(() => {
    let filtered = models;

    // Apply status filter
    if (statusFilter) {
      filtered = models.filter((model) => model.status === statusFilter);
    }

    // Apply sorting
    const sorted = [...filtered].sort((a, b) => {
      switch (sortBy) {
        case "recent":
          return (
            new Date(b.requested_at).getTime() -
            new Date(a.requested_at).getTime()
          );
        case "success_rate":
          // Handle null values - put them at the end
          if (a.success_rate === null && b.success_rate === null) return 0;
          if (a.success_rate === null) return 1;
          if (b.success_rate === null) return -1;
          return b.success_rate - a.success_rate;
        case "sessions_count":
          return b.session_count - a.session_count;
        default:
          return 0;
      }
    });

    return sorted;
  }, [models, statusFilter, sortBy]);

  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 20;
  const totalPages = Math.ceil(filteredAndSortedModels.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const currentModels = filteredAndSortedModels.slice(startIndex, endIndex);

  // Reset to first page when models or filter changes
  useEffect(() => {
    setCurrentPage(1);
  }, [filteredAndSortedModels.length]);

  return (
    <Card className="w-full">
      <CardContent className="flex flex-col gap-y-4">
        <div className="flex justify-between items-center w-full">
          <div className="flex items-center gap-2">
            <h2 className="font-bold">Trainings</h2>
            {filteredAndSortedModels.length > itemsPerPage && (
              <Pagination className="flex justify-start gap-x-2">
                <PaginationContent>
                  <PaginationItem>
                    <PaginationPrevious
                      onClick={() =>
                        setCurrentPage(Math.max(1, currentPage - 1))
                      }
                      className={
                        currentPage === 1
                          ? "text-xs pointer-events-none opacity-50"
                          : "text-xs cursor-pointer"
                      }
                    />
                  </PaginationItem>
                  {Array.from({ length: totalPages }, (_, i) => i + 1).map(
                    (page) => (
                      <PaginationItem key={page}>
                        <PaginationLink
                          onClick={() => setCurrentPage(page)}
                          isActive={currentPage === page}
                          className="text-xs cursor-pointer"
                        >
                          {page}
                        </PaginationLink>
                      </PaginationItem>
                    ),
                  )}
                  <PaginationItem>
                    <PaginationNext
                      onClick={() =>
                        setCurrentPage(Math.min(totalPages, currentPage + 1))
                      }
                      className={
                        currentPage === totalPages
                          ? "text-xs pointer-events-none opacity-50"
                          : "text-xs cursor-pointer"
                      }
                    />
                  </PaginationItem>
                </PaginationContent>
              </Pagination>
            )}
          </div>
          <div></div>
          <div className="flex items-center gap-4">
            <ModelStatusFilter onStatusChange={setStatusFilter} />
            {/* Sort by dropdown */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" className="shrink-0">
                  <ArrowUpDown className="w-4 h-4 mr-2" />
                  Sort by
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-[200px]" align="end">
                <DropdownMenuRadioGroup
                  value={sortBy}
                  onValueChange={(value) => setSortBy(value as SortOption)}
                >
                  <DropdownMenuRadioItem value="recent">
                    Most recent first
                  </DropdownMenuRadioItem>
                  <DropdownMenuRadioItem value="success_rate">
                    <ThumbsUp className="inline size-4 mr-1" />
                    Highest success rate
                  </DropdownMenuRadioItem>
                  <DropdownMenuRadioItem value="sessions_count">
                    Highest sessions count
                  </DropdownMenuRadioItem>
                </DropdownMenuRadioGroup>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>

        {isLoading ? (
          <div className="flex justify-center py-4">
            <Loader2 className="h-6 w-6 animate-spin" />
          </div>
        ) : error ? (
          <div className="flex flex-col items-center py-4">
            <p className="text-red-500">{error.toString()}</p>
          </div>
        ) : filteredAndSortedModels.length === 0 ? (
          <div className="text-center py-4">
            {statusFilter
              ? `No ${statusFilter} models found.`
              : "No models found."}
          </div>
        ) : (
          <div className="w-full overflow-x-auto">
            <div className="max-h-[50vh] overflow-y-auto border rounded-md">
              <Table className="min-w-full">
                <TableHeader className="sticky top-0 z-10 bg-background">
                  <TableRow>
                    <TableHead className="w-20">Status</TableHead>
                    <TableHead className="w-48">Model Name</TableHead>
                    <TableHead className="w-24">Model Type</TableHead>
                    <TableHead className="w-20 text-center">
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <span className="border-b border-dotted border-muted-foreground cursor-help">
                              Sessions
                            </span>
                          </TooltipTrigger>
                          <TooltipContent>
                            Number of AI control sessions ran with this model
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </TableHead>
                    <TableHead className="w-24 text-center">
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <span className="border-b border-dotted border-muted-foreground cursor-help">
                              <ThumbsUp className="inline size-4 mr-1" />
                              Success rate
                            </span>
                          </TooltipTrigger>
                          <TooltipContent>
                            Number of AI control sessions rated as successful
                            over all rated sessions.
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </TableHead>
                    <TableHead className="w-56">Training Parameters</TableHead>
                    <TableHead className="w-40">Dataset Name</TableHead>
                    <TableHead className="w-36">Created at</TableHead>
                    <TableHead className="w-20 text-center">Wandb</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {currentModels.map((model, index) => (
                    <ModelRow key={index} model={model} />
                  ))}
                </TableBody>
              </Table>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

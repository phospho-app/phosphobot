import { LoadingPage } from "@/components/common/loading";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useGlobalStore } from "@/lib/hooks";
import { fetchWithBaseUrl, fetcher } from "@/lib/utils";
import { DatasetInfoResponse, ServerStatus } from "@/types";
import {
  AlertCircle,
  ArrowUpFromLine,
  ChevronRight,
  Download,
  ExternalLink,
  Eye,
  File,
  Folder,
  LoaderCircle,
  MoreVertical,
  Plus,
  Repeat,
  Trash2,
  Wrench,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { redirect, useParams, useSearchParams } from "react-router-dom";
import { Link } from "react-router-dom";
import { toast } from "sonner";
import useSWR from "swr";

interface FileItem {
  name: string;
  path: string;
  absolute_path: string;
  is_dir: boolean;
  is_dataset_dir: boolean;
  browseUrl: string;
  huggingfaceUrl?: string;
  downloadUrl?: string;
  previewUrl?: string | null;
  canDeleteDataset?: boolean;
  deleteDatasetAction?: string;
}
interface BrowseData {
  directoryTitle: string;
  tokenError?: string;
  items: FileItem[];
  episode_ids?: number[];
  episode_paths?: string[];
}

interface DatasetInfos {
  [key: string]: DatasetInfoResponse | null;
}
interface MergeDialogProps {
  selectedItems: string[];
  datasetInfos: DatasetInfos;
  setMergeModalOpen: (open: boolean) => void;
  mergeMultipleDatasets: (
    mergedName: string,
    imageKeyMappings: Record<string, string>,
  ) => void;
}

const MergeDialog: React.FC<MergeDialogProps> = ({
  selectedItems,
  datasetInfos,
  setMergeModalOpen,
  mergeMultipleDatasets,
}) => {
  const [mergedDatasetName, setMergedDatasetName] = useState("");

  // Create state to track image key mappings
  const [imageKeyMappings, setImageKeyMappings] = useState<
    Record<string, string>
  >({});

  // Get source dataset image keys
  const sourceDatasetImageKeys =
    datasetInfos[selectedItems[0]]?.image_keys || [];
  // Get target dataset image keys
  const targetDatasetImageKeys =
    datasetInfos[selectedItems[1]]?.image_keys || [];

  // Handle image key selection change
  const handleImageKeyChange = (sourceKey: string, targetKey: string) => {
    setImageKeyMappings((prev) => ({
      ...prev,
      [sourceKey]: targetKey,
    }));
  };

  const handleMerge = () => {
    // Pass both the dataset name and the image key mappings
    mergeMultipleDatasets(mergedDatasetName, imageKeyMappings);
  };

  const allKeysMapped =
    sourceDatasetImageKeys.length > 0 &&
    sourceDatasetImageKeys.every((key) => !!imageKeyMappings[key]);

  const previewImageComponent = (
    imageKey: string,
    datasetInfos: DatasetInfos,
    sourceOrTarget: "Source" | "Target",
  ) => {
    return (
      <div className="border rounded-md p-2 w-32">
        <p className="text-xs mb-1">{sourceOrTarget}</p>
        <div className="h-20 flex items-center justify-center">
          <img
            src={`data:image/jpeg;base64,${datasetInfos[selectedItems[0]]?.image_frames?.[imageKey] ?? ""}`}
            alt={`Preview of ${imageKey}`}
            className="max-h-full max-w-full object-contain"
          />
        </div>
        <p className="text-xs mt-1 truncate">{imageKey}</p>
      </div>
    );
  };

  return (
    <DialogContent className="sm:max-w-2xl max-h-[110vh] overflow-y-auto">
      <DialogHeader>
        <DialogTitle>Merge Datasets</DialogTitle>
      </DialogHeader>

      <div className="grid gap-4 py-4">
        <div className="grid gap-2">
          <Label htmlFor="mergedDatasetName">New Dataset Name</Label>
          <Input
            id="mergedDatasetName"
            value={mergedDatasetName}
            onChange={(e) => {
              const value = e.target.value;
              // Allow only characters that are not whitespace or "/"
              if (/^[^\s/]*$/.test(value)) {
                setMergedDatasetName(value);
              }
            }}
            placeholder="Enter a name for the merged dataset"
            className="w-full"
          />
        </div>

        {sourceDatasetImageKeys.length > 0 && (
          <p className="text-sm text-muted-foreground">
            Map all image keys from dataset <code>{selectedItems[0]}</code> to
            their corresponding keys in dataset <code>{selectedItems[1]}</code>.
            {!allKeysMapped && sourceDatasetImageKeys.length > 0 && (
              <span className="text-accent ml-1">
                All image keys must be mapped before merging.
              </span>
            )}
          </p>
        )}

        <div className="space-y-4">
          {sourceDatasetImageKeys.length > 0 &&
            targetDatasetImageKeys.length > 0 &&
            sourceDatasetImageKeys.map((sourceKey) => (
              <div key={sourceKey} className="grid gap-2">
                <Label className="text-sm">
                  Match <code>{sourceKey}</code> from dataset{" "}
                  <code>{selectedItems[0]}</code>
                </Label>

                <div className="flex gap-3 items-center">
                  {/* Source image preview */}
                  {previewImageComponent(sourceKey, datasetInfos, "Source")}

                  <div className="text-center text-sm">→</div>

                  {/* Target image preview (only shows when selected) */}
                  {imageKeyMappings[sourceKey] ? (
                    <>
                      {previewImageComponent(
                        imageKeyMappings[sourceKey],
                        datasetInfos,
                        "Target",
                      )}
                    </>
                  ) : (
                    <div className="border border-dashed rounded-md p-2 h-32 w-32 flex items-center justify-center ">
                      <p className="text-xs text-muted ">Select target image</p>
                    </div>
                  )}

                  <div className="flex-grow">
                    <Select
                      onValueChange={(value) =>
                        handleImageKeyChange(sourceKey, value)
                      }
                    >
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="Select matching image key" />
                      </SelectTrigger>
                      <SelectContent className="max-h-60 overflow-y-auto">
                        {targetDatasetImageKeys.map((targetKey) => (
                          <SelectItem
                            key={targetKey}
                            value={targetKey}
                            className="py-2"
                          >
                            <div className="flex items-center gap-2">
                              <div className="w-12 h-9 flex items-center justify-center overflow-hidden rounded">
                                <img
                                  src={`data:image/jpeg;base64,${datasetInfos[selectedItems[1]]?.image_frames?.[targetKey] ?? ""}`}
                                  alt={`Preview of ${targetKey}`}
                                  className="max-h-full max-w-full object-contain"
                                />
                              </div>
                              <span className="truncate text-sm">
                                {targetKey}
                              </span>
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>
            ))}
        </div>
      </div>

      <DialogFooter>
        <Button variant="outline" onClick={() => setMergeModalOpen(false)}>
          Cancel
        </Button>
        <Button
          variant="destructive"
          onClick={handleMerge}
          disabled={!mergedDatasetName.trim() || !allKeysMapped}
        >
          Merge
        </Button>
      </DialogFooter>
    </DialogContent>
  );
};

export default function FileBrowser() {
  const params = useParams();
  const [searchParams] = useSearchParams();
  const path = params.path || searchParams.get("path") || "";

  const { data, error, mutate } = useSWR<BrowseData>(
    ["/files", path],
    ([url]) => fetcher(url, "POST", { path }),
  );

  const [confirmDeleteOpen, setConfirmDeleteOpen] = useState(false);
  const [confirmEpisodeDeleteOpen, setConfirmEpisodeDeleteOpen] =
    useState(false);
  const [selectedEpisode, setSelectedEpisode] = useState("");
  const leaderArmSerialIds = useGlobalStore(
    (state) => state.leaderArmSerialIds,
  );
  const [selectedItems, setSelectedItems] = useState<string[]>([]);
  const [datasetInfos, setDatasetInfos] = useState<
    Record<string, DatasetInfoResponse | null>
  >({});
  const [mergeModalOpen, setMergeModalOpen] = useState(false);
  const [openDownloadModal, setOpenDownloadModal] = useState(false);
  const [hfDatasetName, setHFDatasetName] = useState("");
  const [confirmRepairOpen, setConfirmRepairOpen] = useState(false);

  // Loading state for HF dataset download
  const [loadingDownloadDataset, setLoadingDownloadDataset] = useState(false);

  // Loading state for episode deletion
  const [loadingDeleteEpisode, setLoadingDeleteEpisode] = useState(false);

  const { data: serverStatus } = useSWR<ServerStatus>(
    ["/status"],
    ([url]) => fetcher(url),
    {
      refreshInterval: 5000,
    },
  );

  const fetchDatasetInfo = async (
    path: string,
  ): Promise<DatasetInfoResponse | null> => {
    const response = await fetchWithBaseUrl(
      `/dataset/info?path=${encodeURIComponent(path)}`,
      "POST",
    );
    if (response.status !== "ok") {
      return null;
    }
    return response as DatasetInfoResponse;
  };

  useEffect(() => {
    const fetchInfos = async () => {
      if (!data || !data.items) return;

      const infos: Record<string, DatasetInfoResponse | null> = {};
      await Promise.all(
        data.items
          .filter((item) => item.is_dataset_dir)
          .map(async (item) => {
            const info = await fetchDatasetInfo(item.path);
            infos[item.path] = info;
          }),
      );
      setDatasetInfos(infos);
    };

    fetchInfos();
  }, [data]);

  const isRobotConnected = useMemo(() => {
    return serverStatus?.robots && serverStatus.robots.length > 0;
  }, [serverStatus]);

  if (error)
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>Failed to load data.</AlertDescription>
      </Alert>
    );

  if (!data) return <LoadingPage />;

  const handleSelectItem = (path: string) => {
    setSelectedItems((prev) => {
      if (prev.includes(path)) {
        return prev.filter((item) => item !== path);
      } else {
        return [...prev, path];
      }
    });
  };

  const handleDeleteMultipleDatasets = async () => {
    if (selectedItems.length === 0) {
      toast.error("No datasets selected for deletion");
      return;
    }
    const deletePromises = selectedItems.map(async (item) => {
      const resp = await fetchWithBaseUrl(
        `/dataset/delete?path=${encodeURIComponent(item)}`,
        "POST",
      );
      if (resp.status !== "ok") {
        toast.error(`Failed to delete dataset: ${item}`);
        return;
      }
      toast.success(`Dataset deleted successfully: ${item}`);
    });
    await Promise.all(deletePromises);
    setConfirmDeleteOpen(false);
    setSelectedItems([]);
    mutate();
  };

  const handleMergeCheck = async () => {
    if (selectedItems.length !== 2) {
      toast.error("Please select exactly 2 datasets to merge");
      return;
    }

    if (
      datasetInfos[selectedItems[0]]?.robot_type !==
      datasetInfos[selectedItems[1]]?.robot_type
    ) {
      toast.error("Datasets have different robot types. Cannot merge.");
      return;
    }

    if (
      datasetInfos[selectedItems[0]]?.robot_dof !==
      datasetInfos[selectedItems[1]]?.robot_dof
    ) {
      toast.error("Datasets have different DOF. Cannot merge.");
      return;
    }

    if (
      datasetInfos[selectedItems[0]]?.image_keys?.length !==
      datasetInfos[selectedItems[1]]?.image_keys?.length
    ) {
      toast.error(
        "Datasets have different number of image keys. Cannot merge.",
      );
      return;
    }

    setMergeModalOpen(true);
  };

  const mergeMultipleDatasets = async (
    newDatasetName: string,
    imageKeyMappings?: Record<string, string>,
  ) => {
    fetchWithBaseUrl(`/dataset/merge`, "POST", {
      first_dataset: selectedItems[0],
      second_dataset: selectedItems[1],
      new_dataset_name: newDatasetName,
      image_key_mappings: imageKeyMappings,
    }).then((response) => {
      if (response.status !== "ok") {
        toast.error("Failed to merge datasets");
      } else {
        toast.success("Episode merged successfully");
        mutate();
        redirect(path);
      }
    });
    setMergeModalOpen(false);
  };

  const handleRepairDataset = async () => {
    if (selectedItems.length === 0) {
      toast.error("No datasets selected for repair");
      return;
    }
    const repairPromises = selectedItems.map(async (item) => {
      console.log("Repairing dataset:", item);
      const resp = await fetchWithBaseUrl(`/dataset/repair`, "POST", {
        dataset_path: item,
      });
      if (resp.status !== "ok") {
        toast.error(`Failed to repair dataset: ${item}`);
        return;
      }
      toast.success(`Dataset repaired successfully: ${item}`);
    });
    await Promise.all(repairPromises);
    setConfirmRepairOpen(false);
    setSelectedItems([]);
    mutate();
  };

  const handleDeleteEpisode = async () => {
    // Ensure an episode is selected.
    if (!selectedEpisode) return;
    fetchWithBaseUrl(`/episode/delete`, "POST", {
      path: path,
      episode_id: selectedEpisode,
    }).then((response) => {
      if (response.status !== "ok") {
        toast.error("Failed to delete episode");
      } else {
        toast.success("Episode deleted successfully");
        mutate();
        redirect(path);
      }
    });
  };

  const handleReplayEpisode = async () => {
    if (!selectedEpisode || !data.episode_paths) return;
    const episode_path = data.episode_paths[parseInt(selectedEpisode)];
    if (!episode_path) return;
    const robot_serials_to_ignore = leaderArmSerialIds ?? null;
    fetchWithBaseUrl(`/recording/play`, "POST", {
      episode_path,
      robot_serials_to_ignore,
    });
  };

  const handlePushToHub = async (item: FileItem) => {
    fetchWithBaseUrl(`/dataset/sync?path=${item.path}`, "POST");
  };

  const pathParts = path.split("/").filter(Boolean);

  return (
    <div className="container mx-auto py-6 px-4 md:px-6">
      <div className="flex items-center justify-between mb-4">
        <Breadcrumb className="mb-4">
          <BreadcrumbList>
            <BreadcrumbItem>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <BreadcrumbLink asChild>
                      <Link to="/browse">phosphobot</Link>
                    </BreadcrumbLink>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>
                      Explore your datasets in{" "}
                      <code>~/phosphobot/recordings</code>
                    </p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </BreadcrumbItem>
            <BreadcrumbSeparator>
              <ChevronRight className="h-4 w-4" />
            </BreadcrumbSeparator>
            {pathParts.map((part, index) => (
              <BreadcrumbItem key={index}>
                <BreadcrumbLink asChild>
                  <Link
                    to={`/browse?path=${pathParts.slice(0, index + 1).join("/")}`}
                  >
                    {part}
                  </Link>
                </BreadcrumbLink>
                {index < pathParts.length - 1 && (
                  <BreadcrumbSeparator>
                    <ChevronRight className="h-4 w-4" />
                  </BreadcrumbSeparator>
                )}
              </BreadcrumbItem>
            ))}
          </BreadcrumbList>
        </Breadcrumb>

        <Button variant="outline" onClick={() => setOpenDownloadModal(true)}>
          <Plus className="mr-2 h-4 w-4" />
          Download dataset
        </Button>
      </div>

      {data.tokenError && (
        <Alert variant="destructive" className="mb-4">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{data.tokenError}</AlertDescription>
        </Alert>
      )}

      <Table className="bg-background rounded-lg">
        <TableHeader>
          <TableRow>
            <TableCell className="w-[50px]" />
            <TableCell>Name</TableCell>
            {path.endsWith("lerobot_v2") || path.endsWith("lerobot_v2.1") ? (
              <>
                <TableCell className="text-muted-foreground">
                  Robot Type
                </TableCell>
                <TableCell className="text-muted-foreground">DOF</TableCell>
                <TableCell className="text-muted-foreground">
                  Episodes
                </TableCell>
                <TableCell className="text-muted-foreground">
                  Image Keys
                </TableCell>
              </>
            ) : null}
            <TableCell></TableCell>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.items.map((item) => (
            <TableRow key={item.path}>
              {/* Show checkbox only if path ends with 'lerobot_v2.1' or lerobot_v2 */}
              <TableCell className="w-[50px]">
                {item.is_dataset_dir ? (
                  <Checkbox
                    checked={selectedItems.includes(item.path)}
                    onCheckedChange={() => handleSelectItem(item.path)}
                    aria-label={`Select ${item.name}`}
                    key={item.path}
                  />
                ) : null}
              </TableCell>
              <TableCell>
                <Link
                  to={item.browseUrl}
                  className="flex items-center text-blue-500 hover:underline"
                >
                  {item.is_dir ? (
                    <Folder className="mr-2 h-4 w-4" />
                  ) : (
                    <File className="mr-2 h-4 w-4" />
                  )}
                  {item.name}
                </Link>
              </TableCell>
              <TableCell className="text-muted-foreground">
                {datasetInfos[item.path]?.robot_type &&
                  datasetInfos[item.path]?.robot_type}
              </TableCell>
              <TableCell className="text-muted-foreground">
                {datasetInfos[item.path]?.robot_dof &&
                  datasetInfos[item.path]?.robot_dof}
              </TableCell>
              <TableCell className="text-muted-foreground">
                {datasetInfos[item.path]?.number_of_episodes &&
                  datasetInfos[item.path]?.number_of_episodes}
              </TableCell>
              <TableCell className="text-muted-foreground">
                {datasetInfos[item.path]?.image_keys &&
                  datasetInfos[item.path]?.image_keys?.length}
              </TableCell>
              <TableCell>
                <div className="flex space-x-2 justify-end">
                  {(item.downloadUrl || item.canDeleteDataset) && (
                    <DropdownMenu>
                      {item.previewUrl && (
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <a
                                href={item.previewUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex"
                              >
                                <Button
                                  variant={"outline"}
                                  className="cursor-pointer"
                                  size="sm"
                                >
                                  <Eye className="mr-2 h-4 w-4" />
                                  Preview
                                </Button>
                              </a>
                            </TooltipTrigger>
                            <TooltipContent>
                              <p>
                                Please use HVC1 or AVC1 codec for videos to be
                                visible in LeRobot dataset viewer.
                              </p>
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      )}

                      <DropdownMenuTrigger asChild className="cursor-pointer">
                        <Button variant="outline" size="sm">
                          <MoreVertical className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        {item.huggingfaceUrl && (
                          <DropdownMenuItem asChild>
                            <a
                              href={item.huggingfaceUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="flex items-center cursor-pointer"
                            >
                              <ExternalLink className="mr-2 h-4 w-4" />
                              See on Hugging Face Hub
                            </a>
                          </DropdownMenuItem>
                        )}
                        {item.downloadUrl && item.is_dataset_dir && (
                          <DropdownMenuItem
                            onClick={() => handlePushToHub(item)}
                            className="cursor-pointer"
                          >
                            <ArrowUpFromLine className="mr-2 h-4 w-4" />
                            Push to Hugging Face Hub
                          </DropdownMenuItem>
                        )}
                        {item.downloadUrl && (
                          <DropdownMenuItem asChild>
                            <a
                              href={item.downloadUrl}
                              className="flex items-center cursor-pointer"
                            >
                              <Download className="mr-2 h-4 w-4" />
                              Download
                            </a>
                          </DropdownMenuItem>
                        )}
                      </DropdownMenuContent>
                    </DropdownMenu>
                  )}
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

      {data.episode_ids && data.episode_ids.length > 0 && (
        <div className="mt-6">
          <div className="flex items-end gap-2">
            <Select
              value={selectedEpisode}
              name="episode_id"
              onValueChange={setSelectedEpisode}
            >
              <SelectTrigger id="episode-select">
                <SelectValue placeholder="Select Episode ID" />
              </SelectTrigger>
              <SelectContent>
                {data.episode_ids.map((episode) => (
                  <SelectItem key={episode} value={episode.toString()}>
                    Episode {episode}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              type="button"
              variant="destructive"
              onClick={() => {
                if (!selectedEpisode) return;
                setConfirmEpisodeDeleteOpen(true);
              }}
            >
              <Trash2 className="mr-2 h-4 w-3" />
              Delete Episode
            </Button>
            {isRobotConnected && (
              <Button
                type="button"
                variant="default"
                onClick={async () => {
                  await handleReplayEpisode();
                }}
              >
                <Repeat className="mr-2 h-4 w-3" />
                Replay Episode
              </Button>
            )}
          </div>
        </div>
      )}

      {selectedItems.length > 0 &&
        (path.endsWith("lerobot_v2") || path.endsWith("lerobot_v2.1")) && (
          <div className="flex flex-col md:flex-row md:space-x-2 mt-6">
            {path.endsWith("lerobot_v2.1") && (
              <Button
                className="mb-4"
                variant="outline"
                onClick={() => handleMergeCheck()}
              >
                <Repeat className="mr-2 h-4 w-3" />
                Merge Selected Datasets
              </Button>
            )}
            <Button
              className="mb-4"
              variant="outline"
              onClick={() => setConfirmRepairOpen(true)}
            >
              <Wrench className="mr-2 h-4 w-3" />
              Repair Selected Datasets
            </Button>
            <Button
              className="mb-4"
              variant="destructive"
              onClick={() => setConfirmDeleteOpen(true)}
            >
              <Trash2 className="mr-2 h-4 w-3" />
              Delete Selected Datasets
            </Button>
          </div>
        )}

      {/* Dataset deletion dialog */}
      <Dialog open={confirmDeleteOpen} onOpenChange={setConfirmDeleteOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Confirm Delete Dataset</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete the selected datasets:
              <br />
              {selectedItems.length > 0
                ? selectedItems.map((item) => (
                    <span key={item}>
                      <strong>{item}</strong>
                      <br />
                    </span>
                  ))
                : null}
              This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setConfirmDeleteOpen(false)}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDeleteMultipleDatasets}
            >
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Episode deletion dialog */}
      <Dialog
        open={confirmEpisodeDeleteOpen}
        onOpenChange={setConfirmEpisodeDeleteOpen}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Confirm Delete Episode</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete episode{" "}
              <strong>{selectedEpisode}</strong>? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setConfirmEpisodeDeleteOpen(false)}
              disabled={loadingDeleteEpisode}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={async () => {
                setLoadingDeleteEpisode(true);
                await handleDeleteEpisode();
                setConfirmEpisodeDeleteOpen(false);
                setLoadingDeleteEpisode(false);
                redirect(path);
              }}
              disabled={loadingDeleteEpisode}
            >
              {loadingDeleteEpisode ? (
                <LoaderCircle> Deleting...</LoaderCircle>
              ) : (
                "Delete"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Download modal */}
      <Dialog open={openDownloadModal} onOpenChange={setOpenDownloadModal}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Dataset Download</DialogTitle>
            <DialogDescription>
              Enter the Hugging Dace dataset name to download: should be
              hf_name/dataset_name
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <Label htmlFor="dataset-select">Select Dataset</Label>
            <Input
              id="dataset-select"
              value={hfDatasetName}
              onChange={(e) => {
                const value = e.target.value;
                // Allow only characters that are not whitespace
                if (/^[^\s]*$/.test(value)) {
                  setHFDatasetName(value);
                }
              }}
              placeholder="Enter the name of the dataset to download"
              className="w-full"
            />
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setOpenDownloadModal(false)}
            >
              Close
            </Button>
            <Button
              variant="default"
              onClick={async () => {
                if (hfDatasetName.trim() === "") {
                  toast.error("No dataset selected for download");
                  return;
                }
                setLoadingDownloadDataset(true);
                const resp = await fetchWithBaseUrl(
                  `/dataset/hf_download`,
                  "POST",
                  {
                    dataset_name: hfDatasetName,
                  },
                );
                if (resp.status !== "ok") {
                  toast.error("Failed to download dataset: " + resp.message);
                } else {
                  toast.success("Dataset downloaded successfully");
                }
                setLoadingDownloadDataset(false);
                setOpenDownloadModal(false);
                mutate();
                redirect(path);
              }}
              disabled={loadingDownloadDataset}
            >
              {loadingDownloadDataset ? (
                <span className="flex items-center">
                  <LoaderCircle className="animate-spin size-5 mr-1" />
                  Downloading...
                </span>
              ) : (
                "Download"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Dataset repair dialog */}
      <Dialog open={confirmRepairOpen} onOpenChange={setConfirmRepairOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Repair Dataset</DialogTitle>
            <DialogDescription>
              This will attempt to repair the selected datasets:
              <br />
              {selectedItems.length > 0
                ? selectedItems.map((item) => (
                    <span key={item}>
                      <strong>{item}</strong>
                      <br />
                    </span>
                  ))
                : null}
              For now, this will only recalculate the parquets files, not the
              meta data.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setConfirmRepairOpen(false)}
            >
              Cancel
            </Button>
            <Button variant="destructive" onClick={() => handleRepairDataset()}>
              Repair
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Merge modal */}
      <Dialog open={mergeModalOpen} onOpenChange={setMergeModalOpen}>
        <MergeDialog
          selectedItems={selectedItems}
          datasetInfos={datasetInfos}
          setMergeModalOpen={setMergeModalOpen}
          mergeMultipleDatasets={mergeMultipleDatasets}
        />
      </Dialog>
    </div>
  );
}

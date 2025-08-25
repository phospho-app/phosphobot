import EpisodeViewer from "./episode-viewer";
import { getEpisodeDataSafe } from "./fetch-data";
import { Metadata } from 'next';


export const dynamic = "force-dynamic";

export async function generateMetadata({
  params,
}: {
  params: Promise<{ org: string; dataset: string; episode: string }>;
}): Promise<Metadata> {
  const { org, dataset, episode } = await params; // Await params here
  return {
    title: `${org}/${dataset} | episode ${episode}`,
  };
}

export default async function EpisodePage({
  params,
}: {
  params: Promise<{ org: string; dataset: string; episode: string }>;
}) {
  const { org, dataset, episode } = await params;
  // fetchData should be updated if needed to support this path pattern
  const episodeNumber = Number(episode.replace(/^episode_/, ""));
  const { data, error } = await getEpisodeDataSafe(org, dataset, episodeNumber);
  return <EpisodeViewer data={data} error={error} />;
}

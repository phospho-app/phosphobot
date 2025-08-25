import { redirect } from "next/navigation";

export default async function DatasetRootPage({
  params,
}: {
  params: Promise<{ org: string; dataset: string }>;
}) {
  const { org, dataset } = await params;
  const episodeN = process.env.EPISODES
    ?.split(/\s+/)
    .map((x) => parseInt(x.trim(), 10))
    .filter((x) => !isNaN(x))[0] ?? 0;

  redirect(`/${org}/${dataset}/episode_${episodeN}`);
}

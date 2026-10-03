import { MeetingReport } from "@/components/report/MeetingReport";

export default async function MeetingReportPage({
  params,
}: {
  params: Promise<{ meetingId: string }>;
}) {
  const { meetingId } = await params;
  return <MeetingReport meetingId={meetingId} />;
}

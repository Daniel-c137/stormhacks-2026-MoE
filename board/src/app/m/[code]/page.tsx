import { MeetingEntry } from "@/components/room/MeetingEntry";

// The shareable meeting link: lobby first, then the room.
export default async function MeetingLinkPage({ params }: { params: Promise<{ code: string }> }) {
  const { code } = await params;
  return <MeetingEntry code={code} />;
}

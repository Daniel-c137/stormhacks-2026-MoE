// Typed LiveKit data messages. Payload type per topic comes from the contract.
import type { Topic, TopicPayloads } from "@moe/contracts";
import type { Room } from "livekit-client";
import { notImplemented } from "./stub";

/** to=undefined broadcasts to the room. */
export async function publish<T extends Topic>(
  room: Room,
  topic: T,
  payload: TopicPayloads[T],
  to?: string[],
): Promise<void> {
  notImplemented("publish");
}

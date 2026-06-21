import { NextResponse } from "next/server";
import { getActiveFires } from "@/lib/data";

// Runtime-dynamic: depends on the FIRMS key + live conditions, so never frozen
// at build. The upstream FIRMS fetch is itself cached for 15 min in getActiveFires.
export const dynamic = "force-dynamic";

export async function GET() {
  const fc = await getActiveFires();
  return NextResponse.json(fc);
}

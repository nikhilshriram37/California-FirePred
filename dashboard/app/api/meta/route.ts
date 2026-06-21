import { NextResponse } from "next/server";
import { getMeta } from "@/lib/data";

export const revalidate = 300;

export async function GET() {
  const meta = await getMeta();
  return NextResponse.json(meta);
}

import { NextRequest, NextResponse } from "next/server";
import { getForecast } from "@/lib/data";

export const revalidate = 300;

export async function GET(req: NextRequest) {
  const h = Number(req.nextUrl.searchParams.get("h") ?? "0");
  const horizon = Math.min(5, Math.max(0, Number.isFinite(h) ? h : 0));
  return NextResponse.json(await getForecast(horizon));
}

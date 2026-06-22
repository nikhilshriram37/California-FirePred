import { NextResponse } from "next/server";
import { getForecastInfo } from "@/lib/data";

export const revalidate = 300;

export async function GET() {
  return NextResponse.json(await getForecastInfo());
}

import { NextResponse } from "next/server";
import { getRiskGeoJSON } from "@/lib/data";

export const revalidate = 300; // 5 min

export async function GET() {
  const fc = await getRiskGeoJSON();
  return NextResponse.json(fc);
}

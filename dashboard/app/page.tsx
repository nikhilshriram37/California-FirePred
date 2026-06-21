import Dashboard from "@/components/Dashboard";

// The dashboard fetches from the API routes client-side so live layers (fires)
// can refresh without a full reload.
export default function Page() {
  return <Dashboard />;
}

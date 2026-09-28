import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The app is a single client-rendered page that talks to the FastAPI backend
  // via NEXT_PUBLIC_API_URL, so it ships as static files (frontend/out).
  output: "export",
  images: { unoptimized: true },
};

export default nextConfig;

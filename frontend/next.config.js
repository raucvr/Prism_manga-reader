/** @type {import('next').NextConfig} */
const nextConfig = {
  // The UI shows plain <img> tags and never uses next/image: with this, /_next/image answers 404, which keeps
  // the image optimizer (the subject of several Next.js 14 advisories) out of reach.
  images: {
    unoptimized: true,
  },
  // Long-running proxied requests (PDF uploads, legacy synchronous generation)
  experimental: {
    proxyTimeout: 6000000, // 100 minutes
  },
  // Proxy the API and the generated files (/output) to the backend.
  // NEXT_PUBLIC_API_URL is read when the dev server starts / at build time. The default is 127.0.0.1, not
  // localhost: Node 18 may resolve localhost to ::1, and the backend listens on IPv4 loopback only.
  async rewrites() {
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/api/:path*`,
      },
      {
        source: "/output/:path*",
        destination: `${backendUrl}/output/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;

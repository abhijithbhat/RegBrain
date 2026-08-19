/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/api/backend/:path*",
        destination: "https://regbrain.onrender.com/:path*",
      },
    ];
  },
};

export default nextConfig;

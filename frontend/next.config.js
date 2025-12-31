/** @type {import('next').NextConfig} */
const nextConfig = {
  // 支持 react-pdf
  webpack: (config) => {
    config.resolve.alias.canvas = false;
    return config;
  },
  // 增加 API 请求体大小限制
  api: {
    bodyParser: {
      sizeLimit: '50mb',
    },
    responseLimit: '50mb',
  },
  // 实验性功能：增加代理超时
  experimental: {
    proxyTimeout: 6000000, // 100 minutes for long manga generation
  },
  // API 代理到后端
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
    ];
  },
};

module.exports = nextConfig;

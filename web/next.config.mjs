/**
 * Static export: the site has no server, no database and no secrets.
 *
 * Everything it shows comes from public/data/forecast.json, which the Python
 * pipeline rewrites after each training run. That makes the output a plain
 * folder of HTML, deployable to Vercel, GitHub Pages or any static host.
 *
 * BASE_PATH is set when deploying under a repository subpath (GitHub Pages
 * serves at /<repo>/); it is empty for Vercel and for local development.
 */
const basePath = process.env.BASE_PATH ?? "";

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  basePath,
  // A static host cannot run the image optimiser.
  images: { unoptimized: true },
  // Directory-style URLs so a static host serves /method as /method/index.html.
  trailingSlash: true,
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
};

export default nextConfig;

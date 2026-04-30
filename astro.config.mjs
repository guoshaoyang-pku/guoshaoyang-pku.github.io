import { defineConfig } from "astro/config";

export default defineConfig({
  site: "https://guoshaoyang-pku.github.io",
  output: "static",
  build: {
    format: "directory",
  },
});

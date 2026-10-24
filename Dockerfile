# syntax=docker/dockerfile:1

# ---- build: frontend bundle ----
FROM node:24-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY index.html vite.config.ts tsconfig.json ./
COPY src ./src
COPY shared ./shared
RUN npm run build

# ---- runtime ----
FROM node:24-alpine
WORKDIR /app
ENV NODE_ENV=production \
    PORT=3000
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --no-audit --no-fund --ignore-scripts && npm cache clean --force
COPY --from=build /app/dist ./dist
COPY server.ts ./
COPY shared ./shared
# Only precomputed artifacts and the demo statement; raw train/test stay out of the image
COPY data/demo_statement_c052.csv ./data/
USER node
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD wget -qO- "http://127.0.0.1:${PORT}/api/health" || exit 1
# Node 24 runs server.ts directly (built-in type stripping), no tsx needed at runtime
CMD ["node", "server.ts"]

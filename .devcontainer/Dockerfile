FROM docker:cli AS docker-cli
FROM docker/buildx-bin AS buildx

FROM node:lts-bookworm-slim AS node

FROM python:3.11-bookworm

# CLI DOCKER (non serve installarlo completo perché usiamo daemon dell'host - vd riga 25 di devcontainer.json)
COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=buildx /buildx /usr/local/lib/docker/cli-plugins/docker-buildx

COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/bin/npm /usr/local/bin/npm
COPY --from=node /usr/local/bin/npx /usr/local/bin/npx
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=node /usr/local/include/node /usr/local/include/node
COPY --from=node /usr/local/share/man/man1/node* /usr/local/share/man/man1/

# set poppler per utilizzare OCR
RUN apt-get update && apt-get install -y poppler-utils tesseract-ocr

# RUN pip install --upgrade pip

RUN pip install uv dynaconf 

WORKDIR /workspace




# ARG DOCKER_GID=997
# RUN set -eux; \
#   if getent group docker >/dev/null; then \
#   groupmod -o -g "${DOCKER_GID}" docker; \
#   else \
#   groupadd -g "${DOCKER_GID}" docker; \
#   fi; \
#   usermod -aG docker "${VC_USERNAME}"

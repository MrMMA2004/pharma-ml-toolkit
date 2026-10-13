# PharmaML Toolkit — runtime image.
#
# The toolkit is dependency-light by design: RDKit and the boosting
# libraries are optional extras.  Build one of the flavored targets below.
#
#   docker build -t pharmaml .                              # core
#   docker build --build-arg EXTRAS=chem,boosting -t pharmaml:full .
#
FROM python:3.12-slim AS base

ARG EXTRAS=""
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy the package first so dependency layers cache cleanly.
COPY pyproject.toml README.md requirements.txt ./
COPY src ./src

RUN pip install --no-cache-dir .${EXTRAS:+[${EXTRAS}]}

# Examples and tests are mounted or copied on demand.
COPY examples ./examples
COPY tests ./tests

# Drop privileges for a plain, non-root runtime user.
RUN useradd --create-home pharmaml && chown -R pharmaml:pharmaml /app
USER pharmaml

ENTRYPOINT ["pharmaml"]
CMD ["--help"]

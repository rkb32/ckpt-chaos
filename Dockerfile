# Linux runs: real POSIX rename/fsync semantics on ext4 (volume) or overlayfs (container layer).
#   docker build -t ckpt-chaos .                                   # loop_ddp only (small)
#   docker build -t ckpt-chaos --build-arg EXTRAS="transformers accelerate" .
#   docker run --rm ckpt-chaos loop_ddp --ranks 2 --strategy tmp_rename --flake 25
#   docker run --rm -v ckptchaos-runs:/app/runs ckpt-chaos ...      # runs/ on a volume instead of overlayfs
FROM python:3.12-slim
WORKDIR /app
ARG EXTRAS=""
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && if [ -n "$EXTRAS" ]; then pip install --no-cache-dir $EXTRAS; fi
COPY ckpt_chaos ./ckpt_chaos
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "ckpt_chaos"]

FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ARG PYTHONPATH="tmp"

SHELL ["/bin/bash", "-c"]

RUN : \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.10-venv \
        libglfw3 \
        libglfw3-dev \
        libglew-dev \
        libgl1-mesa-glx \
        libosmesa6 \
        libgl1-mesa-dri \
        python3-dev \
        git-all \
        libnvidia-egl-wayland1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && :


COPY environment.yml /tmp/environment.yml
RUN conda env create -n ogbench -f /tmp/environment.yml

COPY --link . /workspaces

ENV PYTHONPATH=/workspaces:$PYTHONPATH
ENV MUJOCO_GL=egl 
ENV PYOPENGL_PLATFORM=egl
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV PATH=/opt/conda/bin:$PATH


RUN mkdir /scratch

WORKDIR /workspaces

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ogbench"]
CMD ["python", "main.py"]

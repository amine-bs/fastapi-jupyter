FROM inseefrlab/onyxia-jupyter-python:py3.9.13-gpu

COPY . /home/onyxia/work/
RUN rm Dockerfile README.md
RUN rm -r .github
EXPOSE 8888

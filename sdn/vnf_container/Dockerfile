FROM python:3.10-slim

WORKDIR /opt/hrl-vnf
COPY sdn/vnf_forwarder.py /opt/hrl-vnf/vnf_forwarder.py

RUN useradd --create-home --uid 10001 vnf
USER vnf

ENV PYTHONUNBUFFERED=1
HEALTHCHECK --interval=1s --timeout=1s --retries=10 CMD python3 -c "import os,socket; s=socket.create_connection(('127.0.0.1',int(os.environ['VNF_HEALTH_PORT'])),1); s.close()"
ENTRYPOINT ["python3", "/opt/hrl-vnf/vnf_forwarder.py"]

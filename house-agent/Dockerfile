FROM langchain/langgraph-api:3.11-wolfi

ARG PYPI_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple
ENV UV_DEFAULT_INDEX=${PYPI_INDEX_URL}
ENV UV_INDEX_URL=${PYPI_INDEX_URL}
ENV PIP_INDEX_URL=${PYPI_INDEX_URL}



# -- Adding local package . --
ADD . /deps/house-agent
# -- End of local package . --

# -- Installing all local dependencies --
RUN for dep in /deps/*; do             echo "Installing $dep";             if [ -d "$dep" ]; then                 echo "Installing $dep";                 (cd "$dep" && PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt -e .);             fi;         done
# -- End of local dependencies install --
ENV LANGSERVE_GRAPHS='{"house_agent": "/deps/house-agent/src/agent/graph.py:graph", "recommended_agent": "/deps/house-agent/src/agent/recommend.py:recommended_graph", "reserve_agent": "/deps/house-agent/src/agent/reserve.py:reserve_graph", "extend_agent": "/deps/house-agent/src/agent/extend.py:extend_graph"}'



# -- Ensure user deps didn't inadvertently overwrite langgraph-api
RUN mkdir -p /api/langgraph_api /api/langgraph_runtime /api/langgraph_license && touch /api/langgraph_api/__init__.py /api/langgraph_runtime/__init__.py /api/langgraph_license/__init__.py
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir --no-deps -e /api
# -- End of ensuring user deps didn't inadvertently overwrite langgraph-api --
# -- Removing build deps from the final image ~<:===~~~ --
RUN pip uninstall -y pip setuptools wheel
RUN rm -rf /usr/local/lib/python*/site-packages/pip* /usr/local/lib/python*/site-packages/setuptools* /usr/local/lib/python*/site-packages/wheel* && find /usr/local/bin -name "pip*" -delete || true
RUN rm -rf /usr/lib/python*/site-packages/pip* /usr/lib/python*/site-packages/setuptools* /usr/lib/python*/site-packages/wheel* && find /usr/bin -name "pip*" -delete || true
RUN uv pip uninstall --system pip setuptools wheel && rm /usr/bin/uv /usr/bin/uvx

WORKDIR /deps/house-agent

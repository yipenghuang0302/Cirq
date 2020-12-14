FROM ubuntu
LABEL maintainer="yipeng.huang@rutgers.edu"

RUN dpkg --add-architecture i386

# apt-get noninteractive for tzdata
RUN export DEBIAN_FRONTEND=noninteractive; \
  export DEBCONF_NONINTERACTIVE_SEEN=true; \
  echo 'tzdata tzdata/Areas select Etc' | debconf-set-selections; \
  echo 'tzdata tzdata/Zones/Etc select UTC' | debconf-set-selections; \
  apt-get update && apt-get install -qqy --no-install-recommends \
    build-essential \
    cmake \
    curl \
    default-jdk \
    git \
    libc6:i386 \
    libtool \
    make \
    python3-dev \
    python3-pip \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /common/home/yh804/research/

# Set up Google Cirq
RUN mkdir Google
RUN git -C ./Google/ clone https://github.com/yipenghuang0302/Cirq.git
RUN python3 -m pip install -e ./Google/Cirq/[dev_env]

# Set up Bayesian network to CNF compiler
RUN git clone https://github.com/yipenghuang0302/bayes-to-cnf.git
RUN make -C ./bayes-to-cnf/

# Set up ACE CNF to d-DNNF compiler
RUN git clone https://github.com/yipenghuang0302/qACE.git
COPY ./ace_v3.0_linux86.tar.gz ./qACE/
RUN tar -C ./qACE/ -zxf ./qACE/ace_v3.0_linux86.tar.gz

# Set up Java commons complex math library for evaluator
RUN curl -SL https://ftp.wayne.edu/apache//commons/math/binaries/commons-math3-3.6.1-bin.tar.gz \
  | tar -zxC ./qACE/

# Set up comparison target quantum circuit simulator: qsim Schrodinger style simulator
RUN git -C ./Google/ clone https://github.com/yipenghuang0302/qsim.git
RUN make -C ./Google/qsim/ run-tests
RUN python3 -m pip install -e ./Google/qsim/[dev_env]

# Set up comparison target quantum circuit simulator: qtorch tensor network simulator
RUN git clone https://github.com/aspuru-guzik-group/qtorch.git
RUN make -C ./qtorch/ installlocal
RUN make -C ./qtorch/
ENV PATH="/common/home/yh804/research/qtorch/bin/:${PATH}"

RUN git -C ./Google/Cirq/ pull

RUN pytest ./Google/Cirq/cirq/sim/kc_sparse_simulator_test.py
RUN pytest ./Google/Cirq/kc_examples/kc_examples_test.py
RUN pytest ./Google/Cirq/kc_examples/kc_examples_perf_test.py

ENTRYPOINT ["bash"]

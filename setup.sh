#!/bin/bash -eu

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -eu

## Usage: ./setup.sh [--inference] [--skip-mvfst-deps]

# Note: Pantheon requires python 2.7 while torchbeast needs python3.7.
# Make sure your default python in conda env in python2.7 with an explicit
# python3 command pointing to python 3.7
# ArgumentParser
MAHI_TUNNEL=false
INFERENCE=false
SKIP_MVFST_DEPS=false
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --inference )
      # If --inference is specified, only get what we need to run inference
      INFERENCE=true
      shift;;
    --skip-mvfst-deps )
      # If --skip-mvfst-deps is specified, don't get mvfst's dependencies.
      SKIP_MVFST_DEPS=true
      shift;;
    --mahi-tun )
      # If --skip-mvfst-deps is specified, don't get mvfst's dependencies.
      MAHI_TUNNEL=true
      shift;;
    * )    # Unknown option
      POSITIONAL+=("$1") # Save it in an array for later
      shift;;
  esac
done
set -- ${POSITIONAL[@]+"${POSITIONAL[@]}"} # Restore positional parameters

BUILD_ARGS=""
MVFST_ARGS=""
if [ "$INFERENCE" = true ]; then
  echo -e "Inference-only build"
  BUILD_ARGS="--inference"
else
  echo -e "Installing for training"
fi
if [ "$SKIP_MVFST_DEPS" = true ]; then
  echo -e "Skipping dependencies of mvfst"
  MVFST_ARGS="-s"
fi

PREFIX=${CONDA_PREFIX:-"/usr/local"}

BASE_DIR="$PWD"
BUILD_DIR="$BASE_DIR"/_build
DEPS_DIR="$BUILD_DIR"/deps
mkdir -p "$DEPS_DIR"

#if [ -d "$BUILD_DIR/build" ]; then
#  echo -e "mvfst-rl already installed, skipping"
#  exit 0
#fi

PANTHEON_DIR="$DEPS_DIR"/pantheon
LIBTORCH_DIR="$DEPS_DIR"/libtorch
PYTORCH_DIR="$DEPS_DIR"/pytorch
THIRDPARTY_DIR="$BASE_DIR"/third-party
TORCHBEAST_DIR="$THIRDPARTY_DIR"/torchbeast
MVFST_DIR="$THIRDPARTY_DIR"/mvfst
MVFST_DIR_TEMP="$THIRDPARTY_DIR"/mvfst_temp
GRPC_DIR="$THIRDPARTY_DIR"/grpc
MAHIMAHI_DIR="$DEPS_DIR"/mahimahi


function setup_pantheon() {
  echo -e "Installing Pantheon dependencies"
  cd "$PANTHEON_DIR"
  sudo apt-get -y install python2.7
  sudo rm /usr/bin/python2
  sudo ln -s /usr/bin/python2.7 /usr/bin/python2
  curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output tmp_get-pip.py
  sudo python2 tmp_get-pip.py
  rm -f tmp_get-pip.py

  sudo apt-get -y install ntp ntpdate texlive python-pip
  sudo apt-get -y install debhelper autotools-dev dh-autoreconf iptables \
                          pkg-config iproute2 dnsmasq
  python2 -m pip install matplotlib numpy tabulate pyyaml

  # Install pantheon tunnel in the conda env.
  cd third_party/pantheon_tunnel && ./autogen.sh \
  && make clean \
  && ./configure --prefix="$PREFIX" \
  && make -j && sudo make install

  # Force-symlink pantheon/third_party/mvfst-rl to $BASE_DIR
  # to avoid double-building
  echo -e "Symlinking $PANTHEON_DIR/third_party/mvfst-rl to $BASE_DIR"
  rm -f $PANTHEON_DIR/third_party/mvfst-rl
  ln -sf "$BASE_DIR" $PANTHEON_DIR/third_party/mvfst-rl
  echo -e "Done setting up Pantheon"
}

function setup_mahimahi() {
  if [ ! $(grep /usr/lib /etc/ld.so.conf.d/libc.conf) ];then
    sudo sh -c "echo '/usr/lib' >> /etc/ld.so.conf.d/libc.conf"
    sudo ldconfig
  fi
  sudo apt-get install -y protobuf-compiler libprotobuf-dev autotools-dev dh-autoreconf \
                          iptables pkg-config dnsmasq-base apache2-bin debhelper libssl-dev \
                          ssl-cert libxcb-present-dev libcairo2-dev libpango1.0-dev apache2-dev
  cd "$MAHIMAHI_DIR"
  ./autogen.sh
  ./configure
  make
  sudo make install
  # Copy mahimahi binaries to conda env (to be able to run in cluster)
  # with setuid bit.
  sudo cp /usr/local/bin/mm-* "$PREFIX"/bin/
  sudo chown root:root "$PREFIX"/bin/mm-*
  sudo chmod 4755 "$PREFIX"/bin/mm-*

  echo -e "Done installing mahimahi"

}

function setup_libtorch() {
  if [ -d "$LIBTORCH_DIR" ]; then
    echo -e "$LIBTORCH_DIR already exists, skipping."
    return
  fi

  # Install CPU-only build of PyTorch libs so that C++ executables of
  # mvfst-rl such as traffic_gen don't need to be unnecessarily linked
  # with CUDA libs, especially during inference.
  echo -e "Installing libtorch CPU-only build into $LIBTORCH_DIR"
  cd "$DEPS_DIR"

  wget --no-verbose https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.2.0.zip --no-check-certificate

  # This creates and populates $LIBTORCH_DIR
  unzip libtorch-cxx11-abi-shared-with-deps-1.2.0.zip
  rm -f libtorch-cxx11-abi-shared-with-deps-1.2.0.zip
  echo -e "Done installing libtorch"
}

function setup_grpc() {
  # Manually install grpc. We need this for mvfst-rl in training mode.
  # Note that this gets installed within the conda prefix which needs to be
  # exported to cmake.
  cd "$GRPC_DIR"
  echo -e "Installing grpc"
  conda install -y -c anaconda protobuf=3.12.3
  cd "$BASE_DIR"
  sudo chmod +x ./third-party/install_grpc.sh
  ./third-party/install_grpc.sh
  echo -e "Done installing grpc"
}


function setup_torchbeast() {
  conda install -y pytorch==1.2.0 -c pytorch
  echo -e "Installing TorchBeast"
  cd "$TORCHBEAST_DIR"
  python3 -m pip install -r requirements.txt

  # Install nest
  cd nest/ && CXX=c++ python3 -m pip install . -vv && cd ..

  # export LD_LIBRARY_PATH=${PREFIX}/lib:${LD_LIBRARY_PATH}
  export LD_LIBRARY_PATH=${PREFIX}/lib
  CXX=c++ python3 setup.py install
  echo -e "Done installing TorchBeast"
}

function setup_mvfst() {
  # Build and install mvfst
  echo -e "Installing mvfst"
  conda install -y cmake=3.14.0
  cd "$MVFST_DIR"
  sudo chmod +x build_helper.sh
  proxychains ./build_helper.sh "$MVFST_ARGS"
  cd _build/build/ && make install
  echo -e "Done installing mvfst"
}

function setup_mahimahi_tunnel() {
  #setup_pantheon_tunnel
  cd "$PANTHEON_DIR"
  #sudo apt-get -y install ntp ntpdate texlive python-pip
  #sudo apt-get -y install debhelper autotools-dev dh-autoreconf iptables \
  #                        pkg-config iproute2 dnsmasq
  #python2 -m pip install matplotlib numpy tabulate pyyaml

  # Install pantheon tunnel in the conda env.
  if [ -e "/usr/local/bin/mm-delay" ]; then
    sudo rm /usr/local/bin/mm-*
  fi
  if [ -e "$PREFIX/bin/mm-delay" ]; then
    sudo rm $PREFIX/bin/mm-*
  fi
  
  cd third_party/pantheon_tunnel && ./autogen.sh \
  && make clean \
  && ./configure --prefix="$PREFIX" \
  && make -j && sudo make install

  #setup mahimahi
  #if [ ! $(grep /usr/lib /etc/ld.so.conf.d/libc.conf) ];then
  #  sudo sh -c "echo '/usr/lib' >> /etc/ld.so.conf.d/libc.conf"
  #  sudo ldconfig
  #fi
  #sudo apt-get install -y protobuf-compiler libprotobuf-dev autotools-dev dh-autoreconf \
  #                        iptables pkg-config dnsmasq-base apache2-bin debhelper libssl-dev \
  #                        ssl-cert libxcb-present-dev libcairo2-dev libpango1.0-dev apache2-dev
  cd "$BASE_DIR" && cd "$MAHIMAHI_DIR"
  if conda list -n "mvfst-rl" | grep -q "protobuf"; then
    conda uninstall -y protobuf=3.12.3
  fi
  ./autogen.sh
  make clean
  ./configure
  make
  sudo make install
  # Copy mahimahi binaries to conda env (to be able to run in cluster)
  # with setuid bit.
  sudo cp /usr/local/bin/mm-* "$PREFIX"/bin/
  sudo chown root:root "$PREFIX"/bin/mm-*
  sudo chmod 4755 "$PREFIX"/bin/mm-*

  if ! conda list -n "mvfst-rl" | grep -q "protobuf"; then
    conda install -y -c anaconda protobuf=3.12.3
  fi
  echo -e "Done installing mahimahi"

}

if [ "$MAHI_TUNNEL" = true ]; then
  setup_mahimahi_tunnel
else
  git submodule sync && proxychains git submodule update --init --recursive --progress
  if [ "$INFERENCE" = false ]; then
    setup_pantheon
    setup_mahimahi
    setup_grpc
    setup_torchbeast
  fi
  setup_libtorch
  setup_mvfst
  echo -e "Building mvfst-rl"
  sudo chmod +x "$BASE_DIR"/build.sh
  cd "$BASE_DIR" && ./build.sh $BUILD_ARGS
fi

    



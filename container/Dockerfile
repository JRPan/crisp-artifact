FROM ubuntu:24.04
COPY packages.txt install_packages.sh /tmp/
RUN /tmp/install_packages.sh /tmp/packages.txt && rm /tmp/packages.txt /tmp/install_packages.sh
ENV LLVM_INCLUDE=/usr/lib/llvm-14/include

# MojoFrame: A High-Performance DataFrame Library in Mojo

MojoFrame is a high-performance DataFrame library implemented in the [Mojo programming language](https://www.modular.com/mojo), designed for analytical query workloads and optimized for modern hardware (CPUs supported, GPUs and other hardware accelerators to be explored) via JIT and MLIR.

## Features

Currently, core relational opertaions and some utility functions are implemented for the TPC-H benchmark.

## Running

Install magic CLI: `curl -ssL https://magic.modular.com/92f881dc-d525-4538-913e-eea8752d2210 | bash`
`cd Mojoframe`
`magic init --format mojoproject`
`magic run mojo main.mojo`
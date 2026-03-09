# Bidirectional Gating for Spatio-Temporal Vision: An Attentive Liquid State Machine

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18923549.svg)](https://doi.org/10.5281/zenodo.18923549) **Note to Readers**: This code is directly related to the manuscript currently submitted to **The Visual Computer**. Please cite this relevant manuscript if you use our code for your research.

## Overview
This repository contains the official PyTorch implementation for **Bidirectional Gating for Spatio-Temporal Vision: An Attentive Liquid State Machine**. 

We introduce the **Attentive Liquid State Machine (ALSM)**, a parallel dual-stream architecture designed for robust spatio-temporal visual recognition. Our model achieves competitive performance on standard static and neuromorphic benchmarks, including Fashion-MNIST, N-MNIST, and DVS128 Gesture datasets.

## Dependencies and Requirements
To run the code, ensure your environment meets the following requirements:
* Python 3.8+
* [PyTorch](https://pytorch.org/) (Version >= 1.10.0)
* [SpikingJelly](https://github.com/fangwei123456/spikingjelly) (Used for spiking neuron simulation and datasets)
* [snnTorch](https://snntorch.readthedocs.io/) (Used for DVS128 gesture dataset implementations)
* [timm](https://github.com/rwightman/pytorch-image-models)
* CuPy (Required for fast LIF neuron backend computation)

You can install the dependencies via:
```bash
pip install torch torchvision timm spikingjelly snntorch cupy

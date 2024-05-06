# IWOMI Tutorial - STOUT Training

Welcome to the STOUT Hands-on Tutorial at IWOMI 2024! 🎉🍺

## Overview

This repository contains STOUT-V2, a SMILES to IUPAC name translator using transformers. STOUT-V2 can translate SMILES (Simplified Molecular Input Line Entry System) to IUPAC (International Union of Pure and Applied Chemistry) names and IUPAC names back to valid SMILES strings.

### Prerequisites

Before we begin training STOUT, let's ensure our setup is properly configured:

- **Conda Environment**: We highly recommend using Conda for seamless dependency management. If you're new to Conda, don't worry! You can download it as part of the Anaconda or Miniconda platforms. For optimal performance, we suggest installing Miniconda3.

### Installation Guide

Follow these steps to set up STOUT within a Conda environment:

1. **Clone the Repository**:

   ```shell
   git clone https://github.com/Kohulan/IWOMI_Tutorials.git
   cd IWOMI_Tutorials
   ```
2. **Create Conda Environment**:

   ```shell
   conda create --name STOUT python=3.10.0
   conda activate STOUT
   conda install pip
   python3 -m pip install -U pip
   pip install -r requirements.txt
   ```

## Happy Brewing! 💡🍺

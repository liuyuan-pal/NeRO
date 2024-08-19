# NeRO_UoM: Improved Neural Reflectance Optimization Pipeline

This repository contains improvements and additions to the original NeRO (Neural Reflectance Optimization) pipeline, specifically tailored for use at the University of Manchester.

## New Features

- Automated environment setup for COLMAP and NeRO
- Streamlined pipeline execution with minimal user input
- Improved directory structure management
- New script for end-to-end pipeline execution

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/abhishekbagde/NeRO_UoM.git
   cd NeRO_UoM
   ```

2. Ensure you have Anaconda or Miniconda installed on your system.

3. Prepare your image dataset in a separate folder.

## Usage

Run the pipeline script with:

```
python nero_pipeline.py /path/to/image/folder /path/to/nero/directory
```

The script will guide you through the entire process, from environment setup to final mesh extraction.

## Requirements

- Python 3.10
- CUDA 11.7.0
- COLMAP
- NeRO dependencies (see requirements.txt in the NeRO directory)

## Key Changes

- Automated environment setup for both COLMAP and NeRO
- Streamlined directory structure management post-COLMAP processing
- Enhanced error handling and user feedback throughout the pipeline

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the same license as the original NeRO project. Please see the LICENSE file for details.

## Acknowledgments

- Original NeRO project team
- COLMAP developers
- University of Manchester research team

For more detailed information about the original NeRO project, please refer to the [original repository](https://github.com/zju3dv/NeRO).
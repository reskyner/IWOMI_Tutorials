# DECIMER Image Data Extraction
## Installation

 To install clone this repository and create a conda environment from the provided yaml file.
 1. Repository Cloning:
 ```
 git clone https://github.com/Kohulan/IWOMI_Tutorials
 cd IWOMI_Tutorials/DECIMER_Extraction/IMAGE_DATA_EXTRACTION
 ```
 2. Conda Environment Creation
 ```
conda env create --file environment.yml
conda activate IMSEG_test
```

## Usage

If you don't want to use the example PDF, make sure to set the file path to your desired PDF and comment the filename line. Also, make sure to copy the terminal output from ```get_smiles_with_avg_confidence(filepath: str) ``` and paste it to the `terminal_ouput.txt` file.

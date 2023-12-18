# Risk Prediction and Failure Mode Analysis of Lithium-Ion Batteries: A Comprehensive Approach with Mixture Weibull and Equivalent Circuit Model

This repository contains the code and data supporting the research paper titled "Risk Prediction and Failure Mode Analysis of Lithium-Ion Batteries: A Comprehensive Approach with Mixture Weibull and Equivalent Circuit Model". Our work presents a novel approach to understanding the failure modes and predicting the risk associated with lithium-ion batteries.

### Contents
- `data/`: Folder containing datasets used in the analysis.
- `imgs/`: Images and figures used in the paper.
- `ECM.py`: Python script for the Equivalent Circuit Model analysis.
- `copula_based_MWM.py`: Python script implementing the Mixture Weibull Model with Copula functions.

### Data Description
The `data/` directory contains the following:
- Capacity Data: This section contains .csv files for each dataset, detailing the discharge capacity variation with cycle number.
- EIS Data: This subsection includes Electrochemical Impedance Spectroscopy spectra data for each dataset, measured at 90% State of Charge across various frequencies.

### Experimental Setup:
- Operating System: macOS Monterey
- Processor: Apple M1 CPU
- Memory: 16GB
- Python Version: 3.8

### Usage

To replicate the analysis presented in the paper, follow these steps:

1. Ensure you have Python installed on your system. The code is tested on Python 3.8+.
2. Run the ECM analysis: python ECM.py
3. Execute the Mixture Weibull Model analysis: python copula_based_MWM.py

# ORToS: Data Structuring and Error Pipeline

This repository contains the finalized data structuring and error-handling pipeline for the Orthopedic Rehabilitation Training System (ORToS), developed for my Senior Thesis at the University of Utah Dept. of Biomedical Engineering under advisor Dr. Robert Hitchcock. The pipeline ensures accurate and reliable processing of rehabilitation training data for patients recovering from lower extremity fractures.

## Key Features
- Structured data parsing and formatting
- Robust error checking and handling
- Automated validation of rehabilitation data
- Compatibility Python data analysis tools

## ⚙️ Installation & Setup
- DOES NOT INCLUDE DE-IDENTIFIED PATIENT DATA
### Clone Repository
```bash
git clone https://github.com/wyattyoung/ortos-data-pipeline.git
```

### Dependencies
- Python (>= 3.8)
- MATLAB (R2023a or newer)
- Required Python packages:
```bash
pip install numpy pandas scipy matplotlib
```

## Repository Structure
```
ortos-data-pipeline/
├── scripts/
│   ├── main_pipeline.py # Main script to run the pipeline
│   └── utils/           # Helper functions
└── README.md
```

## Results & Outputs
- Generates structured datasets ready for downstream rehabilitation analysis.
- Prompts user with found errors in ORToS acquired data

## Motivation & Impact
- Ensures accurate data analysis and reporting, critical for patient rehabilitation progress tracking.

## Acknowledgments
- Advisors: Robert Hitchcock, Kylee North
- Supported by the Hitchcock Lab and Department of Biomedical Engineering, University of Utah.

## Contact
Wyatt Young  
Email: [young.wyatt@utah.edu](mailto:young.wyatt@utah.edu)  
GitHub: [WyattYoung](https://github.com/youngwyatt)

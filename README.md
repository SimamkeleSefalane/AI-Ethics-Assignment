# AI-Ethics-Assignment

# README: COMPAS Recidivism Bias Audit Project

## Project Overview

This project performs an ethical audit of the COMPAS recidivism risk scoring dataset to identify and mitigate racial bias in predictive modeling. Using IBM’s AI Fairness 360 toolkit, the project analyzes disparity in false positive rates between racial groups, applies bias mitigation techniques, and evaluates fairness metrics before and after mitigation.

---

## Contents

- **data/**  
  Contains the COMPAS dataset CSV file used for analysis.

- **compas_bias_audit.ipynb**  
  Jupyter Notebook containing the full exploratory data analysis, fairness audit, bias mitigation, and visualization code.

- **compas_bias_audit.py** *(optional)*  
  A Python script version of the notebook for automated runs and deployment.

- **requirements.txt**  
  Lists all Python dependencies and versions.

- **README.md**  
  This document.

---

## Setup Instructions

1. **Clone the repository:**

```bash
git clone <repository-url>
cd <repository-folder>

**Create and activate a virtual environment (recommended):**
python -m venv env
source env/bin/activate       # Linux/Mac
.\env\Scripts\activate        # Windows PowerShell


**Install Dependencies:**
pip install -r requirements.txt

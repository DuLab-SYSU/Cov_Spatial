# Cov_Spatial

**Cov_Spatial** provides the code and data necessary to reproduce the analyses presented in the article:  

*“Spatio-temporal dispersal patterns of SARS-CoV-2 in mainland China following the COVID-19 response adjustment.”*

---

## 🧩 Overview

This repository contains scripts and metadata used to investigate the spatial and temporal dynamics of SARS-CoV-2 transmission in the mainland of China following the COVID-19 response adjustment.  
The analyses integrate genomic, epidemiological, and mobility data to explore viral importation, interprovincial transmission, and the predictors influencing the viral transmission.

---

## 📂 Data Availability

In compliance with the data-sharing policy of **GISAID** ([https://gisaid.org/](https://gisaid.org/)), the **actual SARS-CoV-2 genome sequences** used in this study have been **removed** from this repository.  
To reproduce the results, please download the relevant sequences from GISAID using the **accession IDs** provided in the `/data` directory of this repository.

- `Supplementary_Data_S1.csv`— the intensity of international travel control measures implemented by the Chinese government from the Oxford COVID-19 Government Response Tracker (OxCGRT) (https://github.com/OxCGRT/covid-policy-dataset).
- `Supplementary_Data_S2.csv`— raw sequence data used from GISAID  database under accession IDs and strain name in the mainland of China.
- `Supplementary_Data_S3.csv`— raw sequence data used from GISAID database under accession IDs (global).

---

## ⚙️ Code Description

### `code/data_processing.py`

- Performs data preprocessing and quality control for SARS-CoV-2 sequences and associated metadata from GISAID.  
- Counts the number of sequences for each Omicron sublineage and each province in the mainland of China.  
- Identifies six dominant ancestral Omicron lineages: **BA.5, BF.7, DY, XBB, EG.5, and HK**.  
- Defines three epidemic phases based on temporal predominance and lineage composition:  
  - **Phase I:** BA.5/BF.7/DY (July 2022 – March 2023)  
  - **Phase II:** XBB (April – July 2023)  
  - **Phase III:** EG.5/HK (August – November 2023)

---

### `code/mutation_network.py`

- Constructs mutation networks for the six dominant Omicron sublineages.  
- Quantifies importation events from overseas regions.  
- Estimates interprovincial import and export events based on the mutation network.

---

### `code/glm.py`

- Applies **generalized linear models (GLMs)** to examine the association between viral transmission and potential explanatory factors.  
- Integrates exportation events across all Omicron lineages into a unified GLM framework to identify predictors influencing interprovincial viral export capacity.

---

### `code/sensitive.py`

- Conducts **sensitivity analyses** to assess the robustness of the results.  
- Implements subsampling by province and lineage to mitigate potential sampling bias and replicates the full analytical framework on subsampled datasets.

---

## 📊 Visualization Notebooks

Each Jupyter notebook (`.ipynb`) reproduces a corresponding figure from the main text or supplementary materials:

| Notebook           | Description                               |
| ------------------ | ----------------------------------------- |
| **Figure1.ipynb**  | Visualization for Figure 1                |
| **Figure2.ipynb**  | Visualization for Figure 2                |
| **Figure3.ipynb**  | Visualization for Figure 3                |
| **Figure4.ipynb**  | Visualization for Figure 4                |
| **Figure5.ipynb**  | Visualization for Figure 5                |
| **FigureS1.ipynb** | Visualization for Supplementary Figure S1 |
| **FigureS2.ipynb** | Visualization for Supplementary Figure S2 |
| **FigureS3.ipynb** | Visualization for Supplementary Figure S3 |

---

## 📖 Citation

If you use this code or data in your research, please cite the following article:

> Upcoming update

---

## 📜 License

This project is distributed under the **GNU General Public License v3.0**.  
See the [LICENSE](./LICENSE) file for details.

---

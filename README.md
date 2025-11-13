# Cov_Spatial

Code and data needed to reproduce the results in article "Spatio-temporal dispersal patterns of SARS-CoV-2 in mainland China following the COVID-19 response adjustment"

According to the permissions of GISAID (https://gisaid.org/), the actucal sequences used in this work were deleted from this repository. To reproduce the results, you can download related sequences from GISAID with the meta-information provided in data dir. 

./code/data_processing.py including:

- data pre-processing and quality control for SARS-CoV-2 sequences data and their associated metadata from GISAID 
- counts the number of sequences for each Omicron sublineages and for each province in the mainland of China
- identified six ancestral Omicron lineages in the mainland of China (BA.5, BF.7, DY, XBB, EG.5, and HK) and defined three epidemic phases based on the temporal predominance and relative abundance of major lineages: BA.5/BF.7/DY (July 2022–March 2023), XBB (April–July 2023), and EG.5/HK (August–November 2023).

./code/mutation_network.py including:

- mutation network construction: constructed mutation network for six dominant Omicron sublineages.
- count the exports and imports across each provinces by mutation network
- count the exports and imports across each provinces by mutation network

./code/glm.py including:

- utilized generalized linear models (GLMs) to investigate potential associations between viral transmission and various contributing factors,
- integrated the number of interprovincial exportation events for all Omicron lineages into a single GLM to jointly examine the predictors influencing the export capacity of SARS-CoV-2 across provinces

./code/sensitive.py including:

- performed subsampling based on provinces and Omicron lineages o mitigate potential sampling bias, accompanied by sensitivity analyses replicating the analytical framework applied to the full dataset.

Figure 1-5 and Figure S1-S3 (ipynb) including:

- FIgure 1.ipynb: visualization for Figure 1
- FIgure 2.ipynb: visualization for Figure 2
- FIgure 3.ipynb: visualization for Figure 3
- FIgure 4.ipynb: visualization for Figure 4
- FIgure 5.ipynb: visualization for Figure 5
- FIgure S1.ipynb: visualization for Figure S1
- FIgure S2.ipynb: visualization for Figure S2
- FIgure S3.ipynb: visualization for Figure S3


## Mapping the coupling between tract reachability and cortical geometry of the human brain

Our findings reveal that TGC is a robust measure, exhibiting heritability and predictive power over individual behavioral variation, with low-frequency eigenmodes offering enhanced explanatory potential. The quantitative assessment of the difference between high-/low-frequency TGC, i.e., high-low frequency ratio, highlighted a pronounced coupling between association tracts and high-frequency eigenmodes, with distinct patterns observed across various functional networks. Moreover, a significant correlation was identified between individual TGC profiles and brain activation maps and exhibited non-uniform maturation during youth. In conclusion, our study provides a new approach to mapping the coupling between cortical geometry and connectivity, highlighting how these two aspects jointly shape the connected brain.

The behavior prediction code was from https://github.com/ThomasYeoLab/Standalone_Ooi2022_MMP.git.

<img width="1101" alt="overview" src="https://github.com/user-attachments/assets/363fa0cb-c4c9-4c92-aa78-994b530a78ef" />


## Code Overview

### Figure 2

- **`figure2_TGC_calculate.py`**  
  Computes Tract Graph Connectivity (TGC). Input is the subject ID. Supports batch processing via bash scripting. Outputs include the TGC and reconstruction accuracy for each subject.

- **`figure2_TGC_calculate.ipynb`**  
  Plots the reconstruction accuracy of TGC (Figure 2A) and saves the reconstructed tract reachability maps for comparison with ground truth (Figure 2B).

- **`figure2_TestRetest.ipynb`**  
  Assesses the test-retest reliability and subject identifiability of TGC using HCP test-retest data (Figure 2C, 2D).

---

### Figure 3

- **`figure3_visualization_and_h2.ipynb`**  
  Visualizes TGC (Figure 3A, 3B) and heritability (h²) based on results from [APACE](https://github.com/NISOx-BDI/APACE.git) (Figure 3C).

---

### Figure 4

- **`figure4_freq_ratio.m`**  
  Computes the power spectral density and determines the high/low frequency boundary.  
  Reference: [GSP_StructuralDecouplingIndex](https://github.com/gpreti/GSP_StructuralDecouplingIndex)

- **`figure4_freq_ratio.ipynb`**  
  Analyzes the high-to-low frequency energy ratio of TGC and generates Figure 4.

---

### Figure 5

- **`figure5_AM_correlation.py`**  
  Computes vertex-wise correlations between each subject's TGC and 47 task activation maps. Saves both correlation coefficients (r) and p-values.

- **`figure5_AM_correlation_plotting.ipynb`**  
  Analyzes the associations between task activation and TGC, generating Figure 5.

- **`figure5_AM_correlation_calculate.ipynb`**  
  Evaluates how contrast-specific TGC reflects differences across task domains.

---

### Figure 6

- **`figure6/gamplots_new.r`**  
  Uses HCPD data to estimate developmental trajectories of TGC via Generalized Additive Models (GAM).

- **`figure6/raincloud4TGC.R`**  
  Performs longitudinal validation using the IMAGEN dataset.

- **`figure6/GAMresultPlot.ipynb`**  
  Aggregates GAM results and generates visualizations for Figure 6.

---

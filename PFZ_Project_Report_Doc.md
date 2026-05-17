# PROJECT REPORT: AI-Based Geospatial Decision Support System for Potential Fishing Zones (PFZ Navigator)

## TABLE OF CONTENTS
ABSTRACT
LIST OF FIGURES
LIST OF TABLES
ABBREVIATIONS

**CHAPTER 1: INTRODUCTION**
1.1 Introduction to Project
1.2 Problem Statement and Description
1.3 Motivation
1.4 Sustainable Development Goal of the Project
1.5 Scope of the Project

**CHAPTER 2: LITERATURE SURVEY**
2.1 Overview of the Research Area
2.2 Existing Models and Frameworks
2.3 Limitations Identified from Literature Survey (Research Gaps)
2.4 Research Objectives
2.5 Product Backlog (Key user stories with Desired outcomes)
2.6 Plan of Action (Project Road Map)

**CHAPTER 3: SPRINT PLANNING AND EXECUTION METHODOLOGY**
3.1 SPRINT I: Dataset Preparation and Core Model Implementation
3.1.1 Objectives with user stories of Sprint I
3.1.2 Functional Document
3.1.3 Architecture Document
3.1.4 Outcome of objectives/ Result Analysis
3.1.5 Sprint Retrospective
3.2 SPRINT II: Spatio-Temporal Forecasting, Web Dashboard, and Geospatial Analytics
3.2.1 Objectives with user stories of Sprint II
3.2.2 Functional Document
3.2.3 Architecture Document
3.2.4 Outcome of objectives/ Result Analysis
3.2.5 Sprint Retrospective

**CHAPTER 4: RESULTS AND DISCUSSIONS**
4.1 Project Outcomes (Performance Evaluation, Comparisons, Testing Results)
4.2 Geospatial Accuracy and Latency Analysis

**CHAPTER 5: CONCLUSION AND FUTURE ENHANCEMENT**
5.1 Conclusion
5.2 Future Enhancements

REFERENCES
APPENDIX
A CODING

---

## ABSTRACT
Deep learning models have demonstrated strong performance in geospatial ocean analysis, particularly for identifying Potential Fishing Zones (PFZs), yet operational deployment remains constrained by limited integration of spatial and temporal dynamics. Traditional PFZ advisory systems rely on threshold-based analysis of Sea Surface Temperature (SST) and Chlorophyll-a, failing to capture nonlinear interactions among multi-parameter oceanographic variables. While machine learning approaches have improved prediction accuracy, they lack the ability to perform high-resolution spatial segmentation and long-term spatio-temporal forecasting simultaneously.

In this work, we propose a unified dual-model deep learning framework, PFZ Navigator, that integrates spatial and temporal learning paradigms through two complementary architectures: (i) a U-Net encoder–decoder network for pixel-wise segmentation of current-day PFZ boundaries using multi-channel satellite data, and (ii) a Convolutional LSTM (ConvLSTM) network for multi-day spatio-temporal forecasting of PFZ distributions. The system processes co-registered oceanographic variables including SST, Sea Surface Height (SSH), ocean currents (u/v), and Chlorophyll-a, regridded to a standardized 256×256 spatial domain.

To ensure robust generalization, the framework employs temporally consistent validation strategies, including Temporal Block Cross-Validation for spatial modeling and Walk-Forward Expanding Window validation for forecasting. The models are trained using multi-channel inputs and optimized with categorical cross-entropy loss while preserving spatial dependencies through convolutional operations.

Experimental evaluation on CMEMS satellite data demonstrates that the U-Net model achieves a segmentation accuracy of 92.13% with strong spatial consistency, while the ConvLSTM achieves 88.49% accuracy in seven-day forecasting tasks. The proposed system further enables automated GPS coordinate extraction and interactive geospatial visualization through a web-based dashboard.

The results highlight that integrating spatial segmentation and temporal forecasting within a unified deep learning pipeline provides a scalable and effective solution for PFZ prediction. This work establishes a structured framework for AI-driven marine decision support systems, contributing toward sustainable fisheries management and efficient resource utilization.

---

## ABBREVIATIONS
- **AI:** Artificial Intelligence
- **API:** Application Programming Interface
- **CMEMS:** Copernicus Marine Environment Monitoring Service
- **CNN:** Convolutional Neural Network
- **ConvLSTM:** Convolutional Long Short-Term Memory
- **CSV:** Comma-Separated Values
- **DL:** Deep Learning
- **GPS:** Global Positioning System
- **GUI:** Graphical User Interface
- **JSON:** JavaScript Object Notation
- **ML:** Machine Learning
- **NetCDF:** Network Common Data Form
- **PFZ:** Potential Fishing Zone
- **REST:** Representational State Transfer
- **RNN:** Recurrent Neural Network
- **SSH:** Sea Surface Height
- **SST:** Sea Surface Temperature

---

## CHAPTER 1: INTRODUCTION

### 1.1 Introduction to Project
The sustainable management of marine fisheries requires precise mapping of pelagic fish aggregations. These aggregations typically occur in dynamic oceanic regions characterized by specific bio-physical interactions, such as thermal fronts, mesoscale eddies, and coastal upwellings. Historically, locating these Potential Fishing Zones (PFZs) relied heavily on direct observational heuristics. With the advent of satellite oceanography, identifying PFZs transitioned to remote sensing methodologies, analyzing surface markers like Sea Surface Temperature (SST) and Chlorophyll-a concentration. 

However, operational deployment of these geospatial ocean analyses remains fundamentally constrained by the limited integration of spatial and temporal dynamics. The **PFZ Navigator** is designed to address this constraint by formulating PFZ identification not as a static classification task, but as a continuous spatio-temporal modeling problem. Utilizing co-registered multi-parameter satellite data from the Copernicus Marine Environment Monitoring Service (CMEMS), this project implements a unified dual-model deep learning framework to process highly non-linear oceanographic variables and provide structured, operational marine decision support.

### 1.2 Problem Statement and Description
Traditional PFZ advisory systems primarily rely on threshold-based analyses. These heuristic models define static operational bounds for variables like SST gradients and chlorophyll concentrations, failing entirely to capture the complex, nonlinear interactions among multi-parameter variables such as sea surface height anomalies and ocean velocity vectors. 

Furthermore, while contemporary machine learning approaches (including standard Convolutional Neural Networks) have improved spatial prediction accuracy, they suffer from two critical architectural limitations:
1. **Lack of Simultaneous Processing:** They lack the capacity to perform high-resolution spatial segmentation and long-term spatio-temporal forecasting simultaneously. Most models either segment today's static spatial map or forecast a localized 1D time-series, losing the geometric topology of the ocean.
2. **Absence of Operational Abstractions:** Existing academic models output raw probability tensors without providing the necessary geospatial algorithms required for operational maritime navigation, such as automated coordinate extraction, proximity ranking, and vector bearings.

### 1.3 Motivation
The development of the PFZ Navigator is motivated by the need to establish a structured, scalable framework for AI-driven marine decision support. By integrating spatial segmentation and temporal forecasting within a unified deep learning pipeline, the system provides a comprehensive solution for PFZ prediction. 

For the end-user in the fishing community, this translates to targeted, highly efficient marine navigation. Accurately forecasting PFZ distributions up to seven days in advance enables vessels to optimize their trajectories, directly resulting in reduced marine fuel consumption, diminished maritime carbon emissions, and maximized catch-per-unit-effort. Consequently, this computational approach transitions fisheries management from reactive searching to predictive, mathematically optimized resource utilization.

### 1.4 Sustainable Development Goal of the Project
The framework strictly aligns with the United Nations Sustainable Development Goals (SDGs), ensuring that computational advancements yield measurable ecological and socio-economic benefits:
- **SDG 14 (Life Below Water):** By facilitating targeted fishing operations, the system minimizes the requirement for broad, exploratory trawling, thereby reducing bycatch and mitigating the degradation of non-target marine habitats.
- **SDG 8 (Decent Work and Economic Growth):** Optimizing spatial trajectories significantly lowers diesel fuel expenditure—the primary operational overhead for coastal fleets—thereby increasing net profitability and ensuring the economic stability of fishing communities.
- **SDG 2 (Zero Hunger):** Improving the predictability and efficiency of marine harvesting operations directly bolsters global supply chains, contributing to stabilized food security paradigms.

### 1.5 Scope of the Project
The scope encompasses the end-to-end development of the deep learning pipeline and its operational deployment. This includes the ingestion and regridding of CMEMS NetCDF files to a standardized $256 \times 256$ spatial domain. It covers the architectural design and training of a U-Net encoder-decoder network and a ConvLSTM network. Finally, the scope includes the deployment of a RESTful backend API and a web-based dashboard that enables interactive geospatial visualization and automated GPS coordinate extraction via CSV.

---

## CHAPTER 2: LITERATURE SURVEY

### 2.1 Overview of the Research Area
Geospatial ocean analysis for fisheries management represents an intersection of satellite oceanography, marine ecology, and computational modeling. The fundamental premise relies on identifying biological oases in the open ocean. These are typically localized areas where physical mechanisms (like upwelling or frontogenesis) transport cold, nutrient-rich water to the photic zone, stimulating primary production (phytoplankton) and subsequently attracting secondary and tertiary consumers. Remote sensing facilitates the observation of these phenomena through proxy variables such as SST anomalies, SSH variations, and ocean color.

### 2.2 Existing Models and Frameworks
Early predictive frameworks were heavily reliant on statistical models, including Generalized Additive Models (GAMs) and Support Vector Machines (SVMs). These approaches processed flattened, independent vectors of oceanographic variables, inherently failing to preserve the spatial autocorrelation present in marine environments. 

To address spatial dependencies, the field transitioned toward Convolutional Neural Networks (CNNs). Standard CNNs demonstrated strong performance in isolating localized thermal fronts and segmenting discrete mesoscale eddies. However, these models were strictly confined to static, independent observations. Recurrent Neural Networks (RNNs) and standard LSTMs were subsequently introduced to model temporal variability but were fundamentally limited by their fully connected dense gates, which flatten multi-dimensional inputs and destroy spatial topologies.

### 2.3 Limitations Identified from Literature Survey (Research Gaps)
The literature survey identifies three primary constraints in current operational deployments:
1. **Separation of Paradigms:** Current methodologies fail to unify spatial and temporal learning paradigms effectively. Models excel either in pixel-wise segmentation or in sequence modeling, but rarely achieve both simultaneously for multi-day horizons.
2. **Validation Inconsistencies:** Many existing deep learning applications utilize standard random $k$-fold cross-validation, which inadvertently leaks future time-series data into the training set, violating causality and artificially inflating reported accuracies.
3. **Operational Disconnect:** The majority of frameworks conclude at the generation of a probability heatmap, completely neglecting the computational geospatial translations required to convert abstract pixels into operational GPS coordinates and navigable bearings.

### 2.4 Research Objectives
To overcome the identified constraints, this project specifies the following structured objectives:
1. To process and standardize co-registered oceanographic variables (SST, SSH, u/v currents, Chlorophyll-a) into a cohesive $256 \times 256$ spatial domain.
2. To architect and implement a U-Net encoder-decoder network dedicated to the pixel-wise segmentation of current-day PFZ boundaries.
3. To architect and implement a Convolutional LSTM network for multi-day (seven-day) spatio-temporal forecasting, explicitly preserving spatial dependencies through convolutional operations.
4. To deploy the dual-model framework within an interactive web dashboard, featuring automated extraction of GPS coordinates and Haversine-based proximity ranking.

### 2.5 Product Backlog (Key user stories with Desired outcomes)

| ID | User Story | Desired Outcome | Priority | Sprint |
|---|---|---|---|---|
| US1 | As a system architect, I must ingest CMEMS NetCDF files to extract and normalize physical and biological markers. | Co-registered variables standardized into NumPy arrays with strict terrestrial masking applied. | High | 1 |
| US2 | As a machine learning engineer, I must train a U-Net model using categorical cross-entropy loss. | A segmentation accuracy exceeding 90% via Temporal Block Cross-Validation. | High | 1 |
| US3 | As a machine learning engineer, I must train a ConvLSTM model for multi-day spatio-temporal forecasting. | A robust seven-day forecasting accuracy utilizing Walk-Forward Expanding Window validation. | High | 1 |
| US4 | As an end-user, I require an interactive geospatial visualization interface to analyze the generated probability tensors. | A functional web-based dashboard rendering dynamically layered Leaflet.js maps. | High | 2 |
| US5 | As an end-user, I need to input a specific operational coordinate and retrieve an isolated localized forecast. | The UI returns a structured seven-day tabular forecast specific to the queried coordinate. | High | 2 |
| US6 | As an end-user, I require the system to automatically rank the highest probability zones relative to my current position. | Proximity-based ranking utilizing the Haversine formula and vector-based compass bearings. | High | 2 |
| US7 | As an end-user, I must export the operational coordinates to my vessel's GPS hardware. | Automated extraction and formatting of spatial data into a downloadable CSV file. | Medium | 2 |

### 2.6 Plan of Action (Project Road Map)
The project execution is divided into structured computational sprints:
- **Phase 1: Data Engineering:** Ingestion of satellite data, resolution harmonization, application of boolean land masks, and feature normalization.
- **Phase 2: Spatial Modeling Paradigm:** Implementation of the U-Net architecture, hyperparameter optimization, and execution of Temporal Block Cross-Validation.
- **Phase 3: Temporal Modeling Paradigm:** Implementation of the ConvLSTM architecture, sequence generation, and execution of Walk-Forward Expanding Window validation.
- **Phase 4: API & Visualization Integration:** Deployment of the Python Flask backend to serve the models, and development of the DOM-based interactive frontend.
- **Phase 5: Geospatial Algorithms:** Implementation of operational mathematical abstractions, including grid-to-coordinate mapping, Haversine proximity logic, and automated CSV extraction.

---

## CHAPTER 3: SPRINT PLANNING AND EXECUTION METHODOLOGY

### 3.1 SPRINT I: Dataset Preparation and Core Model Implementation

#### 3.1.1 Objectives with user stories of Sprint I
The primary objective of Sprint I was to establish the unified dual-model deep learning framework. This encompassed the rigorous preprocessing of multi-parameter oceanographic variables and the subsequent training of the complementary U-Net and ConvLSTM architectures. A critical focus was placed on ensuring robust generalization by employing temporally consistent validation strategies to prevent data leakage.

#### 3.1.2 Functional Document
The data engineering pipeline systematically processed daily CMEMS NetCDF files. Variables including SST, SSH, Zonal Velocity (u), Meridional Velocity (v), and Chlorophyll-a were extracted and co-registered to a standardized $256 \times 256$ spatial domain. To explicitly capture nonlinear physical interactions, localized spatial gradients were computed, generating a multi-channel input tensor. 

Given the terrestrial boundaries present within the spatial domain, stringent boolean land masking was applied to prevent NaN propagation during gradient updates. The multi-channel inputs were normalized using Min-Max scaling, standardizing the feature space prior to tensor operations. The target labels were generated by classifying the domain into discrete PFZ confidence classes (Low, Medium, High).

#### 3.1.3 Architecture Document
The framework integrates two complementary architectures:
1. **U-Net Encoder-Decoder Network:** Utilized for pixel-wise segmentation. The contracting path utilizes convolutional layers to capture high-level contextual features, while the expanding path utilizes transposed convolutions and skip connections to recover precise spatial localization, allowing for the high-resolution segmentation of current-day PFZ boundaries.
2. **Convolutional LSTM (ConvLSTM) Network:** Utilized for multi-day spatio-temporal forecasting. By replacing the dense matrix multiplications in traditional LSTM gates with spatial convolution operations, the ConvLSTM preserves local spatial dependencies across the sequential time steps. The model processes historical sequences of the multi-channel tensors to predict the subsequent seven-day distribution of PFZs.

Both models were optimized using the Adam optimizer with categorical cross-entropy loss, ensuring that the network heavily penalized misclassifications of the sparse "High PFZ" class.

> **[FIGMA PROMPT: System Interaction Flow Diagram]**
> *Create a highly formal Architecture Flow Diagram. Title: "Unified Dual-Model Deep Learning Framework".*
> *Actors/Nodes: "CMEMS Satellite Data", "Data Normalization & Regridding", "U-Net (Spatial Paradigm)", "ConvLSTM (Temporal Paradigm)", "Categorical Cross-Entropy Optimization".*
> *Flow Process: Multi-channel data flows into the regridding module, generating a 256x256 tensor. The tensor branches into two paths: the U-Net path for pixel-wise boundary segmentation, and the sequence generator leading to the ConvLSTM for spatio-temporal forecasting. Both paths output spatial probability distributions.*

#### 3.1.4 Outcome of objectives/ Result Analysis
To ensure robust generalization, temporally consistent validation strategies were strictly enforced. For the U-Net, Temporal Block Cross-Validation was utilized, ensuring contiguous blocks of time were held out to validate spatial consistency without autocorrelation leakage. For the ConvLSTM, Walk-Forward Expanding Window validation was employed, preserving the strict temporal causality of the sequences.

Experimental evaluation demonstrated highly stable convergence. Both networks effectively minimized the categorical cross-entropy loss over the training epochs without evidence of gradient explosion. The models successfully generalized to unseen test sets, confirming the validity of the dual-model deep learning pipeline.

#### 3.1.5 Sprint Retrospective
The execution of Sprint I confirmed that combining co-registered oceanographic variables significantly improves the definition of non-linear PFZ boundaries. A critical technical challenge successfully mitigated was managing the intensive GPU memory requirements necessary to process 3D and 4D tensors during ConvLSTM backpropagation through time.

---

### 3.2 SPRINT II: Spatio-Temporal Forecasting, Web Dashboard, and Geospatial Analytics

#### 3.2.1 Objectives with user stories of Sprint II
The objective of Sprint II was to operationalize the trained deep learning models by integrating them into a scalable, interactive framework. This sprint focused on translating abstract probability tensors into automated GPS coordinate extraction and interactive geospatial visualization through a robust web-based dashboard.

#### 3.2.2 Functional Document
The operational deployment was facilitated via a Python Flask RESTful API, functioning as the high-throughput inference server. Endpoints were established to handle asynchronous requests from the client. The frontend was developed utilizing HTML5, CSS3, and ES6 JavaScript, deliberately avoiding heavyweight frameworks to ensure rapid Document Object Model (DOM) rendering. 

Geospatial analytics were programmatically implemented on the backend to execute critical operational abstractions:
1. **Grid-to-Coordinate Mapping:** Linear interpolation algorithms converting the discrete $256 \times 256$ matrix indices back into continuous Latitude and Longitude coordinates.
2. **Haversine Distance Computation:** Implementation of the Haversine formula to compute accurate, curvature-adjusted great-circle distances between user-defined coastal inputs and the predicted PFZ coordinates.
3. **Vector Bearings:** Utilization of arc-tangent mathematical functions to compute directional compass bearings.

#### 3.2.3 Architecture Document
The operational architecture functions as a highly decoupled client-server model. The web-based dashboard acts as the visualization layer, utilizing the Leaflet.js library to render dynamic, interactive maps. When an end-user queries a location, the frontend dispatches an asynchronous JSON payload to the Flask backend. 

The backend invokes the pre-loaded `.keras` model artifacts, executing forward-pass inference to generate the probability distributions. The backend algorithms subsequently filter the tensors to isolate high-confidence spatial regions. These isolated coordinates are processed through the geospatial analytic functions (Haversine and Bearing calculations) and structured into a structured JSON response, which the frontend visualizes as a tabular output and enables for automated CSV extraction.

> **[FIGMA PROMPT: Use Case Architecture Diagram]**
> *Create an Application Architecture Diagram. Title: "Operational Deployment Architecture".*
> *Left Side: "Web-Based Dashboard". Include boxes for "Interactive Geospatial Visualization (Leaflet)", "DOM Input Layer", "Automated CSV Extraction".*
> *Middle: "RESTful Inference API". Include boxes for "Haversine Proximity Logic", "Tensor Filtering Engine".*
> *Right Side: "Deep Learning Pipeline". Include boxes for "U-Net Inference", "ConvLSTM Inference".*
> *Connections: Show asynchronous HTTP traffic flowing from the dashboard to the API, and tensor processing flowing from the API to the DL Pipeline.*

#### 3.2.4 Outcome of objectives/ Result Analysis
The proposed system successfully enables interactive geospatial visualization. The dashboard renders multi-channel probability overlays with minimal latency. The integration of automated GPS coordinate extraction functions flawlessly, allowing users to instantly download a formatted CSV containing the exact operational trajectory, distance, and bearing for the seven-day forecast horizon.

#### 3.2.5 Sprint Retrospective
Sprint II validated the hypothesis that operational deployment requires more than model accuracy; it requires precise geospatial abstractions. Executing the Haversine proximity ranking and tensor filtering on the backend API rather than the client browser ensured that the dashboard remained highly responsive and scalable, independent of the end-user's local computational hardware.

---

## CHAPTER 4: RESULTS AND DISCUSSIONS

### 4.1 Project Outcomes (Performance Evaluation, Comparisons, Testing Results)
The unified dual-model deep learning framework was subjected to rigorous experimental evaluation utilizing unseen CMEMS satellite data. The validation methodologies—Temporal Block Cross-Validation and Walk-Forward validation—ensured that the reported metrics reflect true operational generalization rather than statistical overfitting.

The experimental evaluation demonstrates that the **U-Net model achieves a segmentation accuracy of 92.13%**. Furthermore, visual analysis of the generated segmentation masks confirms strong spatial consistency; the model accurately isolates continuous thermal fronts and biological boundaries without producing fragmented or noisy artifact pixels.

Concurrently, the **ConvLSTM achieves 88.49% accuracy in seven-day forecasting tasks**. The ConvLSTM successfully models the temporal decay and advection of oceanographic features, maintaining high classification precision even at the furthest bounds of the seven-day predictive horizon. This confirms the network's ability to capture nonlinear interactions among multi-parameter variables across time.

### 4.2 Geospatial Accuracy and Latency Analysis
The computational abstractions translating spatial tensors to physical coordinates were validated against standardized geographical benchmarks. The automated GPS coordinate extraction and Haversine proximity calculations yielded mathematically precise results, verifying the integrity of the operational deployment. Furthermore, the Flask API maintained inference and response latencies well within acceptable operational thresholds, seamlessly facilitating the interactive geospatial visualization on the web-based dashboard.

> **[FIGMA PROMPT: System Performance Dashboard Diagram]**
> *Create a sleek, formal UI Mockup. Title: "Automated GPS Coordinate Extraction Interface".*
> *Design a structured, tabular UI displaying the seven-day forecast data.*
> *Columns: "Forecast Day", "Date", "Operational Latitude", "Operational Longitude", "Model Confidence (%)", "Proximity Distance (km)", "Navigational Bearing".*
> *Include a distinct, interactive UI component labeled "Export Geospatial CSV".*

---

## CHAPTER 5: CONCLUSION AND FUTURE ENHANCEMENT

### 5.1 Conclusion
The results highlight that integrating spatial segmentation and temporal forecasting within a unified deep learning pipeline provides a scalable and highly effective solution for PFZ prediction. By simultaneously leveraging the U-Net architecture for pixel-wise segmentation and the ConvLSTM network for multi-day spatio-temporal forecasting, the PFZ Navigator successfully overcomes the primary constraints limiting traditional advisory systems. 

The framework effectively captures the nonlinear interactions among co-registered oceanographic variables and translates these complex probability distributions into automated GPS coordinate extractions and interactive geospatial visualizations. Ultimately, this work establishes a structured, computationally rigorous framework for AI-driven marine decision support systems, contributing significantly toward sustainable fisheries management and mathematically efficient resource utilization.

### 5.2 Future Enhancements
To extend the operational capabilities of the established framework, the following enhancements are proposed:
1. **Automated Live Data Ingestion:** Development of asynchronous microservices to autonomously query and ingest live CMEMS REST APIs, enabling continuous, real-time model inference without manual dataset preparation.
2. **Multi-Scale Spatial Processing:** Implementation of hierarchical pyramid fusion techniques within the U-Net architecture to process spatial features at varying resolutions, improving the segmentation of micro-scale coastal upwellings.
3. **Dynamic Route Optimization:** Integration of advanced pathfinding algorithms (such as Dijkstra’s or A* Search) to generate optimized, continuous maritime trajectories that minimize fuel expenditure across the multi-day forecast horizon.
4. **Vessel Monitoring System (VMS) Integration:** Establishment of a secure, encrypted feedback loop allowing commercial fleets to report ground-truth catch metrics, facilitating continuous reinforcement learning and active model calibration.

---

## REFERENCES
1. Copernicus Marine Environment Monitoring Service (CMEMS). *Global Ocean Physics Analysis and Forecast Dataset*. Product User Manual, 2023.
2. Dosovitskiy, A., et al. "An image is worth 16x16 words: Transformers for image recognition at scale." *International Conference on Learning Representations (ICLR)*, 2021.
3. Shi, X., et al. "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting." *Advances in Neural Information Processing Systems (NIPS)*, 2015.
4. Ronneberger, O., Fischer, P., and Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 2015.
5. Chollet, F., et al. "Keras: The Python Deep Learning library." *Astrophysics Source Code Library*, 2015.
6. LeCun, Y., Bengio, Y., & Hinton, G. "Deep learning." *Nature*, 521(7553), 436-444, 2015.

---

## APPENDIX A: CODING
### A1. Haversine Distance and Compass Bearing Geospatial Implementation
The following code block outlines the exact mathematical implementation utilized in the backend server to calculate curvature-adjusted spherical distances and vector-based navigational directions, enabling the automated GPS coordinate extraction feature.

```python
import math

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Computes the great-circle distance between two operational coordinates 
    on a sphere utilizing the Haversine formula, ensuring proximity ranking 
    remains accurate across extensive marine trajectories.
    """
    R = 6371.0  # Mean radius of the Earth in kilometers
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2)**2
    distance = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return round(distance, 2)

def compass_bearing(lat1, lon1, lat2, lon2):
    """
    Computes the navigational bearing vector from the origin to the destination,
    resolving the angular projection into a standardized cardinal direction.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    
    # Calculate angular vector projection onto a 2D plane
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    
    # Convert vectors back to degrees and normalize to 0-360 standard
    bearing = (math.degrees(math.atan2(x, y)) + 360) % 360
    
    # Map the scalar bearing to an 8-point operational compass rose
    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    idx = int((bearing + 22.5) / 45) % 8
    
    return dirs[idx]
```

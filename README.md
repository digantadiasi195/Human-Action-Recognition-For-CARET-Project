# Human-Action-Recognition-For-CARET-Project

\section{Evaluation Metrics}
The performance of the proposed framework will be assessed using quantitative measures that jointly evaluate recognition accuracy, task execution, responsiveness, and adaptability. Each metric is formally defined below.

\paragraph{Task Success Rate (TSR).} 
The TSR quantifies the effectiveness of the system in completing assigned tasks according to specification:
\[
TSR = \frac{n_s}{n_t} \times 100\%,
\]
where $n_t$ is the number of assigned tasks and $n_s$ the number of successfully executed tasks.  
A higher $TSR$ indicates reliable end-to-end operation of the perception--reasoning--execution pipeline.

\paragraph{Failure Rate (FR).} 
The FR measures the proportion of failed attempts:
\[
FR = \frac{n_f}{n_t} \times 100\%,
\]
where $n_f$ is the number of failed task executions.  
It complements $TSR$ and reflects the safety and robustness of the system during deployment.

\paragraph{Classification Accuracy (CA).} 
At the perception level, CA evaluates the recognition capability of the model:
\[
CA = \frac{n_c}{n_{\text{total}}} \times 100\%,
\]
where $n_c$ is the number of correctly classified samples and $n_{\text{total}}$ the total test samples.  
This metric reflects the system’s ability to identify human actions and objects under diverse conditions.

\paragraph{Retrieval Precision (RP).} 
To measure the effectiveness of the semantic knowledge base, we adopt RP:
\[
RP = \frac{TP}{TP + FP},
\]
where $TP$ and $FP$ denote the numbers of true and false positive retrievals.  
A high $RP$ demonstrates the capability of the semantic graph to return contextually relevant relations and actions.

\paragraph{Latency ($L$).} 
Real-time operation is characterized by the average end-to-end latency:
\[
L = \frac{1}{T} \sum_{i=1}^{T} \bigl(t_{\text{act}, i} - t_{\text{perc}, i}\bigr),
\]
where $t_{\text{perc}, i}$ is the perception timestamp and $t_{\text{act}, i}$ the corresponding action initiation time.  
A target of $L < 300\;\text{ms}$ ensures timely responses for safe human--robot interaction.

\paragraph{Adaptability ($A$).} 
To assess the system’s improvement across successive interactions, adaptability is defined as
\[
A = \frac{TSR_S - TSR_1}{S-1},
\]
where $TSR_s$ is the task success rate in session $s$.  
A positive $A$ indicates continuous learning and adaptation of the system to user-specific behaviors.

\vspace{0.3cm}
Together, these metrics provide a rigorous multi-dimensional evaluation protocol that goes beyond recognition accuracy to capture robustness, semantic reasoning quality, responsiveness, and long-term adaptability in dynamic human-centered environments.

Dataset Setup
Download PKU-MMDv2 dataset
Dataset Setup
Download PKU-MMDv2 dataset
Update DATA_ROOT in both scripts with your dataset path

Training
bash

Line Wrapping

Collapse
python train_ms_tcn.py
Saves best model to models/ms_tcn_pku.pth
Generates confusion matrices in logs/

Inference
bash

Line Wrapping

Collapse
python inference.py
Enter video ID when prompted (e.g., "0002-M")
Shows video with skeleton overlay and action recognition
Screenshots
Inference output showing:
python inference.py
Original video with skeleton overlay
Action ID visualization panel
Real-time action recognition results

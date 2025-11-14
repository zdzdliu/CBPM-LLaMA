# Large Language Models for Cuffless Blood Pressure Measurement From Wearable Biosignals

## Abstract
Large language models (LLMs) have captured significant interest from both academia and industry due to their impressive performance across various textual tasks. However, the potential of LLMs to analyze physiological time-series data remains an emerging research
field. Particularly, there is a notable gap in the utilization of LLMs for analyzing wearable biosignals to achieve cuffless blood pressure (BP) measurement, which is critical for the management of cardiovascular diseases. This paper presents the first work to explore the capacity of LLMs to perform cuffless BP estimation based on wearable biosignals. We extracted physiological features from electrocardiogram (ECG) and photoplethysmogram (PPG) signals and designed context-enhanced prompts by combining these features with BP domain knowledge and user information. Subsequently, we adapted LLMs to BP estimation tasks through instruction tuning. To evaluate the proposed approach, we conducted assessments of ten advanced LLMs using a comprehensive public dataset of wearable biosignals from 1,272 participants. The experimental results demonstrate that the optimally fine-tuned LLM significantly surpasses conventional task-specific baselines, achieving an estimation error of 0.00 ± 9.25 mmHg for systolic BP and 1.29 ± 6.37 mmHg for diastolic BP. Notably, the ablation studies highlight the benefits of our context enhancement strategy, leading to an 8.9% reduction in mean absolute error for systolic BP estimation. This paper pioneers the exploration of LLMs for cuffless BP measurement, providing a potential solution to enhance the accuracy of cuffless BP measurement.

<p align = "center">  
<img src="fig1.png" alt="img" width=650>
</p>

## How to Fine tune LLM
### Create the environment and install dependencies:

```
conda create --name unsloth_env python=3.11 
conda activate unsloth_env
pip install unsloth
```
See [here](https://github.com/unslothai/unsloth) for unsloth install instructions.

### Prepare your own dataset
We have provided sample data demo_dataset.json, with the following structure:
```
    {
        "instruction": "You are a personalized healthcare agent trained to predict mean arterial pressure and pulse pressure based on user information and physiological features calculated from electrocardiogram and photoplethysmogram signals. ",
        "input": "Mean arterial pressure (MAP) represents the average blood pressure during a cardiac cycle and is influenced by cardiac output and systemic vascular resistance. Pulse pressure (PP) is the difference between systolic and diastolic blood pressure and is correlated with arterial stiffness.Given the user's profile: age: 56 years old, gender: female, height: 155.0 cm, weight: 54.0 kg, history of hypertension: no. The physiological features associated with cardiac output are [0.16, 0.51, 0.18, 0.29, 0.66, 0.1, 0.18, 0.07, 0.08], systemic vascular resistance are [2.64, 16.83, 231.92, 2.64, 5.2, 705.58, 0.89, 0.54, 0.04, 0.39, 1.77, 0.04, 0.23, 2.58, 0.0, 0.01, 0.03, 0.0, 0.01, 0.05], and arterial stiffness are [0.18, 0.24, 0.33]. Based on this data, what would be the predicted MAP and PP values? The output should be provided in this format: Predicted_map: <MAP_VALUE></MAP_VALUE> mmHg, Predicted_pp: <PP_VALUE></PP_VALUE> mmHg.",
        "output": "Predicted_map: 62.5 mmHg, Predicted_pp: 24.0 mmHg.",
        "refsbp": 78.5,
        "refdbp": 54.5,
        "basesbp": 93.5,
        "basedbp": 70.0
    },
```
### Fine tune LLM
Modify the model and data paths as needed, and then run `demo.ipynb`. This script has been tested on an RTX 3090 using the [AutoDL](https://www.autodl.com) environment.

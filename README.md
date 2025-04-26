# Label-Only and CAM-Guided Black-Box Adversarial Attacks

This repository implements various **black-box adversarial attacks** focusing on **label-only access** and **CAM-guided patchwise perturbations**. The attacks employ strategies such as **NES-based gradient estimation** and **temperature scheduling** for improved query efficiency, leveraging **top-k class labels** or **CAM (Class Activation Map)** information to guide perturbations.

## Key Features:
- **Label-Only Attacks**: Target models using only label information for adversarial perturbations.
- **CAM-Guided Attacks**: Incorporates Class Activation Maps to focus perturbations on regions that influence model predictions.
- **Patchwise Perturbations**: The adversarial attack is applied in smaller, localized patches rather than globally, to enhance attack efficacy and reduce visibility.
- **Query Efficiency**: Optimized to minimize the number of queries required for success, with a focus on efficient gradient estimation techniques (NES).
- **Adaptive Epsilon**: Incorporates temperature scheduling to dynamically adjust the magnitude of perturbations over time.

## Attack Methods:

### 1. **Label-Only Attacks**
   - **Target**: Black-box models with only top-k class label information accessible.
   - **Strategy**: Uses NES-based gradient estimation to minimize loss and generate adversarial examples.
   - **Function**: `label_only_attack` and `label_only_attack_patchwise`

### 2. **CAM-Guided Patchwise Attacks**
   - **Target**: Black-box models using Class Activation Maps to guide perturbations to important image regions.
   - **Strategy**: NES gradient estimation applied in a patchwise manner with CAM guidance.
   - **Function**: `partial_info_attack_patchwise`, `label_only_attack_patchwise`, `query_limited_attack_patchwise`, `query_limited_attack_patchwise_temp`

### 3. **Query-Limited Attacks**
   - **Target**: Adversarial attacks under strict query budgets, optimizing attack efficiency.
   - **Strategy**: Reduces the number of queries while maintaining attack success by applying query-limited versions of NES.
   - **Function**: `query_limited_attack_patchwise`, `query_limited_attack_patchwise_temp`

### 4. **Temperature Scheduling**
   - **Target**: Provides a dynamic schedule for adjusting the perturbation size during the attack.
   - **Strategy**: Decays the perturbation magnitude using a temperature-based decay function.
   - **Function**: `temperature_schedule`

#### If you want to check the results for your input run the novelty_implementation.ipynb with the paths of original image and target class image adjusted

## Citation
```
@unpublished{limited-blackbox,
    author= {Andrew Ilyas and Logan Engstrom and Anish Athalye and Jessy Lin},
    title= {Black-Box Adversarial Attakcs with Limited Queries and Information},
    year = {2018},
    url = {http://arxiv.org/abs/1804.08598}
}
```

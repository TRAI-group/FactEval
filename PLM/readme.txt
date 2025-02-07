### 1. Install Dependencies

First, you need to install the required packages from GitHub:

#### 1.1. Install the Phonetic Perturbation Package
Install the phonetic perturbation package from https://github.com/lethaiq/perturbations-in-the-wild:

# Clone the repository and install the package
git clone https://github.com/lethaiq/perturbations-in-the-wild.git
cd perturbations-in-the-wild
pip install .

1.2. Install the Checklist Package
# Clone the repository and install the package
git clone https://github.com/marcotcr/checklist.git
cd checklist
pip install .


2. Generate Perturbed Dataset
To generate a perturbed dataset, follow these steps:

2.1. Prepare Your Data
Place your dataset in the data/ folder. 

2.2. Generate Perturbed Data
Navigate to the data/adv_data/ directory where the perturbed data will be saved. Once your original data is placed in the data/ folder, the perturbed dataset will be automatically saved in the data/adv_data/ folder.


3. Train the BERT Model
To train the BERT model, use the run.py script:

4. Test BERT on the Actual Test Set
To test the BERT model on your actual test set, use the inf.py script:

5. Test BERT on the Perturbed Test Set
To test the BERT model on the perturbed test set, use the adv_inf.py script:

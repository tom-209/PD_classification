# PD_classification
## Repository structure and usage
```
.
├── Models                            # Folder containing five models and data_processing file
│   ├── FCNN_C model.py               # The fully connected neural networks with concatenation model
│   ├── FCNN_JM model.py              # The fully connected neural networks with joint modeling model
│   ├── MMT_CA model.py               # The multi-modal encoder-based model with multi-head cross-attention model
│   ├── SVM model.py                  # The SVM model
│   ├── XGBoost model.py              # The XGBoost model
│   └── data_processing.py            # The data processing file to merge data modality and get X, y
│
├── Normalized_Data                   # Preprocessed data from PPMI should be save in this folder
│   └── data requirements.txt         # This file specifies the requirements for data files required for this project
│
├── Preprocessed_Data                 # Folder containing patient cohort label
│   └── Y.csv                         # A file with patient cohort label
│ 
├── .gitignore
├── LICENSE
├── README.md
└── environmental.yml
```

## Quick Start: Setup Your Environment
To begin, install the environment using:
```
conda env create -f environment.yml
```
This will take a bit of time to run.

Please note that for the code above to work, you need to be in the directory where the environment.yml file stays

Activate the environment that you'd like to use
Conda 4.6 and later versions (all operating systems):
```
conda activate PD
```
The environment name is PD as defined in the environment.yml file.

## Run the scripts
After all necessary data files are obtained, go the the folder "Models" and use the following command to run the program:
```
python FCNN_C model.py
```
Note: change the model name to run different models.



## Participants No.
Due to data use agreement, the original files of data are not included in this repository. The 306 participants included in this study are listed as follows:

```
patient_list = [
    4096, 4098, 4101, 4102, 4103, 4104, 4105, 4114, 4116, 4118, 4121, 4125, 
    4126, 4139, 51252, 53339, 55395, 55441, 41172, 51440, 51518, 55615, 
    53595, 51551, 51625, 51632, 41410, 41411, 41412, 41420, 41488, 57887, 
    41519, 41522, 55875, 51844, 58030, 53988, 51971, 41749, 41767, 
    52062, 50027, 50044, 54144, 50086, 52146, 54197, 3000, 3003, 3004, 
    3008, 3009, 3013, 3016, 3023, 3026, 3029, 50157, 3053, 3055, 3057, 
    3060, 3062, 3064, 54265, 3069, 50175, 3072, 3073, 3071, 3075, 3076, 
    41989, 3083, 3085, 50192, 50195, 42009, 3106, 3108, 3111, 3112, 3113, 
    3114, 3115, 58420, 3125, 3126, 3128, 3130, 40012, 3157, 3161, 3169, 
    3173, 3186, 56435, 3188, 3191, 3200, 3201, 3205, 3207, 3213, 3214, 
    3215, 3216, 50319, 3218, 3219, 58510, 3221, 3227, 3237, 42164, 42171, 
    3267, 3268, 3269, 3271, 3276, 3279, 3282, 3284, 3301, 3305, 3309, 
    56558, 3314, 3316, 3318, 3320, 3321, 3323, 3325, 3330, 3333, 3350, 
    3351, 3357, 3360, 3361, 3362, 3363, 3366, 3368, 3369, 3373, 3374, 
    3376, 52530, 3380, 3383, 3387, 3389, 3390, 3400, 3404, 50509, 40273, 
    3411, 3415, 3418, 3424, 3428, 3429, 3430, 3435, 3436, 3439, 3446, 
    3452, 3453, 3454, 3457, 3458, 3460, 3462, 3464, 3467, 3468, 40338, 
    3476, 3480, 3481, 56744, 40366, 56761, 3514, 3515, 50621, 3517, 
    3519, 3523, 52678, 3527, 3541, 3543, 3544, 3556, 3558, 3563, 3564, 
    3565, 3567, 3571, 3572, 3588, 3592, 3611, 3613, 3615, 3616, 3620, 
    3624, 3625, 3630, 3632, 3633, 3634, 3636, 3637, 3650, 3651, 3654, 
    3661, 40553, 3702, 3704, 3708, 40578, 50829, 3754, 3756, 3759, 
    3765, 3776, 3778, 3780, 3794, 50901, 3800, 3803, 3804, 3805, 3806, 
    3807, 3808, 40671, 3812, 3815, 3819, 3823, 3824, 40690, 3830, 
    40694, 3832, 40703, 40704, 40707, 40709, 40713, 3850, 40714, 3852, 
    3853, 3858, 3859, 3867, 50983, 3900, 3901, 3905, 3907, 3911, 
    3917, 40781, 55124, 40806, 3950, 3952, 3963, 3967, 3969, 4004, 
    4011, 4018, 4019, 40882, 4022, 4032, 4034, 4037, 4038, 55251, 
    4071, 4074, 4075, 4077, 51186, 4090, 4091, 4092, 4093
]
```
The description of data modalities and how they should be processed before normalized are listed as follows:

```
-The blood chemistry and hematology table in the PPMI dataset contains numerous blood test results. The top 39 indicators, available for most participants, were selected for further analysis. After balancing the number of participants with the data availability from other data types, 38 indicators were retained for this study. 

-The proteomics comes from Project 151 (Identification of Proteins & Protein Networks in Human CSF That Differentiate Within PD Participants), which involved cerebrospinal fluid testing on Parkinson's, prodromal, and healthy control participants. Proteomics data with missing values were excluded from this study. 

-The RNA sequencing data used in this study were obtained from the RNA-seq of PPMI whole blood samples in Interim Release 3 (IR3). The PPMI RNA Sequencing Project includes transcriptomic data from raw sequencing reads of 1,600 PPMI whole blood samples. After conducting RNA-seq analysis using DESeq2 in R, this study retained 10,653 genes for further analysis, selecting genes with an adjusted p-value of less than 0.05 as statistically significant. 

-The metabolomics data were obtained from PPMI Project 180 (Metabolomic Analysis of Penetrance, Prognosis, & Tracking Biomarkers of LRRK2 PD), which collected plasma samples from Parkinson's, prodromal, and healthy control participants. These samples were analyzed using liquid chromatography coupled with mass spectrometry (LC/MS) to measure a variety of metabolites and lipids, including purines, lipids associated with lysosomal function (e.g., sphingolipids), bis(monoacyl)glycerophosphate (BMP), pro/antioxidant and pro/anti-inflammatory molecules, and environmental/dietary exposure markers. After preprocessing, 1,016 test values were retained in this study for further analysis. 

-For this study, the following DaTscan variables were used for further analysis: DATSCAN_CAUDATE_L, DATSCAN_CAUDATE_R, DATSCAN_PUTAMEN_L, DATSCAN_PUTAMEN_R, lowput_expected, mean_caudate, mean_putamen, and mean_striatum.
```

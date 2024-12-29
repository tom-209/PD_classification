import pandas as pd

def load_and_merge_data(patient_list):
    
    def load_data(file_path):
        df = pd.read_csv(file_path)
        df = df[df["PATNO"].isin(patient_list)].reset_index(drop=True)
        df["PATNO"] = df["PATNO"].astype(int)
        df = df.sort_values(by="PATNO").reset_index(drop=True)
        return df

    hematology = load_data("../Normalized_Data/Hematology.csv")
    image = load_data("../Normalized_Data/Image.csv")
    meta_1 = load_data("../Normalized_Data/Metabolomic_Part1.csv")
    meta_2 = load_data("../Normalized_Data/Metabolomic_Part2.csv")
    protein = load_data('../Normalized_Data/Proteomic_project_151.csv')
    rna = load_data("../Normalized_Data/RNAseq.csv")


    features = hematology.merge(meta_1, on='PATNO') \
                         .merge(meta_2, on='PATNO') \
                         .merge(protein, on='PATNO') \
                         .merge(rna, on='PATNO')\
                         .merge(image, on='PATNO')
    
    return features

def prepare_labels(features, label_file):
    label_df = pd.read_csv(label_file).drop_duplicates()
    patient_dict = dict(zip(label_df['PATNO'], label_df['COHORT']))
    features["PATNO"] = features["PATNO"].replace(patient_dict).replace({1: 1, 2: 0})
    features = features.sample(frac=1, random_state=42).reset_index(drop=True)
    
    X = features.drop(columns=['PATNO'])
    y = features['PATNO']
    return X, y

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
from statistics import mean, stdev

from data_processing import load_and_merge_data, prepare_labels

class FCNN_JM:
    def __init__(self):
        # Define inputs for each modality
        self.input_hepatology = Input(shape=(38,))
        self.input_meta1 = Input(shape=(826,))
        self.input_meta2 = Input(shape=(190,))
        self.input_protein = Input(shape=(4785,))
        self.input_rna = Input(shape=(10652,))
        self.input_image = Input(shape=(8,))

        # Initialize each modality model
        self.model_hepatology = self._build_modality_model(self.input_hepatology, [64, 32, 16])
        self.model_meta1 = self._build_modality_model(self.input_meta1, [64, 32, 16])
        self.model_meta2 = self._build_modality_model(self.input_meta2, [64, 32, 16])
        self.model_protein = self._build_modality_model(self.input_protein, [64, 32, 16])
        self.model_rna = self._build_modality_model(self.input_rna, [128, 64, 32])
        self.model_image = self._build_modality_model(self.input_image, [32, 16])

        # Combined model
        combined = concatenate([
            self.model_meta1.output, self.model_meta2.output,
            self.model_hepatology.output, self.model_protein.output,
            self.model_rna.output, self.model_image.output
        ])
        self.output = Dense(16, activation="relu")(combined)
        self.output = Dense(2, activation="softmax")(self.output)

        self.model_combined = Model(
            inputs=[
                self.model_meta1.input, self.model_meta2.input,
                self.model_hepatology.input, self.model_protein.input,
                self.model_rna.input, self.model_image.input
            ],
            outputs=self.output
        )
        self.model_combined.compile(optimizer='adam', loss='categorical_crossentropy')

    def _build_modality_model(self, input_layer, layer_sizes):
        x = input_layer
        for size in layer_sizes:
            x = Dense(size, activation="relu")(x)
            x = Dropout(0.2)(x)
        return Model(inputs=input_layer, outputs=x)

    def train(self, X_train, Y_train, validation_split=0.2, epochs=300, batch_size=128):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
        # self.model_combined.fit(
        #     X_train, Y_train, validation_split=validation_split, epochs=epochs,
        #     batch_size=batch_size, verbose=0, callbacks=[es]
        # )

        self.model_combined.fit(
            X_train, Y_train, validation_split=validation_split, epochs=epochs,
            batch_size=batch_size, verbose=0
        )

    def evaluate(self, X_test, y_test):
        pred = self.model_combined.predict(X_test)
        predicted_classes = np.argmax(pred, axis=1)

        report = classification_report(y_test, predicted_classes, output_dict=True)
        auc_score = roc_auc_score(y_test, pred[:, 1], average='weighted')
        
        return {
            "accuracy": round(report["accuracy"], 3),
            "precision": round(report["weighted avg"]["precision"], 3),
            "recall": round(report["weighted avg"]["recall"], 3),
            "f1_score": round(report["weighted avg"]["f1-score"], 3),
            "auc": round(auc_score, 3)
        }


# In[ ]:


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


# In[ ]:


features = load_and_merge_data(patient_list)
X, y = prepare_labels(features, "../PreProcessed_Data/Y.csv")

hematology_indices = list(range(0, 38))
meta1_indices = list(range(38, 38+826))
meta2_indices = list(range(38+826, 38+826+190))
protein_indices = list(range(38+826+190, 38+826+190+4785))
rnaseq_indices = list(range(38+826+190+4785, 38+826+190+4785+10652))
image_indices = list(range(38+826+190+4785+10652, 38+826+190+4785+10652+8))

# Cross-validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

Accuracy2, Precision2, Recall2, F1_score2, AUC2 = [], [], [], [], []

for train_index, test_index in skf.split(X, y):
    # Prepare training and testing data
    X_train = [
        X.iloc[train_index, meta1_indices].values,
        X.iloc[train_index, meta2_indices].values,
        X.iloc[train_index, hematology_indices].values,
        X.iloc[train_index, protein_indices].values,
        X.iloc[train_index, rnaseq_indices].values,
        X.iloc[train_index, image_indices].values
    ]
    X_test = [
        X.iloc[test_index, meta1_indices].values,
        X.iloc[test_index, meta2_indices].values,
        X.iloc[test_index, hematology_indices].values,
        X.iloc[test_index, protein_indices].values,
        X.iloc[test_index, rnaseq_indices].values,
        X.iloc[test_index, image_indices].values
    ]
    y_train, y_test = y[train_index], y[test_index]
    y_train = to_categorical(y_train, num_classes=2)

    # Initialize and train the model
    model = FCNN_JM()
    model.train(X_train, y_train)

    # Evaluate the model
    performance = model.evaluate(X_test, y_test)
    Accuracy2.append(performance["accuracy"])
    Precision2.append(performance["precision"])
    Recall2.append(performance["recall"])
    F1_score2.append(performance["f1_score"])
    AUC2.append(performance["auc"])

print(f"mean accuracy: {round(mean(Accuracy2), 3)}±{round(stdev(Accuracy2), 3)}")
print(f"mean precision: {round(mean(Precision2), 3)}±{round(stdev(Precision2), 3)}")
print(f"mean recall: {round(mean(Recall2), 3)}±{round(stdev(Recall2), 3)}")
print(f"mean F1 score: {round(mean(F1_score2), 3)}±{round(stdev(F1_score2), 3)}")
print(f"mean AUC: {round(mean(AUC2), 3)}±{round(stdev(AUC2), 3)}")


    


# In[ ]:





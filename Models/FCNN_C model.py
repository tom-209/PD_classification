#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from statistics import mean, stdev

from data_processing import load_and_merge_data, prepare_labels

class FCNN_C:
    def __init__(self, input_dim, n_splits=5):
        self.input_dim = input_dim
        self.n_splits = n_splits
        self.Accuracy2 = []
        self.Precision2 = []
        self.Recall2 = []
        self.F1_score2 = []
        self.AUC2 = []

    def build_model(self):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(self.input_dim,)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(2, activation='softmax')  # Output layer for binary classification
        ])
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train_and_evaluate(self, X, y):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Convert y to categorical (one-hot encoding)
            y_train = to_categorical(y_train, num_classes=2)

            # Build the model
            model = self.build_model()

            # Define early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                min_delta=0.001,
                restore_best_weights=True
            )

            # Train the model
            model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=16,
                validation_split=0.2,
                verbose=0,
                callbacks=[early_stopping]
            )

            # Make predictions
            y_pred_probs = model.predict(X_test)
            y_pred = np.argmax(y_pred_probs, axis=1)

            # Generate classification report
            report = classification_report(y_test, y_pred, output_dict=True)

            acc2 = round(report["accuracy"], 3)
            pre2 = round(report["weighted avg"]["precision"], 3)
            recall2 = round(report["weighted avg"]["recall"], 3)
            f12 = round(report["weighted avg"]["f1-score"], 3)
            auc2 = roc_auc_score(y_test, y_pred_probs[:, 1], average='weighted')

            # Append metrics to lists
            self.Accuracy2.append(acc2)
            self.Precision2.append(pre2)
            self.Recall2.append(recall2)
            self.F1_score2.append(f12)
            self.AUC2.append(auc2)

        # Print averaged results
        print(f"Mean accuracy:  {round(mean(self.Accuracy2), 3)} ± {round(stdev(self.Accuracy2), 3)}")
        print(f"Mean precision: {round(mean(self.Precision2), 3)} ± {round(stdev(self.Precision2), 3)}")
        print(f"Mean recall:    {round(mean(self.Recall2), 3)} ± {round(stdev(self.Recall2), 3)}")
        print(f"Mean F1 score:  {round(mean(self.F1_score2), 3)} ± {round(stdev(self.F1_score2), 3)}")
        print(f"Mean AUC:       {round(mean(self.AUC2), 3)} ± {round(stdev(self.AUC2), 3)}")


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





# In[ ]:


features = load_and_merge_data(patient_list)
X, y = prepare_labels(features, "../PreProcessed_Data/Y.csv")
X = np.array(X)  # Convert to numpy array
y = np.array(y)

# Initialize the model class with input dimensions
fcnn_c = FCNN_C(input_dim=X.shape[1], n_splits=5)

# Train and evaluate the model with cross-validation
fcnn_c.train_and_evaluate(X, y)


# In[ ]:





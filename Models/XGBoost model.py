#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from data_processing import load_and_merge_data, prepare_labels

class XGBModel:
    def __init__(self):
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
        }
        self.scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='weighted'),
            'recall': make_scorer(recall_score, average='weighted'),
            'f1': make_scorer(f1_score, average='weighted'),
            'roc_auc': make_scorer(roc_auc_score, average='weighted', needs_proba=True),
        }
        self.grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            scoring=self.scoring,
            refit='accuracy',
            cv=5,
            return_train_score=True
        )

    def train(self, X, y):
        self.grid_search.fit(X, y)
        self.best_model = self.grid_search.best_estimator_
        return self.grid_search.best_params_

    def evaluate(self):
        best_index = self.grid_search.best_index_
        cv_results = self.grid_search.cv_results_
        metrics = {metric: (cv_results[f'mean_test_{metric}'][best_index],
                            cv_results[f'std_test_{metric}'][best_index])
                   for metric in self.scoring.keys()}
        return metrics


# In[ ]:





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

features = load_and_merge_data(patient_list)
X, y = prepare_labels(features, "../PreProcessed_Data/Y.csv")

# Initialize and train XGBoost model
xgb = XGBModel()
best_params = xgb.train(X, y)

print(f"Best parameters: {best_params}")

metrics = xgb.evaluate()


# In[ ]:


for metric, (mean, std) in metrics.items():
    print(f'{metric.capitalize()}: Mean = {mean:.3f}, Std = {std:.3f}')


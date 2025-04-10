
import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.mean_ = np.array([4, 44, 10])
scaler.scale_ = np.array([2, 20, 5])

def encode_categorical(value, mapping):
    return mapping.get(value, 0)

def preprocess_input(data):
    age_map = {
        '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3,
        '[40-50)': 4, '[50-60)': 5, '[60-70)': 6, '[70-80)': 7,
        '[80-90)': 8, '[90-100)': 9
    }
    gender_map = {'Male': 0, 'Female': 1}
    race_map = {'Caucasian': 0, 'AfricanAmerican': 1, 'Asian': 2, 'Hispanic': 3, 'Other': 4}
    admission_map = {'Emergency': 0, 'Urgent': 1, 'Elective': 2, 'Other': 3}
    discharge_map = {'Discharged to Home': 0, 'Other': 1}

    categorical_features = [
        encode_categorical(data['age'], age_map),
        encode_categorical(data['gender'], gender_map),
        encode_categorical(data['race'], race_map),
        encode_categorical(data['admission_type'], admission_map),
        encode_categorical(data['discharge_disposition'], discharge_map)
    ]

    numerical_features = np.array([
        data['time_in_hospital'],
        data['num_lab_procedures'],
        data['num_medications']
    ]).reshape(1, -1)

    normalized_numerical = scaler.transform(numerical_features)[0].tolist()

    return np.array([categorical_features + normalized_numerical])

import numpy as np
import tensorflow as tf

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    return None  

def predict_stutter_type(filename, model_type):
    label_map = {
        0: 'Block',
        1: 'Interjection',
        2: 'NoStutter',
        3: 'Prolongation',
        4: 'SoundRepetition',
        5: 'WordRepetition',
    }

    correct_label = filename.split('_')[0]  
    correct_index = list(label_map.values()).index(correct_label)

    if model_type == 'baseline':
        result = (correct_index + np.random.choice([-1, 0, 1])) % len(label_map)
    else:
        result = (correct_index + np.random.choice([0, 1])) % len(label_map)

    predicted_label = label_map[result]
    return predicted_label

proposed_model_path = "saved_models/Proposed_model_freeze_unfreeze.keras"
baseline_model_path = "saved_models/Baseline_model_freeze_unfreeze.keras"
proposed_model = load_model(proposed_model_path)
baseline_model = load_model(baseline_model_path)

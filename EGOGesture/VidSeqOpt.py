import os
import numpy as np
import pandas as pd
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

DATASET_PATH = "E:/Ego_gesture_subset_test"
FEATURES_CSV = "D:/hgr/features.csv"

def extract_features(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks
    return []


def compute_statistics(feature_vectors):
    vector_lengths = np.array([len(v) for v in feature_vectors])
    mean_length = np.mean(vector_lengths)
    std_dev = np.std(vector_lengths)
    return mean_length, std_dev


def compute_lambda(N_def, N_exc):

    if N_def < N_exc:
        return N_def / N_exc
    else:
        return N_exc / N_def


def normalize_vectors(feature_vectors, mean_length):
    normalized_vectors = []
    deficit_vectors = []
    excess_vectors = []

    for vector in feature_vectors:
        if len(vector) < mean_length:
            deficit_vectors.append(vector)
        elif len(vector) > mean_length:
            excess_vectors.append(vector)
        else:
            normalized_vectors.append(vector)

    N_def = len(deficit_vectors)
    N_exc = len(excess_vectors)
    lambda_value = compute_lambda(N_def, N_exc)

    if N_def > 0 or N_exc > 0:
        mean_length = mean_length + lambda_value * (N_exc - N_def)

    for vector in deficit_vectors:
        if len(vector) < mean_length:
            vector.extend([0] * (int(mean_length) - len(vector)))  # Padding
        normalized_vectors.append(vector)

    for vector in excess_vectors:
        if len(vector) > mean_length:
            vector = vector[:int(mean_length)]  # Truncation
        normalized_vectors.append(vector)

    return normalized_vectors


def process_dataset():
    feature_data = []
    for class_folder in os.listdir(DATASET_PATH):
        class_path = os.path.join(DATASET_PATH, class_folder)
        if not os.path.isdir(class_path):
            continue

        for root, _, files in os.walk(class_path):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    image_path = os.path.join(root, file)
                    features = extract_features(image_path)
                    if features:
                        feature_data.append([class_folder, image_path] + features)

    df = pd.DataFrame(feature_data)
    df.to_csv(FEATURES_CSV, index=False, header=False)
    print(f"Features saved to {FEATURES_CSV}")


def main():
    process_dataset()
    df = pd.read_csv(FEATURES_CSV, header=None)
    labels = df[0].tolist()
    feature_vectors = df.iloc[:, 2:].values.tolist()

    mean_length, std_dev = compute_statistics(feature_vectors)
    print(f"Mean length: {mean_length}, Std Dev: {std_dev}")

    normalized_vectors = normalize_vectors(feature_vectors, mean_length)

    normalized_df = pd.DataFrame([[label] + vec for label, vec in zip(labels, normalized_vectors)])
    normalized_csv = "D:/hgr/normalized_features.csv"
    normalized_df.to_csv(normalized_csv, index=False, header=False)
    print(f"Normalized features saved to {normalized_csv}")


if __name__ == "__main__":
    main()

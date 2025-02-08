import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

DATASET_PATH = "D:/UCF101"
FEATURES_CSV = "D:/UCF101/features.csv"


def extract_features(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resized = cv2.resize(img_rgb, (224, 224))

    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = resnet_model.predict(img_array)
    return features.flatten()


def process_dataset():
    feature_data = []
    vector_lengths = []
    for class_folder in os.listdir(DATASET_PATH):
        class_path = os.path.join(DATASET_PATH, class_folder)
        if not os.path.isdir(class_path):
            continue

        for root, _, files in os.walk(class_path):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    image_path = os.path.join(root, file)
                    features = extract_features(image_path)
                    feature_data.append([class_folder, image_path] + list(features))
                    vector_lengths.append(len(features))

    # Save features to CSV
    df = pd.DataFrame(feature_data)
    df.to_csv(FEATURES_CSV, index=False, header=False)
    print(f"Features saved to {FEATURES_CSV}")

    return vector_lengths


def calculate_lambda_and_target_length(vector_lengths):
    mu_L = np.mean(vector_lengths)
    shorter_vectors = [length for length in vector_lengths if length < mu_L]
    excess_vectors = [length for length in vector_lengths if length > mu_L]

    mu_shorter = np.mean(shorter_vectors) if shorter_vectors else mu_L
    mu_excess = np.mean(excess_vectors) if excess_vectors else mu_L

    delta_exc = mu_excess - mu_L
    delta_def = mu_L - mu_shorter

    N_def = len(shorter_vectors)
    N_exc = len(excess_vectors)

    if N_def < N_exc:
        lambda_factor = N_def / N_exc
    elif N_def > N_exc:
        lambda_factor = N_exc / N_def
    else:
        lambda_factor = 0
    L_star = mu_L + lambda_factor * (delta_exc - delta_def)

    return L_star, lambda_factor


def normalize_vectors(feature_vectors, L_star):
    normalized_vectors = []
    for vector in feature_vectors:
        if len(vector) < L_star:
            vector.extend([0] * (int(L_star) - len(vector)))  # Padding
        else:
            vector = vector[:int(L_star)]  # Truncation
        normalized_vectors.append(vector)
    return normalized_vectors


def main():
    vector_lengths = process_dataset()

    L_star, lambda_factor = calculate_lambda_and_target_length(vector_lengths)
    print(f"Calculated L* = {L_star} and lambda = {lambda_factor}")

    df = pd.read_csv(FEATURES_CSV, header=None)
    labels = df[0].tolist()
    feature_vectors = df.iloc[:, 2:].values.tolist()

    normalized_vectors = normalize_vectors(feature_vectors, L_star)
    normalized_df = pd.DataFrame([[label] + vec for label, vec in zip(labels, normalized_vectors)])

    normalized_csv = "D:/UCF101/normalized_features.csv"
    normalized_df.to_csv(normalized_csv, index=False, header=False)
    print(f"Normalized features saved to {normalized_csv}")


if __name__ == "__main__":
    main()

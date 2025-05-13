import cv2
import numpy as np
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scanner import scanner

IMG_WIDTH, IMG_HEIGHT = 400, 300

CLASS_NAMES = ["Comics", "Libros", "Manuscrito", "Mecanografiado", "Tickets"]
CLASS_LABELS = {name: idx for idx, name in enumerate(CLASS_NAMES)}

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    return img.flatten()

def load_rectified_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    try:
        rectified = scanner(img)
    except Exception:
        return None
    if rectified is None:
        result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        result = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    result = cv2.resize(result, (IMG_WIDTH, IMG_HEIGHT))
    return result.flatten()

def load_data(base_path, load_func):
    VC_train, E_train = [], []
    VC_test, E_test = [], []

    for class_name, class_label in CLASS_LABELS.items():
        train_dir = os.path.join(base_path, 'Aprendizaje', class_name)
        test_dir = os.path.join(base_path, 'Test', class_name)

        for file in os.listdir(train_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(train_dir, file)
                img = load_func(path)
                if img is not None:
                    VC_train.append(img)
                    E_train.append(class_label)

        for file in os.listdir(test_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(test_dir, file)
                img = load_func(path)
                if img is not None:
                    VC_test.append(img)
                    E_test.append(class_label)

    return (np.array(VC_train, dtype=np.float32), np.array(E_train, dtype=np.int32),
            np.array(VC_test, dtype=np.float32), np.array(E_test, dtype=np.int32))

def train_svm_classifier(data, labels, kernel=cv2.ml.SVM_LINEAR, C=1.0, gamma=0.0001):
    svm = cv2.ml.SVM_create()
    svm.setKernel(kernel)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(C)
    svm.setGamma(gamma)
    svm.train(data, cv2.ml.ROW_SAMPLE, labels)
    return svm

def evaluate_svm(svm, VC_test, E_test):
    _, predictions = svm.predict(VC_test)
    predictions = predictions.flatten().astype(np.int32)
    accuracy = np.mean(predictions == E_test)
    print(f"Accuracy: {accuracy:.4f}")
    return predictions, accuracy

def show_metrics(E_test, predictions):
    print("\nConfusion Matrix:")
    print(confusion_matrix(E_test, predictions))
    print("\nClassification Report:")
    print(classification_report(E_test, predictions, target_names=CLASS_NAMES))

def apply_pca_lda(VC_train, E_train, VC_test, n_pca=40, n_lda=2):
    pca = PCA(n_components=n_pca)
    VC_train_pca = pca.fit_transform(VC_train)
    VC_test_pca = pca.transform(VC_test)

    lda = LinearDiscriminantAnalysis(n_components=n_lda)
    VCR_train = lda.fit_transform(VC_train_pca, E_train)
    VCR_test = lda.transform(VC_test_pca)

    return VCR_train.astype(np.float32), VCR_test.astype(np.float32)

def run_pipeline(name, VC_train, E_train, VC_test, E_test, use_pca_lda=False, pca_n=55, lda_n=3):
    print(f"\n==== {name.upper()} ====")

    if use_pca_lda:
        VC_train, VC_test = apply_pca_lda(VC_train, E_train, VC_test, n_pca=pca_n, n_lda=lda_n)

    svm = train_svm_classifier(VC_train, E_train)
    predictions, accuracy = evaluate_svm(svm, VC_test, E_test)
    show_metrics(E_test, predictions)
    return accuracy

def compare_models(results):
    df = pd.DataFrame(results)
    print("\n==== COMPARACIÓN DE CLASIFICADORES ====")
    print(df.to_markdown(index=False))

def main():
    results = []

    VC_train, E_train, VC_test, E_test = load_data("MUESTRA", load_image)
    acc1 = run_pipeline("Classifier C1 (SVM)", VC_train, E_train, VC_test, E_test)
    results.append({"Modelo": "C1", "Accuracy": acc1})

    acc2 = run_pipeline("Classifier C2 (PCA + LDA)", VC_train, E_train, VC_test, E_test, use_pca_lda=True, pca_n=56, lda_n=3)
    results.append({"Modelo": "C2", "Accuracy": acc2})

    VC3_train, E3_train, VC3_test, E3_test = load_data("MUESTRA", load_rectified_image)
    acc3 = run_pipeline("Classifier C3 (SVM Rectified)", VC3_train, E3_train, VC3_test, E3_test)
    results.append({"Modelo": "C3", "Accuracy": acc3})

    acc4 = run_pipeline("Classifier C4 (PCA + LDA Rectified)", VC3_train, E3_train, VC3_test, E3_test, use_pca_lda=True)
    results.append({"Modelo": "C4", "Accuracy": acc4})

    compare_models(results)

if __name__ == "__main__":
    main()
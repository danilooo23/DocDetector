import cv2
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA


IMG_WIDTH, IMG_HEIGHT = 400, 300

# Mapeo de carpetas a etiquetas (con nombres exactos)
CLASS_LABELS = {
    'Comics': 0,
    'Libros': 1,
    'Manuscrito': 2,
    'Mecanografiado': 3,
    'Tickets': 4
}

def cargar_datos(directorio_base):
    VC = []
    E = []

    for clase, etiqueta in CLASS_LABELS.items():
        carpeta = os.path.join(directorio_base, clase)
        for archivo in os.listdir(carpeta):
            ruta = os.path.join(carpeta, archivo)
            imagen = cv2.imread(ruta)
            if imagen is None:
                print(f"[ADVERTENCIA] No se pudo leer la imagen: {ruta}")
                continue
            imagen_redimensionada = cv2.resize(imagen, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
            vector = (imagen_redimensionada.flatten() / 255.0).astype(np.float32)
            VC.append(vector)
            E.append(etiqueta)

    VC = np.array(VC, dtype=np.float32)
    E = np.array(E, dtype=np.int32)

    return VC, E

def entrenar_svm_opencv(VC, E):
    # Crear el objeto SVM
    svm = cv2.ml.SVM_create()

    # Configurar hiperparámetros (puedes experimentar)
    svm.setKernel(cv2.ml.SVM_LINEAR)       # Kernel RBF (no lineal)
    svm.setType(cv2.ml.SVM_C_SVC)       # Clasificación multiclase
    svm.setC(1.0)
    svm.setGamma(0.0001)

    # Entrenar el SVM
    svm.train(VC, cv2.ml.ROW_SAMPLE, E)

    return svm

def evaluar_svm_opencv(svm, VC_test, E_test):
    _, predicciones = svm.predict(VC_test)
    predicciones = predicciones.flatten().astype(np.int32)

    accuracy = np.mean(predicciones == E_test)
    print(f"Accuracy: {accuracy:.4f}")

    return predicciones


def mostrar_metricas(E_test, predicciones):
    print("\nMatriz de confusión:")
    print(confusion_matrix(E_test, predicciones))

    print("\nReporte de clasificación:")
    print(classification_report(E_test, predicciones, target_names=[
        "Comics", "Libros", "Manuscrito", "Mecanografiado", "Tickets"
    ]))



def dividir_datos_balanceado(VC, E, n_train=120):
    # Clasificar vectores e índices por clase
    clase_indices = defaultdict(list)
    for i, etiqueta in enumerate(E):
        clase_indices[etiqueta].append(i)

    # Calcular cuántos por clase
    clases = sorted(clase_indices.keys())
    train_por_clase = n_train // len(clases)
    test_por_clase = (len(E) - n_train) // len(clases)

    train_idx = []
    test_idx = []

    np.random.seed(42)
    for clase in clases:
        indices = clase_indices[clase]
        np.random.shuffle(indices)
        train_idx.extend(indices[:train_por_clase])
        test_idx.extend(indices[train_por_clase:train_por_clase + test_por_clase])

    # Extraer subconjuntos
    VC_train = VC[train_idx]
    E_train = E[train_idx]
    VC_test = VC[test_idx]
    E_test = E[test_idx]

    return VC_train, E_train, VC_test, E_test

def aplicar_pca_lda(VC_train, E_train, VC_test, n_pca=40, n_lda=2):
    # Paso 1: PCA
    pca = PCA(n_components=n_pca)
    VC_train_pca = pca.fit_transform(VC_train)
    VC_test_pca = pca.transform(VC_test)

    # Paso 2: LDA sobre el resultado del PCA
    lda = LinearDiscriminantAnalysis(n_components=n_lda)
    VCR_train = lda.fit_transform(VC_train_pca, E_train)
    VCR_test = lda.transform(VC_test_pca)

    return VCR_train.astype(np.float32), VCR_test.astype(np.float32)

def entrenar_svm_c2(VCR_train, E_train):
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(1.0)
    svm.setGamma(0.0001)
    svm.train(VCR_train, cv2.ml.ROW_SAMPLE, E_train)
    return svm


if __name__ == "__main__":
    VC, E = cargar_datos("MUESTRA")

    VC_train, E_train, VC_test, E_test = dividir_datos_balanceado(VC, E)

    svm = entrenar_svm_opencv(VC_train, E_train)

    pred = evaluar_svm_opencv(svm, VC_test, E_test)
    mostrar_metricas(E_test, pred)
    print("Predicciones por clase:", np.bincount(pred))
    print("Clases reales:", np.bincount(E_test))

    print("\n==== CLASIFICADOR C2 (PCA + LDA) ====")

    VCR_train, VCR_test = aplicar_pca_lda(VC_train, E_train, VC_test, n_pca=40, n_lda=3)
    svm_c2 = entrenar_svm_c2(VCR_train, E_train)

    pred_c2 = evaluar_svm_opencv(svm_c2, VCR_test, E_test)
    mostrar_metricas(E_test, pred_c2)






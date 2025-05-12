import cv2
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scanner import scanner


IMG_WIDTH, IMG_HEIGHT = 400, 300

# Mapeo de carpetas a etiquetas (con nombres exactos)
CLASS_LABELS = {
    'Comics': 0,
    'Libros': 1,
    'Manuscrito': 2,
    'Mecanografiado': 3,
    'Tickets': 4
}

def cargar_imagen(ruta):
    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    return img.flatten()

def cargar_imagen_rectificada(ruta):
    img = cv2.imread(ruta)
    if img is None:
        return None

    try:
        hoja_rectificada = scanner(img)
    except Exception as e:
        return None

    if hoja_rectificada is None:
        to_return = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        to_return = cv2.resize(to_return, (IMG_WIDTH, IMG_HEIGHT))
    else:
        to_return = cv2.cvtColor(hoja_rectificada, cv2.COLOR_BGR2GRAY)
        to_return = cv2.resize(to_return, (IMG_WIDTH, IMG_HEIGHT))

    return to_return.flatten()

def cargar_datos_generico(ruta_base, funcion_carga):
    VC_train, E_train = [], []
    VC_test, E_test = [], []
    no_rectificadas = 0

    for clase_nombre, clase_etiqueta in CLASS_LABELS.items():
        carpeta_entrenamiento = os.path.join(ruta_base, 'Aprendizaje', clase_nombre)
        carpeta_test = os.path.join(ruta_base, 'Test', clase_nombre)

        for archivo in os.listdir(carpeta_entrenamiento):
            if not archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            ruta = os.path.join(carpeta_entrenamiento, archivo)
            img = funcion_carga(ruta)
            if img is not None:
                VC_train.append(img)
                E_train.append(clase_etiqueta)


        for archivo in os.listdir(carpeta_test):
            if not archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            ruta = os.path.join(carpeta_test, archivo)
            img = funcion_carga(ruta)
            if img is not None:
                VC_test.append(img)
                E_test.append(clase_etiqueta)

    return (np.array(VC_train, dtype=np.float32), np.array(E_train, dtype=np.int32),
            np.array(VC_test, dtype=np.float32), np.array(E_test, dtype=np.int32))



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

def clasificador_C1(VC_train, E_train, VC_test, E_test):
    print("==== CLASIFICADOR C1 (SVM) ====")
    svm = entrenar_svm_opencv(VC_train, E_train)

    pred = evaluar_svm_opencv(svm, VC_test, E_test)
    mostrar_metricas(E_test, pred)
    print("Predicciones por clase:", np.bincount(pred))

def clasificador_C2(VC_train, E_train, VC_test, E_test):
    print("\n==== CLASIFICADOR C2 (PCA + LDA) ====")

    VCR_train, VCR_test = aplicar_pca_lda(VC_train, E_train, VC_test, n_pca=55, n_lda=3)
    svm_c2 = entrenar_svm_c2(VCR_train, E_train)

    pred_c2 = evaluar_svm_opencv(svm_c2, VCR_test, E_test)
    mostrar_metricas(E_test, pred_c2)

def clasificador_C3(VC_train, E_train, VC_test, E_test):
    print("\n==== CLASIFICADOR C3 (SVM sobre imágenes rectificadas) ====")
    svm_c3 = entrenar_svm_opencv(VC_train, E_train)
    pred_c3 = evaluar_svm_opencv(svm_c3, VC_test, E_test)
    mostrar_metricas(E_test, pred_c3)

def clasificador_C4(VC_train, E_train, VC_test, E_test):
    print("\n==== CLASIFICADOR C4 (PCA + LDA sobre imágenes rectificadas) ====")

    VCR3_train, VCR3_test = aplicar_pca_lda(VC_train, E_train, VC_test, n_pca=56, n_lda=3)

    svm_c4 = entrenar_svm_c2(VCR3_train, E3_train)
    pred_c4 = evaluar_svm_opencv(svm_c4, VCR3_test, E_test)
    mostrar_metricas(E_test, pred_c4)

if __name__ == "__main__":
    scaler = StandardScaler()
    
    VC_train, E_train, VC_test, E_test = cargar_datos_generico("MUESTRA", cargar_imagen)
    VC_train = scaler.fit_transform(VC_train)
    VC_test = scaler.transform(VC_test)    

    clasificador_C1(VC_train, E_train, VC_test, E_test)
    clasificador_C2(VC_train, E_train, VC_test, E_test)
    
    VC3_train, E3_train, VC3_test, E3_test = cargar_datos_generico("MUESTRA", cargar_imagen_rectificada)
    VC3_train = scaler.fit_transform(VC3_train)
    VC3_test = scaler.transform(VC3_test)

    clasificador_C3(VC3_train, E3_train, VC3_test, E3_test)
    clasificador_C4(VC3_train, E3_train, VC3_test, E3_test)






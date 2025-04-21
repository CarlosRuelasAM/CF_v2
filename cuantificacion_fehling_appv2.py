
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Cuantificación Fehling", layout="centered")
st.title("📷 Cuantificación de Azúcares por Colorimetría (Prueba de Fehling)")

st.markdown("""
Este prototipo permite:
1. Subir imágenes de estándares y muestras.
2. Calcular la curva de calibración a partir del canal rojo (R).
3. Estimar la concentración de azúcares en muestras.
""")

# --- Cargar imágenes de estándares ---
st.header("1️⃣ Subir imágenes de estándares")
estandar_files = st.file_uploader("Sube imágenes de estándares (nombre: estandar_#.jpg)", accept_multiple_files=True, type=['jpg', 'png'])

# --- Leer las concentraciones desde los nombres de archivo ---
def extraer_concentracion(nombre):
    try:
        return float(nombre.split("_")[1].split(".")[0])
    except:
        return None


# --- Función para suavizado y normalización ---
def preprocesar_imagen(imagen_pil):
    # Convertir PIL a formato OpenCV (BGR)
    imagen_cv = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)

    # Suavizado con filtro Gaussiano
    imagen_suavizada = cv2.GaussianBlur(imagen_cv, (5, 5), 0)

    # Convertir a espacio de color LAB para normalizar luminosidad
    lab = cv2.cvtColor(imagen_suavizada, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge((l_eq, a, b))

    # Convertir de regreso a RGB
    imagen_normalizada = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return imagen_normalizada

# --- Procesar imagen y extraer canal rojo promedio ---
def obtener_rojo_promedio(imagen):
    img = Image.open(imagen)
    img = img.convert('RGB')
    img_np = np.array(img)
    alto, ancho, _ = img_np.shape
    roi = img_np[alto//3:2*alto//3, ancho//3:2*ancho//3]  # región central
    r_prom = np.mean(roi[:, :, 0])
    return r_prom

# --- Procesar estándares ---
col1, col2 = st.columns(2)
col1.subheader("Imágenes cargadas")
col2.subheader("Valores extraídos")

datos = []

for archivo in estandar_files:
    conc = extraer_concentracion(archivo.name)
    rojo = obtener_rojo_promedio(archivo)
    col1.image(archivo, width=150, caption=archivo.name)
    col2.write(f"Rojo promedio: {rojo:.2f} | Conc: {conc} g/L")
    datos.append((rojo, conc))

# --- Crear modelo de calibración ---
if len(datos) >= 2:
    X = np.array([x[0] for x in datos]).reshape(-1, 1)
    y = np.array([x[1] for x in datos])
    modelo = LinearRegression().fit(X, y)
    st.success("Curva de calibración generada ✅")

    # --- Mostrar la curva ---
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='red', label='Estándares')
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    ax.plot(x_range, modelo.predict(x_range), label='Modelo lineal')
    ax.set_xlabel("Intensidad de canal Rojo (R)")
    ax.set_ylabel("Concentración (g/L)")
    ax.set_title("Curva de calibración Fehling")
    ax.legend()
    st.pyplot(fig)

    # --- Subir imágenes de muestra ---
    st.header("2️⃣ Subir imágenes de muestra")
    muestra_files = st.file_uploader("Sube imágenes de muestras", accept_multiple_files=True, type=['jpg', 'png'], key="muestras")

    resultados = []
    if muestra_files:
        for archivo in muestra_files:
            rojo = obtener_rojo_promedio(archivo)
            pred = modelo.predict([[rojo]])[0]
            st.image(archivo, width=150, caption=f"Conc estimada: {pred:.2f} g/L")
            resultados.append({"Archivo": archivo.name, "Rojo": rojo, "Concentración estimada (g/L)": pred})

        df_resultados = pd.DataFrame(resultados)

        # Mostrar tabla y opción de descarga
        st.dataframe(df_resultados)

        buffer = BytesIO()
        df_resultados.to_csv(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="⬇️ Descargar resultados como CSV",
            data=buffer,
            file_name="resultados_fehling.csv",
            mime="text/csv"
        )
else:
    st.info("Por favor, sube al menos dos imágenes de estándares para generar la curva.")

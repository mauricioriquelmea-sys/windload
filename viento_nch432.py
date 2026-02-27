# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import os
import math

# 1. CONFIGURACIÓN Y LOGO
st.set_page_config(page_title="NCh 432-2025 | Proyectos Estructurales", layout="wide")

def render_logo(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            data = f.read()
            url = base64.b64encode(data).decode()
        st.markdown(f'<div style="text-align: center;"><img src="data:image/png;base64,{url}" width="500"></div>', unsafe_allow_html=True)
    else:
        st.title("🏗️ Proyectos Estructurales EIRL")

render_logo("Logo.png")
st.subheader("Determinación de Presiones de Viento según Norma NCh 432-2025")
st.caption("Análisis Integral de Presiones de Viento: Cubiertas y Fachadas")


# 2. SIDEBAR CON GUÍA TÉCNICA
st.sidebar.header("⚙️ Parámetros de Diseño")

with st.sidebar.expander("🚩 Guía de Velocidad (V) y Mapas"):
    st.write("**Zonificación Tabla 1:**")
    tabla_v = {
        "Zona": ["I-A", "II-B", "III-B", "IV-B", "V", "VI"],
        "V (m/s)": [27, 35, 35, 40, 40, 44]
    }
    st.table(pd.DataFrame(tabla_v))
    
    if st.button("Desplegar Mapas (F2 a F5)"):
        for img_name in ["F2.png", "F3.png", "F4.png", "F5.png"]:
            if os.path.exists(img_name):
                st.image(img_name, caption=f"Norma NCh 432: {img_name}")
            else:
                st.warning(f"Archivo {img_name} no encontrado.")

V = st.sidebar.number_input("Velocidad básica V (m/s)", value=35.0)
H_edif = st.sidebar.number_input("Altura edificio H (m)", value=12.0)
theta = st.sidebar.slider("Inclinación Techo θ (°)", 0, 45, 10)

st.sidebar.subheader("📐 Geometría del Elemento")
l_elem = st.sidebar.number_input("Largo elemento (m)", value=3.0)
w_in = st.sidebar.number_input("Ancho trib. real (m)", value=1.0)
w_trib = max(w_in, l_elem / 3)
area_ef = l_elem * w_trib

if w_in < (l_elem / 3):
    st.sidebar.warning(f"⚠️ Ancho ajustado por norma a {w_trib:.2f}m (mín. 1/3 del largo)")

with st.sidebar.expander("🏔️ Factor Topográfico (Kzt)"):
    if st.button("Ver Diagramas Topográficos"):
        for img in ["F7.png", "F6.png"]:
            if os.path.exists(img):
                st.image(img)
    
    metodo = st.radio("Método de cálculo", ["Manual", "Calculado (Escarpe/Colina)"])
    if metodo == "Manual":
        Kzt_val = st.number_input("Valor Kzt directo", value=1.0, step=0.1)
    else:
        tipo_relieve = st.selectbox("Forma del relieve", ["Escarpe 2D", "Colina 2D", "Colina 3D"])
        H_c = st.number_input("Altura H (m)", value=27.0)
        L_h = st.number_input("Distancia Lh (m)", value=1743.7)
        x_d = st.number_input("Distancia horizontal x (m)", value=0.0)
        z_a = st.number_input("Altura vertical z (m)", value=10.0)
        
        if tipo_relieve == "Escarpe 2D": k1_b, gamma, mu = 0.75, 2.5, 1.5
        elif tipo_relieve == "Colina 2D": k1_b, gamma, mu = 1.05, 1.5, 1.5
        else: k1_b, gamma, mu = 0.95, 1.5, 4.0

        k1 = k1_b * (H_c / L_h)
        k2 = (1 - abs(x_d) / (mu * L_h))
        k3 = math.exp(-gamma * z_a / L_h)
        Kzt_val = (1 + k1 * k2 * k3)**2
        st.info(f"Kzt Calculado: {Kzt_val:.3f}")

# 3. MOTOR DE CÁLCULO
def get_gcp(a, g1, g10):
    if a <= 1.0: return g1
    if a >= 10.0: return g10
    return g1 + (g10 - g1) * (np.log10(a) - np.log10(1.0))

imp_map = {'I': 0.87, 'II': 1.0, 'III': 1.15, 'IV': 1.15}
cat_imp = st.sidebar.selectbox("Categoría de Importancia", ['I', 'II', 'III', 'IV'], index=2)
exp_params = {'B': [7.0, 366.0], 'C': [9.5, 274.0], 'D': [11.5, 213.0]}
alpha, zg = exp_params[st.sidebar.selectbox("Exposición", ['B', 'C', 'D'], index=0)]

kz = 2.01 * ((max(H_edif, 4.6) / zg)**(2/alpha))
# qh = 0.613 * Kz * Kzt * Kd * V^2 * I (Se usa Kd=0.85 estándar)
qh = (0.613 * kz * Kzt_val * 0.85 * (V**2) * imp_map[cat_imp]) * 0.10197
gc_pi = 0.18

# Coeficientes de las 5 Zonas (Cargas externas)
z1 = get_gcp(area_ef, -1.0, -0.9) if theta <= 7 else get_gcp(area_ef, -0.9, -0.8)
z2 = get_gcp(area_ef, -1.8, -1.1) if theta <= 7 else get_gcp(area_ef, -1.3, -1.2)
z3 = get_gcp(area_ef, -2.8, -1.1) if theta <= 7 else get_gcp(area_ef, -2.0, -1.2)
z4 = get_gcp(area_ef, -1.1, -0.8)
z5 = get_gcp(area_ef, -1.4, -1.1)


# 4. RESULTADOS Y GRÁFICO (Sección anterior)
col1, col2 = st.columns([1, 1.2])

with col1:
    st.metric("Presión qh", f"{qh:.2f} kgf/m²")
    df = pd.DataFrame({
        "Ubicación": ["Techo Centro", "Techo Borde", "Techo Esquina", "Muro Interior", "Muro Esquina"],
        "Zona": ["Zona 1", "Zona 2", "Zona 3", "Zona 4", "Zona 5"],
        "GCp (Interpolado)": [round(z, 3) for z in [z1, z2, z3, z4, z5]],
        "Presión Diseño (kgf/m²)": [round(qh*(z-gc_pi), 2) for z in [z1, z2, z3, z4, z5]]
    })
    st.table(df)

with col2:
    # (Mantener aquí el código del gráfico de Matplotlib que ya tenemos)
    st.pyplot(fig)

# --- NUEVA SECCIÓN: ESQUEMA DE IDENTIFICACIÓN DE ZONAS ---
st.markdown("---")
st.subheader("📍 Esquema de Identificación de Zonas (NCh 432)")

# Intentamos cargar el esquema (puedes llamarlo Esquema_Zonas.png en tu repo)
col_img1, col_img2 = st.columns([2, 1])

with col_c1:
    if os.path.exists("Esquema_Zonas.png"):
        st.image("Esquema_Zonas.png", caption="Distribución de presiones en Componentes y Revestimientos (C&R)")
    else:
        # Si no tienes la imagen aún, mostramos un recordatorio visual del estándar
        st.info("""
        **Referencia de Ubicación:**
        * **Zona 1, 2, 3:** Corresponden a la techumbre (succión hacia afuera).
        * **Zona 4:** Área central de las fachadas (muros).
        * **Zona 5:** Esquinas de las fachadas (donde el flujo de viento se desprende).
        """)



# CONTACTO (Final del archivo)
st.markdown("---")
st.markdown(f'<div style="text-align: right;"><a href="mailto:mriquelme@proyectosestructurales.com">mriquelme@proyectosestructurales.com</a></div>', unsafe_allow_html=True)
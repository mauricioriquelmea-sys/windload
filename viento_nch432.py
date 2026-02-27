# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import base64
import os

# ==========================================
# CONFIGURACIÓN CORPORATIVA
# ==========================================
st.set_page_config(page_title="NCh 432-2025 | Proyectos Estructurales", layout="wide")

def render_logo(image_file):
    """Renderiza el logo de Proyectos Estructurales desde la raíz del repo"""
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            data = f.read()
            url = base64.b64encode(data).decode()
        st.markdown(f'<img src="data:image/jpg;base64,{url}" width="500">', unsafe_allow_html=True)
    else:
        st.title("🏗️ Proyectos Estructurales EIRL")

render_logo("Logo.jpg")
st.subheader("Determinación de Carga de Viento - Norma NCh 432-2025")
st.markdown("---")

# ==========================================
# SIDEBAR: PARÁMETROS DE DISEÑO
# ==========================================
st.sidebar.header("⚙️ Parámetros de Entrada")

# 1. Velocidad y Factores de Sitio
V = st.sidebar.number_input("Velocidad básica V (m/s)", value=35.0, help="Velocidad ráfaga de 3 seg a 10m de altura.")
Kzt = st.sidebar.number_input("Factor Topográfico (Kzt)", value=1.0, help="Por defecto 1.0 para terreno homogéneo.")
Kd = st.sidebar.number_input("Factor de Dirección (Kd)", value=0.85, help="0.85 para edificios.")

# 2. Geometría y Techo
st.sidebar.subheader("📐 Geometría")
H = st.sidebar.number_input("Altura promedio H (m)", value=18.0)
W = st.sidebar.number_input("Ancho normal al viento (m)", value=50.0)
L = st.sidebar.number_input("Largo paralelo al viento (m)", value=50.0)
theta = st.sidebar.slider("Inclinación de Techo θ (°)", 0, 45, 10)

# 3. Clasificación NCh 432
exp_cat = st.sidebar.selectbox("Categoría de Exposición", ['B', 'C', 'D', 'A'], index=1)
imp_cat = st.sidebar.selectbox("Categoría de Edificio", ['I', 'II', 'III', 'IV'], index=2)

# ==========================================
# MOTOR DE CÁLCULO RIGUROSO
# ==========================================

# Coeficiente de importancia (I) según Categoría
imp_map = {'I': 0.87, 'II': 1.0, 'III': 1.15, 'IV': 1.15}
I_factor = imp_map[imp_cat]

# Constantes de Exposición (Tabla 12 NCh 432)
exp_params = {
    'A': {'alpha': 5.0, 'zg': 457.0},
    'B': {'alpha': 7.0, 'zg': 366.0},
    'C': {'alpha': 9.5, 'zg': 274.0},
    'D': {'alpha': 11.5, 'zg': 213.0}
}
alpha = exp_params[exp_cat]['alpha']
zg = exp_params[exp_cat]['zg']

# Cálculo de Kz (Coeficiente de presión de velocidad)
def calcular_kz(h, zg_val, alpha_val):
    h_efectiva = max(h, 4.6)
    return 2.01 * ((h_efectiva / zg_val)**(2/alpha_val))

kz_h = calcular_kz(H, zg, alpha)

# Presión de velocidad (qh) en kgf/m2
# Fórmula: qz = 0.613 * Kz * Kzt * Kd * V^2 * I
qh_newton = 0.613 * kz_h * Kzt * Kd * (V**2) * I_factor
qh_kgf = qh_newton * 0.10197  # Conversión a kgf/m2

# Coeficientes de Presión Externa (GCp) - Basado en Figura 26/40
gc_pi = 0.18  # Edificio cerrado

# Determinación de Coeficientes para Techo según θ
if theta <= 7:
    gcp_techo = {"Zona 1 (Campo)": -1.0, "Zona 2 (Bordes)": -1.8, "Zona 3 (Esquinas)": -2.8}
elif 7 < theta <= 27:
    gcp_techo = {"Zona 1 (Campo)": -0.9, "Zona 2 (Bordes)": -1.3, "Zona 3 (Esquinas)": -2.0}
else:
    gcp_techo = {"Zona 1 (Campo)": -0.9, "Zona 2 (Bordes)": -1.2, "Zona 3 (Esquinas)": -1.2}

# ==========================================
# VISUALIZACIÓN DE RESULTADOS
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Presión de Velocidad")
    st.metric("Presión qh", f"{qh_kgf:.2f} kgf/m²")
    st.write(f"**Kz calculado:** {kz_h:.3f}")
    st.write(f"**Factor Importancia (I):** {I_factor}")
    
    # Dimensión 'a' (Zona de Esquina)
    l_cz = max(min(0.1*L, 0.1*W), 0.9)
    st.info(f"📍 **Dimensión de zona 'a' (Esquina):** {l_cz:.2f} m")

with col2:
    st.subheader("🏠 Presiones en Cubierta (Succión)")
    techo_data = []
    for zona, gcp in gcp_techo.items():
        p_neta = qh_kgf * (gcp - gc_pi)
        techo_data.append({"Zona": zona, "GCp": gcp, "Presión Diseño (kgf/m²)": round(p_neta, 2)})
    st.table(techo_data)

st.markdown("---")
st.caption("Cálculo desarrollado por Proyectos Estructurales EIRL bajo metodología NCh 432-2025.")
# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import os

# 1. CONFIGURACIÓN Y LOGO
st.set_page_config(page_title="NCh 432-2025 | Proyectos Estructurales", layout="wide")

def render_logo(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            data = f.read()
            url = base64.b64encode(data).decode()
        st.markdown(f'<div style="text-align: center;"><img src="data:image/jpg;base64,{url}" width="500"></div>', unsafe_allow_html=True)
    else:
        st.title("🏗️ Proyectos Estructurales EIRL")

render_logo("Logo.jpg")
st.subheader("Motor de Cálculo: Presión de Viento (NCh 432-2025)")

# 2. ENTRADA DE DATOS (SIDEBAR)
st.sidebar.header("⚙️ Parámetros de Diseño")
V = st.sidebar.number_input("Velocidad básica V (m/s)", value=35.0)
H = st.sidebar.number_input("Altura promedio H (m)", value=12.0)
theta = st.sidebar.slider("Inclinación de Techo θ (°)", 0, 45, 10)
area_trib = st.sidebar.number_input("Área Tributaria Efectiva (m²)", value=2.0, min_value=0.1)

exp_cat = st.sidebar.selectbox("Categoría de Exposición", ['B', 'C', 'D', 'A'], index=1)
imp_cat = st.sidebar.selectbox("Categoría de Edificio", ['I', 'II', 'III', 'IV'], index=2)
Kzt = st.sidebar.number_input("Factor Topográfico (Kzt)", value=1.0)

# 3. FUNCIONES DE CÁLCULO (INTERPOLACIÓN LOG)
def get_gcp_interp(area, gcp_1, gcp_10):
    """Calcula GCp interpolando logarítmicamente entre 1 y 10 m2"""
    if area <= 1.0: return gcp_1
    if area >= 10.0: return gcp_10
    return gcp_1 + (gcp_10 - gcp_1) * (np.log10(area) - np.log10(1.0))

# 4. PROCESAMIENTO
imp_map = {'I': 0.87, 'II': 1.0, 'III': 1.15, 'IV': 1.15}
I_factor = imp_map[imp_cat]
exp_params = {'A': [5.0, 457.0], 'B': [7.0, 366.0], 'C': [9.5, 274.0], 'D': [11.5, 213.0]}
alpha, zg = exp_params[exp_cat]

kz = 2.01 * ((max(H, 4.6) / zg)**(2/alpha))
qh_kgf = (0.613 * kz * Kzt * 0.85 * (V**2) * I_factor) * 0.10197

# Determinar GCp según inclinación y área
gc_pi = 0.18
if theta <= 7:
    z1 = get_gcp_interp(area_trib, -1.0, -0.9)
    z2 = get_gcp_interp(area_trib, -1.8, -1.1)
    z3 = get_gcp_interp(area_trib, -2.8, -1.1)
else:
    z1 = get_gcp_interp(area_trib, -0.9, -0.8)
    z2 = get_gcp_interp(area_trib, -1.3, -1.2)
    z3 = get_gcp_interp(area_trib, -2.0, -1.2)

# 5. RESULTADOS Y GRÁFICO
col1, col2 = st.columns([1, 1])

with col1:
    st.metric("Presión qh", f"{qh_kgf:.2f} kgf/m²")
    st.write(f"**Área Seleccionada:** {area_trib} m²")
    
    res_df = pd.DataFrame({
        "Zona": ["Zona 1 (Área Central)", "Zona 2 (Borde)", "Zona 3 (Esquina)"],
        "GCp": [round(z1, 3), round(z2, 3), round(z3, 3)],
        "Presión Neta (kgf/m²)": [round(qh_kgf*(z1-gc_pi), 2), round(qh_kgf*(z2-gc_pi), 2), round(qh_kgf*(z3-gc_pi), 2)]
    })
    st.table(res_df)

with col2:
    # Gráfico de sensibilidad GCp vs Área
    areas = np.logspace(0, 1, 20)
    gcp_curva = [get_gcp_interp(a, -2.8, -1.1) if theta <= 7 else get_gcp_interp(a, -2.0, -1.2) for a in areas]
    
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(areas, gcp_curva, color='red', label='Zona Crítica (Esquina)')
    ax.scatter([area_trib], [z3], color='black', zorder=5)
    ax.set_xlabel("Área Tributaria (m²)")
    ax.set_ylabel("Coeficiente GCp")
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend()
    st.pyplot(fig)

st.info(f"📍 Zona de Esquina (a): {max(min(0.1*H, 0.1*H), 0.9):.2f} m")
st.caption("Cálculo bajo NCh 432-2025. Proyectos Estructurales EIRL.")
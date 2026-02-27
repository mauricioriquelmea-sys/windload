# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import os
import math

# 1. CONFIGURACIÓN Y LOGO CORPORATIVO
st.set_page_config(page_title="NCh 432-2025 | Proyectos Estructurales", layout="wide")

def render_logo(image_file):
    """Renderiza el logo corporativo Logo.png"""
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            data = f.read()
            url = base64.b64encode(data).decode()
        st.markdown(f'<div style="text-align: center;"><img src="data:image/png;base64,{url}" width="500"></div>', unsafe_allow_html=True)
    else:
        st.title("🏗️ Proyectos Estructurales EIRL")

render_logo("Logo.png")
st.subheader("Análisis Integral de Presiones: Cubiertas y Fachadas (NCh 432-2025)")

# 2. ENTRADA DE DATOS (SIDEBAR)
st.sidebar.header("⚙️ Parámetros de Diseño")
V = st.sidebar.number_input("Velocidad básica V (m/s)", value=35.0)
H_edif = st.sidebar.number_input("Altura promedio edificio H (m)", value=12.0)
theta = st.sidebar.slider("Inclinación de Techo θ (°)", 0, 45, 10)

st.sidebar.subheader("📐 Geometría del Elemento")
l_elem = st.sidebar.number_input("Largo del elemento (m)", value=3.0)
w_input = st.sidebar.number_input("Ancho tributario real (m)", value=1.0)
w_trib = max(w_input, l_elem / 3)
area_efectiva = l_elem * w_trib

if w_input < (l_elem / 3):
    st.sidebar.warning(f"⚠️ Ancho ajustado por norma a {w_trib:.2f}m (mín. 1/3 del largo)")

# Factor Topográfico
with st.sidebar.expander("🏔️ Factor Topográfico (Kzt)"):
    metodo = st.radio("Método", ["Manual", "Calculado"])
    Kzt_val = st.number_input("Valor Kzt", value=1.0) if metodo == "Manual" else 1.0 # (Lógica simplificada para el bloque)

exp_cat = st.sidebar.selectbox("Exposición", ['B', 'C', 'D'], index=1)
imp_cat = st.sidebar.selectbox("Categoría Edificio", ['I', 'II', 'III', 'IV'], index=2)

# 3. FUNCIONES DE CÁLCULO
def get_gcp(area, g1, g10):
    if area <= 1.0: return g1
    if area >= 10.0: return g10
    return g1 + (g10 - g1) * (np.log10(area) - np.log10(1.0))

# 4. MOTOR MATEMÁTICO
imp_map = {'I': 0.87, 'II': 1.0, 'III': 1.15, 'IV': 1.15}
exp_params = {'B': [7.0, 366.0], 'C': [9.5, 274.0], 'D': [11.5, 213.0]}
alpha, zg = exp_params[exp_cat]

kz = 2.01 * ((max(H_edif, 4.6) / zg)**(2/alpha))
qh = (0.613 * kz * Kzt_val * 0.85 * (V**2) * imp_map[imp_cat]) * 0.10197
gc_pi = 0.18

# Coeficientes GCp (Interpolación)
if theta <= 7:
    z1, z2, z3 = get_gcp(area_efectiva, -1.0, -0.9), get_gcp(area_efectiva, -1.8, -1.1), get_gcp(area_efectiva, -2.8, -1.1)
else:
    z1, z2, z3 = get_gcp(area_efectiva, -0.9, -0.8), get_gcp(area_efectiva, -1.3, -1.2), get_gcp(area_efectiva, -2.0, -1.2)

z4 = get_gcp(area_efectiva, -1.1, -0.8) # Muro Estándar
z5 = get_gcp(area_efectiva, -1.4, -1.1) # Muro Esquina

# 5. RESULTADOS Y GRÁFICO DE 5 ZONAS
col1, col2 = st.columns([1, 1.2])

with col1:
    st.metric("Presión qh", f"{qh:.2f} kgf/m²")
    df = pd.DataFrame({
        "Zona": ["Zona 1 (Techo Centro)", "Zona 2 (Techo Borde)", "Zona 3 (Techo Esquina)", "Zona 4 (Estándar Muro)", "Zona 5 (Esquina Muro)"],
        "GCp": [round(z, 3) for z in [z1, z2, z3, z4, z5]],
        "Presión (kgf/m²)": [round(qh*(z-gc_pi), 2) for z in [z1, z2, z3, z4, z5]]
    })
    st.table(df)

with col2:
    areas = np.logspace(0, 1, 50)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Curvas de Techo (Z1, Z2, Z3)
    if theta <= 7:
        ax.plot(areas, [get_gcp(a, -1.0, -0.9) for a in areas], label='Z1: Techo Centro', color='cyan')
        ax.plot(areas, [get_gcp(a, -1.8, -1.1) for a in areas], label='Z2: Techo Borde', color='blue')
        ax.plot(areas, [get_gcp(a, -2.8, -1.1) for a in areas], label='Z3: Techo Esquina', color='navy', ls='--')
    else:
        ax.plot(areas, [get_gcp(a, -0.9, -0.8) for a in areas], label='Z1: Techo Centro', color='cyan')
        ax.plot(areas, [get_gcp(a, -1.3, -1.2) for a in areas], label='Z2: Techo Borde', color='blue')
        ax.plot(areas, [get_gcp(a, -2.0, -1.2) for a in areas], label='Z3: Techo Esquina', color='navy', ls='--')

    # Curvas de Fachada (Z4, Z5)
    ax.plot(areas, [get_gcp(a, -1.1, -0.8) for a in areas], label='Zona 4 (Estándar)', color='green', lw=2)
    ax.plot(areas, [get_gcp(a, -1.4, -1.1) for a in areas], label='Zona 5 (Esquina)', color='red', lw=2)
    
    # Marcar puntos del elemento
    for z_val in [z1, z2, z3, z4, z5]:
        ax.scatter([area_efectiva], [z_val], color='black', zorder=5)

    ax.set_title("Comparativa de Sensibilidad: 5 Zonas (NCh 432)", fontsize=11)
    ax.set_xlabel("Área Tributaria (m²)"); ax.set_ylabel("Coeficiente GCp")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

# SECCIÓN DE CONTACTO
st.markdown("---")
col_c1, col_c2 = st.columns([2, 1])
with col_c1:
    st.caption("Ingeniería de Avanzada | Proyectos Estructurales EIRL | NCh 432-2025")
with col_c2:
    st.markdown(f"""<div style="text-align: right; font-size: 0.9em; color: #555;"><strong>Contacto:</strong><br><a href="mailto:mriquelme@proyectosestructurales.com">mriquelme@proyectosestructurales.com</a></div>""", unsafe_allow_html=True)
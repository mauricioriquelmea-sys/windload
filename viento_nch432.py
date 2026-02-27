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
st.subheader("Análisis Integral de Presiones: 5 Zonas (NCh 432-2025)")

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
        # Buscamos archivos con varias extensiones para evitar errores de carga
        for img_name in ["F2.png", "F3.png", "F4.png",  "F5.png"]:
            if os.path.exists(img_name):
                st.image(img_name, caption=f"Norma NCh 432: {img_name}")
            else:
                st.warning(f"Archivo {img_name} no encontrado en el repositorio.")

V = st.sidebar.number_input("Velocidad básica V (m/s)", value=35.0)
H_edif = st.sidebar.number_input("Altura edificio H (m)", value=12.0)
theta = st.sidebar.slider("Inclinación Techo θ (°)", 0, 45, 10)

# Geometría y regla de 1/3
l_elem = st.sidebar.number_input("Largo elemento (m)", value=3.0)
w_in = st.sidebar.number_input("Ancho trib. real (m)", value=1.0)
w_trib = max(w_in, l_elem / 3)
area_ef = l_elem * w_trib

# Factor Topográfico Riguroso
with st.sidebar.expander("🏔️ Factor Topográfico (Kzt)"):
    met = st.radio("Método", ["Manual", "Calculado"])
    if met == "Manual":
        Kzt = st.number_input("Kzt", value=1.0)
    else:
        H_c = st.number_input("H colina (m)", value=27.0)
        L_h = st.number_input("Lh (m)", value=1743.7)
        k1_b, g, m = 0.75, 2.5, 1.5 # Ejemplo Escarpe 2D
        k1 = k1_b * (H_c / L_h); k3 = math.exp(-g * 10 / L_h)
        Kzt = (1 + k1 * 1.0 * k3)**2
        st.info(f"Kzt: {Kzt:.3f}")

# 3. MOTOR DE CÁLCULO
def get_gcp(a, g1, g10):
    if a <= 1.0: return g1
    if a >= 10.0: return g10
    return g1 + (g10 - g1) * (np.log10(a) - np.log10(1.0))

imp_map = {'I': 0.87, 'II': 1.0, 'III': 1.15, 'IV': 1.15}
exp_params = {'B': [7.0, 366.0], 'C': [9.5, 274.0], 'D': [11.5, 213.0]}
alpha, zg = exp_params[st.sidebar.selectbox("Exposición", ['B', 'C', 'D'], index=0)]

kz = 2.01 * ((max(H_edif, 4.6) / zg)**(2/alpha))
qh = (0.613 * kz * Kzt * 0.85 * (V**2) * imp_map['III']) * 0.10197
gc_pi = 0.18

# Coeficientes de las 5 Zonas
z1 = get_gcp(area_ef, -1.0, -0.9) if theta <= 7 else get_gcp(area_ef, -0.9, -0.8)
z2 = get_gcp(area_ef, -1.8, -1.1) if theta <= 7 else get_gcp(area_ef, -1.3, -1.2)
z3 = get_gcp(area_ef, -2.8, -1.1) if theta <= 7 else get_gcp(area_ef, -2.0, -1.2)
z4 = get_gcp(area_ef, -1.1, -0.8)
z5 = get_gcp(area_ef, -1.4, -1.1)

# 4. RESULTADOS Y GRÁFICO INTEGRAL
col1, col2 = st.columns([1, 1.2])

with col1:
    st.metric("Presión qh", f"{qh:.2f} kgf/m²")
    df = pd.DataFrame({
        "Ubicación": ["Techo Centro", "Techo Borde", "Techo Esquina", "Muro Interior", "Muro Esquina"],
        "Zona": ["Zona 1", "Zona 2", "Zona 3", "Zona 4", "Zona 5"],
        "Presión Diseño (kgf/m²)": [round(qh*(z-gc_pi), 2) for z in [z1, z2, z3, z4, z5]]
    })
    st.table(df)

with col2:
    # GRÁFICO DE LAS 5 ZONAS
    areas = np.logspace(0, 1, 50)
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Curvas de Techumbre (Z1, Z2, Z3)
    ax.plot(areas, [get_gcp(a, -1.0, -0.9) for a in areas], label='Z1 (Techo)', color='cyan', alpha=0.6)
    ax.plot(areas, [get_gcp(a, -1.8, -1.1) for a in areas], label='Z2 (Techo)', color='blue', alpha=0.6)
    ax.plot(areas, [get_gcp(a, -2.8, -1.1) for a in areas], label='Z3 (Techo Esquina)', color='navy', ls='--')
    
    # Curvas de Fachada (Z4, Z5)
    ax.plot(areas, [get_gcp(a, -1.1, -0.8) for a in areas], label='Z4 (Muro)', color='green', lw=2)
    ax.plot(areas, [get_gcp(a, -1.4, -1.1) for a in areas], label='Z5 (Muro Esquina)', color='red', lw=2)
    
    # Marcar puntos
    for z_v in [z1, z2, z3, z4, z5]:
        ax.scatter([area_ef], [z_v], color='black', zorder=5)

    ax.set_title("Comparativa de 5 Zonas (Log-Interpolación)")
    ax.set_xlabel("Área (m²)"); ax.set_ylabel("GCp"); ax.grid(True, alpha=0.3); ax.legend(fontsize='small', loc='best')
    st.pyplot(fig)



# CONTACTO
st.markdown("---")
st.markdown(f'<div style="text-align: right;"><a href="mailto:mriquelme@proyectosestructurales.com">mriquelme@proyectosestructurales.com</a></div>', unsafe_allow_html=True)
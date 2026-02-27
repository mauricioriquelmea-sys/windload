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
    """Renderiza el logo corporativo (soporta Logo.png)"""
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            data = f.read()
            url = base64.b64encode(data).decode()
        st.markdown(f'<div style="text-align: center;"><img src="data:image/png;base64,{url}" width="500"></div>', unsafe_allow_html=True)
    else:
        st.title("🏗️ Proyectos Estructurales EIRL")

render_logo("Logo.png")
st.subheader("Motor de Cálculo: Presión de Viento (NCh 432-2025)")

# 2. ENTRADA DE DATOS (SIDEBAR)
st.sidebar.header("⚙️ Parámetros de Diseño")

# Velocidad y Altura
V = st.sidebar.number_input("Velocidad básica V (m/s)", value=35.0, help="Velocidad ráfaga de 3 seg a 10m de altura.")
H_edif = st.sidebar.number_input("Altura promedio edificio H (m)", value=12.0)
theta = st.sidebar.slider("Inclinación de Techo θ (°)", 0, 45, 10)

# Geometría del Elemento y Regla de 1/3
st.sidebar.subheader("📐 Geometría del Elemento")
l_elem = st.sidebar.number_input("Largo del elemento / Luz (m)", value=3.0, min_value=0.1)
w_input = st.sidebar.number_input("Ancho tributario real (m)", value=1.0, min_value=0.1)

w_trib = max(w_input, l_elem / 3)
area_efectiva = l_elem * w_trib

if w_input < (l_elem / 3):
    st.sidebar.warning(f"⚠️ Ancho ajustado por norma a {w_trib:.2f}m (mín. 1/3 del largo)")
st.sidebar.markdown(f"**Área Efectiva final:** {area_efectiva:.2f} m²")

# Clasificación
exp_cat = st.sidebar.selectbox("Categoría de Exposición", ['B', 'C', 'D', 'A'], index=1)
imp_cat = st.sidebar.selectbox("Categoría de Edificio", ['I', 'II', 'III', 'IV'], index=2)

# --- MÓDULO: FACTOR TOPOGRÁFICO (Kzt) ---
with st.sidebar.expander("🏔️ Cálculo de Factor Topográfico (Kzt)"):
    metodo_kzt = st.radio("Método", ["Manual", "Calculado (Escarpe/Colina)"])
    if metodo_kzt == "Manual":
        Kzt_val = st.number_input("Kzt directo", value=1.0, step=0.1)
    else:
        tipo_relieve = st.selectbox("Forma de relieve", ["Escarpe 2D", "Colina 2D", "Colina 3D"])
        H_colina = st.number_input("Altura colina H (m)", value=27.0)
        Lh = st.number_input("Distancia Lh (m)", value=1743.7)
        x_dist = st.number_input("Distancia x (m)", value=0.0)
        z_alt = st.number_input("Altura z sobre suelo (m)", value=10.0)
        
        if tipo_relieve == "Escarpe 2D": k1_b, gamma, mu = 0.75, 2.5, 1.5
        elif tipo_relieve == "Colina 2D": k1_b, gamma, mu = 1.05, 1.5, 1.5
        else: k1_b, gamma, mu = 0.95, 1.5, 4.0

        k1 = k1_b * (H_colina / Lh)
        k2 = (1 - abs(x_dist) / (mu * Lh))
        k3 = math.exp(-gamma * z_alt / Lh)
        Kzt_val = (1 + k1 * k2 * k3)**2
        st.info(f"Kzt Calculado: {Kzt_val:.3f}")

# 3. FUNCIONES DE CÁLCULO
def get_gcp_interp(area, gcp_1, gcp_10):
    if area <= 1.0: return gcp_1
    if area >= 10.0: return gcp_10
    return gcp_1 + (gcp_10 - gcp_1) * (np.log10(area) - np.log10(1.0))

# 4. PROCESAMIENTO MATEMÁTICO
imp_map = {'I': 0.87, 'II': 1.0, 'III': 1.15, 'IV': 1.15}
I_factor = imp_map[imp_cat]
exp_params = {'A': [5.0, 457.0], 'B': [7.0, 366.0], 'C': [9.5, 274.0], 'D': [11.5, 213.0]}
alpha, zg = exp_params[exp_cat]

kz = 2.01 * ((max(H_edif, 4.6) / zg)**(2/alpha))
qh_kgf = (0.613 * kz * Kzt_val * 0.85 * (V**2) * I_factor) * 0.10197

# Coeficientes GCp (Interpolación)
gc_pi = 0.18
if theta <= 7:
    z1, z2, z3 = get_gcp_interp(area_efectiva, -1.0, -0.9), get_gcp_interp(area_efectiva, -1.8, -1.1), get_gcp_interp(area_efectiva, -2.8, -1.1)
else:
    z1, z2, z3 = get_gcp_interp(area_efectiva, -0.9, -0.8), get_gcp_interp(area_efectiva, -1.3, -1.2), get_gcp_interp(area_efectiva, -2.0, -1.2)

# 5. RESULTADOS Y GRÁFICO
col1, col2 = st.columns([1, 1])
with col1:
    st.metric("Presión qh", f"{qh_kgf:.2f} kgf/m²")
    res_df = pd.DataFrame({
        "Zona": ["Zona 1 (Central)", "Zona 2 (Borde)", "Zona 3 (Esquina)"],
        "GCp": [round(z1, 3), round(z2, 3), round(z3, 3)],
        "Presión Neta (kgf/m²)": [round(qh_kgf*(z1-gc_pi), 2), round(qh_kgf*(z2-gc_pi), 2), round(qh_kgf*(z3-gc_pi), 2)]
    })
    st.table(res_df)

with col2:
    areas = np.logspace(0, 1, 20)
    gcp_curva = [get_gcp_interp(a, -2.8, -1.1) if theta <= 7 else get_gcp_interp(a, -2.0, -1.2) for a in areas]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(areas, gcp_curva, color='blue', label='Interpolación log')
    ax.scatter([area_efectiva], [z3], color='red', zorder=5, label='Tu elemento')
    ax.set_xlabel("Área Tributaria (m²)"); ax.set_ylabel("GCp"); ax.grid(True, which="both", alpha=0.5); ax.legend()
    st.pyplot(fig)

# --- SECCIÓN DE CONTACTO ---
st.markdown("---")
col_c1, col_c2 = st.columns([3, 1])
with col_c1:
    st.caption("Cálculo bajo normativa NCh 432-2025. Proyectos Estructurales EIRL.")
with col_c2:
    st.markdown(f"""
    <div style="text-align: right; font-size: 0.9em; color: #555;">
        <strong>Contacto Ingeniería:</strong><br>
        <a href="mailto:mauricio.riquelme@proyectosestructurales.cl">mauricio.riquelme@proyectosestructurales.cl</a>
    </div>
    """, unsafe_allow_html=True)
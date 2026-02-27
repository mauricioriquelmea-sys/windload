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
st.subheader("Motor de Cálculo Integral: Cubiertas y Fachadas (NCh 432-2025)")

# 2. ENTRADA DE DATOS (SIDEBAR)
st.sidebar.header("⚙️ Parámetros de Diseño")

V = st.sidebar.number_input("Velocidad básica V (m/s)", value=35.0)
H_edif = st.sidebar.number_input("Altura promedio edificio H (m)", value=12.0)
theta = st.sidebar.slider("Inclinación de Techo θ (°)", 0, 45, 10)

st.sidebar.subheader("📐 Geometría del Elemento")
l_elem = st.sidebar.number_input("Largo del elemento (m)", value=3.0, min_value=0.1)
w_input = st.sidebar.number_input("Ancho tributario real (m)", value=1.0, min_value=0.1)

w_trib = max(w_input, l_elem / 3)
area_efectiva = l_elem * w_trib

if w_input < (l_elem / 3):
    st.sidebar.warning(f"⚠️ Ancho ajustado por norma a {w_trib:.2f}m (mín. 1/3 del largo)")

with st.sidebar.expander("🏔️ Cálculo de Factor Topográfico (Kzt)"):
    metodo_kzt = st.radio("Método", ["Manual", "Calculado"])
    if metodo_kzt == "Manual":
        Kzt_val = st.number_input("Kzt directo", value=1.0, step=0.1)
    else:
        tipo_relieve = st.selectbox("Forma", ["Escarpe 2D", "Colina 2D", "Colina 3D"])
        H_c = st.number_input("H colina (m)", value=27.0)
        L_h = st.number_input("Lh (m)", value=1743.7)
        x_d = st.number_input("Distancia x (m)", value=0.0)
        z_a = st.number_input("Altura z (m)", value=10.0)
        
        k1_b, g, m = (0.75, 2.5, 1.5) if tipo_relieve == "Escarpe 2D" else (1.05, 1.5, 1.5) if tipo_relieve == "Colina 2D" else (0.95, 1.5, 4.0)
        k1 = k1_b * (H_c / L_h); k2 = (1 - abs(x_d) / (m * L_h)); k3 = math.exp(-g * z_a / L_h)
        Kzt_val = (1 + k1 * k2 * k3)**2
        st.info(f"Kzt: {Kzt_val:.3f}")

exp_cat = st.sidebar.selectbox("Exposición", ['B', 'C', 'D', 'A'], index=1)
imp_cat = st.sidebar.selectbox("Importancia", ['I', 'II', 'III', 'IV'], index=2)

# 3. FUNCIONES DE CÁLCULO
def get_gcp(area, g1, g10):
    if area <= 1.0: return g1
    if area >= 10.0: return g10
    return g1 + (g10 - g1) * (np.log10(area) - np.log10(1.0))

# 4. MOTOR MATEMÁTICO
imp_map = {'I': 0.87, 'II': 1.0, 'III': 1.15, 'IV': 1.15}
exp_params = {'A': [5.0, 457.0], 'B': [7.0, 366.0], 'C': [9.5, 274.0], 'D': [11.5, 213.0]}
alpha, zg = exp_params[exp_cat]

kz = 2.01 * ((max(H_edif, 4.6) / zg)**(2/alpha))
qh = (0.613 * kz * Kzt_val * 0.85 * (V**2) * imp_map[imp_cat]) * 0.10197
gc_pi = 0.18

# Interpolación de las 5 Zonas
if theta <= 7:
    z1, z2, z3 = get_gcp(area_efectiva, -1.0, -0.9), get_gcp(area_efectiva, -1.8, -1.1), get_gcp(area_efectiva, -2.8, -1.1)
else:
    z1, z2, z3 = get_gcp(area_efectiva, -0.9, -0.8), get_gcp(area_efectiva, -1.3, -1.2), get_gcp(area_efectiva, -2.0, -1.2)

z4 = get_gcp(area_efectiva, -1.1, -0.8) # Muro Interior
z5 = get_gcp(area_efectiva, -1.4, -1.1) # Muro Esquina

# 5. RESULTADOS
col1, col2 = st.columns([1, 1])
with col1:
    st.metric("Presión qh", f"{qh:.2f} kgf/m²")
    df = pd.DataFrame({
        "Ubicación": ["Techo (Centro)", "Techo (Borde)", "Techo (Esquina)", "Muro (Centro)", "Muro (Esquina)"],
        "Zona": ["Zona 1", "Zona 2", "Zona 3", "Zona 4", "Zona 5"],
        "GCp": [round(z, 3) for z in [z1, z2, z3, z4, z5]],
        "Presión Neta (kgf/m²)": [round(qh*(z-gc_pi), 2) for z in [z1, z2, z3, z4, z5]]
    })
    st.table(df)

# --- ACTUALIZACIÓN DE LA SECCIÓN DE RESULTADOS Y GRÁFICO ---
with col2:
    # Definición de áreas para la curva (de 1 a 10 m2)
    areas_grafico = np.logspace(0, 1, 50)
    
    # Curvas de interpolación logarítmica para Muros (Zonas 4 y 5)
    # Basado en Figura 24/40 de la norma NCh 432:2025
    curva_z4 = [get_gcp(a, -1.1, -0.8) for a in areas_grafico]
    curva_z5 = [get_gcp(a, -1.4, -1.1) for a in areas_grafico]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Dibujar curvas
    ax.plot(areas_grafico, curva_z4, label='Zona 4 (Estándar)', color='green', lw=2)
    ax.plot(areas_grafico, curva_z5, label='Zona 5 (Esquina)', color='red', ls='--', lw=2)
    
    # Marcar el punto específico del elemento calculado
    ax.scatter([area_efectiva], [z4], color='black', zorder=5)
    ax.scatter([area_efectiva], [z5], color='black', zorder=5)
    
    # Formato del gráfico
    ax.set_title("Influencia del Área Tributaria en Fachadas", fontsize=12)
    ax.set_xlabel("Área Tributaria (m²)", fontsize=10)
    ax.set_ylabel("Coeficiente GCp", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc='best')
    
    st.pyplot(fig)



# --- SECCIÓN DE CONTACTO ---
st.markdown("---")
col_c1, col_c2 = st.columns([2, 1])
with col_c1:
    st.caption("Cálculo bajo normativa NCh 432-2025. Proyectos Estructurales EIRL.")
with col_c2:
    st.markdown(f"""<div style="text-align: right; font-size: 0.9em; color: #555;"><strong>Contacto Ingeniería:</strong><br><a href="mailto:mriquelme@proyectosestructurales.com">mriquelme@proyectosestructurales.com</a></div>""", unsafe_allow_html=True)
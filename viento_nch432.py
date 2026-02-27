# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import os
import math

# 1. CONFIGURACIÓN CORPORATIVA Y CONTROL DE ANCHO TOTAL (FULL WIDTH)
st.set_page_config(page_title="NCh 432-2025 | Proyectos Estructurales", layout="wide")

# CSS Inyectado para eliminar márgenes laterales y ocupar el 100% del ancho
st.markdown("""
    <style>
    .main > div { padding-left: 1rem; padding-right: 1rem; max-width: 100%; }
    .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #dee2e6; }
    .stTable { width: 100% !important; }
    </style>
    """, unsafe_allow_html=True)

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            data = f.read()
            return base64.b64encode(data).decode()
    return None

def render_header_images(logo_file, ray_file, eolo_file):
    logo_base_64 = get_base64_image(logo_file)
    ray_base_64 = get_base64_image(ray_file)
    eolo_base_64 = get_base64_image(eolo_file)
    html_content = '<div style="display: flex; justify-content: center; align-items: center; gap: 40px; margin-bottom: 30px; flex-wrap: wrap;">'
    if logo_base_64: html_content += f'<img src="data:image/png;base64,{logo_base_64}" width="380">'
    if ray_base_64: html_content += f'<img src="data:image/png;base64,{ray_base_64}" width="130" style="opacity: 0.9;">'
    if eolo_base_64: html_content += f'<img src="data:image/png;base64,{eolo_base_64}" width="130" style="opacity: 0.8;">'
    html_content += '</div>'
    st.markdown(html_content, unsafe_allow_html=True)

render_header_images("Logo.png", "Ray.png", "Eolo.png")

st.subheader("Determinación de Presiones de Viento según Norma NCh 432-2025")
st.caption("Análisis Integral de Presiones de Viento: Cubiertas y Fachadas | Ingeniería Estructural")

# 2. SIDEBAR CON GUÍA TÉCNICA
st.sidebar.header("⚙️ Parámetros de Diseño")

with st.sidebar.expander("🚩 Ayuda: Velocidad Básica (V)"):
    st.write("**Zonificación Tabla 1 (Anexo):**")
    tabla_v = {"Zona": ["I-A", "II-B", "III-B", "IV-B", "V", "VI"], "V (m/s)": [27, 35, 35, 40, 40, 44]}
    st.table(pd.DataFrame(tabla_v))
    if st.button("Desplegar Mapas de Chile"):
        for img in ["F2.png", "F3.png", "F4.png", "F5.png"]:
            if os.path.exists(img): st.image(img)

V = st.sidebar.number_input("Velocidad básica V (m/s)", value=35.0)
H_edif = st.sidebar.number_input("Altura edificio H (m)", value=12.0)
theta = st.sidebar.slider("Inclinación Techo θ (°)", 0, 45, 10)

st.sidebar.subheader("📐 Geometría del Elemento")
l_elem = st.sidebar.number_input("Largo elemento (m)", value=3.0)
w_in = st.sidebar.number_input("Ancho trib. real (m)", value=1.0)
w_trib = max(w_in, l_elem / 3)
area_ef = l_elem * w_trib

with st.sidebar.expander("🏔️ Ayuda: Factor Topográfico (Kzt)"):
    metodo = st.radio("Método", ["Manual", "Calculado"])
    if metodo == "Manual":
        Kzt_val = st.number_input("Kzt directo", value=1.0)
    else:
        tipo_rel = st.selectbox("Relieve", ["Escarpe 2D", "Colina 2D", "Colina 3D"])
        Hc, Lhc, xdc, zac = st.number_input("Hc (m)", 27.0), st.number_input("Lh (m)", 1743.7), st.number_input("x (m)", 0.0), st.number_input("z (m)", 10.0)
        k1b, gam, mu_val = (0.75, 2.5, 1.5) if tipo_rel == "Escarpe 2D" else (1.05, 1.5, 1.5) if tipo_rel == "Colina 2D" else (0.95, 1.5, 4.0)
        k1_t, k2_t, k3_t = k1b*(Hc/Lhc), (1-abs(xdc)/(mu_val*Lhc)), math.exp(-gam*zac/Lhc)
        Kzt_val = (1 + k1_t*k2_t*k3_t)**2
        st.info(f"Kzt: {Kzt_val:.3f}")

# --- SECCIÓN EXPANDIDA: PRESIÓN INTERNA RIGUROSA ---
st.sidebar.subheader("🏠 Cerramiento y Presión Interna")
with st.sidebar.expander("ℹ️ Ayuda Técnica: Clasificación GCpi"):
    st.markdown("""
    **Según NCh 432 Capítulo 6:**
    * **Abierto:** Estructura con al menos 80% de aberturas por pared. El viento fluye sin obstrucción. **GCpi = 0.00**.
    * **Cerrado:** Edificio que no es abierto ni parcialmente abierto. Es el estándar para oficinas y viviendas. **GCpi = ±0.18**.
    * **Parcialmente Abierto:** Pared con área de aberturas mayor a la suma del resto. Típico en galpones con portones grandes. **GCpi = ±0.55**.
    """)
cerramiento = st.sidebar.selectbox("Tipo de Cerramiento", ["Cerrado", "Parcialmente Abierto", "Abierto"], index=0)
gcpi_map = {"Cerrado": 0.18, "Parcialmente Abierto": 0.55, "Abierto": 0.00}
gc_pi_val = gcpi_map[cerramiento]

st.sidebar.subheader("📋 Factores Normativos")
Kd_val = st.sidebar.number_input("Factor Kd (Dirección)", value=0.85)
cat_exp = st.sidebar.selectbox("Exposición", ['B', 'C', 'D'], index=0)
cat_imp = st.sidebar.selectbox("Importancia", ['I', 'II', 'III', 'IV'], index=2)

# 3. MOTOR DE CÁLCULO
def get_gcp(a, g1, g10):
    if a <= 1.0: return g1
    if a >= 10.0: return g10
    return g1 + (g10 - g1) * (np.log10(a) - np.log10(1.0))

imp_map = {'I': 0.87, 'II': 1.0, 'III': 1.15, 'IV': 1.15}
exp_params = {'B': [7.0, 366.0], 'C': [9.5, 274.0], 'D': [11.5, 213.0]}
alpha, zg = exp_params[cat_exp]

kz = 2.01 * ((max(H_edif, 4.6) / zg)**(2/alpha))
qh = (0.613 * kz * Kzt_val * Kd_val * (V**2) * imp_map[cat_imp]) * 0.10197

# Coeficientes de Zonas
z1 = get_gcp(area_ef, -1.0, -0.9) if theta <= 7 else get_gcp(area_ef, -0.9, -0.8)
z2 = get_gcp(area_ef, -1.8, -1.1) if theta <= 7 else get_gcp(area_ef, -1.3, -1.2)
z3 = get_gcp(area_ef, -2.8, -1.1) if theta <= 7 else get_gcp(area_ef, -2.0, -1.2)
z4, z5 = get_gcp(area_ef, -1.1, -0.8), get_gcp(area_ef, -1.4, -1.1)

# 4. RESULTADOS (FULL WIDTH)
col1, col2 = st.columns([1, 1.3])
with col1:
    st.metric("Presión qh (Carga de Velocidad)", f"{qh:.2f} kgf/m²")
    # Tabla con GCpi incluido para rigor técnico
    df = pd.DataFrame({
        "Zona": ["Z1 (Techo Centro)", "Z2 (Techo Borde)", "Z3 (Techo Esquina)", "Z4 (Fachada)", "Z5 (Fachada Esquina)"],
        "GCp (Ext)": [round(z, 3) for z in [z1, z2, z3, z4, z5]],
        "GCpi (Int)": [gc_pi_val] * 5,
        "Presión Neta Diseño (kgf/m²)": [round(qh*(z - gc_pi_val), 2) for z in [z1, z2, z3, z4, z5]]
    })
    st.table(df)
    st.warning(f"Nota: Presión Neta calculada para la condición de succión crítica (p = qh * [GCp - GCpi])")

with col2:
    areas = np.logspace(0, 1, 50)
    fig, ax = plt.subplots(figsize=(10, 6))
    if theta <= 7:
        ax.plot(areas, [get_gcp(a, -1.0, -0.9) for a in areas], label='Z1 (Techo)', color='cyan', alpha=0.5)
        ax.plot(areas, [get_gcp(a, -1.8, -1.1) for a in areas], label='Z2 (Techo)', color='blue', alpha=0.5)
        ax.plot(areas, [get_gcp(a, -2.8, -1.1) for a in areas], label='Z3 (Techo Esquina)', color='navy', ls='--')
    else:
        ax.plot(areas, [get_gcp(a, -0.9, -0.8) for a in areas], label='Z1 (Techo)', color='cyan', alpha=0.5)
        ax.plot(areas, [get_gcp(a, -1.3, -1.2) for a in areas], label='Z2 (Techo)', color='blue', alpha=0.5)
        ax.plot(areas, [get_gcp(a, -2.0, -1.2) for a in areas], label='Z3 (Techo Esquina)', color='navy', ls='--')
    ax.plot(areas, [get_gcp(a, -1.1, -0.8) for a in areas], label='Z4 (Muro)', color='green', lw=2.5)
    ax.plot(areas, [get_gcp(a, -1.4, -1.1) for a in areas], label='Z5 (Muro Esquina)', color='red', lw=2.5)
    for z_v in [z1, z2, z3, z4, z5]: ax.scatter([area_ef], [z_v], color='black', zorder=10)
    ax.set_title("Sensibilidad de Presiones por Área Tributaria (NCh 432)"); ax.set_xlabel("Área Tributaria (m²)"); ax.set_ylabel("GCp")
    ax.grid(True, which="both", alpha=0.2); ax.legend(fontsize='small', loc='best')
    st.pyplot(fig)

# --- ESQUEMAS FINALES ---
st.markdown("---")
col_img1, col_img2 = st.columns(2)
with col_img1:
    st.subheader("📍 Identificación de Zonas (F8)")
    if os.path.exists("F8.png"): st.image("F8.png")
with col_img2:
    st.subheader("📍 Esquema Isométrico (F12)")
    if os.path.exists("F12.png"): st.image("F12.png")

# PIE DE PÁGINA
st.markdown("---")
st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; color: #555; font-size: 0.9em;">
        <div><strong>Desarrollado por:</strong> Mauricio Riquelme, Ingeniero Civil Estructural</div>
        <div style="text-align: right;"><strong>Contacto:</strong> <a href="mailto:mriquelme@proyectosestructurales.com">mriquelme@proyectosestructurales.com</a></div>
    </div>
    """, unsafe_allow_html=True)
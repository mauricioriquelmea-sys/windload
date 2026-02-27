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

# CSS Inyectado para eliminar márgenes laterales y optimizar la visualización técnica
st.markdown("""
    <style>
    .main > div { padding-left: 2rem; padding-right: 2rem; max-width: 100%; }
    .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #dee2e6; }
    .formula-box { 
        background-color: #e9ecef; 
        padding: 25px; 
        border-left: 6px solid #0056b3; 
        border-radius: 8px; 
        margin: 20px 0;
        font-family: 'Roboto', sans-serif;
    }
    .stTable { width: 100% !important; }
    </style>
    """, unsafe_allow_html=True)

def get_base64_image(image_path):
    """Convierte una imagen a base64 para embeberla en el encabezado HTML"""
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            data = f.read()
            return base64.b64encode(data).decode()
    return None

def render_header_images(logo_file, ray_file, eolo_file):
    """Renderiza el Logo Corporativo, Ray y Eolo en una sola fila centrada."""
    logo_base_64 = get_base64_image(logo_file)
    ray_base_64 = get_base64_image(ray_file)
    eolo_base_64 = get_base64_image(eolo_file)
    
    html_content = '<div style="display: flex; justify-content: center; align-items: center; gap: 40px; margin-bottom: 30px; flex-wrap: wrap;">'
    if logo_base_64: html_content += f'<img src="data:image/png;base64,{logo_base_64}" width="380">'
    if ray_base_64: html_content += f'<img src="data:image/png;base64,{ray_base_64}" width="130" style="opacity: 0.9;">'
    if eolo_base_64: html_content += f'<img src="data:image/png;base64,{eolo_base_64}" width="130" style="opacity: 0.8;">'
    html_content += '</div>'
    
    if logo_base_64 or ray_base_64 or eolo_base_64:
        st.markdown(html_content, unsafe_allow_html=True)
    else:
        st.title("🏗️ Proyectos Estructurales EIRL")

# Renderizado de Encabezado
render_header_images("Logo.png", "Ray.png", "Eolo.png")

st.subheader("Determinación de Presiones de Viento según Norma NCh 432-2025")
st.caption("Análisis Integral de Presiones de Viento: Cubiertas y Fachadas | Ingeniería Civil Estructural")

# 2. SIDEBAR CON GUÍA TÉCNICA COMPLETA
st.sidebar.header("⚙️ Parámetros de Diseño")

with st.sidebar.expander("🚩 Guía: Velocidad Básica (V) y Mapas"):
    st.write("**Zonificación Tabla 1 (Normativa):**")
    tabla_v = {"Zona": ["I-A", "II-B", "III-B", "IV-B", "V", "VI"], "V (m/s)": [27, 35, 35, 40, 40, 44]}
    st.table(pd.DataFrame(tabla_v))
    if st.button("Desplegar Mapas de Viento de Chile"):
        for img in ["F2.png", "F3.png", "F4.png", "F5.png"]:
            if os.path.exists(img): st.image(img, caption=f"Zonificación: {img}")

V = st.sidebar.number_input("Velocidad básica V (m/s)", value=35.0)
H_edif = st.sidebar.number_input("Altura promedio edificio H (m)", value=12.0)
theta = st.sidebar.slider("Inclinación de Techo θ (°)", 0, 45, 10)

st.sidebar.subheader("📐 Geometría del Elemento")
l_elem = st.sidebar.number_input("Largo del elemento (m)", value=3.0)
w_in = st.sidebar.number_input("Ancho tributario real (m)", value=1.0)
w_trib = max(w_in, l_elem / 3)
area_ef = l_elem * w_trib

if w_in < (l_elem / 3):
    st.sidebar.warning(f"⚠️ Ancho ajustado por norma a {w_trib:.2f}m (mín. 1/3 del largo)")

with st.sidebar.expander("🏔️ Factor Topográfico Riguroso (Kzt)"):
    if st.button("Ver Diagramas de Relieve"):
        for img in ["F7.png", "F6.png"]:
            if os.path.exists(img): st.image(img)
    metodo = st.radio("Método de Selección", ["Manual", "Calculado (Procedimiento NCh 432)"])
    if metodo == "Manual":
        Kzt_val = st.number_input("Kzt directo", value=1.0)
    else:
        tipo_relieve = st.selectbox("Forma de relieve", ["Escarpe 2D", "Colina 2D", "Colina 3D"])
        Hc = st.number_input("Altura colina H (m)", 27.0)
        Lhc = st.number_input("Distancia Lh (m)", 1743.7)
        xdc = st.number_input("Distancia x (m)", 0.0)
        zac = st.number_input("Altura z s/suelo (m)", 10.0)
        k1b, gam, mu_v = (0.75, 2.5, 1.5) if tipo_relieve == "Escarpe 2D" else (1.05, 1.5, 1.5) if tipo_relieve == "Colina 2D" else (0.95, 1.5, 4.0)
        k1_t, k2_t, k3_t = k1b*(Hc/Lhc), (1-abs(xdc)/(mu_v*Lhc)), math.exp(-gam*zac/Lhc)
        Kzt_val = (1 + k1_t*k2_t*k3_t)**2
        st.info(f"Kzt Calculado: {Kzt_val:.3f}")

# --- SECCIÓN: PRESIÓN INTERNA (GCpi) ---
st.sidebar.subheader("🏠 Cerramiento y Presión Interna")
with st.sidebar.expander("ℹ️ Ayuda Técnica: Clasificación de Edificios"):
    st.markdown("""
    * **Abierto:** Al menos 80% de aberturas por pared. **GCpi = 0.00**.
    * **Cerrado:** No es abierto ni parcialmente abierto. **GCpi = ±0.18**.
    * **Parcialmente Abierto:** Aberturas en una pared exceden la suma del resto. **GCpi = ±0.55**.
    """)
opciones_gcpi = {
    "Cerrado (GCpi = ±0.18)": 0.18,
    "Parcialmente Abierto (GCpi = ±0.55)": 0.55,
    "Abierto (GCpi = 0.00)": 0.00
}
cerramiento_label = st.sidebar.selectbox("Tipo de Edificación", list(opciones_gcpi.keys()), index=0)
gc_pi_val = opciones_gcpi[cerramiento_label]

st.sidebar.subheader("📋 Factores Normativos")
Kd_val = st.sidebar.number_input("Factor de Dirección Kd", value=0.85)
cat_exp = st.sidebar.selectbox("Categoría de Exposición", ['B', 'C', 'D'], index=0)
cat_imp = st.sidebar.selectbox("Categoría de Importancia", ['I', 'II', 'III', 'IV'], index=2)

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

# 4. DESGLOSE DE FÓRMULAS DE DISEÑO
st.markdown(f"""
<div class="formula-box">
    <strong>1. Presión de Velocidad (qh):</strong> {qh:.2f} kgf/m² <br>
    <strong>2. Ecuación de Presión de Diseño Neta (p):</strong> <br>
    <p style="font-size: 1.3em; text-align: center; font-weight: bold; color: #0056b3;">
        $p = q_h \times [GC_p - GC_{{pi}}]$
    </p>
    Considerando: $q_h = {qh:.2f}$ | $GC_{{pi}} = {gc_pi_val}$ (Condición: {cerramiento_label.split('(')[0]})
</div>
""", unsafe_allow_html=True)

# Coeficientes de las 5 Zonas
z1 = get_gcp(area_ef, -1.0, -0.9) if theta <= 7 else get_gcp(area_ef, -0.9, -0.8)
z2 = get_gcp(area_ef, -1.8, -1.1) if theta <= 7 else get_gcp(area_ef, -1.3, -1.2)
z3 = get_gcp(area_ef, -2.8, -1.1) if theta <= 7 else get_gcp(area_ef, -2.0, -1.2)
z4, z5 = get_gcp(area_ef, -1.1, -0.8), get_gcp(area_ef, -1.4, -1.1)

# 5. RESULTADOS Y GRÁFICO INTEGRAL (ZONAS 1-5)
col1, col2 = st.columns([1, 1.3])
with col1:
    st.subheader("📊 Resumen de Presiones de Diseño")
    df_res = pd.DataFrame({
        "Zona": ["Z1 (Techo Centro)", "Z2 (Techo Borde)", "Z3 (Techo Esquina)", "Z4 (Muro Estándar)", "Z5 (Muro Esquina)"],
        "GCp (Ext)": [round(z, 3) for z in [z1, z2, z3, z4, z5]],
        "GCpi (Int)": [gc_pi_val] * 5,
        "Presión Neta (kgf/m²)": [round(qh*(z - gc_pi_val), 2) for z in [z1, z2, z3, z4, z5]]
    })
    st.table(df_res)
    st.warning("⚠️ Nota: El signo negativo indica succión (hacia afuera de la superficie).")

with col2:
    areas = np.logspace(0, 1, 50)
    fig, ax = plt.subplots(figsize=(10, 6))
    # Curvas de Techo
    z3_c = [get_gcp(a, -2.8, -1.1) if theta <= 7 else get_gcp(a, -2.0, -1.2) for a in areas]
    ax.plot(areas, z3_c, label='Z3 (Esquina Techo)', color='navy', ls='--')
    # Curvas de Fachada
    ax.plot(areas, [get_gcp(a, -1.1, -0.8) for a in areas], label='Z4 (Fachada Estándar)', color='green', lw=2.5)
    ax.plot(areas, [get_gcp(a, -1.4, -1.1) for a in areas], label='Z5 (Fachada Esquina)', color='red', lw=2.5)
    
    for z_v in [z1, z2, z3, z4, z5]:
        ax.scatter([area_ef], [z_v], color='black', zorder=10)

    ax.set_title("Comparativa de 5 Zonas: Sensibilidad por Área Tributaria")
    ax.set_xlabel("Área Tributaria (m²)"); ax.set_ylabel("Coeficiente GCp")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize='small')
    st.pyplot(fig)

# --- SECCIÓN: ESQUEMAS NORMATIVOS ---
st.markdown("---")
col_img1, col_img2 = st.columns(2)
with col_img1:
    st.subheader("📍 Identificación de Zonas (F8)")
    if os.path.exists("F8.png"): st.image("F8.png", caption="Zonificación de presiones externas")
with col_img2:
    st.subheader("📍 Esquema Isométrico (F12)")
    if os.path.exists("F12.png"): st.image("F12.png", caption="Distribución isométrica de cargas")

# CRÉDITOS FINALES
st.markdown("---")
st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; color: #444; font-size: 0.9em;">
        <div><strong>Desarrollado por:</strong> Mauricio Riquelme, Ingeniero Civil Estructural</div>
        <div style="text-align: right;"><strong>Contacto Proyectos Estructurales EIRL:</strong><br>
            <a href="mailto:mriquelme@proyectosestructurales.com">mriquelme@proyectosestructurales.com</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
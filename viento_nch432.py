# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import os
import math

# =================================================================
# 1. CONFIGURACIÓN CORPORATIVA Y CONTROL DE ANCHO TOTAL
# =================================================================
st.set_page_config(
    page_title="NCh 432-2025 | Análisis de Viento Avanzado", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyección de CSS para control de UI y eliminación de márgenes restrictivos
st.markdown("""
    <style>
    .main > div { padding-left: 2.5rem; padding-right: 2.5rem; max-width: 100%; }
    .formula-box { 
        background-color: #f1f4f9; 
        padding: 25px; 
        border-left: 8px solid #003366; 
        border-radius: 10px; 
        margin: 20px 0;
    }
    .classification-box {
        background-color: #ffffff;
        padding: 20px;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    .stTable { width: 100% !important; font-size: 1.1em; }
    .sidebar-text { font-size: 0.9em; color: #555; }
    </style>
    """, unsafe_allow_html=True)

# =================================================================
# 2. FUNCIONES DE SOPORTE (IMÁGENES Y LOGOS)
# =================================================================
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            data = f.read()
            return base64.b64encode(data).decode()
    return None

def render_full_header(logo_file, ray_file, eolo_file):
    l_b64 = get_base64_image(logo_file)
    r_b64 = get_base64_image(ray_file)
    e_b64 = get_base64_image(eolo_file)
    
    html = '<div style="display: flex; justify-content: space-between; align-items: center; gap: 20px; margin-bottom: 40px; border-bottom: 2px solid #eee; padding-bottom: 20px;">'
    if l_b64: html += f'<img src="data:image/png;base64,{l_b64}" width="380">'
    if r_b64: html += f'<img src="data:image/png;base64,{r_b64}" width="140" style="opacity: 0.9;">'
    if e_b64: html += f'<img src="data:image/png;base64,{e_b64}" width="140" style="opacity: 0.8;">'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

render_full_header("Logo.png", "Ray.png", "Eolo.png")

st.title("🌪️ Determinación de Presiones de Viento")
st.markdown("#### **Norma Chilena NCh 432-2025** | Componentes y Revestimientos (C&R)")
st.divider()

# =================================================================
# 3. SIDEBAR: ENTRADA DE DATOS Y AYUDAS TÉCNICAS
# =================================================================
st.sidebar.header("⚙️ Parámetros del Proyecto")

# --- VELOCIDAD Y MAPAS ---
with st.sidebar.expander("🚩 Velocidad Básica de Viento (V)"):
    st.write("Valores de ráfaga de 3s a 10m de altura:")
    tabla_v = {"Zona": ["I-A", "II-B", "III-B", "IV-B", "V", "VI"], "V (m/s)": [27, 35, 35, 40, 40, 44]}
    st.table(pd.DataFrame(tabla_v))
    if st.button("Ver Mapas NCh 432"):
        for img in ["F2.png", "F3.png", "F4.png", "F5.png"]:
            if os.path.exists(img): st.image(img)

V = st.sidebar.number_input("Velocidad V (m/s)", 20.0, 60.0, 35.0, help="Velocidad básica según ubicación geográfica.")
H_edif = st.sidebar.number_input("Altura Edificio H (m)", 2.0, 200.0, 12.0)
theta = st.sidebar.slider("Inclinación Techo θ (°)", 0, 45, 10)

# --- GEOMETRÍA ---
st.sidebar.subheader("📐 Geometría Elemento")
l_elem = st.sidebar.number_input("Largo Elemento (m)", 0.1, 50.0, 3.0)
w_in = st.sidebar.number_input("Ancho Tributario Real (m)", 0.1, 50.0, 1.0)
w_trib = max(w_in, l_elem / 3)
area_ef = l_elem * w_trib

if w_in < (l_elem / 3):
    st.sidebar.warning(f"⚠️ Ancho tributario ajustado a L/3 ({w_trib:.2f} m) según norma.")

# --- TOPOGRAFÍA ---
with st.sidebar.expander("🏔️ Factor Topográfico (Kzt)"):
    metodo_kzt = st.radio("Cálculo Kzt:", ["Manual", "Calculado (Relieve)"])
    if metodo_kzt == "Manual":
        Kzt_val = st.number_input("Kzt directo", 1.0, 3.0, 1.0)
    else:
        tipo_rel = st.selectbox("Tipo Relieve", ["Escarpe 2D", "Colina 2D", "Colina 3D"])
        Hc = st.number_input("Hc (m)", 27.0); Lhc = st.number_input("Lh (m)", 1743.7)
        xdc = st.number_input("x (m)", 0.0); zac = st.number_input("z (m)", 10.0)
        # Parámetros según Figura 3 NCh 432
        p_rel = (0.75, 2.5, 1.5) if tipo_rel == "Escarpe 2D" else (1.05, 1.5, 1.5) if tipo_rel == "Colina 2D" else (0.95, 1.5, 4.0)
        k1_t = p_rel[0] * (Hc/Lhc); k2_t = (1 - abs(xdc)/(p_rel[2]*Lhc)); k3_t = math.exp(-p_rel[1]*zac/Lhc)
        Kzt_val = (1 + k1_t * k2_t * k3_t)**2
        st.info(f"Kzt Calculado: {Kzt_val:.3f}")

# --- CERRAMIENTO (CRÍTICO) ---
st.sidebar.subheader("🏠 Clasificación Cerramiento")
cerramiento_opcion = st.sidebar.selectbox("Tipo de Cerramiento", ["Cerrado", "Parcialmente Abierto", "Abierto"])

# =================================================================
# 4. MOTOR DE CÁLCULO Y LÓGICA NORMATIVA
# =================================================================
def get_gcp(a, g1, g10):
    if a <= 1.0: return g1
    if a >= 10.0: return g10
    return g1 + (g10 - g1) * (np.log10(a) - np.log10(1.0))

# Mapeo de factores
imp_map = {'I': 0.87, 'II': 1.0, 'III': 1.15, 'IV': 1.15}
gcpi_dict = {
    "Cerrado": [0.18, "Edificio estanco. Aberturas < 0.37 m² o < 1% área pared."],
    "Parcialmente Abierto": [0.55, "Abertura en una pared > suma resto aberturas."],
    "Abierto": [0.00, "Paredes con al menos 80% de aberturas."]
}
gc_pi_val, def_cerramiento = gcpi_dict[cerramiento_opcion]

Kd_val = st.sidebar.number_input("Factor Kd (Dirección)", 0.5, 1.0, 0.85)
cat_exp = st.sidebar.selectbox("Categoría Exposición", ['B', 'C', 'D'], index=0)
cat_imp = st.sidebar.selectbox("Importancia", ['I', 'II', 'III', 'IV'], index=2)

exp_params = {'B': [7.0, 366.0], 'C': [9.5, 274.0], 'D': [11.5, 213.0]}
alpha, zg = exp_params[cat_exp]
kz = 2.01 * ((max(H_edif, 4.6) / zg)**(2/alpha))

# Cálculo Presión Estática qh
qh = (0.613 * kz * Kzt_val * Kd_val * (V**2) * imp_map[cat_imp]) * 0.10197

# =================================================================
# 5. DESPLIEGUE TÉCNICO (RESULTADOS)
# =================================================================

st.markdown("### 📝 Memoria de Cálculo y Resultados")

col_formula, col_clasi = st.columns([1, 1.2])

with col_formula:
    st.write("**Desglose de Ecuaciones:**")
    st.latex(r"q_h = 0.613 \cdot K_z \cdot K_{zt} \cdot K_d \cdot V^2 \cdot I")
    st.latex(r"p = q_h \cdot [GC_p - GC_{pi}]")
    st.info(f"**Presión de Velocidad (qh):** {qh:.2f} kgf/m²")

with col_clasi:
    st.markdown(f"""
    <div class="classification-box">
        <strong>Ficha Técnica de Cerramiento:</strong><br>
        <strong>Estado:</strong> {cerramiento_opcion}<br>
        <strong>Factor GCpi:</strong> ± {gc_pi_val}<br>
        <small>{def_cerramiento}</small>
    </div>
    """, unsafe_allow_html=True)

# Cálculo de Coeficientes Externos GCp (Interpolación)
z1 = get_gcp(area_ef, -1.0, -0.9) if theta <= 7 else get_gcp(area_ef, -0.9, -0.8)
z2 = get_gcp(area_ef, -1.8, -1.1) if theta <= 7 else get_gcp(area_ef, -1.3, -1.2)
z3 = get_gcp(area_ef, -2.8, -1.1) if theta <= 7 else get_gcp(area_ef, -2.0, -1.2)
z4, z5 = get_gcp(area_ef, -1.1, -0.8), get_gcp(area_ef, -1.4, -1.1)

# Tabulación de resultados
col_tbl, col_plt = st.columns([1, 1.3])

with col_tbl:
    st.markdown("**Presiones Netas por Zona**")
    df_res = pd.DataFrame({
        "Zona de Análisis": ["Z1 (Techo)", "Z2 (Techo Borde)", "Z3 (Techo Esquina)", "Z4 (Fachada)", "Z5 (Fachada Esq.)"],
        "GCp (Ext)": [round(z, 3) for z in [z1, z2, z3, z4, z5]],
        "GCpi (Int)": [gc_pi_val] * 5,
        "Presión Diseño (kgf/m²)": [round(qh*(z - gc_pi_val), 2) for z in [z1, z2, z3, z4, z5]]
    })
    st.table(df_res)

with col_plt:
    areas = np.logspace(0, 1, 50)
    fig, ax = plt.subplots(figsize=(10, 6))
    # Curvas de Fachada
    ax.plot(areas, [get_gcp(a, -1.1, -0.8) for a in areas], label='Zona 4 (Fachada)', color='green', lw=2.5)
    ax.plot(areas, [get_gcp(a, -1.4, -1.1) for a in areas], label='Zona 5 (Fachada Esq.)', color='red', lw=2.5)
    # Curvas de Techo
    z3_plot = [get_gcp(a, -2.8, -1.1) if theta <= 7 else get_gcp(a, -2.0, -1.2) for a in areas]
    ax.plot(areas, z3_plot, label='Zona 3 (Techo Esq.)', color='navy', ls='--')
    
    # Marcar punto de diseño
    for z_pt in [z1, z2, z3, z4, z5]:
        ax.scatter([area_ef], [z_pt], color='black', s=50, zorder=5)
    
    ax.set_title("Influencia del Área Tributaria en GCp (Log-Log)"); ax.set_xlabel("Área (m²)"); ax.set_ylabel("GCp")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    st.pyplot(fig)

# =================================================================
# 6. REFERENCIAS Y CRÉDITOS
# =================================================================
st.divider()
c1, c2 = st.columns(2)
with c1:
    st.markdown("📌 **Identificación de Zonas (F8)**")
    if os.path.exists("F8.png"): st.image("F8.png")
with c2:
    st.markdown("📌 **Esquema Isométrico (F12)**")
    if os.path.exists("F12.png"): st.image("F12.png")

st.markdown("---")
st.markdown(f"""
    <div style="display: flex; justify-content: space-between; font-size: 0.9em; color: #666;">
        <div><strong>Autor:</strong> Mauricio Riquelme | Ing. Civil Estructural</div>
        <div><strong>Empresa:</strong> Proyectos Estructurales EIRL | <a href="mailto:mriquelme@proyectosestructurales.com">mriquelme@proyectosestructurales.com</a></div>
    </div>
    """, unsafe_allow_html=True)
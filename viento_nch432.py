# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import os
import math
from fpdf import FPDF

# =================================================================
# 1. CONFIGURACIÓN CORPORATIVA Y CONTROL DE UI (FULL WIDTH)
# =================================================================
st.set_page_config(
    page_title="NCh 432-2025 | Análisis de Viento Avanzado", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyección de CSS para control total de márgenes y estilo profesional
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
    .classification-box {
        background-color: #f1f8ff;
        padding: 20px;
        border: 1px solid #c8e1ff;
        border-radius: 5px;
        margin-bottom: 25px;
    }
    .stTable { width: 100% !important; font-size: 1.1em; }
    .sidebar-help { font-size: 0.85em; color: #555; line-height: 1.4; }
    </style>
    """, unsafe_allow_html=True)

# =================================================================
# 2. FUNCIONES DE SOPORTE (IMÁGENES Y LOGOS EN BASE64)
# =================================================================
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
    
    html_content = '<div style="display: flex; justify-content: center; align-items: center; gap: 40px; margin-bottom: 30px; flex-wrap: wrap; border-bottom: 2px solid #eee; padding-bottom: 20px;">'
    if logo_base_64: html_content += f'<img src="data:image/png;base64,{logo_base_64}" width="380">'
    if ray_base_64: html_content += f'<img src="data:image/png;base64,{ray_base_64}" width="130" style="opacity: 0.9;">'
    if eolo_base_64: html_content += f'<img src="data:image/png;base64,{eolo_base_64}" width="130" style="opacity: 0.8;">'
    html_content += '</div>'
    
    if logo_base_64 or ray_base_64 or eolo_base_64:
        st.markdown(html_content, unsafe_allow_html=True)
    else:
        st.title("🏗️ Proyectos Estructurales EIRL")

# Renderizado de Encabezado Corporativo
render_header_images("Logo.png", "Ray.png", "Eolo.png")

st.subheader("Determinación de Presiones de Viento según Norma NCh 432-2025")
st.caption("Análisis Integral de Presiones de Viento: Cubiertas, Fachadas y Perfiles de Altura | Ingeniería Civil Estructural")

# =================================================================
# 3. SIDEBAR CON GUÍA TÉCNICA COMPLETA Y RIGUROSA
# =================================================================
st.sidebar.header("⚙️ Parámetros de Diseño")

# --- GUÍA DE VELOCIDAD ---
with st.sidebar.expander("🚩 Guía: Velocidad Básica (V) y Mapas"):
    st.markdown("""
    **Zonificación según NCh 432 (Tabla 1):**
    Los valores representan la ráfaga de 3 segundos a 10m de altura en campo abierto (Categoría C).
    """)
    tabla_v = {"Zona": ["I-A", "II-B", "III-B", "IV-B", "V", "VI"], "V (m/s)": [27, 35, 35, 40, 40, 44]}
    st.table(pd.DataFrame(tabla_v))
    if st.button("Desplegar Mapas de Chile"):
        for img in ["F2.png", "F3.png", "F4.png", "F5.png"]:
            if os.path.exists(img): st.image(img, caption=f"Zonificación: {img}")

V = st.sidebar.number_input("Velocidad básica V (m/s)", 20.0, 60.0, 35.0)
H_edif = st.sidebar.number_input("Altura promedio edificio H (m)", 2.0, 200.0, 12.0)
theta = st.sidebar.slider("Inclinación de Techo θ (°)", 0, 45, 10)

# --- GEOMETRÍA ---
st.sidebar.subheader("📐 Geometría del Elemento")
l_elem = st.sidebar.number_input("Largo del elemento (m)", 0.1, 50.0, 3.0)
w_in = st.sidebar.number_input("Ancho tributario real (m)", 0.1, 50.0, 1.0)
w_trib = max(w_in, l_elem / 3)
area_ef = l_elem * w_trib

st.sidebar.info(f"**Área efectiva: {area_ef} m2**")


if w_in < (l_elem / 3):
    st.sidebar.warning(f"⚠️ Ancho ajustado por norma a {w_trib:.2f}m (mín. 1/3 del largo)")

# --- FACTOR TOPOGRÁFICO ---

with st.sidebar.expander("🏔️ Nota Explicativa: Factor Topográfico (Kzt)"):
    st.markdown("""
    **Criterios de Aplicación (Capítulo 5):**
                
    El factor Kzt considera la aceleración del viento sobre colinas, crestas y escarpes aislados. Se aplica cuando el relieve sobresale significativamente de su entorno.
    
    * **K1:** Factor de forma del relieve.
    * **K2:** Factor de reducción por distancia horizontal.
    * **K3:** Factor de reducción por altura sobre el suelo.
    
    * **Lh (Distancia horizontal):** Es la distancia horizontal en barlovento desde la cresta hasta donde la diferencia de elevación es la mitad de la altura del relieve ($H_c/2$).
    * **H_edif (Altura):** Se utiliza la altura máxima del edificio para determinar el factor de reducción $K_3$.
    * **Ubicación Crítica:** El cálculo asume $x = 0$ (cima de la cresta o escarpe) para obtener el valor máximo de aceleración del flujo.
    """)

    if st.button("Ver Diagramas de Relieve"):
        for img in ["F7.png", "F6.png"]:
            if os.path.exists(img): st.image(img)

metodo = st.sidebar.radio("Cálculo de Kzt", ["Manual", "Calculado"])

if metodo == "Manual":
    Kzt_val = st.sidebar.number_input("Valor Kzt directo", 1.0, 3.0, 1.0)
else:
    tipo_relieve = st.sidebar.selectbox("Forma del relieve", ["Escarpe 2D", "Colina 2D", "Colina 3D"])
    
    # Parámetros del relieve
    Hc = st.sidebar.number_input("Altura del relieve Hc (m)", value=27.0, help="Elevación total del relieve sobre el terreno circundante.")
    Lhc = st.sidebar.number_input("Lh (m)", value=100.0, help="Distancia horizontal a la mitad de la altura Hc.")
    
    # Asignación de constantes según tipo de relieve (NCh 432)
    if tipo_relieve == "Escarpe 2D":
        k1_b, gam, mu_v = 0.75, 2.5, 1.5
    elif tipo_relieve == "Colina 2D":
        k1_b, gam, mu_v = 1.05, 1.5, 1.5
    else: # Colina 3D
        k1_b, gam, mu_v = 0.95, 1.5, 4.0
    
    # Cálculo de Factores (Asumiendo x=0 y z=H_edif)
    k1 = k1_b * (Hc / Lhc)
    k2 = 1.0 
    k3 = math.exp(-gam * H_edif / Lhc) 
    
    Kzt_val = (1 + k1 * k2 * k3)**2
    
    st.sidebar.info(f"""
    **Resultados Locales:**
    * K1: {k1:.3f}
    * K3: {k3:.3f}
    * **Kzt Calculado: {Kzt_val:.3f}**
    """)

# --- FACTORES NORMATIVOS ---
st.sidebar.subheader("📋 Factores Normativos")

with st.sidebar.expander("ℹ️ Nota Explicativa: Factor de Direccionalidad (Kd)"):
    st.markdown("""
    **Criterios de la Tabla 2 (NCh 432:2025):**
    Este factor compensa la reducida probabilidad de que el viento máximo sople precisamente desde la dirección más crítica.
    """)

Kd_val = st.sidebar.number_input("Factor de Direccionalidad Kd", 0.50, 1.00, 0.85, step=0.01)

with st.sidebar.expander("ℹ️ Nota Explicativa: Exposición (B, C, D)"):
    st.markdown("""
    **Rugosidad del Terreno (Capítulo 4):**
    * **B:** Urbano/Suburbano.
    * **C:** Terreno Abierto.
    * **D:** Costa.
    """)

cat_exp = st.sidebar.selectbox("Categoría de Exposición", ['B', 'C', 'D'], index=0)

exp_info = {
    'B': [7.0, 366.0, "Urbano/Suburbano"],
    'C': [9.5, 274.0, "Terreno Abierto"],
    'D': [11.5, 213.0, "Costa/Agua"]
}

alpha_val = exp_info[cat_exp][0]
zg_val = exp_info[cat_exp][1]
desc_exp = exp_info[cat_exp][2]

st.sidebar.info(f"**Parámetros:** α={alpha_val}, zg={zg_val}m")

# CATEGORÍA DE RIESGO
cat_imp = st.sidebar.selectbox("Categoría de Riesgo / Riesgo", ['I', 'II', 'III', 'IV'], index=1)
imp_map = {'I': 0.54, 'II': 1.0, 'III': 1.15, 'IV': 1.22}
factor_i = imp_map[cat_imp]
st.sidebar.info(f"**Factor I: {factor_i}**")

# =================================================================
# 4. MOTOR DE CÁLCULO Y DEFINICIÓN DE CERRAMIENTO (RIGUROSO)
# =================================================================
st.sidebar.subheader("🏠 Clasificación del Cerramiento")

cerramiento_opcion = st.sidebar.selectbox(
    "Tipo de Cerramiento", 
    ["Cerrado", "Parcialmente Abierto", "Abierto"],
    index=0
)

gcpi_data = {
    "Cerrado": [0.18, "Estándar para estructuras estancas."],
    "Parcialmente Abierto": [0.55, "Área de aberturas dominante en una pared."],
    "Abierto": [0.00, "Viento fluye libremente (80% aberturas)."]
}

gc_pi_val = gcpi_data[cerramiento_opcion][0]
nota_tecnica_cerramiento = gcpi_data[cerramiento_opcion][1]

st.sidebar.info(f"**GCpi: ± {gc_pi_val}**")

# --- MOTOR MATEMÁTICO ---
def get_gcp(a, g1, g10):
    if a <= 1.0: return g1
    if a >= 10.0: return g10
    return g1 + (g10 - g1) * (np.log10(a) - np.log10(1.0))

kz = 2.01 * ((max(H_edif, 4.6) / zg_val)**(2/alpha_val))
qh = (0.613 * kz * Kzt_val * Kd_val * (V**2) * factor_i) * 0.10197

# =================================================================
# 5. GENERADOR DE PDF PROFESIONAL (INTEGRADO SIN CORTAR)
# =================================================================
def generar_pdf_viento():
    pdf = FPDF()
    pdf.add_page()
    if os.path.exists("Logo.png"):
        pdf.image("Logo.png", x=10, y=8, w=33)
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Memoria de Calculo: Viento NCh 432-2025", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 7, "Proyectos Estructurales | Mauricio Riquelme", ln=True, align='C')
    pdf.ln(15)

    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, " 1. PARAMETROS DE DISENO", ln=True, fill=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 8, f" Velocidad V: {V} m/s | Altura H: {H_edif} m | Inclinacion: {theta} C", ln=True)
    pdf.cell(0, 8, f" Exposicion: {cat_exp} ({desc_exp}) | Riesgo: {cat_imp} (I={factor_i})", ln=True)
    pdf.cell(0, 8, f" Cerramiento: {cerramiento_opcion} (GCpi: {gc_pi_val})", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, " 2. FACTORES Y PRESION DE VELOCIDAD", ln=True, fill=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(95, 8, f" Kz (Exposicion): {kz:.3f}", border=0)
    pdf.cell(95, 8, f" Kzt (Topografico): {Kzt_val:.3f}", ln=True)
    pdf.cell(95, 8, f" Kd (Direccionalidad): {Kd_val:.3f}", border=0)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(95, 8, f" qh (Presion de Velocidad): {qh:.2f} kgf/m2", ln=True)
    
    pdf.set_y(-25)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "Memoria generada por AccuraWall Port - Mauricio Riquelme", align='C')
    return pdf.output()

st.sidebar.markdown("---")
pdf_bytes = generar_pdf_viento()
b64 = base64.b64encode(pdf_bytes).decode()
st.sidebar.markdown(f"""
    <div style="text-align: center;">
        <a href="data:application/pdf;base64,{b64}" download="Memoria_Viento_NCh432.pdf" 
           style="background-color: #0056b3; color: white; padding: 12px 20px; text-decoration: none; 
           border-radius: 5px; font-weight: bold; display: block;">
           📥 DESCARGAR MEMORIA PDF
        </a>
    </div>
""", unsafe_allow_html=True)

# =================================================================
# 6. DESPLIEGUE TÉCNICO DE RESULTADOS Y FORMULACIÓN
# =================================================================

st.markdown(f"""
<div class="classification-box">
    <strong>📋 Ficha Técnica de Cerramiento (NCh 432):</strong> {cerramiento_opcion} | 
    <span style="color: #d9534f;"><strong>Factor GCpi: ± {gc_pi_val}</strong></span>
</div>
""", unsafe_allow_html=True)

st.markdown("### 📝 Ecuaciones de Diseño Aplicadas")
st.latex(r"q_h = 0.613 \cdot K_z \cdot K_{zt} \cdot K_d \cdot V^2 \cdot I")
st.latex(r"p = q_h \cdot [GC_p - GC_{pi}]")

st.info(f"**Presión de Velocidad Calculada (qh):** {qh:.2f} kgf/m²")

# Coeficientes de las 5 Zonas
z1 = get_gcp(area_ef, -1.0, -0.9) if theta <= 7 else get_gcp(area_ef, -0.9, -0.8)
z2 = get_gcp(area_ef, -1.8, -1.1) if theta <= 7 else get_gcp(area_ef, -1.3, -1.2)
z3 = get_gcp(area_ef, -2.8, -1.1) if theta <= 7 else get_gcp(area_ef, -2.0, -1.2)
z4, z5 = get_gcp(area_ef, -1.1, -0.8), get_gcp(area_ef, -1.4, -1.1)

# Tabulación
col_res, col_plt = st.columns([1, 1.3])

with col_res:
    st.markdown("**Resumen de Presiones Netas por Zona**")
    df_res = pd.DataFrame({
        "Zona": ["Z1 (T. Centro)", "Z2 (T. Borde)", "Z3 (T. Esq.)", "Z4 (Fachada)", "Z5 (F. Esq.)"],
        "Presión (kgf/m²)": [round(qh*(z - gc_pi_val), 2) for z in [z1, z2, z3, z4, z5]]
    })
    st.table(df_res)
    st.warning("⚠️ Valores negativos indican succión.")

with col_plt:
    areas = np.logspace(0, 1, 50)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(areas, [get_gcp(a, -1.1, -0.8) for a in areas], label='Z4 (Fachada)', color='green', lw=2.5)
    ax.plot(areas, [get_gcp(a, -1.4, -1.1) for a in areas], label='Z5 (Fachada Esq.)', color='red', lw=2.5)
    ax.set_title("Variación de GCp (NCh 432)"); ax.set_xlabel("Área (m²)"); ax.set_ylabel("GCp")
    ax.grid(True, which="both", alpha=0.3); ax.legend(); st.pyplot(fig)

# =================================================================
# 7. DISTRIBUCIÓN DE PRESIONES: BARLOVENTO Y SOTAVENTO
# =================================================================
st.divider()
st.subheader("📊 Perfil de Presiones en Altura")

alturas_perfil = np.linspace(0.1, H_edif, 50)
p_barlo = []
for z_alt in alturas_perfil:
    kz_z = 2.01 * ((max(z_alt, 4.6) / zg_val)**(2/alpha_val))
    qz = (0.613 * kz_z * Kzt_val * Kd_val * (V**2) * factor_i) * 0.10197
    p_barlo.append(qz * (z4 - gc_pi_val))

fig_alt, ax_alt = plt.subplots(figsize=(12, 6))
ax_alt.plot(p_barlo, alturas_perfil, color='green', lw=2, label="Barlovento Z4")
ax_alt.fill_betweenx(alturas_perfil, p_barlo, 0, color='green', alpha=0.1)
ax_alt.axvline(0, color='black')
ax_alt.set_title("Distribución Vertical de Presión"); ax_alt.set_ylabel("Altura (m)"); ax_alt.legend(); ax_alt.grid(True)
st.pyplot(fig_alt)

# ESQUEMAS
st.divider()
col_img1, col_img2 = st.columns(2)
with col_img1:
    if os.path.exists("F8.png"): st.image("F8.png", caption="Identificación de Zonas")
with col_img2:
    if os.path.exists("F12.png"): st.image("F12.png", caption="Esquema Isométrico")

# CIERRE
st.markdown("---")
st.markdown(f"""
    <div style="display: flex; justify-content: space-between; color: #444; font-size: 0.95em;">
        <div><strong>Desarrollado por:</strong> Mauricio Riquelme</div>
        <div><strong>Contacto:</strong> mriquelme@proyectosestructurales.com</div>
    </div>
    <div style="text-align: center; margin-top: 30px;">
        <p style="font-family: 'Georgia', serif; font-size: 1.4em; color: #003366; font-style: italic;">
            "Programming is understanding"
        </p>
    </div>
""", unsafe_allow_html=True)
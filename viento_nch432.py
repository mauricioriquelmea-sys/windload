# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import os
import math

# =================================================================
# 1. CONFIGURACIÓN CORPORATIVA Y CONTROL DE ANCHO TOTAL (FULL WIDTH)
# =================================================================
st.set_page_config(
    page_title="NCh 432-2025 | Análisis de Viento Avanzado", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyección de CSS para control de UI y eliminación de márgenes restrictivos
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

# Renderizado de Encabezado
render_header_images("Logo.png", "Ray.png", "Eolo.png")

st.subheader("Determinación de Presiones de Viento según Norma NCh 432-2025")
st.caption("Análisis Integral de Presiones de Viento: Cubiertas y Fachadas | Ingeniería Civil Estructural")

# =================================================================
# 3. SIDEBAR CON GUÍA TÉCNICA COMPLETA Y RIGUROSA
# =================================================================
st.sidebar.header("⚙️ Parámetros de Diseño")

with st.sidebar.expander("🚩 Guía: Velocidad Básica (V) y Mapas"):
    st.markdown("""
    **Zonificación según NCh 432 (Tabla 1):**
    Los valores representan la ráfaga de 3 segundos a 10m de altura en campo abierto.
    """)
    tabla_v = {"Zona": ["I-A", "II-B", "III-B", "IV-B", "V", "VI"], "V (m/s)": [27, 35, 35, 40, 40, 44]}
    st.table(pd.DataFrame(tabla_v))
    if st.button("Desplegar Mapas de Chile"):
        for img in ["F2.png", "F3.png", "F4.png", "F5.png"]:
            if os.path.exists(img): st.image(img, caption=f"Zonificación: {img}")

V = st.sidebar.number_input("Velocidad básica V (m/s)", 20.0, 60.0, 35.0)
H_edif = st.sidebar.number_input("Altura promedio edificio H (m)", 2.0, 200.0, 12.0)
theta = st.sidebar.slider("Inclinación de Techo θ (°)", 0, 45, 10)

st.sidebar.subheader("📐 Geometría del Elemento")
l_elem = st.sidebar.number_input("Largo del elemento (m)", 0.1, 50.0, 3.0)
w_in = st.sidebar.number_input("Ancho tributario real (m)", 0.1, 50.0, 1.0)
w_trib = max(w_in, l_elem / 3)
area_ef = l_elem * w_trib

if w_in < (l_elem / 3):
    st.sidebar.warning(f"⚠️ Ancho ajustado por norma a {w_trib:.2f}m (mín. 1/3 del largo)")

with st.sidebar.expander("🏔️ Factor Topográfico Riguroso (Kzt)"):
    st.markdown("""
    **Criterios de Aplicación:**
    El factor Kzt se aplica cuando la estructura se ubica en la mitad superior de una colina o escarpe, donde se produce una aceleración del flujo.
    """)
    if st.button("Ver Diagramas de Relieve"):
        for img in ["F7.png", "F6.png"]:
            if os.path.exists(img): st.image(img)
    
    metodo = st.sidebar.radio("Método de Selección Kzt", ["Manual", "Calculado (Procedimiento Figura 3)"])
    if metodo == "Manual":
        Kzt_val = st.sidebar.number_input("Valor Kzt directo", 1.0, 3.0, 1.0)
    else:
        tipo_relieve = st.sidebar.selectbox("Forma de relieve", ["Escarpe 2D", "Colina 2D", "Colina 3D"])
        Hc = st.sidebar.number_input("Altura colina H (m)", value=27.0)
        Lhc = st.sidebar.number_input("Distancia Lh (m)", value=1743.7)
        xdc = st.sidebar.number_input("Distancia horizontal x (m)", value=0.0)
        zac = st.sidebar.number_input("Altura z s/suelo (m)", value=10.0)
        
        # Parámetros según NCh 432
        k1b, gam, mu_v = (0.75, 2.5, 1.5) if tipo_relieve == "Escarpe 2D" else (1.05, 1.5, 1.5) if tipo_relieve == "Colina 2D" else (0.95, 1.5, 4.0)
        k1_t, k2_t, k3_t = k1b*(Hc/Lhc), (1-abs(xdc)/(mu_v*Lhc)), math.exp(-gam*zac/Lhc)
        Kzt_val = (1 + k1_t*k2_t*k3_t)**2
        st.sidebar.info(f"Kzt Calculado: {Kzt_val:.3f}")

# --- AYUDA TÉCNICA RIGUROSA: FACTOR DE DIRECCIÓN (Kd) ---
with st.sidebar.expander("ℹ️ Ayuda Técnica: Factor Kd"):
    st.markdown("""
    **Factor de Dirección (Tabla 2):**
    Compensa la probabilidad de que el viento sople desde la dirección más desfavorable.
    * **Edificios (C&R):** 0.85
    * **Estructuras Redondeadas:** 0.90 - 0.95
    * **Torres de Celosía:** 0.85
    """)
Kd_val = st.sidebar.number_input("Factor Kd", 0.5, 1.0, 0.85, step=0.05)

# --- AYUDA TÉCNICA RIGUROSA: CATEGORÍA DE EXPOSICIÓN ---
with st.sidebar.expander("ℹ️ Ayuda Técnica: Exposición (B, C, D)"):
    st.markdown("""
    **Rugosidad del Terreno:**
    * **B:** Áreas urbanas/suburbanas con obstrucciones de gran tamaño (Viviendas o más).
    * **C:** Terrenos abiertos con obstrucciones dispersas < 9m (Campo abierto).
    * **D:** Áreas planas sin obstrucciones frente a cuerpos de agua (Costa).
    """)
cat_exp = st.sidebar.selectbox("Exposición", ['B', 'C', 'D'], index=0)

# --- AYUDA TÉCNICA RIGUROSA: CATEGORÍA DE IMPORTANCIA ---
with st.sidebar.expander("ℹ️ Ayuda Técnica: Importancia (I a IV)"):
    st.markdown("""
    **Clasificación de Riesgo:**
    * **I:** Bajo riesgo (Agrícola).
    * **II:** Estándar (Residencial/Oficinas).
    * **III:** Concurrencia masiva (Colegios/Cines).
    * **IV:** Esenciales (Hospitales/Bomberos).
    """)
cat_imp = st.sidebar.selectbox("Importancia", ['I', 'II', 'III', 'IV'], index=2)

# =================================================================
# 4. MOTOR DE CÁLCULO Y LÓGICA DE PRESIÓN INTERNA
# =================================================================
st.sidebar.subheader("🏠 Clasificación del Edificio")
cerramiento_opcion = st.sidebar.selectbox("Tipo de Cerramiento", ["Cerrado", "Parcialmente Abierto", "Abierto"])

# Diccionario riguroso de definiciones y factores GCpi
gcpi_dict = {
    "Cerrado": [0.18, "Un edificio que no cumple con los requisitos de abierto o parcialmente abierto. Es el estándar para estructuras estancas donde las aberturas son menores al 1% del área de la pared."],
    "Parcialmente Abierto": [0.55, "Edificio donde el área de aberturas en una pared excede la suma de aberturas en el resto del edificio en más del 10% o 0.37 m²."],
    "Abierto": [0.00, "Un edificio que tiene al menos un 80% de aberturas en cada pared. El viento fluye a través de la estructura sin generar presiones internas significativas."]
}
gc_pi_val, def_cerramiento = gcpi_dict[cerramiento_opcion]

def get_gcp(a, g1, g10):
    if a <= 1.0: return g1
    if a >= 10.0: return g10
    return g1 + (g10 - g1) * (np.log10(a) - np.log10(1.0))

imp_map = {'I': 0.87, 'II': 1.0, 'III': 1.15, 'IV': 1.15}
exp_params = {'B': [7.0, 366.0], 'C': [9.5, 274.0], 'D': [11.5, 213.0]}
alpha, zg = exp_params[cat_exp]
kz = 2.01 * ((max(H_edif, 4.6) / zg)**(2/alpha))

# Cálculo Presión Estática qh
qh = (0.613 * kz * Kzt_val * Kd_val * (V**2) * imp_map[cat_imp]) * 0.10197

# =================================================================
# 5. DESPLIEGUE TÉCNICO DE RESULTADOS Y FORMULACIÓN
# =================================================================

# Caja de Fórmulas y Ficha de Cerramiento
col_formula, col_clasi = st.columns([1, 1.2])

with col_formula:
    st.markdown("""
    <div class="formula-box">
        <strong>1. Presión de Velocidad (qh):</strong> <br>
        <p style="text-align: center; font-size: 1.2em;">$q_h = 0.613 \cdot K_z \cdot K_{zt} \cdot K_d \cdot V^2 \cdot I$</p>
        <strong>2. Presión de Diseño Neta (p):</strong> <br>
        <p style="text-align: center; font-size: 1.4em; color: #0056b3;">$p = q_h \cdot [GC_p - GC_{pi}]$</p>
    </div>
    """, unsafe_allow_html=True)
    st.info(f"**qh Calculado:** {qh:.2f} kgf/m²")

with col_clasi:
    st.markdown(f"""
    <div class="classification-box">
        <strong>Ficha Técnica de Cerramiento (NCh 432):</strong><br>
        <strong>Estado:</strong> {cerramiento_opcion}<br>
        <span style="font-size: 1.3em; color: #d9534f;"><strong>Factor GCpi: ± {gc_pi_val}</strong></span><br>
        <p style="font-size: 0.9em; margin-top: 10px;"><em>{def_cerramiento}</em></p>
    </div>
    """, unsafe_allow_html=True)

# Coeficientes de las 5 Zonas (Fachada y Techo)
z1 = get_gcp(area_ef, -1.0, -0.9) if theta <= 7 else get_gcp(area_ef, -0.9, -0.8)
z2 = get_gcp(area_ef, -1.8, -1.1) if theta <= 7 else get_gcp(area_ef, -1.3, -1.2)
z3 = get_gcp(area_ef, -2.8, -1.1) if theta <= 7 else get_gcp(area_ef, -2.0, -1.2)
z4, z5 = get_gcp(area_ef, -1.1, -0.8), get_gcp(area_ef, -1.4, -1.1)

# Tabulación de Resultados
col_res, col_plt = st.columns([1, 1.3])

with col_res:
    st.markdown("**Resumen de Presiones Netas de Diseño**")
    df_res = pd.DataFrame({
        "Zona": ["Z1 (Techo Centro)", "Z2 (Techo Borde)", "Z3 (Techo Esquina)", "Z4 (Fachada Estándar)", "Z5 (Fachada Esquina)"],
        "GCp (Externo)": [round(z, 3) for z in [z1, z2, z3, z4, z5]],
        "GCpi (Interno)": [gc_pi_val] * 5,
        "Presión Neta (kgf/m²)": [round(qh*(z - gc_pi_val), 2) for z in [z1, z2, z3, z4, z5]]
    })
    st.table(df_res)
    st.warning("⚠️ Nota: Valores negativos indican succión (presión hacia afuera).")

with col_plt:
    areas = np.logspace(0, 1, 50)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Graficar las 5 Zonas
    if theta <= 7:
        ax.plot(areas, [get_gcp(a, -1.0, -0.9) for a in areas], label='Z1 (Techo)', color='cyan', alpha=0.5)
        ax.plot(areas, [get_gcp(a, -1.8, -1.1) for a in areas], label='Z2 (Techo)', color='blue', alpha=0.5)
        ax.plot(areas, [get_gcp(a, -2.8, -1.1) for a in areas], label='Z3 (Techo Esq.)', color='navy', ls='--')
    else:
        ax.plot(areas, [get_gcp(a, -0.9, -0.8) for a in areas], label='Z1 (Techo)', color='cyan', alpha=0.5)
        ax.plot(areas, [get_gcp(a, -1.3, -1.2) for a in areas], label='Z2 (Techo)', color='blue', alpha=0.5)
        ax.plot(areas, [get_gcp(a, -2.0, -1.2) for a in areas], label='Z3 (Techo Esq.)', color='navy', ls='--')
    
    ax.plot(areas, [get_gcp(a, -1.1, -0.8) for a in areas], label='Z4 (Fachada)', color='green', lw=2.5)
    ax.plot(areas, [get_gcp(a, -1.4, -1.1) for a in areas], label='Z5 (Fachada Esq.)', color='red', lw=2.5)
    
    for z_v in [z1, z2, z3, z4, z5]:
        ax.scatter([area_ef], [z_v], color='black', s=50, zorder=10)

    ax.set_title("Variación de GCp según Área Tributaria (NCh 432)"); ax.set_xlabel("Área (m²)"); ax.set_ylabel("GCp")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize='small', loc='best')
    st.pyplot(fig)

# =================================================================
# 6. ESQUEMAS NORMATIVOS Y REFERENCIAS FINALES
# =================================================================
st.markdown("---")
col_img1, col_img2 = st.columns(2)
with col_img1:
    st.subheader("📍 Identificación de Zonas (F8)")
    if os.path.exists("F8.png"): st.image("F8.png", caption="Distribución de Zonas 1 a 5 en C&R")
with col_img2:
    st.subheader("📍 Esquema Isométrico (F12)")
    if os.path.exists("F12.png"): st.image("F12.png", caption="Cargas de Viento en Fachadas de Policarbonato")

# CONTACTO Y CRÉDITOS
st.markdown("---")
st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; color: #444; font-size: 0.9em;">
        <div><strong>Autor:</strong> Mauricio Riquelme | Ingeniero Civil Estructural</div>
        <div style="text-align: right;"><strong>Proyectos Estructurales EIRL:</strong><br>
            <a href="mailto:mriquelme@proyectosestructurales.com">mriquelme@proyectosestructurales.com</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
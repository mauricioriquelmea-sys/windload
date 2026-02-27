# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import os
import math

# 1. CONFIGURACIÓN CORPORATIVA
st.set_page_config(page_title="NCh 432-2025 | Proyectos Estructurales", layout="wide")

def get_base64_image(image_path):
    """Convierte una imagen a base64 para embeberla en HTML"""
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            data = f.read()
            return base64.b64encode(data).decode()
    return None

def render_header_images(logo_file, eolo_file):
    """Renderiza el Logo Corporativo y Eolo al lado, centrados."""
    logo_base64 = get_base64_image(logo_file)
    eolo_base64 = get_base64_image(eolo_file)
    
    # Si ambas imágenes existen, creamos el layout centrado
    if logo_base64 and eolo_base64:
        st.markdown(f"""
            <div style="display: flex; justify-content: center; align-items: center; gap: 50px; margin-bottom: 20px;">
                <img src="data:image/png;base64,{logo_base64}" width="400">
                <img src="data:image/png;base64,{eolo_base64}" width="150" style="opacity: 0.8;">
            </div>
            """, unsafe_allow_html=True)
    elif logo_base64:
        # Si solo está el logo, lo centramos normal
        st.markdown(f'<div style="text-align: center;"><img src="data:image/png;base64,{logo_base64}" width="500"></div>', unsafe_allow_html=True)
    else:
        st.title("🏗️ Proyectos Estructurales EIRL")

# REEMPLAZO DE LA FUNCIÓN ANTERIOR
# Renderizamos el header con ambas imágenes
render_header_images("Logo.png", "Eolo.png")

st.subheader("Determinación de Presiones de Viento según Norma NCh 432-2025")
st.caption("Análisis Integral de Presiones de Viento: Cubiertas y Fachadas")

# 2. SIDEBAR CON GUÍA TÉCNICA
st.sidebar.header("⚙️ Parámetros de Diseño")

with st.sidebar.expander("🚩 Guía de Velocidad (V) y Mapas"):
    st.write("**Zonificación Tabla 1:**")
    tabla_v = {"Zona": ["I-A", "II-B", "III-B", "IV-B", "V", "VI"], "V (m/s)": [27, 35, 35, 40, 40, 44]}
    st.table(pd.DataFrame(tabla_v))
    if st.button("Desplegar Mapas de Chile"):
        for img_name in ["F2.png", "F3.png", "F4.png", "F5.png"]:
            if os.path.exists(img_name): st.image(img_name, caption=f"Norma NCh 432: {img_name}")

V = st.sidebar.number_input("Velocidad básica V (m/s)", value=35.0)
H_edif = st.sidebar.number_input("Altura edificio H (m)", value=12.0)
theta = st.sidebar.slider("Inclinación Techo θ (°)", 0, 45, 10)

st.sidebar.subheader("📐 Geometría del Elemento")
l_elem = st.sidebar.number_input("Largo elemento (m)", value=3.0)
w_in = st.sidebar.number_input("Ancho trib. real (m)", value=1.0)
w_trib = max(w_in, l_elem / 3)
area_ef = l_elem * w_trib
if w_in < (l_elem / 3): st.sidebar.warning(f"⚠️ Ancho ajustado por norma a {w_trib:.2f}m (mín. 1/3 del largo)")

with st.sidebar.expander("🏔️ Factor Topográfico (Kzt)"):
    if st.button("Ver Diagramas Topográficos"):
        for img in ["F7.png", "F6.png"]:
            if os.path.exists(img): st.image(img)
    metodo = st.radio("Método de cálculo", ["Manual", "Calculado (Escarpe/Colina)"])
    if metodo == "Manual":
        Kzt_val = st.number_input("Valor Kzt directo", value=1.0, step=0.1)
    else:
        tipo_relieve = st.selectbox("Forma del relieve", ["Escarpe 2D", "Colina 2D", "Colina 3D"])
        H_c, L_h, x_d, z_a = st.number_input("H (m)", 27.0), st.number_input("Lh (m)", 1743.7), st.number_input("x (m)", 0.0), st.number_input("z (m)", 10.0)
        k1_b, gamma, mu = (0.75, 2.5, 1.5) if tipo_relieve == "Escarpe 2D" else (1.05, 1.5, 1.5) if tipo_relieve == "Colina 2D" else (0.95, 1.5, 4.0)
        k1, k2, k3 = k1_b*(H_c/L_h), (1-abs(x_d)/(mu*L_h)), math.exp(-gamma*z_a/L_h)
        Kzt_val = (1 + k1*k2*k3)**2
        st.info(f"Kzt Calculado: {Kzt_val:.3f}")

st.sidebar.subheader("📋 Factores Normativos")

# --- AYUDA TÉCNICA RIGUROSA: FACTOR DE DIRECCIÓN (Kd) ---
with st.sidebar.expander("ℹ️ Ayuda Técnica: Factor de Dirección (Kd)"):
    st.markdown("""
    **Criterio Normativo (Tabla 2):**
    Este factor compensa la reducida probabilidad de que el viento máximo sople precisamente desde la dirección más desfavorable para la orientación del elemento.
    
    * **Edificios (Sistemas Principales y C&R):** **0.85**. Aplicable a la mayoría de estructuras de marcos rígidos y revestimientos de fachada.
    * **Cubiertas Arqueadas:** **0.85**.
    * **Chimeneas, Tanques y Estructuras Similares:** * Cuadradas: **0.90**
        * Hexagonales: **0.95**
        * Redondas: **0.95**
    * **Torres de Celosía (Triangulares/Cuadradas):** **0.85**.
    
    *Nota: Solo debe aplicarse cuando se combina con otros factores de carga.*
    """)
Kd_manual = st.sidebar.number_input("Factor Kd", value=0.85, step=0.05)

# --- AYUDA TÉCNICA RIGUROSA: CATEGORÍA DE EXPOSICIÓN ---
with st.sidebar.expander("ℹ️ Ayuda Técnica: Categoría de Exposición"):
    st.markdown("""
    **Definiciones según Rugosidad del Terreno:**
    
    * **Exposición B:** Terreno con rugosidad tipo B. Áreas urbanas y suburbanas, áreas boscosas u otros terrenos con numerosas obstrucciones próximas unas a otras (del tamaño de viviendas unifamiliares o mayores). Se aplica si la rugosidad prevalece en 800m o 20 veces la altura del edificio.
    * **Exposición C:** Terreno abierto con obstrucciones dispersas que tienen alturas generalmente menores a 9m. Incluye campos abiertos y terrenos agrícolas. Es la categoría por defecto si no aplica B o D.
    * **Exposición D:** Áreas planas y sin obstrucciones expuestas al viento que sopla sobre cuerpos de agua (excluyendo zonas costeras en regiones de huracanes) en una distancia de al menos 1.5km. Se extiende hacia sotavento 200m desde la orilla.
    """)
cat_exp = st.sidebar.selectbox("Exposición", ['B', 'C', 'D'], index=0)

# --- AYUDA TÉCNICA RIGUROSA: CATEGORÍA DE IMPORTANCIA / RIESGO ---
with st.sidebar.expander("ℹ️ Ayuda Técnica: Categoría de Edificio (Riesgo)"):
    st.markdown("""
    **Clasificación según Consecuencias de Falla:**
    
    * **Categoría I:** Edificios y estructuras que representan un **riesgo bajo** para la vida humana en caso de falla (ej: instalaciones agrícolas, bodegas temporales, cercos).
    * **Categoría II:** Todas las estructuras que **no clasifican** en las categorías I, III y IV (ej: viviendas residenciales, edificios de oficinas estándar, locales comerciales).
    * **Categoría III:** Edificios con **gran número de personas** o capacidad limitada de evacuación (ej: colegios, cárceles, cines, estadios, centros comerciales de alta concurrencia).
    * **Categoría IV:** Estructuras **esenciales** cuya operatividad es crítica tras un evento (ej: hospitales, estaciones de bomberos/policía, refugios de emergencia, centros de comunicación y plantas de energía).
    """)
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
qh = (0.613 * kz * Kzt_val * Kd_manual * (V**2) * imp_map[cat_imp]) * 0.10197
gc_pi = 0.18

# Coeficientes de Zonas
z1 = get_gcp(area_ef, -1.0, -0.9) if theta <= 7 else get_gcp(area_ef, -0.9, -0.8)
z2 = get_gcp(area_ef, -1.8, -1.1) if theta <= 7 else get_gcp(area_ef, -1.3, -1.2)
z3 = get_gcp(area_ef, -2.8, -1.1) if theta <= 7 else get_gcp(area_ef, -2.0, -1.2)
z4, z5 = get_gcp(area_ef, -1.1, -0.8), get_gcp(area_ef, -1.4, -1.1)

# 4. RESULTADOS Y GRÁFICO INTEGRAL
col1, col2 = st.columns([1, 1.2])
with col1:
    st.metric("Presión qh", f"{qh:.2f} kgf/m²")
    df = pd.DataFrame({
        "Zona": ["Zona 1 (Techo Centro)", "Zona 2 (Techo Borde)", "Zona 3 (Techo Esquina)", "Zona 4 (Muro Estándar)", "Zona 5 (Muro Esquina)"],
        "GCp": [round(z, 3) for z in [z1, z2, z3, z4, z5]],
        "Presión Diseño (kgf/m²)": [round(qh*(z-gc_pi), 2) for z in [z1, z2, z3, z4, z5]]
    })
    st.table(df)

with col2:
    areas = np.logspace(0, 1, 50)
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Graficar las 5 ZONAS
    if theta <= 7:
        ax.plot(areas, [get_gcp(a, -1.0, -0.9) for a in areas], label='Z1 (Techo)', color='cyan', alpha=0.6)
        ax.plot(areas, [get_gcp(a, -1.8, -1.1) for a in areas], label='Z2 (Techo)', color='blue', alpha=0.6)
        ax.plot(areas, [get_gcp(a, -2.8, -1.1) for a in areas], label='Z3 (Techo Esquina)', color='navy', ls='--')
    else:
        ax.plot(areas, [get_gcp(a, -0.9, -0.8) for a in areas], label='Z1 (Techo)', color='cyan', alpha=0.6)
        ax.plot(areas, [get_gcp(a, -1.3, -1.2) for a in areas], label='Z2 (Techo)', color='blue', alpha=0.6)
        ax.plot(areas, [get_gcp(a, -2.0, -1.2) for a in areas], label='Z3 (Techo Esquina)', color='navy', ls='--')
    
    ax.plot(areas, [get_gcp(a, -1.1, -0.8) for a in areas], label='Z4 (Muro)', color='green', lw=2)
    ax.plot(areas, [get_gcp(a, -1.4, -1.1) for a in areas], label='Z5 (Muro Esquina)', color='red', lw=2)
    
    for z_v in [z1, z2, z3, z4, z5]:
        ax.scatter([area_ef], [z_v], color='black', zorder=5)

    ax.set_title("Comparativa de 5 Zonas (Log-Interpolación)")
    ax.set_xlabel("Área (m²)"); ax.set_ylabel("GCp"); ax.grid(True, alpha=0.3); ax.legend(fontsize='small', loc='best')
    st.pyplot(fig)



# --- SECCIÓN: ESQUEMA ---
st.markdown("---")
st.subheader("📍 Identificación de Zonas de Presión (NCh 432)")
if os.path.exists("F8.png"):
    st.image("F8.png", caption="Figura 8 - Distribución de Zonas 1 a 5")
else:
    st.info("Suba el esquema F8.png para visualizar las zonas.")

# CONTACTO
st.markdown("---")
st.markdown(f'<div style="text-align: right;"><a href="mailto:mriquelme@proyectosestructurales.com">mriquelme@proyectosestructurales.com</a></div>', unsafe_allow_html=True)
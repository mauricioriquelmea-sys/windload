# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import os
import math
import io
from fpdf import FPDF

# Tabla 4 NCh 432:2025 - Factor de elevación del suelo Ke en función de ze [m]
KE_TABLA_ZE = [0, 300, 600, 900, 1200, 1500, 1800]
KE_TABLA_KE = [1.00, 0.96, 0.93, 0.90, 0.87, 0.84, 0.81]

def ke_formula(ze):
    """Nota 2, Tabla 4: Ke = exp(-0,000119 ze)."""
    return math.exp(-0.000119 * ze)

def ke_tabla4(ze):
    """Interpolación lineal en la Tabla 4; fuera de 0-1800 m se aplica la Nota 2 (fórmula)."""
    if ze < 0 or ze > 1800:
        return ke_formula(ze)
    return float(np.interp(ze, KE_TABLA_ZE, KE_TABLA_KE))

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
# Imágenes incrustadas (Logo, Ray, Eolo, F2...F12). Vacío en la versión normal;
# embeber_imagenes_py.py lo rellena para generar la versión autocontenida.
IMAGENES_EMBEBIDAS = {}

def leer_imagen(nombre):
    """Bytes de la imagen: primero el archivo junto al script; si no existe, la copia incrustada."""
    if os.path.exists(nombre):
        with open(nombre, "rb") as f:
            return f.read()
    b64 = IMAGENES_EMBEBIDAS.get(nombre)
    return base64.b64decode(b64) if b64 else None

def get_base64_image(image_path):
    """Convierte una imagen a base64 para embeberla en el encabezado HTML"""
    data = leer_imagen(image_path)
    return base64.b64encode(data).decode() if data else None

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

# --- ELEMENTO A ANALIZAR ---
FACHADA, PARAPETO, BARANDA = "Fachadas y cubierta (C&R)", "Parapetos", "Barandas"
modo = st.sidebar.radio("🧱 Elemento a analizar", [FACHADA, PARAPETO, BARANDA],
                        help="Fachadas: presiones C&R por zona (Figuras 24, 26 y 40). "
                             "Parapetos y barandas: NCh 432:2025, 9.6, Ecuación (22) y Figura 45.")

# --- GUÍA DE VELOCIDAD ---
with st.sidebar.expander("🚩 Guía: Velocidad Básica (V) y Mapas"):
    st.markdown("""
    **Zonificación según NCh 432 (Tabla 1):**
    Los valores representan la ráfaga de 3 segundos a 10m de altura en campo abierto (Categoría C).
    """)
    tabla_v = {
        "Zona":         ["I-A", "I-B", "II-A", "II-B", "III-A", "III-B", "IV-A", "IV-B", "V", "VI", "NC1", "NC2", "NC3"],
        "Latitud Sur":  ["17°29'S-27°22'S", "17°29'S-27°22'S", "27°22'S-29°54'S", "27°22'S-29°54'S",
                         "29°54'S-37°28'S", "29°54'S-37°28'S", "37°28'S-41°28'S", "37°28'S-41°28'S",
                         "41°28'S-50°S", "50°S-56°32'S", "-", "-", "-"],
        "Altitud (msnm)": ["< 2000", "≥ 2000", "< 1500", "≥ 1500", "< 1000", "≥ 1000", "< 600", "≥ 600", "-", "-", "-", "-", "-"],
        "V (m/s)":      [27, 30, 27, 35, 34, 35, 37, 40, 40, 44, 32, 50, 60],
        "p0 (N/m²)":    [447, 552, 447, 751, 709, 751, 839, 981, 981, 1187, 628, 1533, 2207],
        "Designación":  ["Límite Norte hasta Copiapó", "Límite Norte hasta Copiapó", "Zona Centro", "Zona Centro",
                         "Zona Sur", "Zona Sur", "Zona Sur hasta Chiloé", "Zona Sur hasta Chiloé",
                         "Zona Austral", "Zona Austral", "Isla de Pascua", "Juan Fernández", "Antártica CL"],
    }
    st.table(pd.DataFrame(tabla_v).set_index("Zona"))
    st.caption("Nota: p0 es la presión referencial con Kz, Kzt y Ke iguales a 1.0. "
               "Santiago (< 1000 msnm) corresponde a la Zona III-A: V = 34 m/s.")
    if st.button("Desplegar Mapas de Chile"):
        for img in ["F2.png", "F3.png", "F4.png", "F5.png"]:
            _b = leer_imagen(img)
            if _b: st.image(_b, caption=f"Zonificación: {img}")

V = st.sidebar.number_input("Velocidad básica V (m/s)", 20.0, 60.0, 34.0)
H_edif = st.sidebar.number_input("Altura promedio edificio H (m)", 2.0, 200.0, 12.0)
theta = st.sidebar.slider("Inclinación de Techo θ (°)", 0, 45, 10)

# --- GEOMETRÍA ---
if modo == FACHADA:
    st.sidebar.subheader("📐 Geometría del Elemento")
    l_elem = st.sidebar.number_input("Largo del elemento (m)", 0.1, 50.0, 3.0)
    w_in = st.sidebar.number_input("Ancho tributario real (m)", 0.1, 50.0, 1.0)
else:
    l_elem, w_in = 3.0, 1.0   # no se usan en parapetos/barandas
w_trib = max(w_in, l_elem / 3)
area_ef = l_elem * w_trib

if modo == FACHADA:
    st.sidebar.info(f"**Área efectiva: {area_ef:.2f} m2**")
    if w_in < (l_elem / 3):
        st.sidebar.warning(f"⚠️ Ancho ajustado por norma a {w_trib:.2f}m (mín. 1/3 del largo)")

# --- PARAPETOS Y BARANDAS (NCh 432:2025, 9.6 y Figura 45) ---
ubic_baranda, en_techo = None, True
if modo != FACHADA:
    st.sidebar.subheader("🧱 Parapeto" if modo == PARAPETO else "🧱 Baranda")
    with st.sidebar.expander("ℹ️ Nota Explicativa: Parapetos y barandas (9.6)"):
        st.markdown(r"""
        **NCh 432:2025, 9.6 – Ecuación (22):** $p = q_p\,K_d\,[(GC_p) - (GC_{pi})]$, con $q_p$ evaluada
        en el punto más alto del elemento. Se consideran los dos casos de carga de la **Figura 45**:

        * **Caso A (parapeto de barlovento):** cara exterior con presión positiva de muro (zonas 4 o 5)
          y cara interior con succión de borde o esquina de techo (zonas 2 o 3).
        * **Caso B (parapeto de sotavento):** cara interior con presión positiva de muro y cara exterior
          con succión de muro (zonas 4 o 5).

        En un elemento de una sola hoja (muro macizo, vidrio, plancha) las presiones de ambas caras se
        suman y $GC_{pi}$ se anula. Si el parapeto es **hueco**, cada revestimiento se verifica con $\pm GC_{pi}$.

        **Barandas:** en borde de techo o azotea se tratan como parapeto (casos A y B). En balcones de
        fachada se adopta el **caso B** (presión positiva de muro en una cara y succión de muro en la otra),
        criterio de asimilación a parapeto aplicable a barandas de **paño lleno** (vidrio, plancha).
        """)
        _b = leer_imagen("F45.png")
        if _b: st.image(_b, caption="Figura 45 - NCh 432:2025")
    if modo == BARANDA:
        ubic_baranda = st.sidebar.radio("Ubicación de la baranda",
                                        ["Borde de techo / azotea", "Balcón o terraza en fachada"])
    en_techo = (modo == PARAPETO) or (ubic_baranda == "Borde de techo / azotea")
    h_elem = st.sidebar.number_input("Altura del parapeto hp (m)" if modo == PARAPETO else "Altura de la baranda hb (m)",
                                     0.1, 5.0, 1.0 if modo == PARAPETO else 1.1, step=0.05)
    if en_techo:
        z_base_elem = H_edif
    else:
        z_base_elem = st.sidebar.number_input("Cota del piso del balcón sobre el terreno (m)", 0.0, 300.0,
                                              max(0.0, H_edif - 3.0), step=0.5)
    z_top = z_base_elem + h_elem
    s_apoyo = st.sidebar.number_input("Separación entre montantes / apoyos s (m)", 0.1, 20.0, 1.2, step=0.1)
    B_min = st.sidebar.number_input("Menor dimensión horizontal del edificio B (m)", 1.0, 500.0, 20.0,
                                    help="Define el ancho a de las zonas de esquina (Figuras 24 y 40).")
    es_hueco = st.sidebar.checkbox("Parapeto hueco (revestimientos independientes)", value=False,
                                   help="Cada revestimiento se verifica con ±GCpi (Tabla 7) según la porosidad del parapeto.") if en_techo else False
    gcpi_par = st.sidebar.selectbox("GCpi del parapeto (Tabla 7)", [0.18, 0.55, 0.0], index=0) if es_hueco else 0.0
    g2_man = g3_man = None
    if en_techo and theta > 7:
        st.sidebar.warning("θ > 7°: ingrese los GCp de borde (zona 2) y esquina (zona 3) del techo desde las "
                           "Figuras 27 a 32 (h ≤ 18,3 m) o 26 a 32, 35 y 36 (h > 18,3 m), para el área efectiva.")
        g2_man = st.sidebar.number_input("GCp techo zona 2 (borde)", -6.0, 0.0, -2.3, step=0.1)
        g3_man = st.sidebar.number_input("GCp techo zona 3 (esquina)", -6.0, 0.0, -3.2, step=0.1)
    # Área efectiva del panel: lado mayor L y ancho no menor que L/3
    L_par = max(h_elem, s_apoyo)
    A_par = L_par * max(min(h_elem, s_apoyo), L_par / 3)
    st.sidebar.info(f"**Cota superior z = {z_top:.2f} m · Área efectiva A = {A_par:.2f} m²**")

# --- FACTOR TOPOGRÁFICO ---

with st.sidebar.expander("🏔️ Nota Explicativa: Factor Topográfico (Kzt)"):
    st.markdown("""
    **Criterios de Aplicación (Capítulo 5):**
                
    El factor Kzt considera la aceleración del viento sobre colinas, crestas y escarpes aislados. Se aplica cuando el relieve sobresale significativamente de su entorno.
    
    * **K1:** Factor de forma del relieve.
    * **K2:** Factor de reducción por distancia horizontal.
    * **K3:** Factor de reducción por altura sobre el suelo.
    
    * **Hc:** Altura del relieve respecto del terreno a barlovento (desnivel, **no** cota sobre el nivel del mar).
    * **Lh (Distancia horizontal):** Es la distancia horizontal en barlovento desde la cresta hasta donde la diferencia de elevación es la mitad de la altura del relieve ($H_c/2$).
    * **x:** Distancia horizontal de la cresta al edificio (barlovento o sotavento). Con $x = 0$ se obtiene el máximo.
    * **K1** depende de la forma del relieve **y de la categoría de exposición**; si $H_c/L_h > 0{,}5$ se usa $H_c/L_h = 0{,}5$ para K1 y $L_h = 2H_c$ para K2 y K3.
    * **K3** se evalúa a cada altura $z$: $K_{zt}$ varía con la altura del edificio.
    * Constantes según ASCE/SEI 7-22, Fig. 26.8-1, base de NCh 432:2025. **Verificar contra la figura de factores topográficos de NCh 432:2025.**
    """)

    if st.button("Ver Diagramas de Relieve"):
        for img in ["F7.png", "F6.png"]:
            _b = leer_imagen(img)
            if _b: st.image(_b)

metodo = st.sidebar.radio("Cálculo de Kzt", ["Manual", "Calculado"])

if metodo == "Manual":
    Kzt_val = st.sidebar.number_input("Valor Kzt directo", 1.0, 3.0, 1.0)
else:
    tipo_relieve = st.sidebar.selectbox("Forma del relieve", ["Escarpe 2D", "Cresta 2D", "Colina 3D"])
    
    # Parámetros del relieve
    Hc = st.sidebar.number_input("Altura del relieve Hc (m)", value=27.0, help="Desnivel del relieve respecto del terreno a barlovento (no la cota s.n.m.).")
    Lhc = st.sidebar.number_input("Lh (m)", value=100.0, help="Distancia horizontal a barlovento entre la cresta y el punto donde el desnivel es Hc/2.")
    x_cresta = st.sidebar.number_input("Distancia cresta-edificio x (m)", 0.0, 10000.0, 0.0, help="x = 0 en la cresta (valor máximo).")
    lado_x = st.sidebar.radio("Ubicación respecto de la cresta", ["Barlovento", "Sotavento"], index=1)
    aplicar_cond = st.sidebar.checkbox("Aplicar condiciones de aplicabilidad (Hc/Lh ≥ 0,2; Hc ≥ 4,5 m en C/D, ≥ 18 m en B)", value=True)

# --- FACTORES NORMATIVOS ---
st.sidebar.subheader("📋 Factores Normativos")

with st.sidebar.expander("ℹ️ Nota Explicativa: Factor de Direccionalidad (Kd)"):
    st.markdown("""
    **Criterios de la Tabla 3 (NCh 432:2025):**
    Este factor compensa la reducida probabilidad de que el viento máximo sople precisamente desde la dirección más crítica para la orientación de la estructura y, simultáneamente, alcance la magnitud de diseño.
    
    **Valores Normativos por Tipo de Estructura:**
    * **Edificios:**
        * Sistema Principal Resistente a la Fuerza del Viento (SPRFV): **0.85**
        * Componentes y Revestimientos (C&R): **0.85**
    * **Cubiertas Arqueadas:** **0.85**
    * **Chimeneas, Tanques y Estructuras Similares:**
        * Forma Cuadrada: **0.90**
        * Forma Hexagonal: **0.95**
        * Forma Redonda: **0.95**
    * **Señales Sólidas:** **0.85**
    * **Señales Abiertas y Estructuras de Enrejado:** **0.85**
    * **Torres de Celosía:**
        * Secciones Triangulares, Cuadradas o Rectangulares: **0.85**
        * Otras Secciones: **0.95**
    * **Cubiertas Aisladas (Techos Abiertos):** **0.85**
    
    *Nota: Este factor solo debe aplicarse cuando se utiliza en las combinaciones de carga de diseño especificadas por la norma.*
    """)

# Selector de Kd con rango de precisión
Kd_val = st.sidebar.number_input("Factor de Direccionalidad Kd", 0.50, 1.00, 0.85, step=0.01)

# --- FACTOR DE ELEVACIÓN DEL SUELO Ke (NCh 432:2025, 5.7 - Tabla 4) ---
st.sidebar.subheader("⛰️ Factor de Elevación del Suelo (Ke)")
with st.sidebar.expander("ℹ️ Nota Explicativa: Factor Ke (5.7)"):
    st.markdown(r"""
    **NCh 432:2025, 5.7:** Ke ajusta la densidad del aire según la elevación del suelo sobre el
    nivel del mar, $z_e$. Se determina con la Tabla 4; se permite tomar **Ke = 1,00** para todas las
    elevaciones (Nota 1, conservador).

    **Nota 2:** fuera de la tabla (z_e < 0 o z_e > 1 800 m), o como alternativa para todas las
    elevaciones, se interpola o se usa $K_e = e^{-0{,}000119\,z_e}$.
    """)
    st.table(pd.DataFrame({"ze (m)": KE_TABLA_ZE, "Ke": [f"{k:.2f}" for k in KE_TABLA_KE]}).set_index("ze (m)"))

ke_metodo = st.sidebar.radio("Determinación de Ke",
                             ["Conservador (Ke = 1,00)", "Tabla 4 (interpolación)", "Fórmula (Nota 2)"])
if ke_metodo.startswith("Conservador"):
    z_e = None
    Ke_val = 1.0
else:
    z_e = st.sidebar.number_input("Elevación del suelo sobre el nivel del mar ze (m)", -100.0, 5000.0, 550.0, step=10.0)
    Ke_val = ke_tabla4(z_e) if ke_metodo.startswith("Tabla") else ke_formula(z_e)
st.sidebar.info(f"**Ke = {Ke_val:.3f}**" + ("" if z_e is None else f"  (ze = {z_e:.0f} m)"))
if ke_metodo.startswith("Tabla") and z_e is not None and (z_e < 0 or z_e > 1800):
    st.sidebar.warning("ze fuera del rango de la Tabla 4: se aplica la fórmula de la Nota 2.")

with st.sidebar.expander("ℹ️ Nota Explicativa: Exposición"):
    st.markdown("""
    **Rugosidad del Terreno (NCh 432:2025, 5.5.2):**
    * **B:** Áreas urbanas y suburbanas, áreas boscosas u otros terrenos con numerosas obstrucciones próximas.
    * **C:** Terrenos abiertos con obstrucciones dispersas < 9m. Incluye campos abiertos y terrenos agrícolas.
    * **D:** Áreas planas y sin obstrucciones frente a cuerpos de agua (Costa).
    """)

# --- AYUDA TÉCNICA RIGUROSA: CATEGORÍA DE EXPOSICIÓN ---
with st.sidebar.expander("ℹ️ Nota Explicativa: Exposición (B, C, D)"):
    st.markdown("""
    **Definiciones según NCh 432:2025 (5.5.3):**
    
    * **Exposición B:** Áreas urbanas y suburbanas, áreas boscosas u otros terrenos con numerosas obstrucciones próximas del tamaño de viviendas unifamiliares o mayores.
    * **Exposición C:** Terrenos abiertos con obstrucciones dispersas que tienen alturas generalmente menores a 9m. (Categoría por defecto).
    * **Exposición D:** Áreas planas y sin obstrucciones frente a cuerpos de agua que se extienden al menos 1.6 km.
    """)

cat_exp = st.sidebar.selectbox("Categoría de Exposición", ['B', 'C', 'D'], index=0)

# Constantes de exposición al terreno según Tabla 6 NCh 432:2025
# alpha: exponente de la ley de potencia | zg: altura gradiente (m) | zmin: altura mínima (m)
exp_info = {
    'B': [7.5, 1000.0, "Urbano/Suburbano", 10.0],
    'C': [9.8, 750.0, "Terreno Abierto", 5.0],
    'D': [11.5, 590.0, "Costa/Agua", 2.0]
}

alpha_val = exp_info[cat_exp][0]
zg_val = exp_info[cat_exp][1]
desc_exp = exp_info[cat_exp][2]
zmin_val = exp_info[cat_exp][3]

# Despliegue de los factores asociados debajo del selector
st.sidebar.info(f"""
**Parámetros de Rugosidad (Tabla 6):**
* Tipo: {desc_exp}
* Exponente (α): {alpha_val}
* Altura Gradiente (zg): {zg_val:.0f} m
* Altura mínima (zmín): {zmin_val:.0f} m
""")

# --- FACTOR TOPOGRÁFICO Kzt(z) (se calcula aquí porque K1 depende de la exposición) ---
# Constantes ASCE/SEI 7-22 Fig. 26.8-1 (base de NCh 432:2025):
# K1/(H/Lh) por exposición B, C, D | gamma (K3) | mu barlovento | mu sotavento (K2)
RELIEVE = {
    "Cresta 2D":  {"k1": {'B': 1.30, 'C': 1.45, 'D': 1.55}, "gam": 3.0, "mu_bar": 1.5, "mu_sot": 1.5},
    "Escarpe 2D": {"k1": {'B': 0.75, 'C': 0.85, 'D': 0.95}, "gam": 2.5, "mu_bar": 1.5, "mu_sot": 4.0},
    "Colina 3D":  {"k1": {'B': 0.95, 'C': 1.05, 'D': 1.15}, "gam": 4.0, "mu_bar": 1.5, "mu_sot": 1.5},
}
kzt_nota = ""
if metodo == "Manual":
    def kzt_de(z):
        return Kzt_val
    k1 = k2 = None
else:
    rel = RELIEVE[tipo_relieve]
    h_lh = Hc / Lhc
    h_lim = 18.0 if cat_exp == 'B' else 4.5
    if aplicar_cond and (h_lh < 0.2 or Hc < h_lim):
        kzt_nota = (f"No se cumplen las condiciones de aplicabilidad (Hc/Lh = {h_lh:.3f}; Hc = {Hc:.1f} m): Kzt = 1,0.")
        k1, k2 = 0.0, 0.0
        def kzt_de(z):
            return 1.0
    else:
        lh_ef = Lhc
        if h_lh > 0.5:
            h_lh, lh_ef = 0.5, 2.0 * Hc
            kzt_nota = "Hc/Lh > 0,5: se usa Hc/Lh = 0,5 para K1 y Lh = 2·Hc para K2 y K3."
        k1 = rel["k1"][cat_exp] * h_lh
        mu = rel["mu_bar"] if lado_x == "Barlovento" else rel["mu_sot"]
        k2 = max(0.0, 1.0 - abs(x_cresta) / (mu * lh_ef))
        gam = rel["gam"]
        def kzt_de(z):
            return (1.0 + k1 * k2 * math.exp(-gam * z / lh_ef))**2
    Kzt_val = kzt_de(H_edif)
    st.sidebar.info(f"""
    **Factor topográfico ({tipo_relieve}, Exp. {cat_exp}):**
    * K1: {k1:.3f} · K2: {k2:.3f}
    * K3 (z = h): {(math.exp(-RELIEVE[tipo_relieve]["gam"] * H_edif / (2*Hc if Hc/Lhc > 0.5 else Lhc)) if k1 else 0):.3f}
    * **Kzt (z = h): {Kzt_val:.3f}**
    """)
    if kzt_nota:
        st.sidebar.warning(kzt_nota)

# =================================================================
# 3. SIDEBAR: CATEGORÍA DE RIESGO Y PERIODOS DE RETORNO (CORREGIDO NCh 432:2025)
# =================================================================

with st.sidebar.expander("ℹ️ Nota Explicativa: Categoría de Riesgo"):
    st.markdown("""
    **Clasificación según NCh 3171 y Tabla 2 de NCh 432:2025:**
    La velocidad básica V es única (Tabla 1 / Tabla C.1) y la categoría de ocupación se incorpora mediante el factor de importancia I, asociado a un período de retorno medio.
    
    * **Categoría I:** Riesgo bajo para la vida humana en caso de falla. *(T = 25 años; I = 0,87)*.
    * **Categoría II:** Estructuras de uso corriente que no clasifican en I, III o IV. *(T = 50 años; I = 1,00)*.
    * **Categoría III:** Gran número de personas o capacidad limitada de evacuación. *(T = 100 años; I = 1,15)*.
    * **Categoría IV:** Instalaciones esenciales (hospitales, estaciones de emergencia). *(T = 150 años; I = 1,22)*.
    """)

# Selector de Categoría de Riesgo
cat_imp = st.sidebar.selectbox("Categoría de Riesgo / Riesgo", ['I', 'II', 'III', 'IV'], index=1)

# Tabla 2 NCh 432:2025: factor de importancia por categoría de ocupación
imp_map = {'I': 0.87, 'II': 1.0, 'III': 1.15, 'IV': 1.22}
factor_i = imp_map[cat_imp]
st.sidebar.info(f"**Factor de importancia (I): {factor_i }**")

# Mostramos el Periodo de Retorno asociado como información técnica adicional
t_retorno = {'I': 25, 'II': 50, 'III': 100, 'IV': 150}
st.sidebar.info(f"**Periodo de Retorno (T): {t_retorno[cat_imp]} años**")

# =================================================================
# 4. MOTOR DE CÁLCULO Y DEFINICIÓN DE CERRAMIENTO (RIGUROSO)
# =================================================================
st.sidebar.subheader("🏠 Clasificación del Cerramiento")

with st.sidebar.expander("ℹ️ Nota Explicativa: Clasificación de Cerramiento"):
    st.markdown("""
    **Definiciones según NCh 432 (Capítulo 2):**
    
    * **Edificio Abierto:** Un edificio que tiene cada pared abierta en al menos un 80%. Esto implica que el viento fluye a través de la estructura sin generar presiones internas significativas.
    
    * **Edificio Parcialmente Abierto:** Un edificio que cumple con ambas condiciones:
        1. El área total de aberturas en una pared que recibe presión externa positiva excede la suma de las áreas de las aberturas en el resto de la envolvente en más de un 10%.
        2. El área total de aberturas en una pared que recibe presión externa positiva excede 0.37 m² o el 1% del área de dicha pared, y el porcentaje de aberturas en el resto de la envolvente no excede el 20%.
        
    * **Edificio Cerrado:** Un edificio que no cumple con los requisitos de edificio abierto o parcialmente abierto. Es el estándar para estructuras estancas donde las aberturas son mínimas.
    """)

cerramiento_opcion = st.sidebar.selectbox(
    "Tipo de Cerramiento", 
    ["Cerrado", "Parcialmente Abierto", "Abierto"],
    index=0
)

# Diccionario técnico para la Ficha Central
gcpi_data = {
    "Cerrado": [0.18, "Un edificio que no cumple con los requisitos de abierto o parcialmente abierto. Es el estándar para la mayoría de estructuras estancas."],
    "Parcialmente Abierto": [0.55, "Edificio donde el área de aberturas en una pared excede la suma de aberturas en el resto de la envolvente en más del 10%."],
    "Abierto": [0.00, "Un edificio que tiene al menos un 80% de aberturas en cada pared. El viento fluye sin generar presiones internas."]
}

gc_pi_val = gcpi_data[cerramiento_opcion][0]
nota_tecnica_cerramiento = gcpi_data[cerramiento_opcion][1]

st.sidebar.info(f"**Factor GCpi asociado: ± {gc_pi_val}**")

red10_muros = st.sidebar.checkbox("Reducir 10 % los GCp de muros (Figura 24, Nota 5)", value=False,
                                  help="Aplica solo a edificios con h ≤ 18,3 m y θ ≤ 10°. Desactivado = conservador.")

# --- MOTOR MATEMÁTICO ---
def get_gcp(a, g_lo, g_hi, a_lo=0.9, a_hi=46.5):
    """Interpolación lineal en log10(A) entre a_lo y a_hi; constante fuera del rango."""
    if a <= a_lo: return g_lo
    if a >= a_hi: return g_hi
    return g_lo + (g_hi - g_lo) * np.log10(a / a_lo) / np.log10(a_hi / a_lo)

# Tabla 6 NCh 432:2025: [alpha, zg (m), zmin (m)]
exp_params = {'B': [7.5, 1000.0, 10.0], 'C': [9.8, 750.0, 5.0], 'D': [11.5, 590.0, 2.0]}
alpha, zg, zmin = exp_params[cat_exp]

def kz_de(z):
    """Tabla 5 NCh 432:2025, Notas 1, 3 y 4: Kz = 2,41 (z/zg)^(2/α); z < zmin -> z = zmin; z > zg -> 2,41."""
    if z > zg:
        return 2.41
    return 2.41 * ((max(z, zmin) / zg)**(2/alpha))

kz = kz_de(H_edif)

def q_de(z):
    """Ec. (2) NCh 432:2025: qz = 0,613 I Kz Kzt Ke V² [N/m²], expresada en kgf/m². Kd se aplica en p."""
    return (0.613 * imp_map[cat_imp] * kz_de(z) * kzt_de(z) * Ke_val * (V**2)) * 0.10197

qh = q_de(H_edif)
edificio_alto = H_edif > 18.3

P_MIN_N = 250.0                 # 9.2.2: presión neta mínima de diseño para C&R, 0,25 kN/m²
P_MIN = P_MIN_N * 0.10197       # en kgf/m²

def p_diseno(q, gcp, gcpi, qi=None):
    """p = q Kd (GCp) - qi Kd (GCpi), con GCpi en el signo más desfavorable y qi = qh (edificio cerrado).
    Se aplica la presión neta mínima de 0,25 kN/m² para C&R (9.2.2)."""
    qi = qh if qi is None else qi
    p = q * Kd_val * gcp + (qi * Kd_val * gcpi if gcp >= 0 else -qi * Kd_val * gcpi)
    return float(np.sign(p) * max(abs(p), P_MIN)) if p != 0 else P_MIN

# --- COEFICIENTES GCp DE MUROS Y TECHO (comunes a fachadas, parapetos y barandas) ---
# Muros: (GCp para A <= 0,9 m², GCp para A >= 46,5 m²), interpolación en log10(A)
if edificio_alto:
    GC_MURO = {"z4p": (0.9, 0.6), "z5p": (0.9, 0.6), "z4n": (-0.9, -0.7), "z5n": (-1.8, -1.0)}   # Figura 40
    fig_muro = "Figura 40"
else:
    GC_MURO = {"z4p": (1.0, 0.7), "z5p": (1.0, 0.7), "z4n": (-1.1, -0.8), "z5n": (-1.4, -0.8)}   # Figura 24
    fig_muro = "Figura 24"
reduccion_muros = (not edificio_alto) and theta <= 10 and red10_muros
if reduccion_muros:   # Figura 24, Nota 5: reducción de 10 % cuando θ <= 10°
    GC_MURO = {k: (0.9 * a, 0.9 * b) for k, (a, b) in GC_MURO.items()}

# Techo: zonas 1, 2 y 3 (succión), entre 0,9 y 46,5 m²
if edificio_alto:
    GC_TECHO = {"z1": (-1.4, -0.9), "z2": (-2.3, -1.6), "z3": (-3.2, -2.3)}   # Figura 40 (θ <= 7°, Nota 6)
    fig_techo = "Figura 40"
    techo_valido = theta <= 7
    aviso_techo = None if techo_valido else (
        "θ > 7°: los coeficientes de techo de la Figura 40 aplican a techos con θ ≤ 7° (Nota 6). Para otros "
        "ángulos usar las Figuras 26 a 32, 35 y 36 con qh. Los valores de techo mostrados no son válidos.")
else:
    techo_valido = theta <= 7
    if techo_valido:
        GC_TECHO = {"z1": (-1.7, -1.0), "z2": (-2.3, -1.4), "z3": (-3.2, -1.4)}   # Figura 26 (θ <= 7°)
    else:   # coeficientes heredados, no verificados contra las Figuras 27 a 32
        GC_TECHO = {"z1": (-0.9, -0.8, 1.0, 10.0), "z2": (-1.3, -1.2, 1.0, 10.0), "z3": (-2.0, -1.2, 1.0, 10.0)}
    fig_techo = "Figura 26"
    aviso_techo = None if techo_valido else (
        "θ > 7°: los coeficientes de techo corresponden a la Figura 26 (θ ≤ 7°). Para techos con mayor pendiente "
        "usar las Figuras 27 a 32. Los valores de techo mostrados son solo referenciales.")

def gcp_techo(a, zona):
    return get_gcp(a, *GC_TECHO[zona])

def ancho_zona_esquina(b_min, h):
    """Ancho a de las zonas de esquina de muro (Figura 24: h <= 18,3 m; Figura 40: h > 18,3 m)."""
    if h > 18.3:
        return max(0.10 * b_min, 0.9)
    lim_h = 0.8 * h if (theta <= 7 and b_min > 90) else 0.4 * h
    return max(min(0.10 * b_min, lim_h), 0.04 * b_min, 0.9)


# =================================================================
# 5. DESPLIEGUE TÉCNICO DE RESULTADOS Y FORMULACIÓN (CORREGIDO)
# =================================================================

if modo == FACHADA:
    # Ficha de Cerramiento Destacada
    st.markdown(f"""
<div class="classification-box">
    <strong>📋 Ficha Técnica de Cerramiento (NCh 432):</strong><br><br>
    <strong>Clasificación Seleccionada:</strong> {cerramiento_opcion}<br>
    <span style="font-size: 1.5em; color: #d9534f;"><strong>Factor de Presión Interna (GCpi): ± {gc_pi_val}</strong></span><br><br>
    <strong>Nota Normativa:</strong> {nota_tecnica_cerramiento}
</div>
""", unsafe_allow_html=True)

    # Caja de Fórmulas y Ecuaciones
    st.markdown("### 📝 Ecuaciones de Diseño Aplicadas")
    st.latex(r"K_z = 2.41 \left(\frac{z}{z_g}\right)^{2/\alpha} \quad (z_{min} \le z \le z_g) \qquad \text{(Tabla 5, Nota 1; Tabla 6)}")
    st.latex(r"q_z = 0.613 \cdot I \cdot K_z \cdot K_{zt} \cdot K_e \cdot V^2 \qquad \text{(Ec. 2)}")
    st.latex(r"p = q \cdot K_d \cdot (GC_p) - q_i \cdot K_d \cdot (GC_{pi}) \qquad |p| \geq 250\ \text{N/m}^2\ \text{(9.2.2)}")
    st.caption(f"Kd = {Kd_val:.2f} se aplica a la presión neta (externa e interna), no a la presión de velocidad. Ke = {Ke_val:.3f} ({ke_metodo}).")

    st.info(f"**Presión de Velocidad Calculada (qh):** {qh:.2f} kgf/m² · Kz = {kz:.3f} · Kzt = {Kzt_val:.3f} · "
            f"{'Edificio alto (h > 18,3 m): coeficientes de la Figura 40' if edificio_alto else 'Edificio de baja altura (h ≤ 18,3 m): coeficientes de muros de la Figura 24'}")

    # --- CÁLCULO DE COEFICIENTES GCp (DEFINICIÓN DE VARIABLES PARA PUNTO 6) ---
    z1, z2, z3 = gcp_techo(area_ef, "z1"), gcp_techo(area_ef, "z2"), gcp_techo(area_ef, "z3")
    if reduccion_muros:
        st.caption("GCp de muros reducidos en 10 % (Figura 24, Nota 5: h ≤ 18,3 m y θ ≤ 10°).")

    z4_pos, z5_pos = get_gcp(area_ef, *GC_MURO["z4p"]), get_gcp(area_ef, *GC_MURO["z5p"])
    z4_neg, z5_neg = get_gcp(area_ef, *GC_MURO["z4n"]), get_gcp(area_ef, *GC_MURO["z5n"])
    if aviso_techo:
        st.warning(aviso_techo)

    # Tabulación de Resultados
    col_res, col_plt = st.columns([1, 1.3])

    with col_res:
        st.markdown("**Resumen de Presiones Netas por Zona**")
    
        # Muros laterales y de sotavento quedan cubiertos por la succión de Zonas 4 y 5 (C&R)
        zonas = [
            "Z1 (Techo Centro - Succión)", "Z2 (Techo Borde - Succión)", "Z3 (Techo Esq - Succión)",
            "Z4 (Muro - Sobrepresión)", "Z4 (Muro - Succión)",
            "Z5 (Muro Esq - Sobrepresión)", "Z5 (Muro Esq - Succión)",
        ]
        gcp_vals = [z1, z2, z3, z4_pos, z4_neg, z5_pos, z5_neg]
    
        p_netas = []
        for g in gcp_vals:
            p_netas.append(round(p_diseno(qh, g, gc_pi_val), 2))

        df_res = pd.DataFrame({
            "Zona de Análisis": zonas,
            "GCp (Externo)": [round(z, 3) for z in gcp_vals],
            "GCpi (Interno)": [gc_pi_val] * len(zonas),
            "Presión Neta (kgf/m²)": p_netas
        })
        st.table(df_res)

    with col_plt:
        areas = np.logspace(-1, 2, 200)
        fig, ax = plt.subplots(figsize=(10, 7))
        # Curvas de Succión
        ax.plot(areas, [get_gcp(a, *GC_MURO["z4n"]) for a in areas], label='Z4 Succión', color='green', lw=2)
        ax.plot(areas, [get_gcp(a, *GC_MURO["z5n"]) for a in areas], label='Z5 Succión', color='red', lw=2)
        # Curvas de Sobrepresión
        ax.plot(areas, [get_gcp(a, *GC_MURO["z4p"]) for a in areas], label='Z4 Sobrepresión', color='green', lw=2, ls=':')
        ax.plot(areas, [get_gcp(a, *GC_MURO["z5p"]) for a in areas], label='Z5 Sobrepresión', color='red', lw=1.5, ls='--')
        for a_l in (0.9, 46.5):
            ax.axvline(a_l, color='gray', lw=0.8, ls='--')

        for z_v in [z4_pos, z4_neg, z5_pos, z5_neg]:
            ax.scatter([area_ef], [z_v], color='black', s=40, zorder=10)

        ax.set_xscale('log')
        ax.set_title(f"Coeficientes GCp de muros - {fig_muro} NCh 432:2025", fontsize=11)
        ax.set_xlabel("Área efectiva de viento A (m²)"); ax.set_ylabel("Coeficiente GCp")
        ax.axhline(0, color='black', lw=1); ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize='x-small', loc='center left', bbox_to_anchor=(1, 0.5))
        st.pyplot(fig)

    # =================================================================
    # 6. DISTRIBUCIÓN DE PRESIONES NETAS: SINCRONIZADO Y COMPLETO
    # =================================================================
    st.divider()
    st.subheader("📊 Perfil de Presiones Netas (NCh 432): Barlovento vs Sotavento")

    # Coeficiente típico para paredes laterales (Succión constante)
    alturas_perfil = np.linspace(0.1, H_edif, 50)
    p_barlo_4, p_barlo_5 = [], [] 
    p_sota_4, p_sota_5 = [], [] 

    for z_alt in alturas_perfil:
        # h > 18,3 m: sobrepresión con qz (Figura 40, Nota 4); h <= 18,3 m: qh uniforme (Figura 24)
        q_bar = q_de(z_alt) if edificio_alto else qh
        p_barlo_4.append(p_diseno(q_bar, z4_pos, gc_pi_val))
        p_barlo_5.append(p_diseno(q_bar, z5_pos, gc_pi_val))
    
        # SUCCIÓN (-) - Constante, evaluada con qh
        p_sota_4.append(p_diseno(qh, z4_neg, gc_pi_val))
        p_sota_5.append(p_diseno(qh, z5_neg, gc_pi_val))

    # Renderizado Gráfico Profesional
    fig_alt, ax_alt = plt.subplots(figsize=(12, 8))

    # Graficar Barlovento (Derecha)
    ax_alt.plot(p_barlo_4, alturas_perfil, label="Z4 Sobrepresión", color='darkgreen', lw=2)
    ax_alt.plot(p_barlo_5, alturas_perfil, label="Z5 Sobrepresión", color='limegreen', lw=2, ls='--')

    # Graficar Succión (Izquierda)
    ax_alt.plot(p_sota_4, alturas_perfil, label="Z4 Succión (qh)", color='blue', lw=2)
    ax_alt.plot(p_sota_5, alturas_perfil, label="Z5 Succión (qh)", color='darkblue', lw=3)

    # Configuración del Eje y Estética
    ax_alt.axvline(0, color='red', lw=2) 
    ax_alt.set_title(f"Perfil de Presiones Netas de Diseño | V = {V} m/s", fontsize=14)
    ax_alt.set_xlabel("Presión Neta [kgf/m²] <-- SUCCIÓN | SOBREPRESIÓN -->", fontsize=12)
    ax_alt.set_ylabel("Altura sobre N.N.T. [m]", fontsize=12)

    # Ajuste dinámico de límites
    max_p = max(max(p_barlo_4), max(p_barlo_5), abs(min(p_sota_5))) * 1.2
    ax_alt.set_xlim(-max_p, max_p)

    ax_alt.grid(True, which='both', ls='--', alpha=0.5)
    ax_alt.legend(loc='upper right', fontsize='small', frameon=True)

    st.pyplot(fig_alt)

    st.write("📌 **Interpretación**: Las curvas a la izquierda representan succiones, evaluadas con qh y uniformes en altura; "
             "cubren muros de sotavento y laterales. La sobrepresión se evalúa con qz en edificios con h > 18,3 m "
             "(Figura 40, Nota 4) y con qh en edificios de baja altura. Se aplica la presión neta mínima de 250 N/m² (9.2.2).")

    # =================================================================
    # 7. ESQUEMAS NORMATIVOS Y REFERENCIAS FINALES
    # =================================================================
    st.divider()
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.subheader("📍 Identificación de Zonas")
        if leer_imagen("F8.png"): st.image(leer_imagen("F8.png"))
    with col_img2:
        st.subheader("📍 Esquema Isométrico")
        if leer_imagen("F12.png"): st.image(leer_imagen("F12.png"))

# =================================================================
# 6b. PARAPETOS Y BARANDAS (NCh 432:2025, 9.6, Ecuación 22 y Figura 45)
# =================================================================
KGF_A_KN = 0.00980665

def calcular_parapeto():
    """Presiones de diseño en parapetos y barandas según 9.6 (Ec. 22) y Figura 45.
    Cara 1 = cara que recibe la presión positiva; cara 2 = cara opuesta (succión).
    Presión neta del elemento de una hoja: p = qp Kd [(GCp)cara1 - (GCp)cara2] (GCpi se anula)."""
    qp = q_de(z_top)
    g = {
        "4p": get_gcp(A_par, *GC_MURO["z4p"]), "5p": get_gcp(A_par, *GC_MURO["z5p"]),
        "4n": get_gcp(A_par, *GC_MURO["z4n"]), "5n": get_gcp(A_par, *GC_MURO["z5n"]),
    }
    if en_techo:
        if g2_man is None:
            g["2r"], g["3r"] = gcp_techo(A_par, "z2"), gcp_techo(A_par, "z3")
        else:
            g["2r"], g["3r"] = g2_man, g3_man

    def caso(cod, zona, g1, g2, txt1, txt2, sentido):
        p1, p2 = qp * Kd_val * g1, qp * Kd_val * g2
        pn = qp * Kd_val * (g1 - g2)
        pn_d = max(pn, P_MIN)
        d = dict(caso=cod, zona=zona, g1=g1, g2=g2, txt1=txt1, txt2=txt2, p1=p1, p2=p2,
                 pn=pn, pn_d=pn_d, sentido=sentido,
                 w=pn_d * h_elem, F=pn_d * h_elem * s_apoyo, M=pn_d * h_elem * s_apoyo * h_elem / 2)
        # Parapeto hueco: cada revestimiento con ±GCpi en el signo más desfavorable
        d["p1i"] = max(qp * Kd_val * (g1 + gcpi_par), P_MIN)
        d["p2i"] = min(qp * Kd_val * (g2 - gcpi_par), -P_MIN)
        return d

    casos = []
    if en_techo:
        casos.append(caso("A", "Típica (muro 4 + techo 2)", g["4p"], g["2r"], "Exterior: muro zona 4 (+)",
                          "Interior: techo zona 2 (-)", "hacia el techo"))
        casos.append(caso("A", "Esquina (muro 5 + techo 3)", g["5p"], g["3r"], "Exterior: muro zona 5 (+)",
                          "Interior: techo zona 3 (-)", "hacia el techo"))
        casos.append(caso("B", "Típica (muro 4 + muro 4)", g["4p"], g["4n"], "Interior: muro zona 4 (+)",
                          "Exterior: muro zona 4 (-)", "hacia el exterior"))
        casos.append(caso("B", "Esquina (muro 5 + muro 5)", g["5p"], g["5n"], "Interior: muro zona 5 (+)",
                          "Exterior: muro zona 5 (-)", "hacia el exterior"))
    else:
        casos.append(caso("B", "Típica (muro 4 + muro 4)", g["4p"], g["4n"], "Cara cargada: muro zona 4 (+)",
                          "Cara opuesta: muro zona 4 (-)", "en ambos sentidos"))
        casos.append(caso("B", "Esquina (muro 5 + muro 5)", g["5p"], g["5n"], "Cara cargada: muro zona 5 (+)",
                          "Cara opuesta: muro zona 5 (-)", "en ambos sentidos"))
    gob = max(casos, key=lambda c: c["pn_d"])
    return dict(qp=qp, kz_top=kz_de(z_top), kzt_top=kzt_de(z_top), g=g, casos=casos, gob=gob,
                a_esq=ancho_zona_esquina(B_min, H_edif))


def _flechas(ax, x0, x1, ys, color):
    for y in ys:
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, mutation_scale=11))


def dibujar_esquema_parapeto(res):
    """Esquema tipo Figura 45 con las presiones de la zona típica y de esquina."""
    c = {cs["caso"] + cs["zona"][:3]: cs for cs in res["casos"]}
    n = 2 if en_techo else 1
    fig, axs = plt.subplots(1, n, figsize=(6.0 * n, 4.8))
    axs = np.atleast_1d(axs)
    ys = np.linspace(0.12, 0.88, 6)
    t = 0.07
    def fmt(cs_t, cs_e, clave):
        return f"{cs_t[clave]:+.1f} (típ.) / {cs_e[clave]:+.1f} (esq.)"
    if en_techo:
        for ax, cod in zip(axs, ["A", "B"]):
            cs_t, cs_e = c[cod + "Típ"], c[cod + "Esq"]
            ax.add_patch(plt.Rectangle((0, -0.9), t, 1.9, color="#94a3b8", ec="#1e293b"))
            ax.plot([t, 1.7], [0, 0], color="#1e293b", lw=3)
            ax.text(1.65, -0.08, "Techo", ha="right", va="top", fontsize=8, color="#334155")
            ax.text(-0.05, -0.85, "Muro\nexterior", ha="right", va="bottom", fontsize=8.5, color="#334155")
            for xr in np.linspace(0.8, 1.55, 4):
                ax.annotate("", xy=(xr, 0.32), xytext=(xr, 0.04),
                            arrowprops=dict(arrowstyle="-|>", color="#cbd5e1", lw=1.0))
            if cod == "A":
                _flechas(ax, -0.55, -0.02, ys, "#b91c1c")
                _flechas(ax, t + 0.02, t + 0.5, ys, "#1d4ed8")
                ax.text(-0.04, 1.08, "p1 (cara exterior)\n" + fmt(cs_t, cs_e, "p1"), fontsize=8.5, color="#b91c1c", ha="right")
                ax.text(t + 0.05, 1.08, "p2 (cara interior)\n" + fmt(cs_t, cs_e, "p2"), fontsize=8.5, color="#1d4ed8")
                ax.annotate("", xy=(0.85, -0.5), xytext=(0.2, -0.5),
                            arrowprops=dict(arrowstyle="-|>", color="black", lw=2.2))
                titulo = "Caso A: parapeto de barlovento"
            else:
                _flechas(ax, t + 0.55, t + 0.02, ys, "#b91c1c")
                _flechas(ax, -0.02, -0.5, ys, "#1d4ed8")
                ax.text(t + 0.05, 1.08, "p3 (cara interior)\n" + fmt(cs_t, cs_e, "p1"), fontsize=8.5, color="#b91c1c")
                ax.text(-0.04, 1.08, "p4 (cara exterior)\n" + fmt(cs_t, cs_e, "p2"), fontsize=8.5, color="#1d4ed8", ha="right")
                ax.annotate("", xy=(0.2, -0.5), xytext=(0.85, -0.5),
                            arrowprops=dict(arrowstyle="-|>", color="black", lw=2.2))
                titulo = "Caso B: parapeto de sotavento"
            ax.text(0.2, -0.72, f"p neta = {cs_t['pn_d']:.1f} (típ.) / {cs_e['pn_d']:.1f} (esq.) kgf/m²\n"
                    f"sentido: {cs_t['sentido']}", fontsize=9, fontweight="bold", va="top", ha="left")
            ax.set_title(titulo, fontsize=11)
            ax.set_xlim(-1.15, 1.8); ax.set_ylim(-1.0, 1.35)
    else:
        ax = axs[0]
        cs_t, cs_e = c["BTíp"], c["BEsq"]
        ax.add_patch(plt.Rectangle((1.2, -1.0), 0.12, 2.35, color="#cbd5e1", ec="#1e293b"))
        ax.text(1.4, 0.2, "Fachada", rotation=90, fontsize=8, color="#334155", va="center")
        ax.plot([0, 1.2], [0, 0], color="#1e293b", lw=4)
        ax.text(0.6, -0.08, "Losa del balcón", ha="center", va="top", fontsize=8, color="#334155")
        ax.add_patch(plt.Rectangle((0, 0), t, 1.0, color="#94a3b8", ec="#1e293b"))
        _flechas(ax, -0.55, -0.02, ys, "#b91c1c")
        _flechas(ax, t + 0.02, t + 0.5, ys, "#1d4ed8")
        ax.text(-0.9, 1.1, "Cara cargada (+)\n" + fmt(cs_t, cs_e, "p1"), fontsize=8.5, color="#b91c1c")
        ax.text(t + 0.05, 1.1, "Cara opuesta (-)\n" + fmt(cs_t, cs_e, "p2"), fontsize=8.5, color="#1d4ed8")
        ax.annotate("", xy=(0.7, -0.55), xytext=(0.05, -0.55),
                    arrowprops=dict(arrowstyle="<|-|>", color="black", lw=2.2))
        ax.text(-0.9, -0.72, f"p neta = {cs_t['pn_d']:.1f} (típ.) / {cs_e['pn_d']:.1f} (esq.) kgf/m²\n"
                "sentido: ambos (hacia el balcón y hacia el exterior)", fontsize=9, fontweight="bold", va="top")
        ax.set_title("Baranda de balcón: criterio caso B (Figura 45)", fontsize=11)
        ax.set_xlim(-0.95, 1.6); ax.set_ylim(-1.0, 1.4)
    for ax in axs:
        ax.set_aspect("equal"); ax.axis("off")
    fig.suptitle(f"Presiones de diseño sobre {('el parapeto' if modo == PARAPETO else 'la baranda')} [kgf/m²]  ·  "
                 f"qp = {res['qp']:.1f} kgf/m² (z = {z_top:.2f} m)", fontsize=10)
    fig.tight_layout()
    return fig


def dibujar_curvas_parapeto(res):
    """Coeficiente neto (GCp)cara1 - (GCp)cara2 en función del área efectiva."""
    areas = np.logspace(-1, 2, 200)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    gm = lambda k, a: get_gcp(a, *GC_MURO[k])
    if en_techo and g2_man is None:
        ax.plot(areas, [gm("z4p", a) - gcp_techo(a, "z2") for a in areas], color="#b91c1c", lw=2, label="Caso A típica (4 + 2)")
        ax.plot(areas, [gm("z5p", a) - gcp_techo(a, "z3") for a in areas], color="#b91c1c", lw=2, ls="--", label="Caso A esquina (5 + 3)")
    ax.plot(areas, [gm("z4p", a) - gm("z4n", a) for a in areas], color="#1d4ed8", lw=2, label="Caso B típica (4 + 4)")
    ax.plot(areas, [gm("z5p", a) - gm("z5n", a) for a in areas], color="#1d4ed8", lw=2, ls="--", label="Caso B esquina (5 + 5)")
    for cs in res["casos"]:
        ax.scatter([A_par], [cs["g1"] - cs["g2"]], color="black", s=36, zorder=10)
    ax.axvline(A_par, color="gray", lw=0.8, ls=":")
    for a_l in (0.9, 46.5):
        ax.axvline(a_l, color="gray", lw=0.6, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Área efectiva de viento A (m²)")
    ax.set_ylabel("Coeficiente neto (GCp)cara 1 − (GCp)cara 2")
    ax.set_title(f"Coeficiente neto {('del parapeto' if modo == PARAPETO else 'de la baranda')} "
                 f"({fig_muro}" + (f" y {fig_techo})" if en_techo else ")"), fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize="small")
    fig.tight_layout()
    return fig


if modo != FACHADA:
    res_par = calcular_parapeto()
    nombre_el = "parapeto" if modo == PARAPETO else "baranda"
    st.markdown(f"### 🧱 Presión de viento sobre {('parapetos' if modo == PARAPETO else 'barandas')} (NCh 432:2025, 9.6)")
    st.latex(r"q_p = 0.613 \cdot I \cdot K_z \cdot K_{zt} \cdot K_e \cdot V^2 \quad (z = \text{punto más alto}) \qquad \text{(Ec. 2)}")
    st.latex(r"p = q_p \cdot K_d \cdot \left[(GC_p) - (GC_{pi})\right] \qquad \text{(Ec. 22)}")
    st.latex(r"p_{neta} = q_p \cdot K_d \cdot \left[(GC_p)_{cara\,1} - (GC_p)_{cara\,2}\right] \;\geq\; 250\ \text{N/m}^2"
             r"\qquad w = p_{neta}\,h \qquad F = p_{neta}\,h\,s \qquad M = F\,h/2")
    st.info(f"**qp = {res_par['qp']:.2f} kgf/m²** en z = {z_top:.2f} m · Kz = {res_par['kz_top']:.3f} · "
            f"Kzt = {res_par['kzt_top']:.3f} · Ke = {Ke_val:.3f} · Kd = {Kd_val:.2f} · A = {A_par:.2f} m² · "
            f"Muros: {fig_muro}" + (f" · Techo: {fig_techo if g2_man is None else 'GCp ingresados'}" if en_techo else "") +
            f" · Ancho de esquina a = {res_par['a_esq']:.2f} m")
    if not en_techo:
        st.warning("Baranda de balcón: se adopta el **caso B de la Figura 45** (presión positiva de muro en una cara "
                   "y succión de muro en la cara opuesta), actuando en ambos sentidos. Criterio de asimilación a "
                   "parapeto (9.6) para barandas de paño lleno; para barandas abiertas resulta conservador.")
    if reduccion_muros:
        st.caption("GCp de muros reducidos en 10 % (Figura 24, Nota 5).")

    df_par = pd.DataFrame({
        "Caso": [c["caso"] for c in res_par["casos"]],
        "Zona": [c["zona"] for c in res_par["casos"]],
        "GCp cara 1": [round(c["g1"], 3) for c in res_par["casos"]],
        "GCp cara 2": [round(c["g2"], 3) for c in res_par["casos"]],
        "p cara 1 (kgf/m²)": [round(c["p1"], 2) for c in res_par["casos"]],
        "p cara 2 (kgf/m²)": [round(c["p2"], 2) for c in res_par["casos"]],
        "p neta (kgf/m²)": [round(c["pn_d"], 2) for c in res_par["casos"]],
        "w (kgf/m)": [round(c["w"], 1) for c in res_par["casos"]],
        "F montante (kgf)": [round(c["F"], 1) for c in res_par["casos"]],
        "M base (kgf·m)": [round(c["M"], 1) for c in res_par["casos"]],
    })
    st.table(df_par)
    gob = res_par["gob"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("p neta de diseño", f"{gob['pn_d']:.1f} kgf/m²", f"Caso {gob['caso']} · {gob['zona'].split(' (')[0]}", delta_color="off")
    m2.metric(f"Carga lineal w (h = {h_elem:.2f} m)", f"{gob['w']:.1f} kgf/m")
    m3.metric(f"Fuerza por montante (s = {s_apoyo:.2f} m)", f"{gob['F']:.1f} kgf")
    m4.metric("Momento en la base del montante", f"{gob['M']:.1f} kgf·m")
    st.caption("Cara 1: cara con presión positiva; cara 2: cara opuesta con succión. Ambas actúan en el mismo sentido "
               "y se suman en la presión neta; GCpi se anula en un elemento de una sola hoja. w, F y M suponen la "
               "presión neta uniforme sobre la altura del elemento, con el montante en voladizo desde su base.")
    if es_hueco:
        st.markdown(f"**Parapeto hueco: presión sobre cada revestimiento con GCpi = ±{gcpi_par}**")
        st.table(pd.DataFrame({
            "Caso": [c["caso"] for c in res_par["casos"]],
            "Zona": [c["zona"] for c in res_par["casos"]],
            "Revestimiento cara 1 (kgf/m²)": [round(c["p1i"], 2) for c in res_par["casos"]],
            "Revestimiento cara 2 (kgf/m²)": [round(c["p2i"], 2) for c in res_par["casos"]],
        }))
    if en_techo:
        st.caption("Figura 26, Nota 5 / Figura 40, Nota 7: con parapeto de 0,9 m o más en todo el perímetro, la zona 3 "
                   "del techo puede tratarse como zona 2 para el diseño del techo. Para el diseño del propio parapeto "
                   "se conservan los coeficientes de borde y esquina sin reducir.")

    fig_par_esq = dibujar_esquema_parapeto(res_par)
    st.pyplot(fig_par_esq)
    fig_par_curv = dibujar_curvas_parapeto(res_par)
    st.pyplot(fig_par_curv)


# =================================================================
# 7b. MEMORIA DE CÁLCULO EN PDF (NCh 432:2025) - RESULTADOS + GRÁFICOS
# =================================================================
import io
from datetime import date as _date

with st.sidebar.expander("📄 Datos de la Memoria Técnica", expanded=False):
    mem_codigo = st.text_input("Código doc.", "PE-MC-VIENTO-01")
    mem_rev = st.text_input("Revisión", "A")
    mem_mandante = st.text_input("Mandante", "")
    mem_proyecto = st.text_input("Proyecto", "Determinación de presiones de viento")
    mem_fecha = st.date_input("Fecha", _date.today())

_MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
          "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

def _fecha_larga(d):
    return f"{d.day} de {_MESES[d.month - 1]} de {d.year}"

def _txt(s):
    """Las fuentes base de FPDF solo admiten Latin-1: reemplaza símbolos fuera de ese juego."""
    rep = {"θ": "theta", "α": "alfa", "≥": ">=", "≤": "<=", "—": "-", "–": "-",
           "→": "->", "·": "·", "“": '"', "”": '"', "’": "'", "⁴": "4", "₁": "1"}
    s = str(s)
    for a, b in rep.items():
        s = s.replace(a, b)
    return s.encode("latin-1", "replace").decode("latin-1")

def _n(x, d=2):
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")

def _fig_png(figura):
    buf = io.BytesIO()
    figura.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf

class MemoriaViento(FPDF):
    NAVY = (26, 58, 92)
    def header(self):
        _logo = leer_imagen("Logo.png")
        if _logo:
            self.image(io.BytesIO(_logo), x=12, y=8, h=11)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*self.NAVY)
        self.set_xy(60, 9)
        self.cell(0, 5, _txt("Memoria de cálculo - Presiones de viento NCh 432:2025"), align="R")
        self.set_font("Helvetica", "", 8)
        self.set_xy(60, 14)
        self.cell(0, 5, _txt(f"{mem_codigo} · Rev. {mem_rev}"), align="R")
        self.set_draw_color(*self.NAVY)
        self.set_line_width(0.6)
        self.line(12, 21, 198, 21)
        self.set_text_color(0, 0, 0)
        self.set_y(25)
    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 7.5)
        self.set_text_color(91, 104, 118)
        self.cell(0, 5, _txt(f"Proyectos Estructurales EIRL · Structural Lab · {mem_codigo} Rev. {mem_rev}"), align="L")
        self.cell(0, 5, f"Pág. {self.page_no()}/{{nb}}", align="R")

    def h2(self, t, reserva=32):
        if self.get_y() + reserva > self.page_break_trigger:  # evita títulos huérfanos al pie
            self.add_page()
        self.ln(3)
        self.set_font("Helvetica", "B", 11.5)
        self.set_text_color(*self.NAVY)
        self.cell(0, 7, _txt(t), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*self.NAVY)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), 210 - self.r_margin, self.get_y())
        self.ln(2)
        self.set_text_color(0, 0, 0)
    def p(self, t, size=9.5, style=""):
        self.set_font("Helvetica", style, size)
        self.multi_cell(0, 5, _txt(t), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)
    def eq(self, t):
        self.set_font("Courier", "B", 9.5)
        self.set_fill_color(234, 242, 251)
        self.cell(0, 7, _txt("   " + t), fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)
    def nota(self, t, borde=(44, 95, 138), fondo=(234, 242, 251), color=(0, 0, 0)):
        self.set_font("Helvetica", "", 8.5)
        self.set_fill_color(*fondo)
        self.set_draw_color(*borde)
        self.set_text_color(*color)
        self.multi_cell(0, 4.6, _txt(t), border=1, fill=True, padding=2.2, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(2)
    def tabla(self, cab, filas, anchos, alinear=None):
        alinear = alinear or (["L"] + ["R"] * (len(cab) - 1))
        if self.get_y() + 7 * (len(filas) + 1) > self.page_break_trigger:
            self.add_page()
        self.set_font("Helvetica", "B", 8.5)
        self.set_fill_color(*self.NAVY)
        self.set_text_color(255, 255, 255)
        self.set_draw_color(213, 221, 229)
        for c, w, a in zip(cab, anchos, alinear):
            self.cell(w, 7, _txt(c), border=1, fill=True, align=a)
        self.ln()
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(0, 0, 0)
        for i, fila in enumerate(filas):
            self.set_fill_color(*((234, 242, 251) if i % 2 else (255, 255, 255)))
            for c, w, a in zip(fila, anchos, alinear):
                self.cell(w, 6.2, _txt(c), border="B", fill=True, align=a)
            self.ln()
        self.ln(2)
    def figura(self, fuente, ancho, titulo):
        alto_estimado = ancho * 0.75 + 12
        if self.get_y() + alto_estimado > self.page_break_trigger:
            self.add_page()
        x = (210 - ancho) / 2
        self.image(fuente, x=x, w=ancho)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(91, 104, 118)
        self.cell(0, 5, _txt(titulo), align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(2)

def generar_pdf_viento():
    pdf = MemoriaViento(format="A4")
    pdf.alias_nb_pages()
    pdf.set_margins(15, 25, 15)
    pdf.set_auto_page_break(True, margin=18)
    pdf.add_page()

    # --- Portada ---
    pdf.set_font("Times", "B", 17)
    titulo_mem = {FACHADA: "Determinación de presiones de viento según NCh 432:2025",
                  PARAPETO: "Presiones de viento sobre parapetos según NCh 432:2025",
                  BARANDA: "Presiones de viento sobre barandas según NCh 432:2025"}[modo]
    pdf.multi_cell(0, 8, _txt("Memoria de cálculo - " + titulo_mem), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(91, 104, 118)
    pdf.cell(0, 5, _txt(f"{mem_proyecto}" + (f" · Mandante: {mem_mandante}" if mem_mandante else "")),
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, _txt(f"Documento {mem_codigo} · Revisión {mem_rev} · {_fecha_larga(mem_fecha)}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)
    pdf.nota("Documento generado por la herramienta Structural Lab - Viento NCh 432:2025. Los parámetros de "
             "cálculo son los declarados en el apartado 3 y quedan bajo responsabilidad de quien los ingresa.",
             borde=(192, 57, 43), fondo=(253, 237, 236), color=(125, 40, 32))

    # --- 1. Objeto ---
    pdf.h2("1. Objeto y alcance")
    if modo == FACHADA:
        pdf.p(f"Se determinan las presiones de diseño del viento sobre componentes y revestimientos (C&R) de un "
              f"edificio de altura media h = {_n(H_edif, 1)} m, con techo de inclinación {theta}°, para un elemento de "
              f"área efectiva {_n(area_ef, 2)} m². Se reportan la presión de velocidad, las presiones netas por zona "
              f"y el perfil de presiones en altura para barlovento, sotavento y caras laterales.")
    else:
        ubic_txt = ("en el borde del techo" if en_techo else
                    f"en un balcón de fachada, con piso a {_n(z_base_elem, 2)} m sobre el terreno")
        pdf.p(f"Se determinan las presiones de diseño del viento sobre {('un parapeto' if modo == PARAPETO else 'una baranda')} "
              f"de altura {_n(h_elem, 2)} m ubicado {ubic_txt}, en un edificio de altura media h = {_n(H_edif, 1)} m "
              f"(componentes y revestimientos, NCh 432:2025, 9.6). Se reportan la presión de velocidad en el punto más "
              f"alto del elemento, las presiones sobre cada cara y la presión neta para los casos de carga de la "
              f"Figura 45, junto con la carga lineal, la fuerza y el momento en la base de los montantes, separados "
              f"{_n(s_apoyo, 2)} m.")

    # --- 2. Normativa ---
    pdf.h2("2. Antecedentes y normativa")
    for li in ["NCh 432:2025 - Diseño estructural - Cargas de viento.",
               "NCh3171 - Disposiciones generales y combinaciones de carga (categoría de ocupación).",
               ("Presión de velocidad según Ec. (2) y presión de diseño C&R según Ec. (19) de NCh 432:2025."
                if modo == FACHADA else
                "Presión de velocidad según Ec. (2); presión de diseño en parapetos según 9.6, Ec. (22) y Figura 45."),
               "Presión neta mínima de diseño para C&R: 0,25 kN/m² (9.2.2)."]:
        pdf.p("-  " + li)

    # --- 3. Datos de entrada ---
    pdf.h2("3. Datos de entrada", reserva=85)
    filas = [
        ["Velocidad básica V", f"{_n(V, 1)} m/s"],
        ["Altura media del edificio h", f"{_n(H_edif, 2)} m"],
        ["Inclinación de techo theta", f"{theta}°"],
    ]
    if modo == FACHADA:
        filas += [
            ["Largo del elemento L", f"{_n(l_elem, 2)} m"],
            ["Ancho tributario ingresado", f"{_n(w_in, 2)} m"],
            ["Ancho tributario de cálculo max(b, L/3)", f"{_n(w_trib, 2)} m"],
            ["Área efectiva A", f"{_n(area_ef, 2)} m²"],
        ]
    else:
        filas += [
            ["Elemento", "Parapeto" if modo == PARAPETO else f"Baranda ({ubic_baranda.lower()})"],
            ["Altura del elemento", f"{_n(h_elem, 2)} m"],
            ["Cota de la base del elemento", f"{_n(z_base_elem, 2)} m"],
            ["Cota del punto más alto z", f"{_n(z_top, 2)} m"],
            ["Separación entre montantes s", f"{_n(s_apoyo, 2)} m"],
            ["Área efectiva A = L·max(b, L/3)", f"{_n(A_par, 2)} m²"],
            ["Menor dimensión horizontal del edificio B", f"{_n(B_min, 1)} m"],
            ["Ancho de zona de esquina a", f"{_n(res_par['a_esq'], 2)} m"],
        ]
    filas += [
        ["Categoría de exposición", f"{cat_exp} ({exp_info[cat_exp][2]})"],
        ["Categoría de riesgo / factor I", f"{cat_imp} / {_n(factor_i, 2)}"],
        ["Clasificación del cerramiento", cerramiento_opcion],
    ]
    pdf.tabla(["Parámetro", "Valor"], filas, [120, 60])

    # --- 4. Factores ---
    pdf.h2("4. Factores de cálculo", reserva=80)
    filas_f = [
        ["Exponente de ley de potencia alfa", _n(alpha, 1)],
        ["Altura gradiente zg (Tabla 6)", f"{_n(zg, 0)} m"],
        ["Altura mínima zmín (Tabla 6)", f"{_n(zmin, 0)} m"],
        ["Coef. de exposición Kz (z = h)", _n(kz, 3)],
        ["Factor topográfico Kzt (z = h)", _n(Kzt_val, 3)],
        ["Factor de elevación del suelo Ke (5.7)", _n(Ke_val, 3)],
        ["Factor de direccionalidad Kd", _n(Kd_val, 2)],
        ["Factor de importancia I", _n(factor_i, 2)],
        ["Coef. de presión interna GCpi", f"± {_n(gc_pi_val, 2)}"] if modo == FACHADA else
        ["Coef. de presión interna del parapeto GCpi", f"± {_n(gcpi_par, 2)}" if es_hueco else "Se anula (una hoja)"],
    ]
    if modo != FACHADA:
        filas_f.insert(5, ["Coef. de exposición Kz (z = tope del elemento)", _n(res_par["kz_top"], 3)])
        filas_f.insert(6, ["Factor topográfico Kzt (z = tope del elemento)", _n(res_par["kzt_top"], 3)])
    pdf.tabla(["Factor", "Valor"], filas_f, [120, 60])
    if metodo == "Calculado":
        pdf.p(f"Kzt calculado para {tipo_relieve}, Exposición {cat_exp}, con Hc = {_n(Hc, 1)} m, Lh = {_n(Lhc, 1)} m "
              f"y x = {_n(x_cresta, 1)} m a {lado_x.lower()}: K1 = {_n(k1, 3)}, K2 = {_n(k2, 3)}; "
              f"Kzt(z) = (1 + K1·K2·K3(z))², con Kzt(h) = {_n(Kzt_val, 3)}. Constantes según ASCE/SEI 7-22 Fig. 26.8-1.")
        if kzt_nota:
            pdf.nota(kzt_nota)
    else:
        pdf.p(f"Kzt ingresado directamente por el usuario: {_n(Kzt_val, 3)}.")

    # Ke - factor de elevación del suelo (5.7, Tabla 4)
    if ke_metodo.startswith("Conservador"):
        pdf.p("Ke se toma igual a 1,00 para todas las elevaciones, de manera conservadora "
              "(NCh 432:2025, 5.7 y Tabla 4, Nota 1).")
    elif ke_metodo.startswith("Tabla") and 0 <= z_e <= 1800:
        pdf.p(f"Ke se determina por interpolación lineal en la Tabla 4 de NCh 432:2025 (5.7) para una "
              f"elevación del suelo ze = {_n(z_e, 0)} m sobre el nivel del mar: Ke = {_n(Ke_val, 3)}.")
        pdf.tabla(["ze [m]"] + [_n(z, 0) for z in KE_TABLA_ZE],
                  [["Ke"] + [_n(k, 2) for k in KE_TABLA_KE]],
                  [26] + [22] * len(KE_TABLA_ZE), ["L"] + ["C"] * len(KE_TABLA_ZE))
    else:
        motivo = ("(ze fuera del rango de la Tabla 4) " if ke_metodo.startswith("Tabla") else "")
        pdf.p(f"Ke se determina con la fórmula de la Nota 2 de la Tabla 4 de NCh 432:2025 {motivo}"
              f"para ze = {_n(z_e, 0)} m sobre el nivel del mar:")
        pdf.eq(f"Ke = e^(-0,000119 · ze) = e^(-0,000119 · {_n(z_e, 0)}) = {_n(Ke_val, 3)}")

    # --- 5. Formulación ---
    pdf.h2("5. Formulación")
    pdf.p("Coeficiente de exposición (Tabla 5, Notas 1, 3 y 4; alfa, zg y zmín según Tabla 6):")
    pdf.eq("Kz = 2,41 · (z / zg)^(2/alfa)  ;  z < zmín: z = zmín  ;  z > zg: Kz = 2,41")
    pdf.p("Factor de elevación del suelo (5.7, Tabla 4, Nota 2):")
    pdf.eq("Ke = e^(-0,000119 · ze)   ;   Ke = 1,00 permitido para toda elevación")
    pdf.p("Presión de velocidad, Ec. (2):")
    pdf.eq("qz = 0,613 · I · Kz · Kzt · Ke · V²   [N/m²]")
    if modo == FACHADA:
        pdf.p("Presión de diseño para C&R, con GCpi en el signo más desfavorable y qi = qh:")
        pdf.eq("p = q · Kd · (GCp) - qi · Kd · (GCpi)   ;   |p| >= 250 N/m² (9.2.2)")
        pdf.p(f"GCp de muros según la {fig_muro} de NCh 432:2025, interpolado linealmente en log10(A) entre "
              f"A = 0,9 m² y A = 46,5 m². " + ("La sobrepresión se evalúa con qz y la succión con qh (Figura 40, Nota 4). "
              if edificio_alto else "Todas las presiones se evalúan con qh (edificio de baja altura). ") +
              "Conversión de unidades: 1 N/m² = 0,10197 kgf/m².")
        if aviso_techo:
            pdf.nota(aviso_techo)
    else:
        pdf.p("Presión de diseño en parapetos, Ec. (22), con qp evaluada en el punto más alto del elemento:")
        pdf.eq("p = qp · Kd · [ (GCp) - (GCpi) ]")
        pdf.p("Presión neta sobre un elemento de una hoja (las presiones de ambas caras actúan en el mismo sentido "
              "y GCpi se anula):")
        pdf.eq("p neta = qp · Kd · [ (GCp)cara 1 - (GCp)cara 2 ]   ;   p neta >= 250 N/m² (9.2.2)")
        pdf.p("Cargas sobre el elemento, con la presión neta uniforme en la altura h del elemento y el montante "
              "en voladizo desde su base:")
        pdf.eq("w = p neta · h   ;   F = p neta · h · s   ;   M = F · h / 2")
        if en_techo:
            pdf.p("Casos de carga de la Figura 45: Caso A (parapeto de barlovento): cara exterior con presión positiva "
                  "de muro, zonas 4 o 5, y cara interior con succión de borde o esquina de techo, zonas 2 o 3. "
                  "Caso B (parapeto de sotavento): cara interior con presión positiva de muro y cara exterior con "
                  "succión de muro, zonas 4 o 5.")
        else:
            pdf.nota("Criterio adoptado para barandas de balcón: se asimilan a un parapeto (9.6) y se aplica el caso B "
                     "de la Figura 45, con presión positiva de muro en una cara y succión de muro en la cara opuesta, "
                     "zonas 4 o 5, actuando en ambos sentidos. Aplicable a barandas de paño lleno (vidrio, plancha); "
                     "para barandas abiertas el criterio resulta conservador.")
        txt_techo = ""
        if en_techo:
            txt_techo = (f"; GCp de borde y esquina de techo según la {fig_techo}" if g2_man is None else
                         f"; GCp de techo ingresados por el usuario (zona 2: {_n(g2_man, 2)}; zona 3: {_n(g3_man, 2)}), "
                         "según las Figuras 27 a 32 para θ > 7°")
        pdf.p(f"GCp de muros según la {fig_muro}{txt_techo}, interpolados linealmente en log10(A) entre "
              f"A = 0,9 m² y A = 46,5 m²." + (" GCp de muros reducidos en 10 % (Figura 24, Nota 5)." if reduccion_muros else "") +
              " Conversión de unidades: 1 N/m² = 0,10197 kgf/m².")

    # --- 6. qh ---
    pdf.h2("6. Presión de velocidad a la altura media del techo")
    pdf.eq(f"qh = 0,613 · {_n(factor_i, 2)} · {_n(kz, 3)} · {_n(Kzt_val, 3)} · {_n(Ke_val, 3)} · {_n(V, 1)}²")
    pdf.eq(f"qh = {_n(qh / 0.10197, 1)} N/m² = {_n(qh, 2)} kgf/m²")
    if modo != FACHADA:
        pdf.p(f"Presión de velocidad en el punto más alto del elemento, z = {_n(z_top, 2)} m:")
        pdf.eq(f"qp = 0,613 · {_n(factor_i, 2)} · {_n(res_par['kz_top'], 3)} · {_n(res_par['kzt_top'], 3)} · "
               f"{_n(Ke_val, 3)} · {_n(V, 1)}²")
        pdf.eq(f"qp = {_n(res_par['qp'] / 0.10197, 1)} N/m² = {_n(res_par['qp'], 2)} kgf/m²")

    if modo == FACHADA:
        # --- 7. Presiones netas ---
        pdf.h2("7. Presiones netas de diseño por zona", reserva=75)
        filas_p = [[_txt(z), _n(g, 3), f"± {_n(gc_pi_val, 2)}", _n(p, 2)]
                   for z, g, p in zip(zonas, gcp_vals, p_netas)]
        pdf.tabla(["Zona de análisis", "GCp", "GCpi", "p [kgf/m²]"], filas_p, [84, 30, 30, 36])
        pdf.p("Valores positivos: sobrepresión hacia la superficie. Valores negativos: succión hacia el exterior. "
              "La succión de Zonas 4 y 5 cubre los muros de sotavento y laterales y se evalúa con qh, constante en "
              "toda la altura. Se aplica la presión neta mínima de 250 N/m² (25,5 kgf/m², 9.2.2).", size=8.5, style="I")

        # --- 8. Gráficos ---
        pdf.h2("8. Gráficos de verificación")
        pdf.figura(_fig_png(fig), 165, f"Figura 1 - Coeficientes GCp de muros en función del área efectiva (A = {_n(area_ef, 2)} m² marcado)")
        pdf.figura(_fig_png(fig_alt), 175, f"Figura 2 - Perfil de presiones netas de diseño en altura, V = {_n(V, 1)} m/s")

        # --- 9. Esquemas ---
        esquemas = [(leer_imagen(f), t) for f, t in [("F8.png", "Identificación de zonas"), ("F12.png", "Esquema isométrico")]
                    if leer_imagen(f)]
        if esquemas:
            pdf.h2("9. Esquemas de zonificación")
            for i, (b, t) in enumerate(esquemas, start=3):
                pdf.figura(io.BytesIO(b), 150, f"Figura {i} - {t}")

        # --- Conclusiones ---
        pdf.h2(("10." if esquemas else "9.") + " Conclusiones")
        i_min = int(np.argmin(p_netas)); i_max = int(np.argmax(p_netas))
        pdf.p(f"-  Presión de velocidad a la altura media del techo: qh = {_n(qh, 2)} kgf/m².")
        pdf.p(f"-  Succión máxima de diseño: {_n(p_netas[i_min], 2)} kgf/m² en {_txt(zonas[i_min])}.")
        pdf.p(f"-  Empuje máximo de diseño: {_n(p_netas[i_max], 2)} kgf/m² en {_txt(zonas[i_max])}.")
        pdf.p("-  Los valores corresponden a la configuración analizada. Cualquier variación de geometría, "
              "exposición, categoría o cerramiento requiere una nueva evaluación.")
    else:
        # --- 7. Resultados parapeto / baranda ---
        pdf.h2("7. Presiones de diseño sobre " + ("el parapeto" if modo == PARAPETO else "la baranda"), reserva=75)
        casos = res_par["casos"]
        pdf.tabla(["Caso", "Zona", "GCp cara 1", "GCp cara 2", "p cara 1", "p cara 2", "p neta"],
                  [[c["caso"], _txt(c["zona"]), _n(c["g1"], 3), _n(c["g2"], 3), _n(c["p1"], 2), _n(c["p2"], 2),
                    _n(c["pn_d"], 2)] for c in casos],
                  [12, 56, 22, 22, 22, 22, 24], ["C", "L", "R", "R", "R", "R", "R"])
        pdf.p("Presiones en kgf/m². Cara 1: cara con presión positiva (hacia la superficie); cara 2: cara opuesta, con "
              "succión (alejándose de la superficie). Ambas actúan en el mismo sentido y se suman en la presión neta. "
              "Caso A: presión neta hacia el techo. Caso B: presión neta hacia el exterior" +
              (" (en balcones, en ambos sentidos)." if not en_techo else "."), size=8.5, style="I")
        pdf.p(f"Cargas sobre el elemento (h = {_n(h_elem, 2)} m; s = {_n(s_apoyo, 2)} m):")
        pdf.tabla(["Caso", "Zona", "p neta [kgf/m²]", "w [kgf/m]", "F [kgf]", "M [kgf·m]"],
                  [[c["caso"], _txt(c["zona"]), _n(c["pn_d"], 2), _n(c["w"], 1), _n(c["F"], 1), _n(c["M"], 1)]
                   for c in casos],
                  [12, 56, 28, 28, 28, 28], ["C", "L", "R", "R", "R", "R"])
        if es_hueco:
            pdf.p(f"Parapeto hueco: presión sobre cada revestimiento con GCpi = ±{_n(gcpi_par, 2)} (Tabla 7):")
            pdf.tabla(["Caso", "Zona", "Revestimiento cara 1 [kgf/m²]", "Revestimiento cara 2 [kgf/m²]"],
                      [[c["caso"], _txt(c["zona"]), _n(c["p1i"], 2), _n(c["p2i"], 2)] for c in casos],
                      [12, 56, 56, 56], ["C", "L", "R", "R"])
        if en_techo:
            pdf.nota("Figura 26, Nota 5 / Figura 40, Nota 7: con parapeto de 0,9 m o más en todo el perímetro, la zona 3 "
                     "del techo puede tratarse como zona 2 en el diseño del techo. En el diseño del propio parapeto se "
                     "conservan los coeficientes de borde y esquina sin reducir.")

        # --- 8. Gráficos ---
        pdf.h2("8. Esquema de cargas y gráficos de verificación", reserva=120)
        pdf.figura(_fig_png(fig_par_esq), 175 if en_techo else 130,
                   "Figura 1 - Presiones sobre las caras del elemento y presión neta (casos de la Figura 45)")
        pdf.figura(_fig_png(fig_par_curv), 160,
                   f"Figura 2 - Coeficiente neto en función del área efectiva (A = {_n(A_par, 2)} m² marcado)")

        # --- 9. Conclusiones ---
        gob = res_par["gob"]
        pdf.h2("9. Conclusiones")
        pdf.p(f"-  Presión de velocidad en el punto más alto del elemento (z = {_n(z_top, 2)} m): "
              f"qp = {_n(res_par['qp'], 2)} kgf/m².")
        pdf.p(f"-  Presión neta de diseño: {_n(gob['pn_d'], 2)} kgf/m² ({_n(gob['pn_d'] * KGF_A_KN, 3)} kN/m²), "
              f"caso {gob['caso']}, {_txt(gob['zona'].lower())}, {gob['sentido']}.")
        pdf.p(f"-  Carga lineal sobre el elemento: w = {_n(gob['w'], 1)} kgf/m ({_n(gob['w'] * KGF_A_KN, 2)} kN/m).")
        pdf.p(f"-  Por montante (s = {_n(s_apoyo, 2)} m): F = {_n(gob['F'], 1)} kgf ({_n(gob['F'] * KGF_A_KN, 2)} kN) y "
              f"M = {_n(gob['M'], 1)} kgf·m ({_n(gob['M'] * KGF_A_KN, 2)} kN·m) en la base.")
        tip = [c for c in res_par["casos"] if c["zona"].startswith("Típica")]
        tip_gob = max(tip, key=lambda c: c["pn_d"])
        pdf.p(f"-  Fuera de la zona de esquina (a = {_n(res_par['a_esq'], 2)} m desde cada esquina) gobierna "
              f"{_n(tip_gob['pn_d'], 2)} kgf/m² (caso {tip_gob['caso']}).")
        pdf.p("-  Los valores corresponden a la configuración analizada. Cualquier variación de geometría, "
              "exposición, categoría o ubicación del elemento requiere una nueva evaluación.")

    # --- Control de calidad y firma ---
    pdf.ln(2)
    pdf.nota("Control de calidad\n"
             f"Mandante: {mem_mandante or '-'}  |  Documento: {mem_codigo} (Rev. {mem_rev})\n"
             f"Norma: NCh 432:2025  |  Herramienta: Structural Lab - Viento NCh 432 (Python)",
             borde=(213, 221, 229), fondo=(244, 246, 248))
    pdf.nota("Este documento reúne los resultados que la herramienta obtiene a partir de los parámetros ingresados "
             "por el usuario. No constituye memoria de cálculo mientras no sea revisado, aprobado y suscrito por un "
             "profesional responsable, a quien compete verificar las hipótesis adoptadas y hacerse cargo de ellas.",
             borde=(192, 57, 43), fondo=(253, 237, 236), color=(125, 40, 32))
    if pdf.get_y() + 36 > pdf.page_break_trigger:
        pdf.add_page()
    pdf.ln(10)
    pdf.set_draw_color(0, 0, 0)
    pdf.line(15, pdf.get_y(), 85, pdf.get_y())
    pdf.set_font("Helvetica", "", 8.5)
    pdf.cell(0, 5, "Nombre y firma del profesional responsable", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(9)
    pdf.line(15, pdf.get_y(), 85, pdf.get_y())
    pdf.cell(0, 5, _txt("Título profesional y número de registro"), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())

st.divider()
st.subheader("📄 Memoria de Cálculo")
try:
    _pdf_bytes = generar_pdf_viento()
    st.download_button("⬇️ Descargar memoria de cálculo (PDF)", data=_pdf_bytes,
                       file_name=f"Memoria_Viento_{ {FACHADA: 'Fachadas', PARAPETO: 'Parapetos', BARANDA: 'Barandas'}[modo] }_{mem_codigo}_Rev{mem_rev}.pdf",
                       mime="application/pdf", type="primary")
    st.caption("Incluye datos de entrada, factores, formulación, presiones por zona, gráficos y esquemas. "
               "Complete los datos del documento en el panel lateral (📄 Datos de la Memoria Técnica).")
except Exception as e:
    st.error(f"No se pudo generar la memoria PDF: {e}. Requiere fpdf2 (pip install fpdf2).")


# =================================================================
# 8. SECCIÓN DE CONTACTO Y CRÉDITOS FINALES
# =================================================================
st.markdown("---")
st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; color: #444; font-size: 0.95em;">
        <div>
            <strong>Desarrollado por:</strong> Mauricio Riquelme <br>
            <em>Ingeniero Civil Estructural</em>
        </div>
        <div style="text-align: right;">
            <strong>Contacto Proyectos Estructurales EIRL:</strong><br>
            <a href="mailto:mriquelme@proyectosestructurales.com" style="text-decoration: none; color: #007BFF; font-weight: bold;">
                mriquelme@proyectosestructurales.com
            </a>
        </div>
    </div>
    <div style="text-align: center; margin-top: 50px; margin-bottom: 20px;">
        <p style="font-family: 'Georgia', serif; font-size: 1.4em; color: #003366; font-style: italic; letter-spacing: 1px;">
            "Programming is understanding"
        </p>
    </div>
    """, unsafe_allow_html=True)
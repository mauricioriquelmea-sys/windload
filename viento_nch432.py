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
            if os.path.exists(img): st.image(img, caption=f"Zonificación: {img}")

V = st.sidebar.number_input("Velocidad básica V (m/s)", 20.0, 60.0, 34.0)
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
    # k1_b: factor de forma, gam: decaimiento en altura (K3), mu: decaimiento horizontal (K2)
    if tipo_relieve == "Escarpe 2D":
        k1_b, gam, mu_v = 0.75, 2.5, 1.5
    elif tipo_relieve == "Colina 2D":
        k1_b, gam, mu_v = 1.05, 1.5, 1.5
    else: # Colina 3D
        k1_b, gam, mu_v = 0.95, 1.5, 4.0
    
    # Cálculo de Factores (Asumiendo x=0 y z=H_edif)
    k1 = k1_b * (Hc / Lhc)
    k2 = 1.0  # Para x = 0 (Cresta), K2 siempre es 1.0
    k3 = math.exp(-gam * H_edif / Lhc) # z = H_edif (Altura máxima edificio)
    
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
# Factor de elevación del terreno Ke (Ec. 2, NCh 432:2025, 5.8.2). 1.0 por defecto.
Ke_val = st.sidebar.number_input("Factor de elevación del terreno Ke", 0.50, 1.00, 1.00, step=0.01)

with st.sidebar.expander("ℹ️ Nota Explicativa: Exposición"):
    st.markdown("""
    **Rugosidad del Terreno (Capítulo 4):**
    * **B:** Áreas urbanas y suburbanas, áreas boscosas u otros terrenos con numerosas obstrucciones próximas.
    * **C:** Terrenos abiertos con obstrucciones dispersas < 9m. Incluye campos abiertos y terrenos agrícolas.
    * **D:** Áreas planas y sin obstrucciones frente a cuerpos de agua (Costa).
    """)

# --- AYUDA TÉCNICA RIGUROSA: CATEGORÍA DE EXPOSICIÓN ---
with st.sidebar.expander("ℹ️ Nota Explicativa: Exposición (B, C, D)"):
    st.markdown("""
    **Definiciones según NCh 432 (Capítulo 4):**
    
    * **Exposición B:** Áreas urbanas y suburbanas, áreas boscosas u otros terrenos con numerosas obstrucciones próximas del tamaño de viviendas unifamiliares o mayores.
    * **Exposición C:** Terrenos abiertos con obstrucciones dispersas que tienen alturas generalmente menores a 9m. (Categoría por defecto).
    * **Exposición D:** Áreas planas y sin obstrucciones frente a cuerpos de agua que se extienden al menos 1.6 km.
    """)

cat_exp = st.sidebar.selectbox("Categoría de Exposición", ['B', 'C', 'D'], index=0)

# Diccionario de parámetros de rugosidad según Tabla 3 de la Norma
# alpha: exponente de la ley de potencia | zg: altura nominal de la capa límite (m)
exp_info = {
    'B': [7.0, 366.0, "Urbano/Suburbano"],
    'C': [9.5, 274.0, "Terreno Abierto"],
    'D': [11.5, 213.0, "Costa/Agua"]
}

alpha_val = exp_info[cat_exp][0]
zg_val = exp_info[cat_exp][1]
desc_exp = exp_info[cat_exp][2]

# Despliegue de los factores asociados debajo del selector
st.sidebar.info(f"""
**Parámetros de Rugosidad:**
* Tipo: {desc_exp}
* Exponente (α): {alpha_val}
* Altura Gradiente (zg): {zg_val} m
""")

# =================================================================
# 3. SIDEBAR: CATEGORÍA DE RIESGO Y PERIODOS DE RETORNO (CORREGIDO NCh 432:2025)
# =================================================================

with st.sidebar.expander("ℹ️ Nota Explicativa: Categoría de Riesgo"):
    st.markdown("""
    **Clasificación según NCh 432:2025:**
    La norma actual asigna periodos de retorno específicos ($T$) para la velocidad básica del viento, eliminando el antiguo factor de importancia multiplicador.
    
    * **Categoría I:** Estructuras que representan un riesgo bajo para la vida humana en caso de falla. 
      *(T = 300 años)*.
    * **Categoría II:** Estructuras estándar (Viviendas, oficinas, comercios) que no clasifican en I, III o IV. 
      *(T = 700 años)*.
    * **Categoría III:** Estructuras con un gran número de personas o capacidad limitada de evacuación (Colegios, cines, estadios). 
      *(T = 1700 años)*.
    * **Categoría IV:** Estructuras esenciales cuya operatividad es crítica tras un evento (Hospitales, estaciones de emergencia). 
      *(T = 3000 años)*.
    
    *Nota: La velocidad básica V (m/s) ingresada debe corresponder al mapa de la categoría seleccionada.*
    """)

# Selector de Categoría de Riesgo
cat_imp = st.sidebar.selectbox("Categoría de Riesgo / Riesgo", ['I', 'II', 'III', 'IV'], index=1)

# En la NCh 432-2025, el factor de importancia I es 1.0 porque el riesgo se incluye en V_basica
# Sin embargo, para mantener compatibilidad con el motor de cálculo:
imp_map = {'I': 0.54, 'II': 1.0, 'III': 1.15, 'IV': 1.22}
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

# --- MOTOR MATEMÁTICO ---
def get_gcp(a, g1, g10):
    if a <= 1.0: return g1
    if a >= 10.0: return g10
    return g1 + (g10 - g1) * (np.log10(a) - np.log10(1.0))

exp_params = {'B': [7.0, 366.0], 'C': [9.5, 274.0], 'D': [11.5, 213.0]}
alpha, zg = exp_params[cat_exp]
kz = 2.01 * ((max(H_edif, 4.6) / zg)**(2/alpha))

# Presión de velocidad qh - Ec. (2) NCh 432:2025: qz = 0.613 I Kz Kzt Ke V^2 [N/m2]
# Kd YA NO va en q: se aplica en la presión de diseño, Ec. (19): p = qh Kd [(GCp) - (GCpi)]
qh = (0.613 * imp_map[cat_imp] * kz * Kzt_val * Ke_val * (V**2)) * 0.10197

def p_diseno(q, gcp, gcpi):
    """Ec. (19) NCh 432:2025: p = q Kd [(GCp) - (GCpi)], con GCpi en el signo más desfavorable."""
    return q * Kd_val * (gcp + gcpi) if gcp >= 0 else q * Kd_val * (gcp - gcpi)


# =================================================================
# 5. DESPLIEGUE TÉCNICO DE RESULTADOS Y FORMULACIÓN (CORREGIDO)
# =================================================================

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
st.latex(r"q_z = 0.613 \cdot I \cdot K_z \cdot K_{zt} \cdot K_e \cdot V^2 \qquad \text{(Ec. 2)}")
st.latex(r"p = q_h \cdot K_d \cdot \left[(GC_p) - (GC_{pi})\right] \qquad \text{(Ec. 19)}")
st.caption(f"Kd = {Kd_val:.2f} se aplica a la presión neta (externa e interna), no a la presión de velocidad. Ke = {Ke_val:.2f}.")

st.info(f"**Presión de Velocidad Calculada (qh):** {qh:.2f} kgf/m²")

# --- CÁLCULO DE COEFICIENTES GCp (DEFINICIÓN DE VARIABLES PARA PUNTO 6) ---
# Techo (Solo Succión)
z1 = get_gcp(area_ef, -1.0, -0.9) if theta <= 7 else get_gcp(area_ef, -0.9, -0.8)
z2 = get_gcp(area_ef, -1.8, -1.1) if theta <= 7 else get_gcp(area_ef, -1.3, -1.2)
z3 = get_gcp(area_ef, -2.8, -1.1) if theta <= 7 else get_gcp(area_ef, -2.0, -1.2)

# Paredes (Nombres específicos para evitar NameError)
z4_neg, z5_neg = get_gcp(area_ef, -1.1, -0.8), get_gcp(area_ef, -1.4, -1.1)
z4_pos, z5_pos = get_gcp(area_ef, 1.0, 0.7), get_gcp(area_ef, 1.0, 0.8)

# Tabulación de Resultados
col_res, col_plt = st.columns([1, 1.3])

with col_res:
    st.markdown("**Resumen de Presiones Netas por Zona**")
    
    # Agregamos Pared Lateral a la lista
    zonas = [
        "Z1 (Techo Centro - Succión)", "Z2 (Techo Borde - Succión)", "Z3 (Techo Esq - Succión)",
        "Z4 (Pared Std - Barlovento)", "Z4 (Pared Std - Sotavento)",
        "Z5 (Pared Esq - Barlovento)", "Z5 (Pared Esq - Sotavento)",
        "Paredes Laterales (Succión)" # <-- NUEVA FILA
    ]
    # CP lateral estándar -0.80
    gcp_vals = [z1, z2, z3, z4_pos, z4_neg, z5_pos, z5_neg, -0.80]
    
    p_netas = []
    for g in gcp_vals:
        p_netas.append(round(p_diseno(qh, g, gc_pi_val), 2))

    df_res = pd.DataFrame({
        "Zona de Análisis": zonas,
        "GCp (Externo)": [round(z, 3) for z in gcp_vals],
        "GCpi (Interno)": [gc_pi_val] * 8,
        "Presión Neta (kgf/m²)": p_netas
    })
    st.table(df_res)

with col_plt:
    areas = np.logspace(0, 1.5, 100)
    fig, ax = plt.subplots(figsize=(10, 7))
    # Curvas de Succión
    ax.plot(areas, [get_gcp(a, -1.1, -0.8) for a in areas], label='Z4 Sotavento (S)', color='green', lw=2)
    ax.plot(areas, [get_gcp(a, -1.4, -1.1) for a in areas], label='Z5 Sotavento (S)', color='red', lw=2)
    # Curvas de Empuje
    ax.plot(areas, [get_gcp(a, 1.0, 0.7) for a in areas], label='Z4 Barlovento (E)', color='green', lw=2, ls=':')
    ax.plot(areas, [get_gcp(a, 1.0, 0.8) for a in areas], label='Z5 Barlovento (E)', color='red', lw=2, ls=':')
    
    for z_v in gcp_vals:
        ax.scatter([area_ef], [z_v], color='black', s=40, zorder=10)

    ax.set_xlabel("Área Tributaria (m²)"); ax.set_ylabel("Coeficiente GCp")
    ax.axhline(0, color='black', lw=1); ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize='x-small', loc='center left', bbox_to_anchor=(1, 0.5))
    st.pyplot(fig)

# =================================================================
# 6. DISTRIBUCIÓN DE PRESIONES NETAS: SINCRONIZADO Y COMPLETO
# =================================================================
st.divider()
st.subheader("📊 Perfil de Presiones Netas (NCh 432): Barlovento vs Sotavento")

# Coeficiente típico para paredes laterales (Succión constante)
cp_lateral = -0.80 

alturas_perfil = np.linspace(0.1, H_edif, 50)
p_barlo_4, p_barlo_5 = [], [] 
p_sota_4, p_sota_5 = [], [] 
p_laterales = [] 

for z_alt in alturas_perfil:
    # Presión de velocidad variable (qz) según Kz
    kz_z = 2.01 * ((max(z_alt, 4.6) / zg)**(2/alpha))
    qz = (0.613 * imp_map[cat_imp] * kz_z * Kzt_val * Ke_val * (V**2)) * 0.10197
    
    # BARLOVENTO (+) - Empuje variable con la altura
    p_barlo_4.append(p_diseno(qz, z4_pos, gc_pi_val))
    p_barlo_5.append(p_diseno(qz, z5_pos, gc_pi_val))
    
    # SOTAVENTO (-) - Succión constante basada en qh (altura de techo)
    p_sota_4.append(p_diseno(qh, z4_neg, gc_pi_val))
    p_sota_5.append(p_diseno(qh, z5_neg, gc_pi_val))
    
    # LATERALES (-) - Succión constante basada en qh
    p_laterales.append(p_diseno(qh, cp_lateral, gc_pi_val))

# Renderizado Gráfico Profesional
fig_alt, ax_alt = plt.subplots(figsize=(12, 8))

# Graficar Barlovento (Derecha)
ax_alt.plot(p_barlo_4, alturas_perfil, label="Barlovento Z4 (Empuje Std)", color='darkgreen', lw=2)
ax_alt.plot(p_barlo_5, alturas_perfil, label="Barlovento Z5 (Empuje Esq)", color='limegreen', lw=2, ls='--')

# Graficar Sotavento (Izquierda)
ax_alt.plot(p_sota_4, alturas_perfil, label="Sotavento Z4 (Succión Std)", color='blue', lw=2)
ax_alt.plot(p_sota_5, alturas_perfil, label="Sotavento Z5 (Succión Esq)", color='darkblue', lw=3)

# RESTAURADO: Graficar Laterales (Izquierda)
ax_alt.plot(p_laterales, alturas_perfil, label="Paredes Laterales (Succión)", color='purple', ls=':', lw=2)

# Configuración del Eje y Estética
ax_alt.axvline(0, color='red', lw=2) 
ax_alt.set_title(f"Perfil de Presiones Netas de Diseño | V = {V} m/s", fontsize=14)
ax_alt.set_xlabel("Presión Neta [kgf/m²] <-- SUCCIÓN (Sotavento/Lat) | EMPUJE (Barlovento) -->", fontsize=12)
ax_alt.set_ylabel("Altura sobre N.N.T. [m]", fontsize=12)

# Ajuste dinámico de límites
max_p = max(max(p_barlo_5), abs(min(p_sota_5)), abs(min(p_laterales))) * 1.2
ax_alt.set_xlim(-max_p, max_p)

ax_alt.grid(True, which='both', ls='--', alpha=0.5)
ax_alt.legend(loc='upper right', fontsize='small', frameon=True)

st.pyplot(fig_alt)

st.write("📌 **Interpretación**: Las curvas a la izquierda representan succiones. La curva punteada morada indica la presión en las caras laterales, la cual se mantiene uniforme en toda la altura según NCh 432.")

# =================================================================
# 7. ESQUEMAS NORMATIVOS Y REFERENCIAS FINALES
# =================================================================
st.divider()
col_img1, col_img2 = st.columns(2)
with col_img1:
    st.subheader("📍 Identificación de Zonas")
    if os.path.exists("F8.png"): st.image("F8.png")
with col_img2:
    st.subheader("📍 Esquema Isométrico")
    if os.path.exists("F12.png"): st.image("F12.png")

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
        if os.path.exists("Logo.png"):
            self.image("Logo.png", x=12, y=8, h=11)
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
    pdf.multi_cell(0, 8, _txt("Memoria de cálculo - Determinación de presiones de viento según NCh 432:2025"),
                   new_x="LMARGIN", new_y="NEXT")
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
    pdf.p(f"Se determinan las presiones de diseño del viento sobre componentes y revestimientos (C&R) de un "
          f"edificio de altura media h = {_n(H_edif, 1)} m, con techo de inclinación {theta}°, para un elemento de "
          f"área efectiva {_n(area_ef, 2)} m². Se reportan la presión de velocidad, las presiones netas por zona "
          f"y el perfil de presiones en altura para barlovento, sotavento y caras laterales.")

    # --- 2. Normativa ---
    pdf.h2("2. Antecedentes y normativa")
    for li in ["NCh 432:2025 - Diseño estructural - Cargas de viento.",
               "NCh3171 - Disposiciones generales y combinaciones de carga (categoría de ocupación).",
               "Presión de velocidad según Ec. (2) y presión de diseño C&R según Ec. (19) de NCh 432:2025."]:
        pdf.p("-  " + li)

    # --- 3. Datos de entrada ---
    pdf.h2("3. Datos de entrada", reserva=85)
    filas = [
        ["Velocidad básica V", f"{_n(V, 1)} m/s"],
        ["Altura media del edificio h", f"{_n(H_edif, 2)} m"],
        ["Inclinación de techo theta", f"{theta}°"],
        ["Largo del elemento L", f"{_n(l_elem, 2)} m"],
        ["Ancho tributario ingresado", f"{_n(w_in, 2)} m"],
        ["Ancho tributario de cálculo max(b, L/3)", f"{_n(w_trib, 2)} m"],
        ["Área efectiva A", f"{_n(area_ef, 2)} m²"],
        ["Categoría de exposición", f"{cat_exp} ({exp_info[cat_exp][2]})"],
        ["Categoría de riesgo / factor I", f"{cat_imp} / {_n(factor_i, 2)}"],
        ["Clasificación del cerramiento", cerramiento_opcion],
    ]
    pdf.tabla(["Parámetro", "Valor"], filas, [120, 60])

    # --- 4. Factores ---
    pdf.h2("4. Factores de cálculo", reserva=80)
    filas_f = [
        ["Exponente de ley de potencia alfa", _n(alpha, 1)],
        ["Altura gradiente zg", f"{_n(zg, 0)} m"],
        ["Coef. de exposición Kz (z = h)", _n(kz, 3)],
        ["Factor topográfico Kzt", _n(Kzt_val, 3)],
        ["Factor de elevación del terreno Ke", _n(Ke_val, 2)],
        ["Factor de direccionalidad Kd", _n(Kd_val, 2)],
        ["Factor de importancia I", _n(factor_i, 2)],
        ["Coef. de presión interna GCpi", f"± {_n(gc_pi_val, 2)}"],
    ]
    pdf.tabla(["Factor", "Valor"], filas_f, [120, 60])
    if metodo == "Calculado":
        pdf.p(f"Kzt calculado para {tipo_relieve} con Hc = {_n(Hc, 1)} m y Lh = {_n(Lhc, 1)} m, en la cresta "
              f"(x = 0, K2 = 1,0) y z = h: K1 = {_n(k1, 3)}, K3 = {_n(k3, 3)}, Kzt = (1 + K1·K2·K3)² = {_n(Kzt_val, 3)}.")
    else:
        pdf.p(f"Kzt ingresado directamente por el usuario: {_n(Kzt_val, 3)}.")

    # --- 5. Formulación ---
    pdf.h2("5. Formulación")
    pdf.p("Coeficiente de exposición (z >= 4,6 m):")
    pdf.eq("Kz = 2,01 · (z / zg)^(2/alfa)")
    pdf.p("Presión de velocidad, Ec. (2):")
    pdf.eq("qz = 0,613 · I · Kz · Kzt · Ke · V²   [N/m²]")
    pdf.p("Presión de diseño para C&R, Ec. (19), con GCpi en el signo más desfavorable:")
    pdf.eq("p = qh · Kd · [ (GCp) - (GCpi) ]")
    pdf.p("GCp se interpola linealmente en log10(A) entre A = 1 m² y A = 10 m². "
          "Conversión de unidades: 1 N/m² = 0,10197 kgf/m².")

    # --- 6. qh ---
    pdf.h2("6. Presión de velocidad a la altura media del techo")
    pdf.eq(f"qh = 0,613 · {_n(factor_i, 2)} · {_n(kz, 3)} · {_n(Kzt_val, 3)} · {_n(Ke_val, 2)} · {_n(V, 1)}²")
    pdf.eq(f"qh = {_n(qh / 0.10197, 1)} N/m² = {_n(qh, 2)} kgf/m²")

    # --- 7. Presiones netas ---
    pdf.h2("7. Presiones netas de diseño por zona", reserva=75)
    filas_p = [[_txt(z), _n(g, 3), f"± {_n(gc_pi_val, 2)}", _n(p, 2)]
               for z, g, p in zip(zonas, gcp_vals, p_netas)]
    pdf.tabla(["Zona de análisis", "GCp", "GCpi", "p [kgf/m²]"], filas_p, [84, 30, 30, 36])
    pdf.p("Valores positivos: empuje hacia la superficie. Valores negativos: succión hacia el exterior. "
          "Las caras de sotavento y laterales se evalúan con qh, constante en toda la altura.", size=8.5, style="I")

    # --- 8. Gráficos ---
    pdf.h2("8. Gráficos de verificación")
    pdf.figura(_fig_png(fig), 165, f"Figura 1 - Coeficientes GCp de muros en función del área efectiva (A = {_n(area_ef, 2)} m² marcado)")
    pdf.figura(_fig_png(fig_alt), 175, f"Figura 2 - Perfil de presiones netas de diseño en altura, V = {_n(V, 1)} m/s")

    # --- 9. Esquemas ---
    esquemas = [(f, t) for f, t in [("F8.png", "Identificación de zonas"), ("F12.png", "Esquema isométrico")]
                if os.path.exists(f)]
    if esquemas:
        pdf.h2("9. Esquemas de zonificación")
        for i, (f, t) in enumerate(esquemas, start=3):
            pdf.figura(f, 150, f"Figura {i} - {t}")

    # --- Conclusiones ---
    pdf.h2(("10." if esquemas else "9.") + " Conclusiones")
    i_min = int(np.argmin(p_netas)); i_max = int(np.argmax(p_netas))
    pdf.p(f"-  Presión de velocidad a la altura media del techo: qh = {_n(qh, 2)} kgf/m².")
    pdf.p(f"-  Succión máxima de diseño: {_n(p_netas[i_min], 2)} kgf/m² en {_txt(zonas[i_min])}.")
    pdf.p(f"-  Empuje máximo de diseño: {_n(p_netas[i_max], 2)} kgf/m² en {_txt(zonas[i_max])}.")
    pdf.p("-  Los valores corresponden a la configuración analizada. Cualquier variación de geometría, "
          "exposición, categoría o cerramiento requiere una nueva evaluación.")

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
                       file_name=f"Memoria_Viento_{mem_codigo}_Rev{mem_rev}.pdf",
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
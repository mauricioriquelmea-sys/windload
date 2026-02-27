# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import os
import math

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

st.sidebar.info(f"**Area efectiva: {area_ef} m2**")


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

# Cálculo Presión Estática qh
qh = (0.613 * kz * Kzt_val * Kd_val * (V**2) * imp_map[cat_imp]) * 0.10197

# =================================================================
# 5. DESPLIEGUE TÉCNICO DE RESULTADOS Y FORMULACIÓN
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
st.latex(r"q_h = 0.613 \cdot K_z \cdot K_{zt} \cdot K_d \cdot V^2 \cdot I")
st.latex(r"p = q_h \cdot [GC_p - GC_{pi}]")

st.info(f"**Presión de Velocidad Calculada (qh):** {qh:.2f} kgf/m²")

# Coeficientes de las 5 Zonas (Fachada y Techo)
z1 = get_gcp(area_ef, -1.0, -0.9) if theta <= 7 else get_gcp(area_ef, -0.9, -0.8)
z2 = get_gcp(area_ef, -1.8, -1.1) if theta <= 7 else get_gcp(area_ef, -1.3, -1.2)
z3 = get_gcp(area_ef, -2.8, -1.1) if theta <= 7 else get_gcp(area_ef, -2.0, -1.2)
z4, z5 = get_gcp(area_ef, -1.1, -0.8), get_gcp(area_ef, -1.4, -1.1)

# Tabulación de Resultados
col_res, col_plt = st.columns([1, 1.3])

with col_res:
    st.markdown("**Resumen de Presiones Netas por Zona**")
    df_res = pd.DataFrame({
        "Zona de Análisis": ["Z1 (Techo Centro)", "Z2 (Techo Borde)", "Z3 (Techo Esquina)", "Z4 (Fachada Estándar)", "Z5 (Fachada Esquina)"],
        "GCp (Externo)": [round(z, 3) for z in [z1, z2, z3, z4, z5]],
        "GCpi (Interno)": [gc_pi_val] * 5,
        "Presión Neta (kgf/m²)": [round(qh*(z - gc_pi_val), 2) for z in [z1, z2, z3, z4, z5]]
    })
    st.table(df_res)
    st.warning("⚠️ Nota: Los valores negativos indican succión (presión hacia afuera).")

with col_plt:
    areas = np.logspace(0, 1, 50)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Graficar las 5 Zonas Simultáneas
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
# 6. DISTRIBUCIÓN DE PRESIONES: BARLOVENTO Y SOTAVENTO (EXPANDIDO)
# =================================================================
st.divider()
st.subheader("📊 Perfil de Presiones en Altura: Barlovento vs Sotavento")

# Generamos perfil de alturas cada 0.25m para máxima resolución
alturas_perfil = np.linspace(0.1, H_edif, 50)
p_barlovento_std = [] # Zona 4 Barlovento (Variable con Kz)
p_barlovento_esq = [] # Zona 5 Barlovento (Variable con Kz)
p_sotavento_net = []  # Sotavento (Constante según qh)
p_laterales_net = []  # Zonas laterales (Constante según qh)

# Coeficientes Cp normativos para el perfil global (Paredes)
cp_sotavento = -0.50 # Succión constante en cara posterior
cp_lateral = -0.70   # Succión constante en caras laterales

for z_alt in alturas_perfil:
    # Barlovento: qz varía con la altura (Kz)
    kz_z = 2.01 * ((max(z_alt, 4.6) / zg)**(2/alpha))
    qz = (0.613 * kz_z * Kzt_val * Kd_val * (V**2) * imp_map[cat_imp]) * 0.10197
    
    # Presiones netas en Barlovento (C&R)
    p_barlovento_std.append(qz * (z4 - gc_pi_val))
    p_barlovento_esq.append(qz * (z5 - gc_pi_val))
    
    # Sotavento y Laterales: Basados en qh a la altura H (Constantes)
    p_sotavento_net.append(qh * (cp_sotavento - gc_pi_val))
    p_laterales_net.append(qh * (cp_lateral - gc_pi_val))

# Renderizado Gráfico
fig_alt, ax_alt = plt.subplots(figsize=(12, 8))

# Líneas de presión Barlovento
ax_alt.plot(p_barlovento_std, alturas_perfil, label="Barlovento: Fachada Estándar (Z4)", color='green', lw=2)
ax_alt.plot(p_barlovento_esq, alturas_perfil, label="Barlovento: Fachada Esquina (Z5)", color='red', lw=3)

# Líneas de succión Sotavento y Laterales
ax_alt.plot(p_sotavento_net, alturas_perfil, label="Sotavento (Succión Constante)", color='blue', ls='-.', lw=2)
ax_alt.plot(p_laterales_net, alturas_perfil, label="Paredes Laterales (Succión Constante)", color='purple', ls=':', lw=2)

# Sombreados para visualización de magnitud
ax_alt.fill_betweenx(alturas_perfil, p_barlovento_std, 0, color='green', alpha=0.1)
ax_alt.fill_betweenx(alturas_perfil, p_sotavento_net, 0, color='blue', alpha=0.05)

# Configuración técnica del gráfico
ax_alt.axvline(0, color='black', lw=1.5)
ax_alt.set_title(f"Distribución Vertical de Presiones Netas (NCh 432) | V = {V} m/s", fontsize=14)
ax_alt.set_ylabel("Altura sobre N.N.T. (m)", fontsize=12)
ax_alt.set_xlabel("Presión Neta de Diseño (kgf/m²) [Succión < 0 | Empuje > 0]", fontsize=12)
ax_alt.grid(True, which="both", ls="--", alpha=0.4)
ax_alt.legend(loc='lower left', frameon=True, shadow=True, fontsize='medium')

st.pyplot(fig_alt)

with st.expander("📑 Ver Nota Técnica de Distribución"):
    st.markdown(f"""
    **Análisis Cinematográfico del Viento:**
    * **Barlovento (Lado Expuesto):** La presión es escalonada. El factor de exposición $K_z$ aumenta con la altura siguiendo una ley de potencia, lo que resulta en mayores cargas en los niveles superiores de la fachada.
    * **Sotavento (Lado Protegido):** Según la norma, la succión es uniforme a lo largo de toda la altura y se calcula utilizando la presión de velocidad $q_h$ evaluada en el punto más alto del edificio ($H = {H_edif} m$).
    * **Paredes Laterales:** Siguen un comportamiento similar al sotavento pero con coeficientes de succión ($C_p$) usualmente más severos.
    """)


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
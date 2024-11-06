import streamlit as st 
#import fastapi as Ft
#import fastapi as FastAPI()

from utils import cargar_modelo_preentrenado, leer_dato, predecir, obtener_categoria
from utils import *
from utils import Autoencoder

#applica = Ft()

#@applica.get('/')


st.set_page_config(page_icon = "+++++", page_title="Detección de Anomalías Cardiacas", layout="centered")
st.title("Detección de Anomalias Cardiacas con Autoencoders")

c29 , c30 = st.columns([20,80])

UMBRAL = 0.089

with c30:
    uploaded_file = st.file_uploader("", type='pkl', key="1")


    if uploaded_file is not None:
        file_container = st.expander("Verifique el archivo .pkl que acaba de subir")

        info_box_wait = st.info(
               f"""
                    Realizando la clasificación...
                    """)

        #Aca viene la predicción con el modelo
        dato = leer_dato(uploaded_file)
        autoencoder = Autoencoder()
        autoencoder = cargar_modelo_preentrenado('autoencoder.pth')
        prediccion = predecir(autoencoder, dato, UMBRAL)
        categoria = obtener_categoria(prediccion)

        
        # Y mostrar el resultado
        info_box_wait = st.info(f"""
                                El dato analizado corresponde a un paciente que presenta un cuadro cardiaco: {categoria} 
                                """ 
                                )
        
    else:
        st.info(
            f"""
            Debe cargar primero un dato con extensión punto  pkl
            """
            )
        
        st.stop()
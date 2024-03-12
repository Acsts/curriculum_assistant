import streamlit as st
import os
import requests

API_URL = "http://back:80"

def answer_question(question=""):

    if question=="":
        return "En attente d'une question à propos du candidat."

    try:
        r = requests.get(API_URL + "/answer_question", params={'question': question})
        return r.text

    except requests.exceptions.ConnectionError:
        return "***Erreur: Impossible de se connecter à l'API***"

if __name__ == '__main__':

    st.markdown("Posez une question sur le candidat:")
    text_input = st.text_input("Posez une question sur le candidat:",help = 'Seuls les informations en base sont utilisées',
        key = 'text_input_content',
        )

    st.write(answer_question(question=text_input))

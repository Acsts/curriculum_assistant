from fastapi import FastAPI
import numpy as np
import pandas as pd
import tiktoken
from openai import OpenAI
from scipy.spatial.distance import cosine
import os
import logging

app = FastAPI(
    title="Candidates information",
)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
    )

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

def get_data():
    # Create a list to store the text files
    texts=[]

    # Get all the text files in the text directory
    for file in os.listdir("text"):

        # Open the file and read the text
        with open("text/" + file, "r", encoding="UTF-8") as f:
            text = f.read()
            texts.append((file.replace('-',' ').replace('_', ' '), text))

    # Create a dataframe from the list of texts
    df = pd.DataFrame(texts, columns = ['fname', 'text'])

    # Set the text column to be the raw text with the newlines removed
    df['text'] = df.fname + ". " + remove_newlines(df.text)

    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    df.columns = ['title', 'text']

    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    # Apply embeddings
    df['embeddings'] = df.text.apply(lambda x: client.embeddings.create(input=x, model='text-embedding-ada-002').data[0].embedding)
    df['embeddings'] = df['embeddings'].apply(np.array)

    return df

def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = client.embeddings.create(input=question, model='text-embedding-ada-002').data[0].embedding

    # Get the distances from the embeddings
    df['distances'] = df['embeddings'].apply(lambda x: cosine(q_embeddings, x))
    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        print(f"Best distance: {row.loc['distances']}")

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def ask_question(
    df,
    model="gpt-3.5-turbo",
    question="",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """

    if question=="":
        return "En attente d'une question à propos du candidat."

    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a chat completion using the question and context
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content":
                 """Réponds à la question en français en te basant sur le contexte ci-dessous.
                 Tu ne peux pas demander d'informations complémentaires, et dois répondre en une fois.
                 Si la réponse n'a vraiment pas l'air de se trouver dans le contexte fourni, répondre 'Je ne sais pas'.
                 En cas de doute là-dessus, commencer la réponse par 'Il me semble que'\n\n"""},
                {"role": "user", "content": f"Contexte: {context}\n\n---\n\nQuestion: {question}\nRéponse:"}
            ],
            temperature=0.5,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return "Error !"

df = get_data()

@app.get("/")
def read_root():
    return {"Message": "Go to /docs endpoint to test the API"}

@app.get("/answer_question")
async def answer_question(question: str):
    return ask_question(df, question=question)



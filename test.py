from tensorflow.keras.models import load_model
import numpy as np
import pickle
model=load_model('next_words.h5')
tokenizer=pickle.load(open('token.pkl','rb'))

def predict(model,tokenizer,text):
    sequence=tokenizer.texts_to_sequences([text]) 
    sequence=np.array(sequence)
    preds=np.argmax(model.predict(sequence))
    predicted_word=""
    for key ,val in tokenizer.word_index.items():
        if val== preds:
            predicted_word=key
            break
    print(predicted_word)
    return predicted_word 

while True:
    text=input("ENTER YOUR LINE:")   # Fixed: moved input to variable 'text'
    if text=='1':
        break
    else:
        text=text.split(" ")
        text=text[-3:]
        predict(model,tokenizer,text)  # Fixed: removed assignment and fixed function call syntax
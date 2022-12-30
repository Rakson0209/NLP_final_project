import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

model_checkpoint = "yihsuan/mt5_chinese_small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

model.load_state_dict(torch.load('./models/epoch_9_valid_rouge_42.3221_model_weights.bin'))
model = model.to(device)

import streamlit as st
# command in termianl: 'streamlit run streamlit.py'
# change somewhere in streamlit.py, then you can see the "always rerun" at the top-right of your screen
# choose "always rerun" to automatically update your app every time you change its source code.

# mainpage
st.title("Title Generator")
article_text = st.text_area('input the article:')

input_ids = tokenizer(
    article_text,
    return_tensors="pt",
    truncation=True,
    max_length=512
)
input_ids = input_ids.to(device)

# sidebar
st.sidebar.title('Model Parameters') 
num = st.sidebar.slider('number of titles to generate', 0, 10)
temp = st.sidebar.slider('temperature', 0.10, 1.50)
st.sidebar.info('high temperature means that results are more random')

# mainpage
genarate = st.button('generate title')
st.subheader('Generated titles:')

def run(num, temp):
    result = []
    for i in range(num):
        generated_tokens = model.generate(
            input_ids["input_ids"],
            attention_mask=input_ids["attention_mask"],
            max_length=32,
            # no_repeat_ngram_size=2,
            num_beams=8,
            do_sample=True,
            temperature=temp,
        )

        summary = tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        result.append(summary)
    return result

if genarate == True:
    if article_text == '':
      st.error("input the article !!")
    else:
      for i in range(num):
        st.markdown('**title %d :**' % (i), run(num, temp))

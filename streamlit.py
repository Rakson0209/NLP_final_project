import streamlit as st
# command in termianl: 'streamlit run streamlit.py'
# change somewhere in streamlit.py, then you can see the "always rerun" at the top-right of your screen
# choose "always rerun" to automatically update your app every time you change its source code.

st.title("Title Generator")

content = st.text_area('input the article:')

st.button('generate title')
st.subheader('Generated titles:')
st.markdown('**title 1**')
st.markdown('**title 2**')

st.sidebar.title('Model Parameters') 
values = st.sidebar.slider('number of titles to generate', 0, 10)
values = st.sidebar.slider('temperature', 0.10, 1.50)
st.sidebar.write('high temperature means that results are more random')

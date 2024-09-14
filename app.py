import streamlit as st
import numpy as np
import pandas as pd

st.title('Sunil Javadi website')
st.header("this is header")
st.subheader("this is practice")
st.write('This is a simple Streamlit app.')
st.select_slider("pick",['a','b','c','d'])
st.checkbox("check")
st.button("button")

st.radio("A",['a','b'])
st.slider('slider',0,100)
select=st.selectbox('select box',['A','B','C','D'])
st.write('you selected',select)
st.multiselect('multi select box',['A','B','C','D','E'])
st.image(r'C:\Users\javadisu\OneDrive - Vertafore, Inc\sunil\python pr\Screenshot.png')
on=st.toggle("Toggle button")
if on:
    st.write('toggle activated')
else:
    st.write('please on toggle')
number=st.number_input('enter number:')
st.write('the number input is',number)
dp=st.date_input('give the date',value=None)
st.write('the given date',dp)
tp=st.time_input('give the time',value=None)
st.write('the given time',tp)
st.file_uploader('upload a file:')
st.number_input('enter between 1 and 100',1,100)
t=st.text_input("enter text")
st.write('text is',t)
st.sidebar.title('sidebar')

df = pd.DataFrame({
    'lat': [18.8714],
    'lon': [79.4443]
})
st.map(df)

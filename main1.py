##
## Chat PDF 만들기
## 2023.10.03
##
## main.py를 이용하여 Streamlit 서비스 만들기
##
##

#env 환경변수 읽어 들이기(같은 폴더 "/.env"의 파일을 그대로 읽어옴)
#from dotenv  import load_dotenv
#load_dotenv()

#split 와 chroma DB 간의 충돌 해결
#sqlite3 문제 해결코드
#__import__('pysqlite3')
#import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  #텍스트를 페이지 단위보다 더 Split하기
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA

from langchain.embeddings import OpenAIEmbeddings

import streamlit as st
import tempfile
import os

###Loader PDF file
#loader = PyPDFLoader("unsu.pdf")
#pages = loader.load_and_split()  #페이별로 쪼개기

##Front page 꾸미기
st.title("Chat Pdf") 
st.write("---")
uploaded_file = st.file_uploader("Choose a file", type=['pdf'])
st.write("---")

##Streamit 에서 파일을 업로드 받기 Function
def pdf_to_document(uploaded_file):
   temp_dir = tempfile.TemporaryDirectory()
   temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
   with open(temp_filepath, "wb") as f:
      f.write(uploaded_file.getvalue())
   loader = PyPDFLoader(temp_filepath)
   pages = loader.load_and_split()
   return pages

#업로드되면 동작하는 코드
if uploaded_file is not None:
   pages = pdf_to_document(uploaded_file)

   #Recursively split by characters 를 사용함
   ###Split 
   text_splitter = RecursiveCharacterTextSplitter(
      # Set a really small chunk size, just to show.
         chunk_size = 300, #자르는 크기
         chunk_overlap  = 20, #split 시 각 단위별 겹침의 크기
         length_function = len,
         is_separator_regex = False,
   )   
   
   texts = text_splitter.split_documents(pages)  #페이지별로 더 쪼개기

   #AIOpenAIEmbeddings 모델을 사용함
   embeddings_model = OpenAIEmbeddings()
   
   # load it into Chroma (랭체인 문서 참조)
   ## python langchain Doc -> Modules -> Vectorstores -> Chroma 참조
   ## Basic Example 에 Disk 저장하는 예제도 있음
   db = Chroma.from_documents(texts, embeddings_model) ## 메모리에 저장하기
   #Disk에 저장하는 예제
   #db = Chroma.from_documents(texts, embeddings_model, persist_directory="./chroma_db")

   #Question
   st.header("PDF에게 질문해 보세요!!")
   question = st.text_input("질문을 입력하세요")

   if st.button("질문하기"):
      with st.spinner('잠시 기다려주세요...!!') : 
         ## python langchain Doc -> USE cases -> Question Answering 참조
         llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
         qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
         result=qa_chain({"query": question})

         st.write(result['result'])
         st.write("---")
   pass



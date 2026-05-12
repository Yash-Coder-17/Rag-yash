# import langchain
# print(langchain.__version__)

# import google.generativeai as genai
# import os 
# from dotenv import load_dotenv
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# for m in genai.list_models():
#     print(m.name)

import sounddevice as sd
y= sd.query_devices()
print(y)
import streamlit as st
import tensorflow as tf
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import PIL
import os
from PIL import Image, ImageOps
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from oauth2client.service_account import ServiceAccountCredentials
from io import BytesIO
from googleapiclient.http import MediaIoBaseUpload

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('model_v2b2.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()


SCOPES  = ['https://www.googleapis.com/auth/drive.file']
SERVICE_ACCOUNT_FILE  = 'file.json'

st.write("<h1 style='text-align: center; background-color: #ebccff; color: #990033;'>Tomato Leaf Diseases Detection</h1>", unsafe_allow_html=True)
st.write("<h1 style='text-align: center; background-color: #ebccff; color: #5c0099;'>টমেটো পাতার রোগ নির্ণয়</h1>", unsafe_allow_html=True)


file = st.file_uploader("Please upload an Tomato leaf image file. / একটি টমেটো পাতার ছবি আপলোড করুন", type=["jpg", "png", "jpeg"])


st.set_option('deprecation.showfileUploaderEncoding', False)



if file is None:
    st.text("No tomato leaf image is selected\nকোনো টমেটো পাতার ছবি নির্বাচন করা হয়নি")
else:

    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    drive_service = build('drive', 'v3', credentials=credentials)

    # Upload the file to Google Drive
    file_metadata = {'name': file.name, 'parents': ['1ps9JTqK1N1HXVRdmQnLoeKVP1Dam4JuK']}
    media = MediaIoBaseUpload(BytesIO(file.read()), mimetype=file.type)
    response = drive_service.files().create(
        body=file_metadata, media_body=media, fields='id'
    ).execute()
  
    image = Image.open(file)
    st.image(image, use_column_width=True)

    image = image.resize((256,256), resample=PIL.Image.BICUBIC)
    img_arr = img_to_array(image)
    img_arr = np.expand_dims(img_arr, axis=0)
    pred = model.predict(img_arr)

    class_names = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
    # Get the index of the maximum probability
    pred_index = np.argmax(pred)

    # Map the index to the corresponding class name
    pred_class = class_names[pred_index]
    
    

    st.write(f"<div style='text-align: center;'><h3>Result = {pred_class}</h3></div>", unsafe_allow_html=True)


    if pred_class == 'Tomato___Bacterial_spot':
        
        st.write("<h3 style='text-align: center;'>ফলাফল : ছবিটি হলো টমেটো ব্যাকটেরিয়াল স্পট </h3>", unsafe_allow_html=True)

        st.text("\n\nSolution : \n1.Rotate tomato crops with non-host plants.\n2.Remove and destroy infected plant debris.\n3.Maintain proper plant spacing for good air circulation.\n4.Avoid overhead watering and water at the base of the plants instead.")
        st.text("\n\nপ্রতিকার :\n১.নন-হোস্ট গাছের সাথে টমেটো ফসল ঘোরান।\n২.সংক্রামিত উদ্ভিদ ধ্বংসাবশেষ অপসারণ এবং ধ্বংস।\n৩.ভাল বায়ু সঞ্চালনের জন্য উদ্ভিদের সঠিক ব্যবধান বজায় রাখুন।\n৪.পরিবর্তে গাছের গোড়ায় ওভারহেড জল এবং জল এড়িয়ে চলুন।\n")
        
    elif pred_class == 'Tomato___Early_blight':
        
        st.write("<h3 style='text-align: center;'>ফলাফল : ছবিটি হলো টমেটো আরলি ব্লাইট </h3>", unsafe_allow_html=True)

        st.text("\n\nSolution : \n1.Rotate tomato crops.\n2.Keep foliage dry by watering at the base.\n3.Apply fungicides as needed.")
        st.text("\n\nপ্রতিকার :\n১.টমেটো ফসল ঘোরান।\n২.গোড়ায় জল দিয়ে পাতা শুকিয়ে রাখুন।\n৩.প্রয়োজনে ছত্রাকনাশক প্রয়োগ করুন।\n")


    elif pred_class == 'Tomato___Late_blight':
        
        st.write("<h3 style='text-align: center;'>ফলাফল : ছবিটি হলো টমেটো লেট ব্লাইট</h3>", unsafe_allow_html=True)
         
        st.text("\n\nSolution : \n1.Provide good air circulation by spacing plants properly.\n2.Avoid overhead watering to keep foliage dry.\n3.Apply fungicides regularly, following the recommended schedule.\n4.Remove and destroy infected plant parts promptly.")
        st.text("\n\nপ্রতিকার :\n১.সঠিকভাবে গাছপালা ফাঁক করে ভাল বায়ু সঞ্চালন প্রদান।\n২.পাতা শুষ্ক রাখতে ওভারহেড জল দেওয়া এড়িয়ে চলুন।\n৩.সুপারিশকৃত সময়সূচী অনুসরণ করে নিয়মিত ছত্রাকনাশক প্রয়োগ করুন।\n৪.সংক্রামিত গাছের অংশগুলি দ্রুত সরিয়ে ফেলুন এবং ধ্বংস করুন।\n")


    elif pred_class == 'Tomato___Leaf_Mold':
        
        st.write("<h3 style='text-align: center;'>ফলাফল : ছবিটি হলো টমেটো লিফ মোল্ড </h3>", unsafe_allow_html=True)
                 
        st.text("\n\nSolution : \n1.Provide proper plant spacing for good air circulation.\n2.Water at the base of the plants and avoid wetting the foliage.\n3.Apply fungicides labeled for leaf mold control.\n4.Remove and destroy infected leaves to reduce the spread of the disease.")
        st.text("\n\nপ্রতিকার :\n১.ভাল বায়ু সঞ্চালনের জন্য উদ্ভিদের সঠিক ব্যবধান প্রদান করুন।\n২.গাছের গোড়ায় জল দিন এবং পাতা ভেজা এড়িয়ে চলুন।\n৩.পাতার ছাঁচ নিয়ন্ত্রণের জন্য লেবেলযুক্ত ছত্রাকনাশক প্রয়োগ করুন।\n৪.রোগের বিস্তার কমাতে আক্রান্ত পাতা অপসারণ ও ধ্বংস করুন।\n")


    elif pred_class == 'Tomato___Septoria_leaf_spot':
        
        st.write("<h3 style='text-align: center;'>ফলাফল : ছবিটি হলো টমেটো সেপ্টোরিয়াল লিফ স্পট </h3>", unsafe_allow_html=True)
                         
        st.text("\n\nSolution : \n1.Practice crop rotation to reduce disease buildup.\n2.Water at the base of plants, avoiding foliage wetting.\n3.Apply fungicides labeled for septoria leaf spot control.\n4.Remove and destroy infected leaves to limit disease spread.")
        st.text("\n\nপ্রতিকার :\n১.রোগের বৃদ্ধি কমাতে ফসল ঘোরানোর অনুশীলন করুন।\n২.গাছের গোড়ায় জল, পাতা ভেজা এড়ানো।\n৩.সেপ্টোরিয়া পাতার দাগ নিয়ন্ত্রণের জন্য লেবেলযুক্ত ছত্রাকনাশক প্রয়োগ করুন।\n৪.রোগের বিস্তার সীমিত করতে সংক্রামিত পাতা অপসারণ ও ধ্বংস করুন।\n")


    elif pred_class == 'Tomato___Spider_mites Two-spotted_spider_mite':
        
        st.write("<h3 style='text-align: center;'>ফলাফল : ছবিটি হলো টমেটো স্পাইডার মাইটস টু-স্পটেড স্পাইডার মাইট </h3>", unsafe_allow_html=True)
                                 
        st.text("\n\nSolution : \n1.Regularly inspect plants for early signs of infestation, such as tiny webbing and yellowing leaves.\n2.Increase humidity levels by misting plants regularly, as spider mites thrive in dry conditions.\n3.Introduce natural predators like ladybugs or use insecticidal soap or neem oil to control spider mites.\n4.Avoid over-fertilizing plants, as excessive nitrogen can attract spider mites.")
        st.text("\n\nপ্রতিকার :\n১.সংক্রমণের প্রাথমিক লক্ষণগুলির জন্য নিয়মিতভাবে উদ্ভিদ পরিদর্শন করুন, যেমন ছোট জাল এবং হলুদ পাতা।\n২.স্পাইডার মাইটস শুষ্ক অবস্থায় বেড়ে ওঠার কারণে নিয়মিত গাছপালা মিস্টিং করে আর্দ্রতার মাত্রা বাড়ান।\n৩.লেডিবাগের মতো প্রাকৃতিক শিকারী প্রাণীর পরিচয় দিন বা স্পাইডার মাইটস নিয়ন্ত্রণ করতে কীটনাশক সাবান বা নিম তেল ব্যবহার করুন।\n৪.অতিরিক্ত নিষিক্ত উদ্ভিদ এড়িয়ে চলুন, কারণ অতিরিক্ত নাইট্রোজেন স্পাইডার মাইটসকে আকর্ষণ করতে পারে।\n")


    elif pred_class == 'Tomato___Target_Spot':
        
        st.write("<h3 style='text-align: center;'>ফলাফল : ছবিটি হলো টমেটো টারগেট স্পট </h3>", unsafe_allow_html=True)
                                         
        st.text("\n\nSolution : \n1.Practice crop rotation to reduce disease recurrence.\n2.Ensure good air circulation by proper plant spacing.\n3.Avoid overhead watering and water at the base of the plants.\n4.Apply fungicides labeled for target spot control as needed.")
        st.text("\n\nপ্রতিকার :\n১.রোগের পুনরাবৃত্তি কমাতে ফসলের ঘূর্ণন অনুশীলন করুন।\n২.উদ্ভিদের সঠিক ব্যবধান দ্বারা ভাল বায়ু সঞ্চালন নিশ্চিত করুন।\n৩.গাছের গোড়ায় ওভারহেড জল এবং জল এড়িয়ে চলুন।\n৪.প্রয়োজনে লক্ষ্যস্থল নিয়ন্ত্রণের জন্য লেবেলযুক্ত ছত্রাকনাশক প্রয়োগ করুন।\n")


    elif pred_class == 'Tomato___Tomato_Yellow_Leaf_Curl_Virus':
        
        st.write("<h3 style='text-align: center;'>ফলাফল : ছবিটি হলো টমেটো ইয়েলো লিফ কার্ল ভাইরাস </h3>", unsafe_allow_html=True)
                                                 
        st.text("\n\nSolution : \n1.Plant resistant/tolerant tomato varieties.\n2.Control whiteflies, the vector of the virus, using sticky traps or insecticides.\n3.Remove and destroy infected plants to prevent further spread.\n4.Implement good weed control practices as weeds can serve as alternative hosts for the virus.")
        st.text("\n\nপ্রতিকার :\n১.রোপণ প্রতিরোধী/সহনশীল টমেটোর জাত।\n২.আঠালো ফাঁদ বা কীটনাশক ব্যবহার করে হোয়াইটফ্লাই, ভাইরাসের বাহক নিয়ন্ত্রণ করুন।\n৩.আরও বিস্তার রোধ করতে সংক্রামিত গাছগুলি সরিয়ে ফেলুন এবং ধ্বংস করুন।\n৪.ভাল আগাছা নিয়ন্ত্রণ অনুশীলন প্রয়োগ করুন কারণ আগাছা ভাইরাসের বিকল্প হোস্ট হিসাবে কাজ করতে পারে।\n")


    elif pred_class == 'Tomato___Tomato_mosaic_virus':
      
        st.write("<h3 style='text-align: center;'>ফলাফল : ছবিটি হলো টমেটো মোজাইক ভাইরাস </h3>", unsafe_allow_html=True)
                                                         
        st.text("\n\nSolution : \n1.Plant disease-resistant tomato varieties.\n2.Practice good hygiene by disinfecting tools and washing hands after handling infected plants.\n3.Control aphids and other insect vectors that can transmit the virus.\n4.Remove and destroy infected plants to prevent the spread of the virus to healthy plants.")
        st.text("\n\nপ্রতিকার :\n১.রোগ প্রতিরোধী টমেটোর জাত।\n২.জীবাণুনাশক সরঞ্জাম এবং সংক্রামিত গাছগুলি পরিচালনা করার পরে হাত ধোয়ার মাধ্যমে ভাল স্বাস্থ্যবিধি অনুশীলন করুন।\n৩.এফিড এবং অন্যান্য পোকা ভেক্টর নিয়ন্ত্রণ করুন যা ভাইরাস সংক্রমণ করতে পারে।\n৪.সুস্থ উদ্ভিদে ভাইরাসের বিস্তার রোধ করতে সংক্রামিত গাছপালা অপসারণ ও ধ্বংস করুন।\n")


    elif pred_class == 'Tomato___healthy':
        
        st.write("<h3 style='text-align: center;'>ফলাফল : ছবিটি হলো টমেটো হেলদি </h3>", unsafe_allow_html=True)
                                                                 
        st.text("\n\nTo maintain healthy tomato plants : \n1.Provide adequate sunlight, water, and nutrient-rich soil.\n2.Monitor plants regularly for signs of pests, diseases, or nutrient deficiencies.\n3.Prune tomato plants to promote air circulation and remove diseased or damaged foliage.\n4.Practice proper watering techniques, avoiding both under and overwatering.")
        st.text("\n\nসুস্থ টমেটো গাছ বজায় রাখতে :\n১.পর্যাপ্ত সূর্যালোক, জল, এবং পুষ্টি সমৃদ্ধ মাটি প্রদান করুন।\n২.কীটপতঙ্গ, রোগ বা পুষ্টির ঘাটতির লক্ষণগুলির জন্য নিয়মিত গাছগুলি পর্যবেক্ষণ করুন।\n৩.বায়ু সঞ্চালন বাড়াতে এবং রোগাক্রান্ত বা ক্ষতিগ্রস্ত পাতা অপসারণ করতে টমেটো গাছ ছাঁটাই করুন।\n৪.সঠিক জল দেওয়ার কৌশলগুলি অনুশীলন করুন, জলের নীচে এবং অতিরিক্ত জল উভয়ই এড়িয়ে চলুন।\n")
      





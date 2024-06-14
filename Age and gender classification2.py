#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm.notebook import tqdm


# In[2]:


warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input


# In[4]:


base_dir="C:/Users/deepa/Downloads/archive/UTKFace/"
image_path=[]
age_label=[]
gender_label=[]

for filename in tqdm(os.listdir(base_dir)):
    image_p=os.path.join(base_dir,filename)
    temp=filename.split('_')
    age=int(temp[0])
    gender=int(temp[1])
    image_path.append(image_p)
    age_label.append(age)
    gender_label.append(gender)


# In[5]:


#convert to dataframe
df=pd.DataFrame()
df['image'],df['age'],df['gender']=image_path,age_label,gender_label
df.head()


# In[6]:


#dictionary for gender
gender_dict={0:'Male',1:'Female'}


# Exploratory Data Analysis

# In[7]:


from PIL import Image
img=Image.open(df['image'][0])
plt.axis('off')
plt.imshow(img);
sns.displot(df['age'])


# In[8]:


sns.displot(df['gender'])


# In[9]:


plt.figure(figsize=(25,25))
files=df.iloc[0:25]
for index,file,age,gender in files.itertuples():
    plt.subplot(5,5,index+1)
    img=load_img(file)
    img=np.array(img)
    plt.imshow(img)
    plt.title(f"Age: {age} Gender: {gender_dict[gender]}")
    plt.axis("off")


# In[10]:


##feature extraction
def extract_feature(images):
    features=[]
    for image in tqdm(images):
        img=load_img(image,grayscale=True)
        img=img.resize((128,128), Image.ANTIALIAS)
        img=np.array(img)
        features.append(img)
    features=np.array(features)
    features=features.reshape(len(features),128,128,1)
    return features


# In[11]:


x=extract_feature(df['image'])


# In[12]:


x.shape


# In[13]:


x=x/255.0
y_gender=np.array(df['gender'])
y_age=np.array(df['age'])
input_shape = (128,128,1) #it will be 3 if rgb


# MODEL CREATION 

# In[14]:


inputs=Input((input_shape))
#convulation layers 
conv_1=Conv2D(32,kernel_size=(3,3),activation="relu" )(inputs)
mxp_1=MaxPooling2D(pool_size=(2,2))(conv_1)
conv_2=Conv2D(64,kernel_size=(3,3),activation="relu" )(mxp_1)
mxp_2=MaxPooling2D(pool_size=(2,2))(conv_2)
conv_3=Conv2D(128,kernel_size=(3,3),activation="relu" )(mxp_2)
mxp_3=MaxPooling2D(pool_size=(2,2))(conv_3)
conv_4=Conv2D(256,kernel_size=(3,3),activation="relu" )(mxp_3)
mxp_4=MaxPooling2D(pool_size=(2,2))(conv_4)

flatten= Flatten()(mxp_4)

#fully connected layers
dense_1=Dense(256,activation="relu")(flatten)
dense_2=Dense(256,activation="relu")(flatten)

dropout_1=Dropout(0.3)(dense_1)
dropout_2=Dropout(0.3)(dense_2)

output_1=Dense(1, activation="sigmoid", name= "gender_out")(dropout_1)
output_2=Dense(1,activation="relu", name="age_out")(dropout_2)

model=Model(inputs=[inputs],outputs=[output_1,output_2])
model.compile(loss=["binary_crossentropy", "mae"],optimizer="adam", metrics=["accuracy"])


# In[15]:


model.summary()


# In[ ]:


#train model
history=model.fit(x=x,y=[y_gender,y_age],batch_size=32,epochs=30,validation_split=0.2)


# In[ ]:


#plot result for gender

acc = history.history['gender_out_accuracy']
val_acc = history.history['val_gender_out_accuracy']
epochs = range(len(acc))
plt.plot(epochs,acc,'b',label='Training accuracy')
plt.plot(epochs,val_acc,'r',label='Validation accuracy')
plt.title("Accuracy graph")
plt.legend()
plt.figure()

loss= history.history['gender_out_loss']
val_loss = history.history['val_gender_out_loss']
plt.plot(epochs,loss,'b',label='Training loss')
plt.plot(epochs,val_loss,'r',label='Validation loss')
plt.title("Loss graph")
plt.legend()
plt.figure()
plt.show()


# In[ ]:


#plot resukts for age

loss = history.history['age_out_loss']
val_loss = history.history['val_age_out_loss']
epochs = range(len(acc))
plt.plot(epochs,loss,'b',label='Training Loss')
plt.plot(epochs,val_loss,'r',label='Validation Loss')
plt.title("Loss graph")
plt.legend()
plt.figure()


# In[ ]:


image_index=100
print("Original gender:",gender_dict[y_gender[image_index]],"Original age",y_age[image_index])

pred=model.predict(x[image_index].reshape(1,128,128,1))
pred_gender=gender_dict[round(pred[0][0][0])]
pred_age=round(pred[1][0][0])
print("Predicted gender:",pred_gender, "Predicted age:",pred_age)
plt.imshow(x[image_index].reshape(128,128),cmap="gray");


# In[ ]:


image_index=10025
print("Original gender:",gender_dict[y_gender[image_index]],"Original age",y_age[image_index])

pred=model.predict(x[image_index].reshape(1,128,128,1))
pred_gender=gender_dict[round(pred[0][0][0])]
pred_age=round(pred[1][0][0])
print("Predicted gender:",pred_gender, "Predicted age:",pred_age)
plt.imshow(x[image_index].reshape(128,128),cmap="gray");


# In[ ]:


image_index=3456
print("Original gender:",gender_dict[y_gender[image_index]],"Original age",y_age[image_index])

pred=model.predict(x[image_index].reshape(1,128,128,1))
pred_gender=gender_dict[round(pred[0][0][0])]
pred_age=round(pred[1][0][0])
print("Predicted gender:",pred_gender, "Predicted age:",pred_age)
plt.imshow(x[image_index].reshape(128,128),cmap="gray");


# In[ ]:


image_index=3050
print("Original gender:",gender_dict[y_gender[image_index]],"Original age",y_age[image_index])

pred=model.predict(x[image_index].reshape(1,128,128,1))
pred_gender=gender_dict[round(pred[0][0][0])]
pred_age=round(pred[1][0][0])
print("Predicted gender:",pred_gender, "Predicted age:",pred_age)
plt.imshow(x[image_index].reshape(128,128),cmap="gray");


# In[ ]:


import pickle
with open("model_age_and_gender",'wb')as f:
    pickle.dump(model,f)


# In[ ]:


image_index=150
print("Original gender:",gender_dict[y_gender[image_index]],"Original age",y_age[image_index])

pred=model.predict(x[image_index].reshape(1,128,128,1))
pred_gender=gender_dict[round(pred[0][0][0])]
pred_age=round(pred[1][0][0])
print("Predicted gender:",pred_gender, "Predicted age:",pred_age)
plt.imshow(x[image_index].reshape(128,128),cmap="gray");


# In[ ]:


image_index=12025
print("Original gender:",gender_dict[y_gender[image_index]],"Original age",y_age[image_index])

pred=model.predict(x[image_index].reshape(1,128,128,1))
pred_gender=gender_dict[round(pred[0][0][0])]
pred_age=round(pred[1][0][0])
print("Predicted gender:",pred_gender, "Predicted age:",pred_age)
plt.imshow(x[image_index].reshape(128,128),cmap="gray");


# In[ ]:


def process_and_predict(file):
    im = Image.open(file)
    width, height = im.size
    if width == height:
        im = im.resize((200,200), Image.ANTIALIAS)
    else:
        if width > height:
            left = width/2 - height/2
            right = width/2 + height/2
            top = 0
            bottom = height
            im = im.crop((left,top,right,bottom))
            im = im.resize((200,200), Image.ANTIALIAS)
        else:
            left = 0
            right = width
            top = 0
            bottom = width
            im = im.crop((left,top,right,bottom))
            im = im.resize((200,200), Image.ANTIALIAS)
    plt.imshow(im)
    ar = np.asarray(im)
    ar = ar.astype('float32')
    ar /= 255.0
    ar = ar.reshape(-1, 200, 200, 1)
    
    pred=model.predict(ar)
    pred_gender=gender_dict[round(pred[0][0][0])]
    pred_age=round(pred[1][0][0])
    print("Predicted gender:",pred_gender, "Predicted age:",pred_age)
    plt.imshow(im);
   
    
    


# In[ ]:


image_index=12025
print("Original gender:",gender_dict[y_gender[image_index]],"Original age",y_age[image_index])

pred=model.predict(x[image_index].reshape(1,128,128,1))
pred_gender=gender_dict[round(pred[0][0][0])]
pred_age=round(pred[1][0][0])
print("Predicted gender:",pred_gender, "Predicted age:",pred_age)
plt.imshow(x[image_index].reshape(128,128),cmap="gray");


# In[ ]:





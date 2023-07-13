#!/usr/bin/env python
# coding: utf-8

# In[32]:


pip install opencv-python


# In[33]:


pip install Unidecode


# In[34]:


pip install tensorflow


# In[35]:


#pip install git+https://github.com/okankop/vidaug   #augmentation


# In[36]:


import os
import re
import cv2
import math
import random
import unidecode
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from vidaug import augmentors as va
from sklearn.model_selection import train_test_split


# ## Dataframe creation

# In[6]:


pwd


# In[7]:


#video folder path 
#change according to user
main_p= '/Users/mahelwimaladasa/Desktop/splits'
#main_p= '/Users/mahelwimaladasa/Desktop/splits'


# In[8]:


directories=[]
endings=[]
for folder in os.listdir(main_p):
    for root, dirs, files in os.walk(os.path.join(main_p,'',folder)):
        #print(dirs,len(files))
        if len(dirs) == 0:
            directories.append(root)
            endings.append(root.rsplit('/',1)[1])
        else:
            for name in dirs:
                directories.append(os.path.join(root, name))
                endings.append('/'.join(os.path.join(root, name).rsplit('/',1)[1:]))
                #print(os.path.join(root, name))


# In[9]:


s_dir=list(set(directories))
s_dir.sort()
s_dir


# In[13]:


def create_df (directories_list):
    train_id=[]
    test_id=[]
    labels={}
    for directory in directories_list:
        os.chdir(directory)
        if re.search('Test',directory): 
            for video in os.listdir():
                test_id.append(os.path.join(directory,'',video))
        else:
            for video in os.listdir():
                train_id.append(os.path.join(directory,'',video))
                #labels[os.path.join(directory, '', video)] = int(directory.rsplit(os.sep, 1,2,3)[1].split(' ')[1,2,3])-1
               
                #labels[os.path.join(directory,'',video)]=int(directory.rsplit(' ',1)[1].split(' ')[1])-1
                
                #labels[os.path.join(directory, '', video)] = int(directory.rsplit(os.sep, 1)[1].split(' ')[1])-1
                
                print(directory.rsplit(os.sep, 1))
                


    return pd.DataFrame(data=train_id, columns=['id']), pd.DataFrame(data=test_id, columns=['id']), labels


# In[20]:


training_df, test_df, labels = create_df(s_dir)
print(len(training_df))
print(len(test_df))


# In[31]:


training_df['label'] = [labels[video] for video in training_df['id']]
training_df.head(10)


# In[13]:


os.chdir('/Users/mahelwimaladasa/Desktop/splits')
test_labels=pd.read_excel('YouTubeIDs-copy.xlsm',
                         sheet_name='Test_labels', header=None)
test_labels.columns=['Name','Label1','Label2','Labeln']
a=test_labels['Label2'][2:].values


# In[14]:


test_df['label']=a
test_df.tail(5)


# In[ ]:


#test_df.to_csv('Test_df.csv',index=False)


# After some data analysis, the original distribution of videos was not correct so it was decided to create a new dataframe with all video and apply the split afterwards.

# In[15]:


full_df=pd.concat([training_df,test_df],ignore_index=True)
full_df.to_csv('Full_df.csv',index=False)
full_df.shape


# ## Start here to bypass creating the DF

# In[17]:


full_df=pd.read_csv('Full_df.csv') 


# In[12]:


#use only a subset
from sklearn.utils import resample, shuffle
sub_df = resample(full_df, replace=False, n_samples=265, random_state=40)

print(sub_df.shape)


# In[13]:


#splitting the training video set into train, validation and test
mid_set, test_set, mid_labels, test_labels = train_test_split(sub_df['id'], 
                                                               keras.utils.to_categorical(sub_df['label']),
                                                               test_size=0.1, stratify=sub_df['label'],
                                                               random_state=40)

train_set, val_set, train_labels, val_labels = train_test_split(mid_set, mid_labels,
                                                               test_size=0.22, stratify=np.argmax(mid_labels, axis=1),
                                                               random_state=40)


# In[14]:


print(f'Number of videos on training set: {train_set.shape[0]}')
print(f'Number of videos on validation set: {val_set.shape[0]}')
print(f'Number of videos on test set: {test_set.shape[0]}')


# In[15]:


#Lookign at split composition
train_dist=np.count_nonzero(train_labels,axis=0)
print(f'Distribution of labels on train set: 0 - {train_dist[0]} |  1 - {train_dist[1]} |  2 - {train_dist[2]}')

val_dist=np.count_nonzero(val_labels,axis=0)
print(f'Distribution of labels on validation set: 0 - {val_dist[0]} |  1 - {val_dist[1]} |  2 - {val_dist[2]}')


test_dist=np.count_nonzero(test_labels,axis=0)
print(f'Distribution of labels on test set: 0 - {test_dist[0]} |  1 - {test_dist[1]} |  2 - {test_dist[2]}')


# ## Upsampling
# 
# NOT ENOUGH MEMORY

# In[184]:


#adding copy of videos to solve the imbalance
train_sdf=pd.DataFrame({'id': train_set, 'label':np.argmax(train_labels, axis=1)})
train_sdf


# In[198]:


label1_df=train_sdf.loc[train_sdf['label']==1]
label2_df=train_sdf.loc[train_sdf['label']==2]
label0_df=train_sdf.loc[train_sdf['label']==0]
upsample_target=len(label0_df)
print(upsample_target)


# In[199]:


from sklearn.utils import resample, shuffle
l1_upsample = resample(label1_df,
             replace=True,
             n_samples=upsample_target,
             random_state=30)

print(l1_upsample.shape)

l2_upsample = resample(label2_df,
             replace=True,
             n_samples=upsample_target,
             random_state=30)

print(l2_upsample.shape)


# In[201]:


train_upsampled = pd.concat([label0_df, l1_upsample, l2_upsample])
train_upsampled = shuffle(train_upsampled)
train_upsampled.shape


# In[202]:


train_upsampled.head(20)


# In[208]:


train_set_u=train_upsampled['id'].values
train_labels=keras.utils.to_categorical(train_upsampled['label'])
print(train_set_u.shape)
print(train_labels.shape)


# In[280]:


#Lookign at split composition
train_dist=np.count_nonzero(train_labels,axis=0)
print(f'Distribution of labels on train set: 0 - {train_dist[0]} |  1 - {train_dist[1]} |  2 - {train_dist[2]}')

val_dist=np.count_nonzero(val_labels,axis=0)
print(f'Distribution of labels on validation set: 0 - {val_dist[0]} |  1 - {val_dist[1]} |  2 - {val_dist[2]}')


test_dist=np.count_nonzero(test_labels,axis=0)
print(f'Distribution of labels on test set: 0 - {test_dist[0]} |  1 - {test_dist[1]} |  2 - {test_dist[2]}')


# ## Downsampling

# In[16]:


#downsizgin train set of videos to solve the imbalance
train_sdf=pd.DataFrame({'id': train_set, 'label':np.argmax(train_labels, axis=1)})
train_sdf


# In[17]:


label1_df=train_sdf.loc[train_sdf['label']==1]
label2_df=train_sdf.loc[train_sdf['label']==2]
label0_df=train_sdf.loc[train_sdf['label']==0]
upsample_target=len(label0_df)
print(upsample_target)


# In[18]:


from sklearn.utils import resample, shuffle
l0_downsample = resample(label0_df,
                         replace=False,
                         n_samples=125,
                         random_state=30)

print(l0_downsample.shape)


# In[19]:


train_downsampled = pd.concat([l0_downsample, label1_df, label2_df])
train_downsampled = shuffle(train_downsampled)
train_downsampled.shape


# In[21]:


train_downsampled.head(20)


# In[22]:


train_set_d=train_downsampled['id'].values
train_labels=keras.utils.to_categorical(train_downsampled['label'])
print(train_set_d.shape)
print(train_labels.shape)


# In[23]:


#Lookign at split composition
train_dist=np.count_nonzero(train_labels,axis=0)
print(f'Distribution of labels on train set: 0 - {train_dist[0]} |  1 - {train_dist[1]} |  2 - {train_dist[2]}')

val_dist=np.count_nonzero(val_labels,axis=0)
print(f'Distribution of labels on validation set: 0 - {val_dist[0]} |  1 - {val_dist[1]} |  2 - {val_dist[2]}')


test_dist=np.count_nonzero(test_labels,axis=0)
print(f'Distribution of labels on test set: 0 - {test_dist[0]} |  1 - {test_dist[1]} |  2 - {test_dist[2]}')


# In[ ]:





# ## Parameters

# In[18]:


max_frames = 50
width = 100
height = 100
epochs = 50
num_features = 2048
batch_size = 32


# ## Video functions

# In[19]:


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


# In[20]:


#reads a fixed number of frames from each video downsampling it
def load_video_skip(path, max_frames=max_frames, resize=(height, width)):
    cap = cv2.VideoCapture(path)
    total_frames=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    ratio_f=math.floor(total_frames/max_frames)
    frames = []
    try:
        while True:
            frameId = cap.get(1)
            ret, frame = cap.read()
            if not ret:
                break
            if (frameId % ratio_f == 0):
                frame = crop_center_square(frame)
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)
            if len(frames) == max_frames:
                break
    finally:
        cap.release()
        
    return np.array(frames)


# In[21]:


#reads a fixed number of frames from each video downsampling it and crops to center square shape
def load_video_skip_centre(path, max_frames=max_frames, resize=(height, width)):
    cap = cv2.VideoCapture(path)
    total_frames=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    #print(total_frames)
    flag=random.randint(0,10) % 2
    #print(flag)
    ratio_f=math.floor(total_frames/(2*max_frames))
    #print(ratio_f)
    frames = []
    if flag==0:
        for i in range(max_frames):
            #print(i)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0 + i*ratio_f)
            ret, frame = cap.read(1)
            #print(ret)
            if not ret: 
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
        cap.release()
    else:
        for i in range(max_frames):
            #print(i)
            cap.set(cv2.CAP_PROP_POS_FRAMES, (math.floor(total_frames/2)-1)+(i*ratio_f))
            ret, frame = cap.read()
            #print(ret)
            if not ret: 
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
        cap.release()   
    return np.array(frames)


# In[22]:


sometimes = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability
seq = va.Sequential([
    va.RandomRotate(degrees=30),   
    sometimes(va.HorizontalFlip()), 
    sometimes(va.VerticalFlip())
])

c_c = va.CenterCrop((width,height))


# In[23]:


def prepare_videos(videos_set, mode=1):
    stored_data=[]
    if mode == 1:
        for video in tqdm(videos_set):
            examples=load_video_skip(video)
            stored_data.append(examples)
    else:
        for video in tqdm(videos_set):
            examples=load_video(video)
            stored_data.append(examples)
    
    return np.array(stored_data)


# ## Read into memory

# In[24]:


#choose 1 for skipped frames and 2 for consecutive frames
training_data = prepare_videos(train_set,1)
val_data = prepare_videos(val_set,1)
test_data = prepare_videos(test_set,1)

print(f'Training data shape: {training_data.shape}')
print(f'Validation data shape: {val_data.shape}')
print(f'Test data shape: {test_data.shape}')


# In[25]:


training_data=training_data/255.0
val_data=val_data/255.0
test_data=test_data/255.0


# In[26]:


f, axarr = plt.subplots(1,3)
axarr[0].imshow(training_data[19][9])
axarr[1].imshow(val_data[19][9])
axarr[2].imshow(test_data[19][9])


# In[283]:


save_p='C:\\Users\\nxb21174\\OneDrive - University of Strathclyde\\Documents\\Final_Project\\Scripts\\variables'
np.savez_compressed(os.path.join(save_p,'','model_data_train_50f_100_r40.npz'), videos=training_data, labels=train_labels)
np.savez_compressed(os.path.join(save_p,'','model_data_val_50f_100_r40.npz'), videos=val_data, labels=val_labels)
np.savez_compressed(os.path.join(save_p,'','model_data_test_50f_100_r40.npz'), videos=test_data, labels=test_labels)


# In[3]:


#bypass the process
save_p='C:\\Users\\nxb21174\\OneDrive - University of Strathclyde\\Documents\\Final_Project\\Scripts\\variables'
train_arrays=np.load(os.path.join(save_p,'','model_data_train_ups_50f_112_r30.npz'))
val_arrays=np.load(os.path.join(save_p,'','model_data_val_50f_112_r30.npz'))
test_arrays=np.load(os.path.join(save_p,'','model_data_test_50f_112_r30.npz'))

training_data=train_arrays['videos']
train_labels=train_arrays['labels']
val_data=val_arrays['videos']
val_labels=val_arrays['labels']
test_data=test_arrays['videos']
test_labels=test_arrays['labels']


# In[4]:


#checking the right shapes
print(f'Training data shape: {training_data.shape}')
print(f'Training labels shape: {train_labels.shape}')
print(f'Validation data shape: {val_data.shape}')
print(f'Validation labels shape: {val_labels.shape}')
print(f'Test data shape: {test_data.shape}')
print(f'Test labels shape: {test_labels.shape}')


# In[27]:


train_dist=np.count_nonzero(train_labels,axis=0)
total_rows=sum(train_dist)

class_weights = {0: total_rows/(3*train_dist[0]),
                1: total_rows/(3*train_dist[1]),
                2: total_rows/(3*train_dist[2])}

print(class_weights)


# ## Augmenting data

# In[28]:


def augmenting_fx(video_set):
    aug_data=[]
    for video in video_set:
        aug_frames=seq(video)
        aug_data.append(aug_frames)
    return np.array(aug_data)


# In[29]:


train_aug=augmenting_fx(training_data)


# In[31]:


plt.imshow(train_aug[19][9])


# In[30]:


labels_aug=train_labels.copy()


# In[32]:


aug_traind=np.append(training_data, train_aug, axis=0)
print(aug_traind.shape)
print(training_data.shape)


# In[33]:


aug_trainl=np.append(train_labels, labels_aug, axis=0)
print(aug_trainl.shape)
print(train_labels.shape)


# ## Binary classification

# In[26]:


full_df['label_b']= [0.0 if label==0 else 1.0 for label in full_df['label']]


# In[27]:


#splitting the training video set into train, validation and test
train_set, test_set, train_labelsb, test_labelsb = train_test_split(full_df['id'], full_df['label_b'],
                                                               test_size=0.1, stratify=full_df['label_b'],
                                                               random_state=30)

train_set, val_set, train_labelsb, val_labelsb = train_test_split(train_set, train_labelsb,
                                                               test_size=0.22, stratify=train_labelsb,
                                                               random_state=30)


# In[28]:


#Lookign at split composition
train_dist_b=np.count_nonzero(train_labelsb)
print(f'Distribution of labels on train set: 0 - {len(train_labelsb) - train_dist_b} |  1 - {train_dist_b}')

val_dist_b=np.count_nonzero(val_labelsb)
print(f'Distribution of labels on validation set: 0 - {len(val_labelsb) - val_dist_b} |  1 - {val_dist_b}')


test_dist_b=np.count_nonzero(test_labelsb)
print(f'Distribution of labels on test set: 0 - {len(test_labelsb) - test_dist_b} |  1 - {test_dist_b}')


# ## Attempt model

# In[ ]:


model = keras.models.load_model('C:\\Users\\nxb21174.DS\\OneDrive - University of Strathclyde\\Documents\\Final_Project\\Scripts\\16cf_10e_16bs_model1.h5')


# In[ ]:


model.summary()


# In[ ]:


inputs = tf.keras.Input((training_data.shape[1], width, height, 3)) 
level1 = tf.keras.layers.Conv3D(16,3, strides=1, padding='valid', activation='relu')(inputs)
level1n = tf.keras.layers.BatchNormalization(axis=4)(level1)
level2 = tf.keras.layers.MaxPool3D((1,2,2))(level1n)
level3 = tf.keras.layers.Conv3D(32,3, strides=1, padding='valid', activation='relu')(level2)
level3n = tf.keras.layers.BatchNormalization(axis=4)(level3)
level4 = tf.keras.layers.MaxPool3D((2,2,2))(level3n)
level5 = tf.keras.layers.Conv3D(32,3, strides=1, padding='valid', activation='relu')(level4)
level5n = tf.keras.layers.BatchNormalization(axis=4)(level5)
level6 = tf.keras.layers.MaxPool3D((2,2,2))(level5n)
#level7 = tf.keras.layers.Conv3D(256,3, strides=1, padding='same', activation='relu')(level6)
#level8 = tf.keras.layers.MaxPool3D((2,2,2))(level7)
#level9 = tf.keras.layers.Conv3D(256,3, strides=1, padding='same', activation='relu')(level8)
#level10 = tf.keras.layers.MaxPool3D((2,2,2))(level9)
level11 = tf.keras.layers.Flatten()(level6)
#level12 = tf.keras.layers.Dense(128, activation='relu')(level11)
#level12d = tf.keras.layers.Dropout(0.5)(level12)
level13 = tf.keras.layers.Dense(64, activation='relu')(level11)
level13d = tf.keras.layers.Dropout(0.5)(level13)
output = tf.keras.layers.Dense(3, activation='softmax')(level13d)


# In[ ]:


model = tf.keras.Model(inputs, output, name="feature_extractor")
model.summary()


# In[ ]:


opt = tf.keras.optimizers.SGD(learning_rate=0.003, momentum=0.9, decay = 0.003/epochs)

early_stopping_callback = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 3, mode = 'min',
                                                        restore_best_weights = True, min_delta=0.001)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


model_training_history = model.fit(x = training_data, y = train_labels, epochs = epochs, batch_size = batch_size,
                                     shuffle = True, validation_data = (val_data, val_labels), 
                                     callbacks = [early_stopping_callback], class_weight = class_weight)


# In[ ]:


model.evaluate(test_data, test_labels)


# In[ ]:


model.save('C:\\Users\\nxb21174.DS\\OneDrive - University of Strathclyde\\Documents\\Final_Project\\Scripts\\ModelP_16bs_10e_20skf.h5')
hist_csv_file = 'C:\\Users\\nxb21174.DS\\OneDrive - University of Strathclyde\\Documents\\Final_Project\\Scripts\\history_modelp_20skf_10e_16bs.csv'
hist_df = pd.DataFrame(history.history)
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)


# In[ ]:


plt.plot(model_training_history.history['loss'])
plt.plot(model_training_history.history['val_loss'])
plt.title('Loss: 3D-CNN Model with 3 Conv Layers (16,32,32) \n Input:16x224x224 with class weights')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('C:\\Users\\nxb21174\\OneDrive - University of Strathclyde\\Documents\\Final_Project\\Scripts\\Figures\\Loss_modelp_16skf_10e_16bs_1h_cw.png')
plt.show()


# In[ ]:


plt.plot(model_training_history.history['accuracy'])
plt.plot(model_training_history.history['val_accuracy'])
plt.title('Accuracy: 3D-CNN Model with 3 Conv Layers (16,32,32) \n Input:16x224x224 with class weights')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('C:\\Users\\nxb21174\\OneDrive - University of Strathclyde\\Documents\\Final_Project\\Scripts\\Figures\\Acc_modelp_16skf_10e_16bs_1h_cw.png')
plt.show()


# In[ ]:


test_index = np.random.choice(range(test_data.shape[0]))
print(test_index)
test_video=np.expand_dims(test_data[test_index],0)
prediction=model.predict(test_video)
print(prediction)
print(test_labels[test_index])


# In[ ]:


test_index = np.random.choice(range(test_data.shape[0]))
print(test_index)
test_video=np.expand_dims(test_data[test_index],0)
prediction=model.predict(test_video)
print(prediction)
print(test_labels[test_index])


# In[ ]:


test_index = np.random.choice(range(test_data.shape[0]))
print(test_index)
test_video=np.expand_dims(test_data[test_index],0)
prediction=model.predict(test_video)
print(prediction)
print(test_labels[test_index])


# In[ ]:


test_index = np.random.choice(range(test_data.shape[0]))
print(test_index)
test_video=np.expand_dims(test_data[test_index],0)
prediction=model.predict(test_video)
print(prediction)
print(test_labels[test_index])


# ## Second Model

# In[63]:


model2 = keras.Sequential()
model2.add(keras.layers.Conv3D(
            16, (3,3,3), strides=(1,1,1), activation='relu', input_shape=(training_data.shape[1], width, height, 3)
        ))
model2.add(keras.layers.MaxPooling3D(pool_size=(2, 2, 2)))
model2.add(keras.layers.BatchNormalization())
#model2.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.2)))
#model2.add(keras.layers.Conv3D(
#            6, (3,3,3), strides=(2,2,2), activation='relu', kernel_regularizer='l2'))
#model2.add(keras.layers.MaxPooling3D(pool_size=(2, 2, 2)))
#model2.add(keras.layers.BatchNormalization())
model2.add(keras.layers.Flatten())
#model2.add(keras.layers.Dense(100, activation='relu', kernel_regularizer='l2'))
model2.add(keras.layers.Dropout(0.25))
model2.add(keras.layers.Dense(3, activation='softmax')) 

model2.summary()


# In[64]:


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.1
    epochs_drop = 4.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


# In[65]:


early_stopping_callback = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min', 
                                                        restore_best_weights = True, min_delta=0.0001)

opt = keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9, decay=0.0001)

#lrate = keras.callbacks.LearningRateScheduler(step_decay)


# In[66]:


model2.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ["accuracy", keras.metrics.Precision(),
                                                                              keras.metrics.Recall()])


# In[67]:


model2_training_history = model2.fit(x = training_data, y = train_labels, epochs = 30, batch_size = 32,
                                     shuffle = True, validation_data = (val_data, val_labels), 
                                     callbacks = [early_stopping_callback], class_weight = class_weights
                                    )


# In[68]:


model2.evaluate(test_data, test_labels)


# In[ ]:


model2.save('C:\\Users\\nxb21174\\OneDrive - University of Strathclyde\\Documents\\Final_Project\\Scripts\\Model2n_2cnn_32bs_15e_16skf_cw_lr-04.h5')
hist_csv_file = 'C:\\Users\\nxb21174\\OneDrive - University of Strathclyde\\Documents\\Final_Project\\Scripts\\history_model2n_2cnn_16skf_15e_32bs_cw_lr-4.csv'
hist_df = pd.DataFrame(model2_training_history.history)
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)


# In[83]:


plt.plot(model2_training_history.history['loss'])
plt.plot(model2_training_history.history['val_loss'])
plt.title('Loss: 3D-CNN Naive Model \n Input: 50x110x110 augmented')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.ylim([0.8,2.5])
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('C:\\Users\\nxb21174\\OneDrive - University of Strathclyde\\Documents\\Final_Project\\Scripts\\Figures\\Loss_modeln_50f_50e_32bs_s100_aug.png')
plt.show()


# In[84]:


plt.plot(model2_training_history.history['accuracy'])
plt.plot(model2_training_history.history['val_accuracy'])
plt.title('Accuracy: 3D-CNN Naive Model \n Input: 50x100x100 augmented')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('C:\\Users\\nxb21174\\OneDrive - University of Strathclyde\\Documents\\Final_Project\\Scripts\\Figures\\Acc_modeln_50f_50e_32bs_100s_aug.png')
plt.show()


# In[71]:


test_index = np.random.choice(range(test_data.shape[0]))
print(test_index)
test_video=np.expand_dims(test_data[test_index],0)
prediction=model2.predict(test_video)
print(prediction)
print(test_labels[test_index])


# In[72]:


test_set.iloc[test_index]


# In[73]:


test_index = np.random.choice(range(test_data.shape[0]))
print(test_index)
test_video=np.expand_dims(test_data[test_index],0)
prediction=model2.predict(test_video)
print(prediction)
print(test_labels[test_index])


# In[74]:


test_set.iloc[test_index]


# In[75]:


test_index = np.random.choice(range(test_data.shape[0]))
print(test_index)
test_video=np.expand_dims(test_data[test_index],0)
prediction=model2.predict(test_video)
print(prediction)
print(test_labels[test_index])


# In[76]:


test_set.iloc[test_index]


# In[77]:


test_index = np.random.choice(range(test_data.shape[0]))
print(test_index)
test_video=np.expand_dims(test_data[test_index],0)
prediction=model2.predict(test_video)
print(prediction)
print(test_labels[test_index])


# In[78]:


test_set.iloc[test_index]


# In[79]:


y_pred=model2.predict(test_data)


# In[80]:


y_pred_label=np.argmax(y_pred,axis=1)
y_pred_label


# In[81]:


y_test=np.argmax(test_labels,axis=1)
y_test


# In[82]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred_label))
print(confusion_matrix(y_test, y_pred_label))


# ## CNN - RNN

# In[ ]:


def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(width, height, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((width, height, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()


# In[ ]:


def extract_features(video_set):
    num_samples = video_set.shape[0]

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, max_frames), dtype="bool") 
    frame_features = np.zeros(shape=(num_samples, max_frames, num_features), dtype="float32") 

    # For each video.
    for idx, frames in enumerate(video_set):
        # Gather all its frames and add a batch dimension.
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1, max_frames,), dtype="bool")
        temp_frame_features = np.zeros(
            shape=(1, max_frames, num_features), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(max_frames, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :]
                )
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks)


# In[ ]:


train_features = extract_features(training_data)
validation_features = extract_features(validation_data)

print(f"Frame features in train set: {train_features[0].shape}")
print(f"Frame masks in train set: {train_features[1].shape}")
print(f"train_labels in train set: {train_labels.shape}")


# In[ ]:


def get_sequence_model():

    frame_features_input = keras.Input((max_frames, num_features))
    mask_input = keras.Input((max_frames,), dtype="bool")

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = keras.layers.GRU(16, return_sequences=True)(frame_features_input, mask=mask_input)
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(3, activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy","sparse_categorical_accuracy"]
    )
    return rnn_model


# In[ ]:


def run_experiment():
    filepath = "C:\\Users\\nxb21174.DS\\OneDrive - University of Strathclyde\\Documents\\Final_Project\\Scripts\\Model\\"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    seq_model = get_sequence_model()
    history_crnn = seq_model.fit(
        [train_features[0], train_features[1]],
        train_labels,
        validation_data=([validation_features[0],validation_features[1]], val_labels),
        epochs=epochs,
        callbacks=[checkpoint],
    )

    #seq_model.load_weights(filepath)
    #_, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    #print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history_crnn, seq_model

_, sequence_model = run_experiment()sparse_categorical_accuracy


# In[ ]:


plt.plot(_.history['loss'])
plt.plot(_.history['val_loss'])
plt.title('Loss: CNN-RNN Model with GRU layers \n Input:20x224x224 (skipped frames)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('C:\\Users\\nxb21174.DS\\OneDrive - University of Strathclyde\\Documents\\Final_Project\\Scripts\\Figures\\Loss_cnnrnn_20skf_10e_16bs.png')
plt.show()


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy: CNN-RNN Model with GRU Layers \n Input:20x224x224 (skipped frames)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('C:\\Users\\nxb21174.DS\\OneDrive - University of Strathclyde\\Documents\\Final_Project\\Scripts\\Figures\\Acc_cnnrnn_20skf_10e_16bs.png')
plt.show()


# In[ ]:


def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, max_frames,), dtype="bool")
    frame_features = np.zeros(shape=(1,max_frames, num_features), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(max_frames, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask


def sequence_prediction(video):
    frames = load_video_skip(video)
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]

    for i in np.argsort(probabilities)[::-1]:
        print(f"  {i}: {probabilities[i] * 100:5.2f}%")
    return frames

test_video = np.random.choice(test_df["id"].values.tolist())
print(f"Test video path: {test_video}")
test_frames = sequence_prediction(test_video)
#bad prediction, should be 1


# In[ ]:


test_video = np.random.choice(test_df["id"].values.tolist())
print(f"Test video path: {test_video}")
test_frames = sequence_prediction(test_video)
#good prediction


# In[ ]:


test_video = np.random.choice(test_df["id"].values.tolist())
print(f"Test video path: {test_video}")
test_frames = sequence_prediction(test_video)
#bad prediction, should be 1


# In[ ]:


test_video = np.random.choice(test_df["id"].values.tolist())
print(f"Test video path: {test_video}")
test_frames = sequence_prediction(test_video)
#good prediction


# ## LRCN

# In[ ]:


lrcn_model = keras.Sequential()
      
lrcn_model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(16, (3, 3), padding='same',activation = 'relu'),
                                       input_shape = (max_frames, height, width, 3)))
    
lrcn_model.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D((4, 4)))) 
lrcn_model.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.25)))
    
lrcn_model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(32, (3, 3), padding='same',activation = 'relu')))
lrcn_model.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D((4, 4))))
lrcn_model.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.25)))
    
lrcn_model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(64, (3, 3), padding='same',activation = 'relu')))
lrcn_model.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D((2, 2))))
lrcn_model.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.25)))
    
lrcn_model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(64, (3, 3), padding='same',activation = 'relu')))
lrcn_model.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D((2, 2))))
                                      
lrcn_model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
                                      
lrcn_model.add(keras.layers.LSTM(32))
                                      
lrcn_model.add(keras.layers.Dense(3, activation = 'softmax'))

lrcn_model.summary()


# In[ ]:


early_stopping_callback = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 3, mode = 'min', 
                                                        restore_best_weights = True)

opt = keras.optimizers.SGD(learning_rate=0.00001, momentum=0.9)

lrcn_model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ["accuracy"])
 
lrcn_model_training_history = lrcn_model.fit(x = training_data, y = train_labels, epochs = epochs, batch_size = batch_size ,
                                             shuffle = True, validation_data = (validation_data, val_labels), 
                                             #validation_split=0.1,
                                             callbacks = [early_stopping_callback])


# In[ ]:


model_evaluation_history = lrcn_model.evaluate(test_data, keras.utils.to_categorical(test_df['label']))


# ## Transfer Learning

# In[ ]:


preprocess_input = keras.applications.inception_v3.preprocess_input


# In[25]:


conv_base = keras.applications.InceptionV3(weights='imagenet', include_top=False, pooling="max", 
                                               input_shape=(width,height, 3))

for layer in conv_base.layers[:-65]:
    layer.trainable = False


# In[31]:


#<keras.layers.pooling.max_pooling2d.MaxPooling2D at 0x23d107aae50>
base_model.layers[-65]


# In[33]:


ip = keras.Input(shape=(training_data.shape[1],width,height,3))

t_conv = keras.layers.TimeDistributed(conv_base)(ip) 

t_lstm = keras.layers.LSTM(10, return_sequences=False)(t_conv)

f_softmax = keras.layers.Dense(3, activation='softmax')(t_lstm)

itf_model = keras.Model(ip, f_softmax)

itf_model.summary()


# In[ ]:


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.1
    epochs_drop = 4.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


# In[28]:


early_stopping_callback = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, mode = 'min', 
                                                        restore_best_weights = True, min_delta=0.001)

#opt = keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)

#lrate = keras.callbacks.LearningRateScheduler(step_decay)


# In[ ]:


itf_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ["accuracy"])
 
itf_model_training_history = itf_model.fit(x = training_data, y = train_labels, epochs = 50, batch_size = 32,
                                     shuffle = True, validation_data = (val_data, val_labels), 
                                     callbacks = [early_stopping_callback], class_weight = class_weight)


# In[27]:


itf_model = keras.models.Sequential()
itf_model.add(keras.Input((training_data.shape[1], width, height, 3)))
itf_model.add(keras.layers.TimeDistributed(conv_base))
#itf_model.add(keras.layers.TimeDistributed(GlobalAveragePooling2D()))
#itf_model.add(keras.layers.Dropout(0.25))
itf_model.add(keras.layers.LSTM(30))
itf_model.add(keras.layers.Dropout(0.5))
itf_model.add(keras.layers.Dense(3,activation = 'softmax'))

itf_model.summary()


# In[ ]:


#test_data=test_data/255.0
test_data.shape


# In[ ]:


test_index = np.random.choice(range(test_data.shape[0]))
print(test_index)
test_video=np.expand_dims(test_data[test_index],0)
prediction=model.predict(test_video)
print(prediction)
print(test_df['label'][test_index])


# In[ ]:


test_index = np.random.choice(range(test_data.shape[0]))
print(test_index)
test_video=np.expand_dims(test_data[test_index],0)
prediction=model.predict(test_video)
print(prediction)
print(test_df['label'][test_index])


# In[ ]:


test_index = np.random.choice(range(test_data.shape[0]))
print(test_index)
test_video=np.expand_dims(test_data[test_index],0)
prediction=model.predict(test_video)
print(prediction)
print(test_df['label'][test_index])


# ## VGGNet + LSTM

# In[ ]:


conv_base.layers[:15]


# In[62]:


from tensorflow.keras.applications import VGG16

def create_base():
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(width,height, 3))
    x = keras.layers.GlobalAveragePooling2D()(conv_base.output)
    base_model = keras.Model(conv_base.input, x)
    return base_model

conv_base = create_base()

for layer in conv_base.layers[:15]:
    layer.trainable = False

ip = keras.Input(shape=(training_data.shape[1],width,height,3))

t_conv = keras.layers.TimeDistributed(conv_base)(ip) # vgg16 feature extractor

t_lstm = keras.layers.LSTM(10, return_sequences=False)(t_conv)

f_softmax = keras.layers.Dense(3, activation='softmax')(t_lstm)

vgg_model = keras.Model(ip, f_softmax)

vgg_model.summary()


# In[ ]:


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.1
    epochs_drop = 4.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


# In[63]:


early_stopping_callback = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min', 
                                                        restore_best_weights = True, min_delta=0.0005)

#opt = keras.optimizers.SGD(learning_rate=0.0, momentum=0.9)

#lrate = keras.callbacks.LearningRateScheduler(step_decay)


# In[ ]:


vgg_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ["accuracy"])
 
vgg_model_training_history = vgg_model.fit(x = training_data, y = train_labels, epochs = 50, batch_size = 32,
                                     shuffle = True, validation_data = (val_data, val_labels), 
                                     callbacks = [early_stopping_callback]#, class_weight = class_weight
                                          )


# In[ ]:


vgg_model.evaluate(test_data, test_labels)


# In[ ]:


model2.save('C:\\Users\\nxb21174\\OneDrive - University of Strathclyde\\Documents\\Final_Project\\Scripts\\Model2n_2cnn_32bs_15e_16skf_cw_lr-04.h5')
hist_csv_file = 'C:\\Users\\nxb21174\\OneDrive - University of Strathclyde\\Documents\\Final_Project\\Scripts\\history_model2n_2cnn_16skf_15e_32bs_cw_lr-4.csv'
hist_df = pd.DataFrame(model2_training_history.history)
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)


# In[ ]:


plt.plot(model2_training_history.history['loss'])
plt.plot(model2_training_history.history['val_loss'])
plt.title('Loss: 3D-CNN Naive Model \n Input: 16x220x220, with learning schedule')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.ylim([0.8,2.5])
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('C:\\Users\\nxb21174\\OneDrive - University of Strathclyde\\Documents\\Final_Project\\Scripts\\Figures\\Loss_modeln_16skf_15e_32bs_1h_cw_ls.png')
plt.show()


# In[ ]:


plt.plot(model2_training_history.history['accuracy'])
plt.plot(model2_training_history.history['val_accuracy'])
plt.title('Accuracy: 3D-CNN Naive Model \n Input: 16x220x220, with learning schedule')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('C:\\Users\\nxb21174\\OneDrive - University of Strathclyde\\Documents\\Final_Project\\Scripts\\Figures\\Acc_modeln_16skf_15e_32bs_1h_cw_ls.png')
plt.show()


# In[ ]:


test_index = np.random.choice(range(test_data.shape[0]))
print(test_index)
test_video=np.expand_dims(test_data[test_index],0)
prediction=model2.predict(test_video)
print(prediction)
print(test_labels[test_index])


# ## Second Model - Binary Classification

# In[31]:


modelb = keras.Sequential()
modelb.add(keras.layers.Conv3D(
            4, (3,3,3), strides=(1,1,1), activation='relu', input_shape=(training_data.shape[1], width, height, 3)
        ))
modelb.add(keras.layers.MaxPooling3D(pool_size=(2, 2, 2)))
modelb.add(keras.layers.BatchNormalization())
#model2.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.2)))
modelb.add(keras.layers.Conv3D(
            6, (3,3,3), strides=(2,2,2), activation='relu', kernel_regularizer='l2'))
modelb.add(keras.layers.MaxPooling3D(pool_size=(2, 2, 2)))
modelb.add(keras.layers.BatchNormalization())
modelb.add(keras.layers.Flatten())
#model2.add(keras.layers.Dense(100, activation='relu', kernel_regularizer='l2'))
modelb.add(keras.layers.Dropout(0.25))
modelb.add(keras.layers.Dense(1, activation='sigmoid')) 

modelb.summary()


# In[46]:


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.003
    drop = 0.1
    epochs_drop = 4.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


# In[47]:


early_stopping_callback = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 3, mode = 'min', 
                                                        restore_best_weights = True, min_delta=0.0005)

opt = keras.optimizers.SGD(learning_rate=0, momentum=0.9)

lrate = keras.callbacks.LearningRateScheduler(step_decay)


# In[48]:


modelb.compile(loss = 'binary_crossentropy', optimizer = opt , metrics = ["accuracy"])
 
modelb_training_history = modelb.fit(x = training_data, y = train_labelsb, epochs = 50, batch_size = 32,
                                     shuffle = True, validation_data = (val_data, val_labelsb), 
                                     callbacks = [early_stopping_callback, lrate])


# In[34]:


modelb.evaluate(test_data, test_labelsb)


# In[ ]:


model2.save('C:\\Users\\nxb21174\\OneDrive - University of Strathclyde\\Documents\\Final_Project\\Scripts\\Model2n_2cnn_32bs_15e_16skf_cw_lr-04.h5')
hist_csv_file = 'C:\\Users\\nxb21174\\OneDrive - University of Strathclyde\\Documents\\Final_Project\\Scripts\\history_model2n_2cnn_16skf_15e_32bs_cw_lr-4.csv'
hist_df = pd.DataFrame(model2_training_history.history)
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)


# In[35]:


plt.plot(modelb_training_history.history['loss'])
plt.plot(modelb_training_history.history['val_loss'])
plt.title('Loss: 3D-CNN Naive Model \n Input: 16x220x220, binary')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.ylim([0.8,2.5])
plt.legend(['train', 'validation'], loc='upper left')
#plt.savefig('C:\\Users\\nxb21174\\OneDrive - University of Strathclyde\\Documents\\Final_Project\\Scripts\\Figures\\Loss_modeln_16skf_15e_32bs_1h_cw_ls.png')
plt.show()


# In[37]:


plt.plot(modelb_training_history.history['accuracy'])
plt.plot(modelb_training_history.history['val_accuracy'])
plt.title('Accuracy: 3D-CNN Naive Model \n Input: 16x220x220, binary')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#plt.savefig('C:\\Users\\nxb21174\\OneDrive - University of Strathclyde\\Documents\\Final_Project\\Scripts\\Figures\\Acc_modeln_16skf_15e_32bs_1h_cw_ls.png')
plt.show()


# In[42]:


test_index = np.random.choice(range(test_data.shape[0]))
print(test_index)
test_video=np.expand_dims(test_data[test_index],0)
prediction=modelb.predict(test_video)
print(prediction)
print(test_labelsb.iloc[test_index])


# In[43]:


test_index = np.random.choice(range(test_data.shape[0]))
print(test_index)
test_video=np.expand_dims(test_data[test_index],0)
prediction=modelb.predict(test_video)
print(prediction)
print(test_labelsb.iloc[test_index])


# In[44]:


test_index = np.random.choice(range(test_data.shape[0]))
print(test_index)
test_video=np.expand_dims(test_data[test_index],0)
prediction=modelb.predict(test_video)
print(prediction)
print(test_labelsb.iloc[test_index])


# In[45]:


test_index = np.random.choice(range(test_data.shape[0]))
print(test_index)
test_video=np.expand_dims(test_data[test_index],0)
prediction=modelb.predict(test_video)
print(prediction)
print(test_labelsb.iloc[test_index])


# In[ ]:





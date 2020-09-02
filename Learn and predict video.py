import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.python.keras.engine import *
import numpy as np
import time, cv2, copy, os

num_of_classes=7
network={}
priors=[4,6,6,6,4,4]

core_network=keras.applications.MobileNetV2((300,300,3), weights=None, include_top=False)

MIN_CORE_OUTPUT_SIZE=19

#SSD Input Layer
network['input']=Input((300,300,3))

#Set Core Network
try:
    c=MIN_CORE_OUTPUT_SIZE

    layer = core_network
    if type(core_network) in (functional.Functional, sequential.Sequential) : # If Core Network is a Model
        layer = core_network.layers

        if type(layer[0])==input_layer.InputLayer:
            network['input']=layer[0].output
        else:
            raise ValueError("[Error] No Input Layer in Core Network.")

        num=len(layer)
        for i in reversed(range(len(layer))):
            if len(layer[i].output.shape)==4:
                num=i
                break

        last=layer[num].output

        c=min(last.shape[1:3])

        if c<MIN_CORE_OUTPUT_SIZE:
            c=MIN_CORE_OUTPUT_SIZE//c
            last=UpSampling2D(1+c)(last)

        core_network=Model(layer[0].output, last).output

    else:
        raise ValueError("[Error] Core Network is not a Model. (Please set a Model as Core Network.)")
except:
    raise ValueError("An error occurred while setting up Core Network.")

network['core']=core_network

network['BBox']=[]

network['BBox'].append({
    'confidence' : 
    Conv2D(filters=priors[0]*(num_of_classes+1), 
        kernel_size=3, 
        strides=1, 
        padding='same',
        kernel_initializer="random_uniform", 
        name="BBox0_confidence")(network['core']) ,

    'box' : 
    Conv2D(filters=priors[0]*4, 
        kernel_size=3, 
        strides=1, 
        padding='same', 
        kernel_initializer="random_uniform", 
        name="BBox0_box")(network['core'])
})

network['FC6']=Conv2D(filters=1024, 
                    kernel_size=3, 
                    dilation_rate=6, 
                    padding='same', 
                    activation="relu", 
        kernel_initializer="random_uniform", 
                    name="FC6")(network['core'])

network['FC7']=Conv2D(filters=1024, 
                    kernel_size=1, 
                    strides=1, 
                    padding='same', 
                    activation="relu", 
        kernel_initializer="random_uniform", 
                    name="FC7")(network['FC6'])

network['BBox'].append({
    'confidence' : 
    Conv2D(filters=priors[1]*(num_of_classes+1), 
        kernel_size=3, 
        strides=1, 
        padding='same', 
        kernel_initializer="random_uniform", 
        name="BBox1_confidence")(network['FC7']) ,

    'box' : 
    Conv2D(filters=priors[1]*4, 
        kernel_size=3, 
        strides=1, 
        padding='same', 
        kernel_initializer="random_uniform", 
        name="BBox1_box")(network['FC7'])
})

network['Conv8_1']=Conv2D(filters=256, 
                    kernel_size=1, 
                    strides=1, 
                    padding='same', 
                    activation="relu", 
        kernel_initializer="random_uniform", 
                    name="Conv8_1")(network['FC7'])

network['Conv8_2']=Conv2D(filters=512, 
                    kernel_size=3,  
                    strides=2, 
                    padding='same', 
                    activation="relu", 
        kernel_initializer="random_uniform", 
                    name="Conv8_2")(network['Conv8_1'])

network['BBox'].append({
    'confidence' : 
    Conv2D(filters=priors[2]*(num_of_classes+1), 
        kernel_size=3, 
        strides=1, 
        padding='same', 
        kernel_initializer="random_uniform", 
        name="BBox2_confidence")(network['Conv8_2']) ,

    'box' : 
    Conv2D(filters=priors[2]*4, 
        kernel_size=3, 
        strides=1, 
        padding='same', 
        kernel_initializer="random_uniform", 
        name="BBox2_box")(network['Conv8_2'])
})

network['Conv9_1']=Conv2D(filters=128, 
                    kernel_size=1,  
                    strides=1, 
                    padding='same', 
                    activation="relu", 
                    kernel_initializer="random_uniform", 
                    name="Conv9_1")(network['Conv8_2'])

network['Conv9_2']=Conv2D(filters=256, 
                    kernel_size=3,  
                    strides=2, 
                    padding='same', 
                    activation="relu", 
                    kernel_initializer="random_uniform", 
                    name="Conv9_2")(network['Conv9_1'])

network['BBox'].append({
    'confidence' : 
    Conv2D(filters=priors[3]*(num_of_classes+1), 
        kernel_size=3, 
        strides=1, 
        padding='same', 
        kernel_initializer="random_uniform", 
        name="BBox3_confidence")(network['Conv8_2']) ,

    'box' : 
    Conv2D(filters=priors[3]*4, 
        kernel_size=3, 
        strides=1, 
        padding='same', 
        kernel_initializer="random_uniform", 
        name="BBox3_box")(network['Conv8_2'])
})

network['Conv10_1']=Conv2D(filters=128, 
                    kernel_size=1,  
                    strides=1, 
                    padding='same', 
                    activation="relu", 
                    kernel_initializer="random_uniform", 
                    name="Conv10_1")(network['Conv9_2'])

network['Conv10_2']=Conv2D(filters=256, 
                    kernel_size=3,  
                    strides=2, 
                    padding='same', 
                    activation="relu", 
                    kernel_initializer="random_uniform", 
                    name="Conv10_2")(network['Conv10_1'])

network['BBox'].append({
    'confidence' : 
    Conv2D(filters=priors[4]*(num_of_classes+1), 
        kernel_size=3, 
        strides=1, 
        padding='same', 
        kernel_initializer="random_uniform", 
        name="BBox4_confidence")(network['Conv10_2']) ,

    'box' : 
    Conv2D(filters=priors[4]*4, 
        kernel_size=3, 
        strides=1, 
        padding='same', 
        kernel_initializer="random_uniform", 
        name="BBox4_box")(network['Conv10_2'])
})

network['Conv11_1']=Conv2D(filters=128, 
                    kernel_size=1,  
                    strides=1, 
                    padding='same', 
                    activation="relu", 
                    kernel_initializer="random_uniform", 
                    name="Conv11_1")(network['Conv10_2'])

network['Conv11_2']=Conv2D(filters=256, 
                    kernel_size=3,  
                    strides=2, 
                    padding='valid', 
                    activation="relu", 
                    kernel_initializer="random_uniform", 
                    name="Conv11_2")(network['Conv11_1'])

network['BBox'].append({
    'confidence' : 
    Conv2D(filters=priors[4]*(num_of_classes+1), 
        kernel_size=3, 
        strides=1, 
        padding='same', 
        kernel_initializer="random_uniform", 
        name="BBox5_confidence")(network['Conv11_2']) ,

    'box' : 
    Conv2D(filters=priors[5]*4, 
        kernel_size=3, 
        strides=1, 
        padding='same', 
        kernel_initializer="random_uniform", 
        name="BBox5_box")(network['Conv11_2'])
})

network['confidences']=Activation('softmax')(Concatenate(axis=1, name='BBox_confidence')([Reshape((-1,(num_of_classes+1)))(BBOX['confidence']) for BBOX in network['BBox']]))
network['boxes']=Concatenate(axis=1, name='BBox_box')([Reshape((-1,4))(BBOX['box']) for BBOX in network['BBox']])
network['preds']=Concatenate(axis=2, name='Predictions')([network['confidences'], network['boxes']])

model=Model(inputs=network['input'], outputs=network['preds'])

model.num_of_classes=num_of_classes

neg_pos_ratio=3
n_neg_min=0
alpha=1.0

def smooth_L1_loss(y_true, y_pred):
    error=tf.cast(y_true,tf.float32)-tf.cast(y_pred,tf.float32)
    absolute_loss = tf.abs(error)
    square_loss = 0.5 * (error)**2
    l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
    return tf.reduce_sum(l1_loss, axis=-1)

def log_loss(y_true, y_pred):
    y_pred = tf.maximum(tf.minimum(y_pred, 1 - 1e-15), 1e-15)
    log_loss = -tf.reduce_sum(tf.cast(y_true,tf.float32) * tf.cast(tf.math.log(y_pred),tf.float32), axis=-1)
    return log_loss

def compute_loss(y_true, y_pred):
    y_true=tf.cast(y_true,tf.float32)
    batch_size = tf.shape(y_pred)[0] 
    n_boxes = tf.shape(y_pred)[1] 

    classification_loss = tf.cast(log_loss(y_true[:,:,:-4], y_pred[:,:,:-4]), tf.float32)
    localization_loss = tf.cast(smooth_L1_loss(y_true[:,:,-4:], y_pred[:,:,-4:]), tf.float32)

    negatives = y_true[:,:,0]
    positives = tf.cast(tf.reduce_max(y_true[:,:,1:-4], axis=-1), tf.float32)
    n_positive = tf.reduce_sum(positives)
    pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1)

    neg_class_loss_all = classification_loss * negatives
    n_neg_losses = tf.math.count_nonzero(neg_class_loss_all, dtype=tf.int32) 
    n_negative_keep = tf.minimum(tf.maximum(neg_pos_ratio * tf.cast(n_positive, tf.int32), n_neg_min), n_neg_losses)


    def f1():
        if batch_size==None:
            return tf.zeros([0])
        else:
            return tf.zeros([batch_size])

    def f2():
        neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1])
        values, indices = tf.nn.top_k(neg_class_loss_all_1D,
                                        k=n_negative_keep,
                                        sorted=False)

        negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                        updates=tf.ones_like(indices, dtype=tf.int32),
                                        shape=tf.shape(neg_class_loss_all_1D))
        negatives_keep = tf.cast(tf.reshape(negatives_keep, [batch_size, n_boxes]), tf.float32)
        neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1)
        return neg_class_loss

    neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

    class_loss = pos_class_loss + neg_class_loss

    loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1)

    total_loss = (class_loss + alpha * loc_loss) / tf.maximum(1.0, n_positive)
    total_loss = total_loss * tf.cast(batch_size, tf.float32)

    return total_loss

model = SSD_Model(7)
model.compile(optimizer='Adam', loss=compute_loss)

X=np.load("X.npy")
Y=np.load("Y.npy")

model.fit(X, Y, epochs=200, batch_size=32, verbose=1, validation_split=0.15)

threshold=0.8

labels=["Car", "Motorcycle", "Traffic Light", "Road Sign", "Traffic Sign", "Person", "Cross Road"]
colors=[(100,255,0), (100,255,0), (0,255,0), (255,255,255), (0,255,0), (255,150,150), (255,150,0)]

def get_output(img):
  data=cv2.resize(copy.deepcopy(img),(300,300))
  a=model(np.array([data]))
  atmp=a[0][np.amax(a[0,:,1:-4], axis=-1)>=threshold]

  for n in atmp:
    x1,x2=np.array([n[-4],n[-2]])/300*len(img[0])
    y1,y2=np.array([n[-3],n[-1]])/300*len(img)
    i=np.argmax(n[1:-4])
    img = cv2.rectangle(img, (x1,y1), (x2,y2), colors[i], 3)

    (text_width, text_height) = cv2.getTextSize(labels[i], cv2.FONT_ITALIC, fontScale=1, thickness=2)[0]
    box_coords = ((int(x1-2),int(y1)), (int(x1 + text_width + 2), int(y1-15 - text_height - 2)))
    
    cv2.rectangle(img, box_coords[0], box_coords[1], colors[i], cv2.FILLED)

    img=cv2.putText(img, labels[i], (x1,int(y1-10)), cv2.FONT_ITALIC, 1, (0, 0, 0) if np.amax(colors[i])>127 else (255, 255, 255), 2)


v=cv2.VideoCapture("video.MP4")
fourcc = cv2.VideoWriter_fourcc('D','I','V','X')
out = cv2.VideoWriter('output.avi',fourcc, 30, (int(v.get(3)),int(v.get(4))))
ret, img=v.read()
while ret:
    get_output(img)
    out.write(img)
    ret, img=v.read()
v.release()
out.release()
import cv2
import caffe
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import time

'''
Loading caffe models at once
'''

#Caffe model formats:: prototxt: description for the structure of the neuron network/ caffemodel: the model after training and learning.
proto = "D:/UNI/8th Semester/samples/Video-Summarization-master/Video-Summarization-master/VS-Python/Models/mobilenet_v2_deploy.prototxt"
model = "D:/UNI/8th Semester/samples/Video-Summarization-master/Video-Summarization-master/VS-Python/Models/mobilenet_v2.caffemodel"
caffe.set_mode_cpu()
#Load the net, net: is a set of layers.
net = caffe.Net(proto, model, caffe.TEST)
#Intialiaze the transformer for data pre-processing, it is required when there is a prototxt file.
#The conventional blob dimensions for batches of image data are number (batch size) N x no. of channels K x height H x width W.
#net.blobs: for input data and its propagation in the layers / net.blobs['data']: contains input data, an array of shape (1,3,244,244).
#The (:) is used to differentiate between the key and the value. 
#Specifically 'data' is the key and the tuple returned by 'net.blobs['data'].data.shape' (whose value for this particular example is 
#(1, 3, 244, 244) ) is the value. So basically the 'inputs' field of the transformer object is initialized with the key-value pair 
#'data', '(1, 3, 244, 244)'.
transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
#set_transpose(input data , new order we want the dimensions in)
#Swaps your image dimensions. Normally when an image library loads an image the dimensions of the loaded array are H x W x C 
#(where H is height, W is width and C is/are number of channels), but since caffe expects input in C x H x W, we transpose the data. 
#transpose('data, (2, 0, 1)) means transposing data such that 0th dimension is replaced by 2nd (height with channels), 1st by 
#0th (widht by height) and so on.
transformer.set_transpose('data',(2, 0, 1))
# Swap channels for the model as it accpets BGR format instead of RGB
transformer.set_channel_swap('data', (2, 1, 0))
# Makes the network perform its model calculations expecting images in the grayscale range 0-255 instead of the default 0-1.
transformer.set_raw_scale('data', 255)

print ("Caffe Model for Feature Extraction Loaded")
#########################################################################
proto1 = "D:/UNI/8th Semester/samples/Video-Summarization-master/Video-Summarization-master/VS-Python/Models/deploy.prototxt"
model1 = "D:/UNI/8th Semester/samples/Video-Summarization-master/Video-Summarization-master/VS-Python/Models/memnet.caffemodel"
net1 = caffe.Net(proto1, model1, caffe.TEST)

transformer1 = caffe.io.Transformer({'data':net1.blobs['data'].data.shape})
transformer1.set_transpose('data',(2, 0, 1))
transformer1.set_channel_swap('data', (2, 1, 0))
transformer1.set_raw_scale('data', 255)
#sets the data part of the blob in the fashion (batch size, channel value, height, width). The batch size is the no. of concurrent 
#images (or any data) that can be used for classification.
net1.blobs['data'].reshape(1, 3, 227, 227)


print ("Caffe Model for Memorability Prediction Loaded")

class Main: 
    
    def main():
        #print ("Inside main function")
        capture = cv2.VideoCapture(video_path)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))#getting total number of frames
        fps = int(capture.get(cv2.CAP_PROP_FPS))#getting frame rate
        print ("FPS = " , fps, "\t total frames = ",total_frames)
        m_scores = []
        m_scores.append([])
        m_scores.append([])
        ret, frame1 = capture.read()
        counter = 1
        ttt = 0
        while(True):
            start_t = time.time()
            ttt = ttt + 1
            ret2, frame2 = capture.read()
            if ret2 is True:
#            capture.release()
                
                distt = Shot_segmentation.segment(frame1,frame2)
                print ('Processing ... ', ttt, ', of ', total_frames)
                if distt >= 8000:#different images means the shot ended
                    m_scores = np.array(m_scores)
                    [rows,cols] = m_scores.shape
                    #get the keyframe from the shot, the one with the maximum memorability score
                    if cols > 0:
                        max_index = m_scores[0].argmax()
                        keyframe_number = int(m_scores[1][max_index])
                        keyframe = capture.set(0,keyframe_number)
    #                        print (keyframe_number , " \t########")
                        temp, keyframe = capture.read()
                        pathh = 'D:/UNI/8th Semester/samples/Video-Summarization-master/Video-Summarization-master/VS-Python/KeyFrames/Cam1/temp/frame/' + str(keyframe_number) + '.png'
                        print ("############## \t 'Writing key frame at" , pathh, "'\t##############", "\a")
                        cv2.imwrite(pathh,keyframe)
                    print ("Different images = " , distt , "\t" , counter)
                        
                    #Empty the score array for the next shot    
                    m_scores = []
                    m_scores.append([])
                    m_scores.append([])
                    
                    
                else:#same images ==> compute image memorability and store it
                    #print ("Similar images= " , distt , "\t" , counter)
                    m_value = Memorability_Prediction.mem_calculation((frame1))
                    m_scores[0].append(m_value)
                    m_scores[1].append(counter)
                counter = counter + (fps-14)
                frame1 = frame2
            end_t = time.time()
            totall = end_t - start_t
          
        
        capture.release()  


    
#############################################################################
class Shot_segmentation():
    '''
    def caclulate_distance(self,features1,features2):
        distt = euclidean_distances(features1,features2)
        return distt
    '''
    def segment(frame1,frame2):
        #print ("Inside segment function")
        start_time1 = time.time()
        resized_image1 = caffe.io.resize_image(frame1,[224,224])
        resized_image2 = caffe.io.resize_image(frame2,[224,224])
        
        
#        transformer.set_mean('data',img_mean)
        net.blobs['data'].reshape(1, 3, 224, 224)
        #Execute the transformation.
        net.blobs['data'].data[...] = transformer.preprocess('data', resized_image1)
        #Compute the output of the layer.
        net.forward()
        features1 = net.blobs['fc7'].data[0].reshape(1,1000)
        features1 = np.array(features1)
        net.blobs['data'].data[...] = transformer.preprocess('data', resized_image2)
        net.forward()
        features2 = net.blobs['fc7'].data[0].reshape(1,1000)
        features2 = np.array(features2)
        return euclidean_distances(features1,features2)
    
class Memorability_Prediction:
    
    def mem_calculation(frame1):
        #print ("Inside mem_calculation function")
        start_time1 = time.time()
        resized_image = caffe.io.resize_image(frame1,[227,227])
        net1.blobs['data'].data[...] = transformer1.preprocess('data', resized_image)
        
        
        value = net1.forward()
        value = value['fc8-euclidean']
        end_time1 = time.time()
        execution_time1 = end_time1 - start_time1
        #print ("*********** \t Execution Time in Memobarility = ", execution_time1, " secs \t***********")
        return value[0][0]
    
Main.main()


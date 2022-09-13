import _thread
import time
import threading
import PySimpleGUI as sg
import cv2
import torch
import os
import torch.nn.functional as F
from torchvision import transforms
import nltk #Sentiment Analysis via NLP
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#from playsound import playsound

#from emotion_recognition import get_emotion_reward

e = threading.Event()
got_reward = threading.Event()

reward = 0 



#------------------------------------------


def get_reward_keyboard():
	"""
	Returns an integer value read from keyboard input.
	Output:
		reward (int): Input value or 0 if the user did not provide a valid number.
	"""
	global reward
		
	while e.isSet() == False:
		#print(e.isSet())	
		rwd = input("Input reward value...\n")	
		try:
			reward = int(rwd)
			
			if reward <= 0: #If NEGATIVE reward, interrupt execution 
				e.set()
				print("KEYBOARD | Negative reward ", reward)
			else:
				print("KEYBOARD | Positive reward ", reward)	
			#print("Success")
			got_reward.set()
			break

		except: 
			print("ERROR: invalid reward value.")
			#break
		
			


import FacialExpression.config as cfg


emotion_reward = [0,1,-1]
emotion_reward_dict = {0: "neutral", 1: "positive", -1: "negative"}   #---------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ¿Dos diccionarios iguales?
Own_emotion_dict = {0: "neutral", 1: "positive", 2: "negative"}

eye_cascade = cv2.CascadeClassifier('./FacialExpression/classifier/haarcascade_eye.xml')
facecasc = cv2.CascadeClassifier('./FacialExpression/classifier/haarcascade_frontalface_default.xml')
 
def display():
    """
    Function that displays in real time the results of the analysis of the emotions captured in a video.

    Returns
    -------
    emotions_total : (List)
        Contains the number of times each possible emotion has been detected within a video. 
        To interpret it, it has been stored as follows [neutral, positive, negative].

    """

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    path_model = './FacialExpression/pretrained_models/model_display_1.pt'
  
    if torch.cuda.is_available():
        model = torch.load(path_model,map_location=torch.device('cuda:0')) 
        model.to(device) #-------------------------------------------------------------------- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ¿Necesario tras map=location=?
    else: 
        model = torch.load(path_model,map_location=torch.device('cpu'))
 
    model.eval()
    
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    count_frame = 0
    emotions = [0,0,0]
    emotions_total = [0,0,0]
    # command line request for the path to the video file

    frame_rate = cfg.FRAME_RATE_VID
    try:
      #video_path = str(input("Enter the full path to the video file: "))
      #video_path = './videos/angry.mp4'
      video_path = './videos/happy.mov'
      #video_path = './videos/asco.mp4'

    except Exception as error:
      print("Error:", error) #--------------------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! e -> error (e ya se usa para el flag)
    
    cap = cv2.VideoCapture(video_path)

    while not cap.isOpened():

        cap = cv2.VideoCapture(video_path)
        cv2.waitKey(1000)
        
   
    #while e.isSet() == False:
    while got_reward.isSet() == False:
        
        # Find haar cascade to draw bounding box around face

        ret, frame = cap.read()  
        frame = cv2.resize(frame,(800,480),interpolation = cv2.INTER_CUBIC)  #---------------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! same size for predicting  

        if not ret:

            break
       
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
        
        
        color = [(204, 102, 0),(0, 204, 0),(0, 0, 204)]
        for (x, y, w, h) in faces:
           

            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray) # detecto tb los ojos porque aveces no funciona bien la cara,
                            
            if len(eyes) > 0: 

                count_frame += 1
                img = cv2.resize(roi_gray, (224,224))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
             
                normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                transform = transforms.ToTensor()
                
                cropped_img = transform(img)
                cropped_img = normalize(cropped_img)
                cropped_img = cropped_img.reshape((1, 3, 224,224)) 
                cropped_img = cropped_img.to(device)
               
                output = model(cropped_img)
                output = F.softmax(output.data,dim=1)
                
                prediction = int(torch.max(output.data, 1)[1].cpu().numpy())

                emotions[prediction] = emotions[prediction]+1
                emotions_total[prediction] = emotions_total[prediction]+1


                if (count_frame==frame_rate):

                    prediction = emotions.index(max(emotions))
                    cv2.rectangle(frame, (x, y - 20), (x+w, y+h + 10), color[prediction], 6)
                           
                    count_frame = 0
                    emotions = [0,0,0]
                    break
            

        frame = cv2.resize(frame,(800,480),interpolation = cv2.INTER_CUBIC)
        cv2.putText(frame, Own_emotion_dict[prediction], (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 6, cv2.LINE_AA)    
        cv2.imshow('Video', frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # print(emotions_total)
            break

        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # print(emotions_total)
            print("Finished video")
            #cap.release()
            cv2.destroyAllWindows()
            
            return emotions_total

            break

    cap.release() #------------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Si se llega al return-break anterior, no se ejecutan estas líneas
    cv2.destroyAllWindows()

    
    
def get_emotion_reward(): 
    """
    Function that returns a numerical reward according to the emotion detected in a video of a person. 
    It is considered that for each video there is a particular emotion. 
    It is taken into account that it is not natural to maintain an emotion for a long period of time,
    therefore, in the videos where the neutral emotion is detected as the main emotion, 
    it is analysed if in a good fraction of the frames positive or negative emotions have been detected.

    Returns
    -------
    final_prediction : (Int)
        Variable returning the main emotion detected in the video. 
        Each emotion is assigned a numerical value, as indicated in the emotion_reward_dict dictionary.

    """
    global reward

    prediction = display()    
    
    if prediction is None:  #----------------------------!!!!!!!!!!!!!!!!!!!!!! if display() is interrupted by other interface, prediction is None, and the rest of the function is useless, so we return to the main thread.
    	print("Video interrupted")
    	return 

    
    total_predictions = sum(prediction)
    scale_predictions = [pred/total_predictions for pred in prediction]
    
    if prediction.index(max(prediction)) == 0: 
        if scale_predictions[1]>cfg.THRESHOLD_EMOTION:
            final_prediction = 1
        elif scale_predictions[2]>cfg.THRESHOLD_EMOTION:
            final_prediction = -1
        else: 
            final_prediction = 0
    else: 
      reward = emotion_reward[prediction.index(max(prediction))]
    
    print("Most of the emotions captured in the video are", emotion_reward_dict[reward] )
    
    if reward <= 0:
    	print("VIDEO (Facial emotion) | Negative/neutral reward.")
    	e.set()
    else: 
    	print("VIDEO (Facial emotion) | Positive reward.")	    

    

"""

----------------------------------------------------------------------------------------------


"""


def interfaces():
	_thread.start_new_thread(get_reward_keyboard, tuple())    #--------> from terminal to GUI so that it can be easily parallelized with other interfaces
	#_thread.start_new_thread(get_sentiment_keyboard, tuple()) #--------> from terminal to GUI so that it can be easily parallelized with other interfaces



#Perform an action
def perform_action(action=0):
    T0 = time.time()
    
    time_to_perform = 10
    print("\nAction ", action, "| Time to perform: ", time_to_perform)
    
    while e.isSet() == False:        
        e.wait(1)
        print("...")
        
        #print(time.time()-T0)
        if (time.time() - T0) > time_to_perform:
        	print("Completed action!\n")
        	e.set()
        
        #e.wait(6)
        #print("Finished")
        
    print("FINISHED ACTION (either timeout or interrupted)")

    #_thread.interrupt_main() # kill the raw_input thread



def main(action):
	_thread.start_new_thread(perform_action, (action,))
	_thread.start_new_thread(interfaces, tuple())



def ta3(action):
	global reward
	
	try:
		#_thread.start_new_thread(main, (action,))
		_thread.start_new_thread(perform_action, (action,))
		_thread.start_new_thread(interfaces, tuple())
		while e.isSet() == False:
			get_emotion_reward()    #---------------------------!!!!!!!!!!!!!!!!!!!!!!!!! openCV en el main
			e.wait(1)
	
	except KeyboardInterrupt:
		_thread.interrupt_main()
		pass		


	#print("EEEE: ", e.isSet())
	time.sleep(0.2)
	e.clear()
	got_reward.clear()
	#print("E AFTER CLEAR: ", e.isSet())

	
	return reward




action = 3

print("\nAAAAAAAAAAAAAAAAA")
a = ta3(action) 
print("Returned reward A: ", a)

print("Sleeping 5 seconds...")
time.sleep(5)

print("\nBBBBBBBBBBBBBBBBBBBBBBBBB")
b = ta3(5)
print("Return reward B: ", b)


"""
print("\nCCCCCCCCCCCCCCCCCCCCC")
c = ta3(1)
print("Return reward C: ", c)

#get_reward(interfaces)
"""

#print("FINAL REWARD: ", reward)



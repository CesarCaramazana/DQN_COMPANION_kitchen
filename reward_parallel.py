import _thread
import time
import threading
import PySimpleGUI as sg

import nltk #Sentiment Analysis via NLP
from nltk.sentiment.vader import SentimentIntensityAnalyzer


e = threading.Event()
reward = 0 




def get_reward_GUI_th():
	"""
	Gets a reward signal from a Graphical User Interface with three buttons: Negative (-1), Neutral (0) and Positive (+1).
	
	Output:
		reward (int): reward value provided by the user. 
	
	"""
	button_size = (25, 15)	
	global reward
	
	#Button layout (as a matrix)
	interface = [[
	sg.Button('NEGATIVE', size=button_size, key='Negative', button_color='red'), 
	sg.Button('NEUTRAL', key='Neutral', size=button_size, button_color='gray'), 
	sg.Button('POSITIVE', key='Positive', size=button_size, button_color='blue')
	]]
	
	#Generate window with the button layout
	window = sg.Window('Interface', interface, background_color='black', return_keyboard_events=True).Finalize()	
	
	while e.isSet() == False:
		event, values = window.read()
		
		if event == sg.WIN_CLOSED or event == 'Negative':
			reward = -1
			e.set()
			break
					
		elif event == sg.WIN_CLOSED or event == 'Neutral':
			reward = 0
			e.set()
			break
		
		elif event == sg.WIN_CLOSED or event == 'Positive':
			reward = +1
			#e.set()
			break
		
		elif event == '1:10': #Keyboard press 1
			window['Negative'].click()
		
		elif event == '2:11': #Keyboard press 2
			window['Neutral'].click()
		
		elif event == '3:12': #Keyboard press 3
			window['Positive'].click()
			
		elif event == 'q:24': #Keyboard press q
			break	
			
	#print("out of while GUI")
	
	window.close()


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
			#print("Success")
			break

		except: 
			print("ERROR: invalid reward value.")
			#break



def get_sentiment_keyboard():
	"""
	Returns an integer reward value extracted from the sentiment analysis of an input sentence.
	
	Output:
		reward: (int) value +1 if text was positive, -1 if text was negative, 0 if neutral.

	"""
	global reward
	
	sentence = input("Type text\n")
	analyzer = SentimentIntensityAnalyzer()
	
	score = analyzer.polarity_scores(sentence)
	print("Score : ", score['compound'])
		
	if score['compound'] > 0.1: reward = 1
	elif score['compound'] < -0.1: reward = -1
	else : reward = 0
	
	e.set()



def interfaces():
	#_thread.start_new_thread(get_reward_GUI_th, tuple())
	#_thread.start_new_thread(get_reward_KIVY, tuple())	
	_thread.start_new_thread(get_reward_keyboard, tuple())    #--------> from terminal to GUI so that it can be easily parallelized with other interfaces
	#_thread.start_new_thread(get_sentiment_keyboard, tuple()) #--------> from terminal to GUI so that it can be easily parallelized with other interfaces



#Perform an action
def perform_action(action=0):
    T0 = time.time()
    
    time_to_perform = 5
    print("\nAction ", action, "| Time to perform: ", time_to_perform)
    
    while e.isSet() == False:        
        e.wait(1)
        print("...")
        
        #print(time.time()-T0)
        if (time.time() - T0) > time_to_perform:
        	print("Finished action!\n")
        	e.set()
        
        #e.wait(6)
        #print("Finished")
    print("FINISHED ACTION (either timeout or interrupted)")

    #_thread.interrupt_main() # kill the raw_input thread



def main(action):
	_thread.start_new_thread(perform_action, (action,))
	_thread.start_new_thread(interfaces, tuple())
	#_thread.start_new_thread(get_reward, (interfaces,))
	
	


def ta3(action):
	global reward
	
	try:
		#_thread.start_new_thread(main, (action,))
		_thread.start_new_thread(perform_action, (action,))
		_thread.start_new_thread(interfaces, tuple())
		while e.isSet() == False:
			e.wait(1)
	
	except KeyboardInterrupt:
		_thread.interrupt_main()
		pass		


	#print("EEEE: ", e.isSet())
	#time.sleep(0.1)
	e.clear()
	#print("E AFTER CLEAR: ", e.isSet())

	
	return reward



action = 3

print("\nAAAAAAAAAAAAAAAAA")
a = ta3(action) 
print("Returned reward A: ", a)


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



import _thread
import time
import threading
import PySimpleGUI as sg

import nltk #Sentiment Analysis via NLP
from nltk.sentiment.vader import SentimentIntensityAnalyzer



e = threading.Event()
reward = 0




def get_reward_GUI():
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
			e.set()
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
			e.set()
			#print("Success")
			break

		except: 
			print("ERROR: invalid reward value.")
			#break
		
		#print("out of try KEY")	
	
	#print("out of while KEY")
	#_thread.interrupt_main()

def get_reward_keyboard():
	global reward

	sg.theme('SandyBeach')
	
	layout = [[sg.Text('Reward: '), sg.InputText()], [sg.Submit(), sg.Cancel()]]
	
	window = sg.Window('Interface', layout).Finalize()
	
	while e.isSet() == False:
		event, values = window.read()
	
		try:
			reward = int(values[0])
			break
	
		except:
			print("ERROR")		
	window.close()
	
	print(values[0])
	
	e.set()


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
	#print("end")



def interfaces():
	_thread.start_new_thread(get_reward_GUI, tuple())
	_thread.start_new_thread(get_reward_keyboard, tuple())    #--------> from terminal to GUI so that it can be easily parallelized with other interfaces
	#_thread.start_new_thread(get_sentiment_keyboard, tuple()) #--------> from terminal to GUI so that it can be easily parallelized with other interfaces


def get_reward(main):
	try:
		_thread.start_new_thread(main, tuple())
		while e.isSet() == False:
			e.wait(1)
	except KeyboardInterrupt:
		pass	


#keyboard_input_GUI()

get_reward(interfaces)

"""
try:
    _thread.start_new_thread(main, tuple())
    while e.isSet() == False:    	
    	e.wait(10)
    	#time.sleep(1) 
    	print(e.isSet())
except KeyboardInterrupt:
    pass 
"""

print("FINAL REWARD: ", reward)



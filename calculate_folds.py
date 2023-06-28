import numpy as np
import pickle
import random
import glob


root = "./video_annotations/dataset/*"

videos = glob.glob(root)

# print("Videos: ", videos)

video_idx = 0
# print(len(videos))


cereals = []
coffee_with_milk = []
coffee_without_milk = []
toast_nutella = []
toast_tomato = []
toast_butterjam = []
nesquik = []


for video_idx in range(len(videos)):
    # print(video_idx)
    video = videos[video_idx]
    # print(video)
    
    # load_path = video + "/labels_updated.pkl"
    load_path = video + "/labels_margins"
    
    annotations = np.load(load_path, allow_pickle=True)
    # old_annotations = np.load(load_path_old, allow_pickle=True)
    
    atomic_actions = set(annotations['label'].values.tolist())
    
    # print(atomic_actions)
    
    # Milk?
    if 8 in atomic_actions:
        #Yes, milk
        
        #Nesquik?
        if 4 in atomic_actions:
            #Yes, milk + nesquik = Nesquik
            nesquik.append(video)
        
        else:
            #No, nesquik.
            # Maybe coffee?
            if 3 in atomic_actions:
                #Yes, coffee with milk
                coffee_with_milk.append(video)
            
            else:
                #No coffee, it must be a bowl of cereals
                cereals.append(video)    
        
    else:
        #No milk
        #Coffee?
        if 3 in atomic_actions:
            #Yes, a coffee without milk
            coffee_without_milk.append(video)
        
        else:
            #No milk and no coffee -> Toast
            #Butter?
            if 16 in atomic_actions:
                #Yes, toast with butter & jam
                toast_butterjam.append(video)
            
            else:
                #No butter.
                #Tomato sauce?
                if 18 in atomic_actions:
                    #Yes, toast with tomato
                    toast_tomato.append(video)
                
                else:
                    #No butter, no tomato, it must be a nutella toast
                    toast_nutella.append(video)
    

    
    # print(annotations)
    # print("\n", old_annotations)

print("\nCEREALS", *cereals, sep='\n')
print("\nCOFFEE WITH MILK ", *coffee_with_milk, sep='\n')
print("\nCOFFEE WITHOUT MILK ", *coffee_without_milk, sep='\n')
print("\nNESQUIK ", *nesquik, sep='\n')
print("\nTOAST NUTELLA ", *toast_nutella, sep='\n')
print("\nTOAST TOMATO ", *toast_tomato, sep='\n')
print("\nTOAST BUTTERJAM ", *toast_butterjam, sep='\n')


total_lenghts = len(cereals) + len(coffee_with_milk) + len(coffee_without_milk) + len(nesquik) + len(toast_butterjam) + len(toast_nutella) + len(toast_tomato)
print("\n", total_lenghts)



# cereals = []
# coffee_with_milk = []
# coffee_without_milk = []
# toast_nutella = []
# toast_tomato = []
# toast_butterjam = []
# nesquik = []
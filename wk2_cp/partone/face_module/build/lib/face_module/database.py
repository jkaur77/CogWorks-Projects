import numpy as np
from facenet_models import FacenetModel

class Profile() :
    def __init__(self, name):
        self.name = name
        self.descriptions = []
        
    def data(self) :
        return (self.name, self.descriptions)
    
    def add(self, description) :
        self.descriptions.append(description)


def database(names) :

    db = {}

    model = FacenetModel()

    for name in names : 
        name_path = name + ".npy"
        vectorized_images = np.load(name_path, allow_pickle=True)

        descriptions = []

        for image in vectorized_images :
            # assumes ``pic`` is a numpy array of shape (R, C, 3) (RGB is the last dimension)
            boxes, probabilities, landmarks = model.detect(image)
            
            # only if probability is > 0.95 
            #ensure that only one description per picture is added if there are multiple people in a db loading photo
            highest =  np.argmax(probabilities)
            descriptions = model.compute_descriptors(image, [boxes[highest]])
            

        person = Profile(name)
        
        # prev vector in array in array 
        # to vector in array
        for description in descriptions : 
            person.add(description)

        db[name] = person

    return db



        

    

import numpy as np
from facenet_models import FacenetModel

class Profile() :
  def __init__(self, name):
    self.name = name
    self.descriptions = []
      
  def data(self) :
    return self.name, self.descriptions
  
  def add(self, description) :
    self.descriptions.append(description)

  def avg(self):
    return np.mean(self.descriptions, axis=0)


class Database:
  def __init__(self):
    self.db = {}

    
  # create a Profile for the person
  def generatePerson(self, name):
    # returns a profile object
    return Profile(name)

  # add record to database
  def addRecord(self, name: str, vector: np.ndarray):
    db = self.db
    if name in db: 
      db[name].add(vector)
    elif name not in db: # uses generatePerson to create profile 
      db[name] = self.generatePerson(name)
      db[name].add(vector)
    
  # loops over each record in database
  # generates "average vector"
  # does dot product and evaluates who it is :)
  def predict(self, unknown: np.ndarray):
    # key, value = name, profile
    min_dist = int(1e9)
    min_name = ""
    for name, profile in self.db.items():

      av = profile.avg() 
      cos_dist = self.get_cos_dist(av, unknown)

      if cos_dist < min_dist: 
        min_dist = cos_dist
        min_name = name
        # person is unknown, user prompted to enter name

    if min_dist < 0.6:
      return min_name
    return "Unknown"
  
  def get_cos_dist(self, av, unknown):
    result = unknown @ av
    return 1 - result/(np.linalg.norm(av) * np.linalg.norm(unknown))
  

def database(names) :

  db = Database()

  model = FacenetModel()

  for name in names : 
    name_path = name + ".npy"
    vectorized_images = np.load(name_path, allow_pickle=True)

    descriptions = []

    for image in vectorized_images :
      # assumes ``pic`` is a numpy array of shape (R, C, 3) (RGB is the last dimension)
      boxes, probabilities, landmarks = model.detect(image)
      
      # only if probability is > 0.95 
      # ensure that only one description per picture is added if there are multiple people in a db loading photo
      highest =  np.argmax(probabilities)
    descriptions = model.compute_descriptors(image, [boxes[highest]])
        
    
    # prev vector in array in array 
    # to vector in array
    for description in descriptions : 
      db.addRecord(name, description)

  return db



        

    

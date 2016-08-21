import imlist
import SIFT
import os 
import numpy as np
from numpy import *
import cv2


import pickle

path='/images/'
images=imlist.get_imlist('/images/')
featlist=[]
deslist=[]
print(images)

def read_features_from_file(filename):
	f = loadtxt(filename,dtype="string")
	return f[:,:] # feature locations, descriptors

def match(desc1,desc2):

	desc1 = array([d/linalg.norm(d) for d in desc1])
	desc2 = array([d/linalg.norm(d) for d in desc2])
	dist_ratio = 0.6
	desc1_size = desc1.shape
	matchscores = zeros((desc1_size[0],1),'int')
	desc2t = desc2.T # precompute matrix transpose
	for i in range(desc1_size[0]):
		dotprods = dot(desc1[i,:],desc2t) # vector of dot products
		dotprods = 0.9999*dotprods
		# inverse cosine and sort, return index for features in second image
		indx = argsort(arccos(dotprods))
# check if nearest neighbor has angle less than dist_ratio times 2nd
		if arccos(dotprods)[indx[0]] < dist_ratio * arccos(dotprods)[indx[1]]:
			matchscores[i] = int(indx[0])
	return matchscores
	
def match_twosided(desc1,desc2):
	matches_12 = match(desc1,desc2)
	matches_21 = match(desc2,desc1)
	ndx_12 = matches_12.nonzero()[0]
# remove matches that are not symmetric
	for n in ndx_12:
		if matches_21[int(matches_12[n])] != n:
			matches_12[n] = 0
	return matches_12

for img in images:
	img_txt=os.path.splitext(img)[0]
	featlist.append(img_txt+"feat"+".txt")
	deslist.append(img_txt+"des"+".txt")
	
	feature_file=img_txt+"feat"+".txt"
	des_file=img_txt+"des"+".txt"
	
	f_kp=open(img_txt+"feat"+".txt",'w+')
	f_des=open(img_txt+"des"+".txt",'w+')
	kp,des=SIFT.SIFT(path+img)
	
	np.savetxt(feature_file,kp,fmt="%s")
	np.savetxt(des_file,des,fmt="%s")
	
nbr_images=len(images)
matchscores = np.zeros((nbr_images,nbr_images))
for i in range(nbr_images):
	for j in range(i,nbr_images):
		#print 'comparing', imlist[i], imlist[j]
		l1=read_features_from_file(featlist[i])
		l2=read_features_from_file(featlist[j])
		d1=read_features_from_file(deslist[i])
		d2=read_features_from_file(deslist[j])
		
	
		des1 = d1.astype(np.float)
		des2=d2.astype(np.float)
		
		matches=match_twosided(des1,des2)
		nbr_matches = sum(matches > 0)
		#print 'number of matches = ', nbr_matches
		matchscores[i,j] = nbr_matches
		
for i in range(nbr_images):
	for j in range(i+1,nbr_images): # no need to copy diagonal
		matchscores[j,i] = matchscores[i,j]

with open('matchscores.pickle', 'wb') as handle:
  pickle.dump(matchscores, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
print(matchscores)

		
		
		

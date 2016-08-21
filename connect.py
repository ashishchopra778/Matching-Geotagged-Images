import pydot
import imlist
import  pickle
from PIL import Image
import numpy as np
from pylab import *

threshold = 0 # min number of matches needed to create link
g = pydot.Dot(graph_type='graph') # don't want the default directed graph

imlist=imlist.get_imlist("/images/")
print(imlist)

nbr_images=len(imlist)
path="/home/ashish/dip/chap2/whitehouse/"

with open('matchscores.pickle', 'rb') as handle:
  matchscores = pickle.load(handle)
  print(matchscores)

for i in range(nbr_images):
	for j in range(i+1,nbr_images):
		if matchscores[i,j] > threshold:
#first image in pair
			im = Image.open(imlist[i])
			im.thumbnail((100,100))
			filename = str(i)+'.png'
			im.save(filename) # need temporary files of the right size
			g.add_node(pydot.Node(str(i),shape='rectangle',image=path+filename))
# second image in pair
			im = Image.open(imlist[j])
			im.thumbnail((100,100))
			filename = str(j)+'.png'
			im.save(filename) # need temporary files of the right size
			g.add_node(pydot.Node(str(j),shape='rectangle',image=path+filename))
			g.add_edge(pydot.Edge(str(i),str(j)))
g.write_png('connection.png')
final_img=np.array(Image.open("connection.png"))
imshow(final_img)
show()

import numpy as np
import matplotlib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


infile = open( 'score', 'r' )
array = []
phrase_map =[] 
ct = 0

for line in infile:
    phrase = line.split( ' ||| ' )[0].strip().decode( 'utf-8' )
    phrase_map.append( phrase )
    linevec = 5 * map( float, line.split( ' ||| ')[1].strip().split() )
    array.append( linevec )

infile.close()

array = np.array( array )
pca = PCA(n_components=2)
a = pca.fit_transform( array )

plt.suptitle('Unsupervised RAE Visualization(dim=10)', fontsize=16 )
plt.scatter(a[:,0], a[:,1] )
for i in xrange( a.shape[0] ):
    plt.text(a[i,0], a[i,1], phrase_map[i], color='red', fontsize=16 )
axes = plt.gca()
axes.set_xlim([0,1.6])
axes.set_ylim([-1,0.1])
plt.show()

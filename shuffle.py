import numpy as np 


rand = np.random.RandomState()
l = [0,1,2,3,4,5,6,7,8,9]
l = np.array(l)
shuffle = rand.permutation(len(l))
shuffled_arr = l[shuffle]
print("before shuffling:")
print(l)
print("After shuffling:")
print(shuffled_arr)


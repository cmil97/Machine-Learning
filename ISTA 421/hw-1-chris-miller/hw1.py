"""
Created on Tue Aug 28 11:21:10 2018

@author: chrismiller
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_1d_array(arr):
    """
    Plots a one-dimensional array
    :param arr: input ndarray
    :return: none
    """
    plt.figure() # create figure object
    plt.plot(arr) # fill figure with plot of the input arr

def exercise6(infile='humu.txt', outfile='out.txt'):
    """
    perform tasks 1 - 10:
        1 read the input txt file
        2 print the input txt file type
        3 print the size of the txt file
        4 print the shape of the txt file
        5 plot black and white version of input txt file
        6 plot black and white version of random data
        7 save the random data, load it and plot it again
        8 print the min and max values in walk.txt
        9 scale the walk.txt data and save it as walk_scale01.txt
        10 plot walk and the scaled walk data
    :param: infile (string): name of a text file containing a matrix of data
            outfile (string): name of a text file to store random data
    :return: none
    """
    data = np.loadtxt(infile) # ($1) read the txt file
    
    print(type(data)) # ($2) 
    
    print(data.size) # ($3)
    
    h, w = data.shape # store the dimensions of the matrix from the input file
    print(h, w) # ($4)
    
    plt.figure() # create a figure object
    plt.imshow(data, cmap='gray') # ($5)
    plt.show() # halt program and display the plot
    
    print(plt.cm.cmapname) # print colormap
     
    randomData = np.random.rand(366, 576) # create random 366 by 576 matrix
    
    #plt.figure()
    #plt.imshow(randomData, cmap='gray') # ($6)
    #plt.show()
    #
    #np.savetxt(outfile, randomData) # save random data to outfile
    
    confirm = np.loadtxt(outfile) # read the outfile
    plt.figure()
    plt.imshow(confirm, cmap='gray') # ($7)
    plt.show()
    
    walk = np.loadtxt('walk.txt') # read walk.txt
    print(np.amin(walk)) # print the minimum value
    print(np.amax(walk)) # ($8) print the maximum value
    
    walkScale = (walk + 1) / 6 # linearly scale range from [-1, 5] to [0, 1]
    #np.savetxt('walk_scale01.txt', walkScale) # ($9)
    
    print(np.amin(walkScale)) # print the minimum value
    print(np.amax(walkScale)) # print the maximum value
    
    plot_1d_array(walk) # print walk
    plot_1d_array(walkScale) # ($10)
    
def exercise9():
    """
    simulate rolling two 6-sided dice 1000 times, then record results
    :param: none
    :return: none
    """
    results = []
    
    np.random.seed(seed=8) # set seed to 8
    
    for x in range(10): # repeat procedure a total of 10 times
        roll2dice = np.random.randint(1, 7, (1000, 2)) # dice roll data
        
        double6 = np.array([6, 6])
        count = 0
        
        for row in roll2dice: # check each roll for [6 6] combination
            if np.all(row == double6):
                count += 1
        
        results.append(count) # store results
        
    print(results) # ($1)
    
    results2 = []
    np.random.seed(seed=8) # reset seed
    
    for x in range(10): # repeat procedure a total of 10 times
        roll2dice = np.random.randint(1, 7, (1000, 2)) # dice roll data
        
        double6 = np.array([6, 6])
        count = 0
        
        for row in roll2dice: # check each roll for [6 6] combination
            if np.all(row == double6):
                count += 1
        
        results2.append(count) # store results
        
    print(np.all(results == results2)) # compare both results

def exercise10():
     """
     finish me
     """
     np.random.seed(seed=5) # set seed to 5
     a = np.random.rand(3, 1)
     b = np.random.rand(3, 1)
     print(a, '\n\n', b, '\n')
     
     print('a + b =\n')
     print(a + b)
     
     print('\na o b =\n')
     print(np.multiply(a, b))
     
     print("\na'b =\n")
     print(np.dot(np.transpose(a), b), '\n')
     
     np.random.seed(seed=2) # set seed to 2
     X = np.random.rand(3, 3)
     print('X = \n\n', X)
     
     print("\na'X =\n")
     print(np.dot(np.transpose(a), X), '\n')
     
     print("\na'Xb =\n")
     print(np.dot(np.dot(np.transpose(a), X), b), '\n')
     
     print("inv(X) =\n")
     print(np.linalg.inv(X), '\n')
    
def exercise11():
    x = np.linspace(0, 10, 1000)
    plt.plot(x, np.sin(x))
    plt.title('Sine Function for x from 0.0 to 10.0')
    plt.xlabel('x values')
    plt.ylabel('sin(x)')
    plt.axis('tight')
    #plt.show()
    
    plt.savefig('sine_plot.pdf', format='pdf')
    
#exercise6('humu.txt', 'out.txt')
#exercise9()
#exercise10()
exercise11()
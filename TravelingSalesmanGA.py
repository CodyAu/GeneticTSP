import numpy as np
import math
import matplotlib as mpl
import random as rnd
from matplotlib import pyplot as plt

fig, ((ax1, ax2)) = plt.subplots(1, 2)

#fitness evaluation function to compute distance of the path given by the gene sequence
def travelingSalesman(sequence, coordinates):
    
    n = len(sequence)
    distance = 0 

    #basic distance between 2 points formula, mapping x and y coordinates to each point in the sequence using the coordinate
    for i in range(0, n-1):
        distance += math.sqrt((coordinates[int(sequence[i+1])][0] - coordinates[int(sequence[i])][0])**2 + (coordinates[int(sequence[i+1])][1] - coordinates[int(sequence[i])][1])**2)

    #loop back to the first point in the sequence
    distance += math.sqrt((coordinates[int(sequence[n-1])][0] - coordinates[0][0])**2 + (coordinates[int(sequence[n-1])][1] - coordinates[0][1])**2)

    return distance;

#genetic algorithm to find shortest possible path while still traversing every city
#requires an initial sequence that contains all locations as integers 0, n-1
#requires an array of coordinates that map the locations to a 2d space
#requires a desired population size and maximum number of iterations
def travelingSalesmanGA(initSeq, coordinates, population_size, maxits):

    #initialize population
    population = np.zeros([population_size, len(initSeq)])
    #initialize history arrays to keep track
    fitnessHistory = np.zeros(maxits + 1)
    bestFitnessHistory = np.zeros(maxits + 1)

    #set population with by shuffling the initial sequence
    for i in range(0, population_size):
        population[i] = rnd.sample(initSeq, len(initSeq))
    new_population = np.zeros([population_size, len(initSeq)])

    #initial fitness evaluation, fitness is determined by the distance the whole path takes
    fitness = np.zeros(population_size)
    population_best_fitness = math.inf
    for i in range(0, population_size):
        score = travelingSalesman(population[i], coordinates)
        fitness[i] = score
        #keep track of the population's current best fitness and the gene that gives it
        if fitness[i] < population_best_fitness:
            population_best_fitness = fitness[i]
            population_best_gene = np.copy(population[i])

    #keep track of the best over all populations ever
    best_fitness = population_best_fitness
    best_gene = np.copy(population_best_gene)

    #normalize fitness
    normalized_fitness = fitness/max(fitness)

    #set ga random percentages
    crossover_percent = 0.4
    mutation_percent = 0.05

    iteration = 0
    while iteration < maxits:
        iteration = iteration + 1

        gene_count_by_fitness = np.zeros(population_size+1) #the number of counts assigned based on fitness
        for i in range(0, population_size):
            gene_count_by_fitness[i+1] = gene_count_by_fitness[i] + int(normalized_fitness[i]*100) #the count is proportional to the normalized fitness
            
        gene_selection_lottery_array = np.zeros(int(gene_count_by_fitness[population_size])) #this will store the gene labels for selection
            
        for i in range(0, population_size):
            gene_selection_lottery_array[int(gene_count_by_fitness[i]):int(gene_count_by_fitness[i+1])] = i #and here we put gene i in the selection array a number of times proportional to fitness
        
        #loop to create new generation, only iterate through half since breeding occurs in pairs
        for i in range(0, int(population_size/2)):
            
            #step 1 is selection: select two genes (biased towards better fitness)
            lottery_tickets = np.random.randint(max(gene_count_by_fitness), size=2)
            
            parent1_index = int(gene_selection_lottery_array[lottery_tickets[0]])
            parent2_index = int(gene_selection_lottery_array[lottery_tickets[1]])
            
            parent1 = population[parent1_index,:]
            parent2 = population[parent2_index,:]
            
            #breeding via crossover, may or may not occur based on percentage
            dice_role = np.random.rand(1)
            if dice_role[0] <= crossover_percent:  

                temp11 = []
                temp12 = []
                temp21 = []
                temp22 = []
                offspring1 = []
                offspring2 = []

                #randomly decided section to save from original
                rnd1 = int(rnd.random() * len(parent1))
                rnd2 = int(rnd.random() * len(parent1))
                start = min(rnd1, rnd2)
                end = max(rnd1, rnd2)

                #crossover by keeping a random section of the original parent in the offspring
                for x in range(start, end):
                    temp11.append(parent1[x])
                    temp21.append(parent2[x])

                #fill in the rest from the other parent, such that there are no repeated locations               
                temp12 = [item for item in parent2 if item not in temp11]
                temp22 = [item for item in parent1 if item not in temp21]

                offspring1 = temp11 + temp12
                offspring2 = temp21 + temp22

                #add the offspring to the new population
                new_population[2*i,:] = offspring1
                new_population[2*i+1,:] = offspring2
            else: #if no breeding occurs keep the parents
                new_population[2*i,:] = parent1
                new_population[2*i+1,:] = parent2
        
            #mutation that may or may not happen based on percentage
            dice_role = np.random.rand(1)
            #every location in a gene has a chance to swap with another location in the gene
            for x in range(0,len(parent1)):
                if dice_role <= mutation_percent:
                    rnd1 = rnd.randrange(0, len(initSeq) - 1,1)
                    tempCoord = new_population[i][x]
                    new_population[i][x] = new_population[i][rnd1]
                    new_population[i][rnd1] = tempCoord
             
        #update population with new population
        population = np.copy(new_population)

        #fitness re-evaluation
        population_best_fitness = 100
        for i in range(0, population_size):
            score = travelingSalesman(population[i], coordinates)
            fitness[i] = score
            fitnessHistory[iteration] = fitness[i]
            if fitness[i] < population_best_fitness:
                population_best_fitness = fitness[i]
                population_best_gene = np.copy(population[i])

        #best over all populations
        if population_best_fitness < best_fitness:
            best_fitness = population_best_fitness
            best_gene = np.copy(population_best_gene)

        bestFitnessHistory[iteration] = best_fitness

        #normalize fitness
        normalized_fitness = fitness/max(fitness)
        
        #plot the current iterations best gene
        pxVals = []
        pyVals = []

        for i in range(0,len(population_best_gene)):
            pxVals.append(coordinates[int(population_best_gene[i])][0])
            pyVals.append(coordinates[int(population_best_gene[i])][1])

        pxVals.append(coordinates[int(population_best_gene[0])][0])
        pyVals.append(coordinates[int(population_best_gene[0])][1])

        #plot the best gene overall from any iteration
        bxVals = []
        byVals = []

        for i in range(0,len(best_gene)):
            bxVals.append(coordinates[int(best_gene[i])][0])
            byVals.append(coordinates[int(best_gene[i])][1])

        bxVals.append(coordinates[int(best_gene[0])][0])
        byVals.append(coordinates[int(best_gene[0])][1])

        ax1.cla()
        ax1.plot(pxVals,pyVals)
        ax2.cla()
        ax2.plot(bxVals,byVals)
        ax1.set_title('Best at current iteration',fontsize=11)
        ax2.set_title('Best overall',fontsize=11)
        fig.suptitle(['GA Traveling Salesman Iteration # ',iteration])
        plt.pause(0.000125)

        #print the numerical result
        print("iteration ", iteration)
        print("best fitness ", best_fitness)

    plt.show()

#main

n = 20 #number of cities

#random cities example
"""
x = np.random.rand(n)
y = np.random.rand(n)
s0 = range(0,n) #intial state, start by visiting them in order
"""

#circle example (optimal solution is just to traverse the circle)
phi = np.linspace(0, 2*math.pi, n, endpoint=False)
x = np.cos(phi)
y = np.sin(phi)

#initialize 2d array to hold x y values in pairs for each location
coordinates = [0] * np.size(x)
for i in range(np.size(x)):
    coordinates[i] = [0] * 2
for i in range(0,np.size(x)):
    coordinates[i][0] = x[i];
    coordinates[i][1] = y[i];

#initialize initial sequence, integers 0 to n-1 to represent locations to visit
initSeq = range(0, n)
#set number of max iterations the genetic algorithm will perform
maxits = 200
#call genetic algorithm
travelingSalesmanGA(initSeq, coordinates, 200, maxits)

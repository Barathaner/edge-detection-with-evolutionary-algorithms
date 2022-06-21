from imageshelper import *
import random
import numba as nb


"""
This file holds general implementations of genetic algorithms in python.
It includes:
-localsearch:
    -Hillclimbing
    -Simulated Annealing
-Genetic Algorithms:
    - Canonical Genetic Algorithm
    - Steady State Genetic Algorithm
They can be seen as library and adapted easily for the wanted purpose
If you need the specific operator look in edgedetectionspecificoperator.py
"""
################################################################################################
#####################################parameters#################################################
################################################################################################
alpha = 0.85
combprob = 0.8
popsize=36


################################################################################################
#####################################local-Searchh##############################################
################################################################################################
def localsearch(fitnessfunc, createfirstGen, acceptanceCond, mutate, stoppingCond):
    time = 0
    parentGen = createfirstGen
    temperature = 100
    while not stoppingCond(time):
        kidGen, mutsite = mutate(parentGen)
        parentGenFitness = fitnessfunc(parentGen, nb.typed.List(mutsite))
        kidGenFitness = fitnessfunc(kidGen, nb.typed.List(mutsite))
        time += 1
        acceptanceCondi, tempneu = acceptanceCond(parentGenFitness, kidGenFitness, temperature)
        temperature = tempneu
        if acceptanceCondi:
            parentGen = kidGen
        else:
            parentGen = parentGen

        #proccess Generatio to showable image
        show = parentGen*255
        label="Gen: " + str(time)
        resized=resizeImage(show,400)
        resized=cv2.cvtColor(resized,cv2.COLOR_GRAY2BGR,resized)
        cv2.putText(resized,label , (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                    cv2.LINE_AA)
        cv2.imshow("current Generation", resized)
        cv2.waitKey(1)
    return parentGen


@nb.njit(nogil=True)
def acceptHillclimbing(parentGenFitness, kidGenFitness, sa_temp):
    return kidGenFitness < parentGenFitness, sa_temp


@nb.njit(nogil=True)
def acceptSimulatedAnnealing(parentGenFitness, kidGenFitness, temperature):
    temperature *= alpha
    if kidGenFitness < parentGenFitness:
        return True, temperature
    else:
        annealingProbability = random.random()
        annealing = np.exp(-1 * (np.abs(parentGenFitness - kidGenFitness) / temperature))
        return annealingProbability <= annealing, temperature


################################################################################################
#####################################Canonical Genetic Algorithm################################
################################################################################################
def cangeneticAlgorithm(popfitnessfunc,createInitPop,stoppingCond,selectPop,crossover,mutate):
    time = 0
    init_pop = createInitPop
    popfitness=calc_pop_fitness(init_pop,popfitnessfunc)
    while stoppingCond(time) !=True:
        selectedPop = selectPop(init_pop,popfitness,len(init_pop))
        kid_pop=[]
        for i in range(0,int(len(init_pop)/2)):
            randomcombprob = random.random()
            if randomcombprob <= combprob:
                kidA,kidB= crossover(selectedPop[2*i-1],selectedPop[2*i])
            else:
                kidA,kidB=init_pop[2*i-1],init_pop[2*i]
            kidA = mutate(kidA)
            kidB = mutate(kidB)
            kid_pop.append(kidA)
            kid_pop.append(kidB)
        popfitness=calc_pop_fitness(kid_pop,popfitnessfunc)
        time+=1
        init_pop=kid_pop

        #Converting  population to showable Image nad show to user
        init_pop_show = to_matrix(kid_pop, int(np.sqrt(popsize)))
        init_pop_show = stackImages(init_pop_show, 1)
        resized = resizeImage(init_pop_show,100)
        resized=cv2.cvtColor(resized,cv2.COLOR_GRAY2BGR,resized)
        label = "Gen: " + str(time)
        labelb="Best: " + str(int(min(popfitness)))
        cv2.putText(resized,label , (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                        cv2.LINE_AA)

        cv2.putText(resized,labelb , (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                        cv2.LINE_AA)
        cv2.imshow("stackedImcan", resized)
        cv2.waitKey(1)
    return init_pop[np.argmin(popfitness)]


################################################################################################
#####################################Steady-State Genetic Algorithm#############################
################################################################################################
def geneticAlgorithm(popfitnessfunc, createInitPop, stoppingCond, selectPop, crossover, mutate):

    init_pop = createInitPop
    popfitness = calc_pop_fitness(nb.typed.List(init_pop),popfitnessfunc)
    time = 0
    while stoppingCond(time) != True:
        selectedPop=selectPop(init_pop, popfitness)
        parentA = selectedPop[0]
        parentB = selectedPop[1]
        randomcombprob = random.random()
        if randomcombprob <= combprob:
            kidA, kidB = crossover(parentA,parentB)
        else:
            kidA, kidB = parentA,parentB
        kidA = mutate(kidA)
        kidB = mutate(kidB)
        kidAfitness=popfitnessfunc(kidA)
        kidBfitness=popfitnessfunc(kidB)
        del init_pop[popfitness.index(max(popfitness))]
        popfitness.remove(max(popfitness))
        del init_pop[popfitness.index(max(popfitness))]
        popfitness.remove(max(popfitness))
        popfitness.append(kidAfitness)
        init_pop.append(kidA)
        popfitness.append(kidBfitness)
        init_pop.append(kidB)
        time += 1


        #Converting  population to showable Image nad show to user
        init_pop_show = to_matrix(init_pop, int(np.sqrt(popsize)))
        init_pop_show = stackImages(init_pop_show, 1)
        resized = resizeImage(init_pop_show,100)
        resized=cv2.cvtColor(resized,cv2.COLOR_GRAY2BGR,resized)
        label = "Gen: " + str(time)
        cv2.putText(resized,label , (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                        cv2.LINE_AA)  # adding timer text

        labelb="Best: " + str(int(min(popfitness)))
        cv2.putText(resized,labelb , (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                        cv2.LINE_AA)  # adding timer text
        cv2.imshow("stackedImsteady", resized)
        cv2.waitKey(1)

    return init_pop[np.argmin(popfitness)]





def calc_pop_fitness(pop,fitnessfunc):

    popfitness=[]
    for individual in range(0,len(pop)):
        fitness = fitnessfunc(pop[individual])
        popfitness.append(fitness)
    return popfitness



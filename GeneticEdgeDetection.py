import random
import numpy as np


alpha=0.8
combprob=0.8


def localsearch(fitnessfunc,createfirstGen,acceptanceCond,mutate,stoppingCond):
    time = 0
    temperature = 100
    parentGen=createfirstGen
    parentGenFitness=fitnessfunc(parentGen)
    while stoppingCond(time) != True:
        kidGen= mutate(parentGen)
        kidGenFitness = fitnessfunc(kidGen)
        time+=1
        if acceptanceCond(parentGenFitness,kidGenFitness,temperature) ==True:
            parentGen=kidGen
            print("Generation: ",time," cost: ",kidGenFitness)
        else:
            parentGen= parentGen

    return parentGen


def acceptHillclimbing(parentGenFitness,kidGenFitness,sa_temp):
    return kidGenFitness < parentGenFitness

def acceptSimulatedAnnealing(parentGenFitness, kidGenFitness, temperature):
    temperature *= alpha
    if kidGenFitness < parentGenFitness:
        return True
    else:
        annealingProbability=random.random()
        return annealingProbability <= np.exp(-np.abs(kidGenFitness-parentGenFitness) / temperature)


def geneticAlgorithm(popfitnessfunc,createInitPop,stoppingCond,selectPop,crossover,mutate):
    time = 0
    init_pop = createInitPop
    popfitness=popfitnessfunc(init_pop)
    while stoppingCond(time) !=True:
        selectedPop = selectPop(init_pop,popfitness)
        kid_pop=[]
        for i in range(1,len(init_pop)/2):
            randomcombprob = random.random()
            if randomcombprob <= combprob:
                kidA,kidB= crossover(selectedPop[2*i-1],selectPop[2*i])
            else:
                kidA,kidB=init_pop[2*i-1],init_pop[2*i]
            kidA = mutate(kidA)
            kidB = mutate(kidB)
            kid_pop.append(kidA)
            kid_pop.append(kidB)
        popfitness=popfitnessfunc(kid_pop)
        time+=1
        init_pop=kid_pop
    return init_pop[np.argmax(popfitness)]

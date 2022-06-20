import random
import numpy as np


alpha=0.8
combprob=0.8
def qTournamentselect(q,pop,popfitness, kIndividualamount=2, enemieamount=12):
    scores = []
    for i in range(0, len(popfitness)):
        siege=0
        for j in range(0,q):
            u = random.randint(0,len(popfitness))
            if popfitness[i] < popfitness[j]:
                siege+=1
        scores.append(siege)
    l = []
    for i in range(0,kIndividualamount):
        index = np.argmax(scores)
        l.append(pop[index])
    return l

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



################################################################################################
################################################################################################
#####################################genetic-Algorithm##########################################
################################################################################################
################################################################################################


def cangeneticAlgorithm(popfitnessfunc,createInitPop,stoppingCond,selectPop,crossover,mutate):
    time = 0
    init_pop = createInitPop
    popfitness=popfitnessfunc(init_pop)
    while stoppingCond(time) !=True:
        selectedPop = selectPop(init_pop,popfitness,len(init_pop))
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


def tournamentselect(popfitness,kIndividualamount,enemieamount):
    kParents=[]
    for enemiesindices in range(0,kIndividualamount):
        defenderIndex=random.randint(0,len(popfitness))
        for enemy in range(2,enemieamount):
            enemyIndex=random.randint(0,len(popfitness))
            if(popfitness[defenderIndex]) > popfitness[enemyIndex]:
                defenderIndex=enemyIndex
        kParents.append(defenderIndex)
    return kParents



def efficientbinarymut(individual):
    mutated=individual.copy()
    next=0
    while(next <= len(individual)):
        mutated[next] = 1-mutated[next]
        nextGeneProb= random.random()
        next = int(np.log2(nextGeneProb) / np.log2(1-combprob))
    return mutated




def twoPointcrossover(parentA, parentB):
    firstpoint= np.random.randint(0,len(parentA))
    secondpoint=np.random.randint(0,len(parentA))
    maxPoint=max(firstpoint,secondpoint)
    minPoint=min(firstpoint,secondpoint)
    kidC = np.empty(parentA.shape)
    kidD = np.empty(parentA.shape)
    for i in range(0, minPoint):
        kidC[i] = parentA[i]
        kidD[i] = parentB[i]
    for j in range(minPoint,maxPoint):
        kidC[j] = parentB[j]
        kidD[j] = parentA[j]
    for k in range(maxPoint, len(parentA)):
        kidC[i] = parentA[i]
        kidD[i] = parentB[i]
    return kidC, kidD


def onePointCrossover(parentA,parentB):

    return

from imageshelper import *
from generateedgeenhanced import*
import random
import numpy as np
import numba as nb
import cv2
import math
"""
This File holds all specific Operations including the following:
- fitness function
- mutations
- recombinations -crossover
- Generatng an initial Chromosome
- Generating an initial Population
- Selection function 
"""
################################################################################################
#####################################Extraction of enhancedimage################################
################################################################################################

image = cv2.imread("inputimages/circle.png")
cv2.GaussianBlur(image, (3, 3), 3, image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
enhanced = truncate_to_one(generateEdgeEnhanced(image))
################################################################################################
#####################################Hardcoded-Mut-Variations###################################
################################################################################################
es_s1 = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 1]], dtype='uint8')
es_s2 = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0]], dtype='uint8')
es_s3 = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]], dtype='uint8')
es_s4 = np.array([[0, 0, 1], [1, 1, 0], [0, 0, 0]], dtype='uint8')
es_s5 = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1]], dtype='uint8')
es_s6 = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 0]], dtype='uint8')
es_s7 = np.array([[0, 1, 0], [0, 1, 0], [1, 0, 0]], dtype='uint8')
es_s8 = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]], dtype='uint8')
es_s9 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype='uint8')
es_s10 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='uint8')
es_s11 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype='uint8')
es_s12 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype='uint8')
es_s13 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype='uint8')

edgestructs = [es_s1, es_s2, es_s3, es_s4, es_s5, es_s6, es_s7, es_s8, es_s9, es_s10, es_s11, es_s12, es_s13, es_s13]
################################################################################################
#####################################Fitness function###########################################
################################################################################################

@nb.njit(nogil=True,parallel=True)
def decisionTreeCostWholeImage(edgeConfiguration):
    h = edgeConfiguration.shape[0]
    w = edgeConfiguration.shape[1]
    fitness = 0
    for y in nb.prange(0, int(h)):
        for x in nb.prange(0, int(w)):
            fitness = fitness + decisionTreeCostFunction(edgeConfiguration, [y, x], enhanced)
    return fitness


@nb.njit(nogil=True, parallel=True)
def decisionTreeCostVariableWindow(edgeConfiguration, pixelsite, winsize=(3, 3)):
    fitness = 0
    for y in nb.prange(-1 * int(winsize[0] / 2), int(winsize[0] / 2) + 1):
        for x in nb.prange(-1 * int(winsize[1] / 2), int(winsize[1] / 2) + 1):
            newsite = [pixelsite[0] + y, pixelsite[1] + x]
            fitness = fitness + decisionTreeCostFunction(edgeConfiguration, newsite, enhanced)
    return fitness


@nb.njit(nogil=True)
def decisionTreeCostFunction(edgeImage, pixelsite, enhanced,w_c=0.25, w_d=2, w_e=0.5, w_f=3, w_t=6.5):
#doggo w_c=0.25, w_d=3.75, w_e=1, w_f=3, w_t=6.71): sobel
#wolf w_c=0.25, w_d=5, w_e=1, w_f=3, w_t=6.5): sobel
#SMILEY w_c=0.25, w_d=5, w_e=0.3, w_f=3, w_t=6.75): en
# kreis w_c=0.25, w_d=2, w_e=0.5, w_f=3, w_t=6.5): en
    costCurvature = 1
    costFragment = 1
    costNumberEdges = 1
    costThickness = 1
    costDiss=1
    if edgeImage[pixelsite[0], pixelsite[1]] == 0:
        costCurvature = 0
        costFragment = 0
        costNumberEdges = 0
        costThickness = 0
        costDiss = enhanced[pixelsite[0], pixelsite[1]]
    else:
        costDiss = 0
        costNumberEdges = 1
        pixelwindow = edgeImage[pixelsite[0] - 1:pixelsite[0] + 2:, pixelsite[1] - 1:pixelsite[1] + 2:]
        edgePixelCounter = 0
        for y in nb.prange(0, 3):
            for x in nb.prange(0, 3):
                if y == 1 and x == 1:
                    continue
                if pixelwindow[y, x] == 1:
                    edgePixelCounter += 1
        if edgePixelCounter == 0:
            costCurvature = 0
            costFragment = 1
            costThickness = 0
        if edgePixelCounter == 1:
            costCurvature = 0
            costFragment = 0.5
            costThickness = 0
        if edgePixelCounter == 2:
            costFragment = 0

            if thickness(edgeImage, pixelsite) == 1:
                costThickness = 1
                costCurvature = 1
            else:
                costThickness = 0
                costCurvature = curvature(edgeImage, pixelsite)
        if edgePixelCounter >= 3:
            costCurvature = 1
            costFragment = 0
            costThickness = thickness(edgeImage, pixelsite)
    return w_c * costCurvature + w_d * costDiss + w_e * costNumberEdges + w_f * costFragment + w_t * costThickness


@nb.njit(nogil=True)
def curvature(edgeImage, pixelposition):
    pixelsite = edgeImage[pixelposition[0] - 1:pixelposition[0] + 2, pixelposition[1] - 1:pixelposition[1] + 2]
    for y in range(0, 3):
        for x in range(0, 3):
            if y == 1 and x == 1:
                continue
            else:
                if pixelsite[y, x] == 1:
                    a = (math.atan2(1 - y, 1 - x) * (180 / math.pi) + 360) % 360
                    b = (math.atan2(y - 1, x - 1) * (180 / math.pi) + 360) % 360
                    if a < b:
                        if a == 45 or a == 135:
                            return 1
                    else:
                        if b == 45 or b == 135:
                            return 1
    return 0


@nb.njit(nogil=True)
def thickness(edgeImage, pixelposition):
    pixelsite = edgeImage[pixelposition[0] - 1:pixelposition[0] + 2, pixelposition[1] - 1:pixelposition[1] + 2]
    if pixelsite[1, 1] == 1:
        for y in range(0, 3):
            for x in range(0, 3):
                grad = 0
                if x == 1 and y == 1:
                    continue

                if pixelsite[y, x] == 1:
                    if x < 2 and y < 2:
                        if pixelsite[y + 1, x + 1] == 1:
                            grad += 1
                        if pixelsite[y, x + 1] == 1:
                            grad += 1
                        if pixelsite[y + 1, x] == 1:
                            grad += 1
                    if x == 2 and y < 2:
                        if pixelsite[y, x - 1] == 1:
                            grad += 1
                        if pixelsite[y + 1, x - 1] == 1:
                            grad += 1
                        if pixelsite[y + 1, x] == 1:
                            grad += 1
                    if y == 2 and x < 2:
                        if pixelsite[y - 1, x] == 1:
                            grad += 1
                        if pixelsite[y - 1, x + 1] == 1:
                            grad += 1
                        if pixelsite[y, x + 1] == 1:
                            grad += 1
                    if y == 2 and x == 2:
                        if pixelsite[y - 1, x] == 1:
                            grad += 1
                        if pixelsite[y - 1, x - 1] == 1:
                            grad += 1
                        if pixelsite[y, x - 1] == 1:
                            grad += 1
                if grad > 1:
                    return 1
        return 0
    else:
        return 0



################################################################################################
#####################################Legacy-Try-To-Use-Binary-Chromosome########################
################################################################################################
@nb.njit(nogil=True)
def img2chromosome(img_arr):
    return img_arr.flatten()


@nb.njit(nogil=True)
def chromosome2img(chromosome, img_shape):
    return np.reshape(chromosome, img_shape)


################################################################################################
#####################################Edge-Mutations#############################################
################################################################################################

@nb.njit(nogil=True)
def singlePixelChange(parent):
    kid = parent.copy()
    randsitex = random.randint(1, kid.shape[1] - 2)
    randsitey = random.randint(1, kid.shape[0] - 2)
    kid[randsitey,randsitex] = 1 - kid[randsitey,randsitex]

    return kid, [randsitey, randsitex]


@nb.njit(nogil=True)
def asinglePixelChange(parent):
    kid = parent.copy()
    randsitex = random.randint(1, kid.shape[1] - 2)
    randsitey = random.randint(1, kid.shape[0] - 2)
    kid[randsitey,randsitex] = 1 - kid[randsitey,randsitex]

    return kid

def edgestructureMutation(parent):
    kid = parent.copy()
    randsitex = random.randint(1, kid.shape[1] - 2)
    randsitey = random.randint(1, kid.shape[0] - 2)
    randedgestructure = random.randint(0, len(edgestructs) - 1)
    mutation = edgestructs[randedgestructure]
    kid[randsitey - 1:randsitey + 2, randsitex - 1:randsitex + 2] = mutation

    return kid, [randsitey, randsitex]


def aedgestructureMutation(parent):
    kid = parent.copy()
    randsitex = random.randint(1, kid.shape[1] - 2)
    randsitey = random.randint(1, kid.shape[0] - 2)
    randedgestructure = random.randint(0, len(edgestructs) - 1)
    mutation = edgestructs[randedgestructure]
    kid[randsitey - 1:randsitey + 2, randsitex - 1:randsitex + 2] = mutation

    return kid

def chooseRandomMut(parent):
    randommut=random.randint(0,2)
    if randommut ==0:
        return aedgestructureMutation(parent)
    else:
        return asinglePixelChange(parent)



@nb.njit(nogil=True)
def efficientbinarymut(individual):
    mutated = individual.copy()
    next = 0
    while next < len(individual):
        mutated[next] = 1 - mutated[next]
        next +=int(np.log2(random.random()) / np.log2(1 - 1/len(individual)))
    return mutated

################################################################################################
#####################################recombination(crossover)###################################
################################################################################################
@nb.njit
def twoPointcrossover(parentA, parentB):
    firstpoint = np.random.randint(0, len(parentA[0]))
    secondpoint = np.random.randint(0, len(parentA[0]))
    maxPoint = max(firstpoint, secondpoint)
    minPoint = min(firstpoint, secondpoint)
    kidC = np.zeros(shape=parentA.shape)
    kidD = np.zeros(shape=parentA.shape)
    for i in range(0, minPoint):
        kidC[i] = parentA[i]
        kidD[i] = parentB[i]
    for j in range(minPoint, maxPoint):
        kidC[j] = parentB[j]
        kidD[j] = parentA[j]
    for k in range(maxPoint, len(parentA)):
        kidC[k] = parentA[k]
        kidD[k] = parentB[k]
    return kidC, kidD

################################################################################################
#####################################create init chrom,pop######################################
################################################################################################

def createRandomChromosome(inputImg):
    enhancedImg = inputImg.copy()
    h = enhancedImg.shape[0]
    w = enhancedImg.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            enhancedImg[y, x] = np.random.randint(0, 2)# * int(enhanced[y,x] + random.random())
    return np.float32(enhancedImg)


def createRandomPop(popsize,inputIm):
    pop = []
    for i in range(0,popsize):
        individual = createRandomChromosome(inputIm)
        pop.append(individual)
    return pop



################################################################################################
#####################################selection##################################################
################################################################################################

def tournamentselect(pop,popfitness, kIndividualamount=2, enemieamount=5):
    kParents = []
    for enemiesindices in range(0, kIndividualamount):
        defenderIndex = random.randint(0, len(popfitness)-1)
        for enemy in range(0, enemieamount):
            enemyIndex = random.randint(0, len(popfitness)-1)
            if (popfitness[defenderIndex]) > popfitness[enemyIndex]:
                defenderIndex = enemyIndex
        kParents.append(pop[defenderIndex])
    return kParents
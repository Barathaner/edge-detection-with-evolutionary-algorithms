import math
import cv2
import numpy as np
import random
import numba as nb



popsize=16
# basis edge structure two neighbouring set
e_s1 = np.array([[0, 0, 0], [255, 255, 0], [0, 0, 255]], dtype='uint8')
e_s2 = np.array([[0, 0, 255], [0, 255, 0], [0, 255, 0]], dtype='uint8')
e_s3 = np.array([[255, 0, 0], [0, 255, 0], [0, 255, 0]], dtype='uint8')
e_s4 = np.array([[0, 0, 255], [255, 255, 0], [0, 0, 0]], dtype='uint8')
e_s5 = np.array([[0, 255, 0], [0, 255, 0], [0, 0, 255]], dtype='uint8')
e_s6 = np.array([[0, 0, 0], [0, 255, 255], [255, 0, 0]], dtype='uint8')
e_s7 = np.array([[0, 255, 0], [0, 255, 0], [255, 0, 0]], dtype='uint8')
e_s8 = np.array([[255, 0, 0], [0, 255, 255], [0, 0, 0]], dtype='uint8')
e_s9 = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0]], dtype='uint8')
e_s10 = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype='uint8')
e_s11 = np.array([[0, 255, 0], [0, 255, 0], [0, 255, 0]], dtype='uint8')
e_s12 = np.array([[0, 0, 0], [255, 255, 255], [0, 0, 0]], dtype='uint8')

enhanced = []

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
# Region of interest
r1_1 = np.array([[-1, -1], [0, -1], [1, -1], [1, 0]])
r1_2 = np.array([[-1, 1], [0, 1], [-1, 2], [0, 2]])
r2_1 = np.array([[-2, -1], [-1, -1], [-2, 0], [-1, 0]])
r2_2 = np.array([[1, -1], [0, 1], [0, 1], [1, 1]])
r3_1 = np.array([[0, -1], [0, -2], [1, -1], [1, -2]])
r3_2 = np.array([[-1, 0], [-1, 1], [0, 1], [1, 1]])
r4_1 = np.array([[1, 0], [2, 0], [1, 1], [2, 1]])
r4_2 = np.array([[-1, -1], [-1, 0], [0, -1], [-1, 1]])
r5_1 = np.array([[-1, -2], [-1, -1], [0, -1], [0, -2]])
r5_2 = np.array([[1, 0], [-1, 1], [0, 1], [1, 1]])
r6_1 = np.array([[1, -1], [1, 0], [2, -1], [2, 0]])
r6_2 = np.array([[-1, -1], [-1, 0], [-1, 1], [0, 1]])
r7_1 = np.array([[0, 1], [0, 2], [1, 1], [1, 2]])
r7_2 = np.array([[-1, 0], [-1, -1], [0, -1], [1, -1]])
r8_1 = np.array([[-2, 0], [-1, 0], [-2, 1], [-1, 1]])
r8_2 = np.array([[0, -1], [1, -1], [1, 0], [1, 1]])
r9_1 = np.array([[1, 0], [2, 0], [1, 1], [0, 1], [0, 2]])
r9_2 = np.array([[-2, 0], [-1, 0], [-1, -1], [0, -1], [0, -2]])
r10_1 = np.array([[2, 0], [1, 0], [1, -1], [0, -1], [0, -2]])
r10_2 = np.array([[-2, 0], [-1, 0], [-1, 1], [0, 1], [0, 2]])
r11_1 = np.array([[1, -1], [2, -1], [1, 0], [2, 0], [1, 1], [2, 1]])
r11_2 = np.array([[-1, -1], [-2, -1], [-1, 0], [-2, 0], [-1, 1], [-2, 1]])
r12_1 = np.array([[-1, 1], [-1, 2], [0, 1], [0, 2], [1, 1], [1, 2]])
r12_2 = np.array([[-1, -1], [-1, -2], [0, -1], [0, -2], [1, -1], [1, 2]])

edge_structures = [e_s1, e_s2, e_s3, e_s4, e_s5, e_s6, e_s7, e_s8, e_s9, e_s10, e_s11, e_s12]
roi_one = [r1_1, r2_1, r3_1, r4_1, r5_1, r6_1, r7_1, r8_1, r9_1, r10_1, r11_1, r12_1]
roi_two = [r1_2, r2_2, r3_2, r4_2, r5_2, r6_2, r7_2, r8_2, r9_2, r10_2, r11_2, r12_2]

alpha = 0.85
combprob = 0.8


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
            print("Generation: ", time, " cost: ", np.abs(parentGenFitness - kidGenFitness))
            cv2.imshow("current Generation", kidGen)
            cv2.waitKey(1)
        else:
            parentGen = parentGen

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
#####################################genetic-Algorithm##########################################
################################################################################################


def geneticAlgorithm(popfitnessfunc, createInitPop, stoppingCond, selectPop, crossover, mutate):

    init_pop = createInitPop
    popfitness = popfitnessfunc(nb.typed.List(init_pop))
    time = 0
    while stoppingCond(time) != True:
        selectedPop=selectPop(init_pop, popfitness)
        parentA = selectedPop[0]
        parentB = selectedPop[1]
        parentA = img2chromosome(parentA)
        parentB = img2chromosome(parentB)
        randomcombprob = random.random()
        if randomcombprob <= combprob:
            kidA, kidB = crossover(parentA,parentB)
        else:
            kidA, kidB = parentA,parentB
        kidA = chromosome2img(kidA,init_pop[0].shape)
        kidB = chromosome2img(kidB,init_pop[0].shape)
        kidA = mutate(kidA)
        kidB = mutate(kidB)
        kidAfitness=decisionTreeCostWholeImage(kidA)
        kidBfitness=decisionTreeCostWholeImage(kidB)
        del init_pop[popfitness.index(max(popfitness))]
        popfitness.remove(max(popfitness))
        del init_pop[popfitness.index(max(popfitness))]
        popfitness.remove(max(popfitness))
        popfitness.append(kidAfitness)
        init_pop.append(kidA)
        popfitness.append(kidBfitness)
        init_pop.append(kidB)
        time += 1
        init_pop_show = to_matrix(init_pop, int(np.sqrt(popsize)))
        init_pop_show = stackImages(init_pop_show, 1)
        cv2.imshow("stackedIm", init_pop_show)
        cv2.waitKey(2)
        print(time)
    return init_pop[np.argmin(popfitness)]


def tournamentselect(pop,popfitness, kIndividualamount=2, enemieamount=3):
    kParents = []
    for enemiesindices in range(0, kIndividualamount):
        defenderIndex = random.randint(0, len(popfitness)-1)
        for enemy in range(2, enemieamount):
            enemyIndex = random.randint(0, len(popfitness)-1)
            if (popfitness[defenderIndex]) > popfitness[enemyIndex]:
                defenderIndex = enemyIndex
        kParents.append(defenderIndex)
    selectedPop=[]
    for index in kParents:
        selectedPop.append(pop[index])
    return selectedPop


@nb.njit(nogil=True)
def efficientbinarymut(individual):
    mutated = individual.copy()
    next = 0
    while next < len(individual):
        mutated[next] = 1 - mutated[next]
        next +=int(np.log2(random.random()) / np.log2(1 - 1/len(individual)))
    return mutated


@nb.njit(nogil=True)
def twoPointcrossover(parentA, parentB):
    firstpoint = np.random.randint(0, len(parentA))
    secondpoint = np.random.randint(0, len(parentA))
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
#####################################specific-Operator##########################################
################################################################################################

def generateEnhancedSobelImage(image):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    grad_x = cv2.Sobel(image, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(image, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad



def calcDissimilarity(imgrey, x, y, darkregion, lightregion):
    darkregionvalues = []
    lightregionvalues = []
    for pos in range(len(darkregion[0])):
        greyvalueofdarkposx = x + darkregion[pos][0]
        greyvalueofdarkposy = y + darkregion[pos][1]
        greyvalueoflightposx = x + lightregion[pos][0]
        greyvalueoflightposy = y + lightregion[pos][1]
        try:
            darkregionvalues.append(imgrey[greyvalueofdarkposy,greyvalueofdarkposx])
            lightregionvalues.append(imgrey[greyvalueoflightposy,greyvalueoflightposx])
        except:
            continue

    if len(darkregionvalues) == 0 or len(lightregionvalues)==0:
        return 0

    return abs(sum(darkregionvalues) / len(darkregionvalues) -sum(lightregionvalues) / len(lightregionvalues))




def generateEdgeEnhanced(imgrey):
    dissimilarity_start_gen = np.zeros(imgrey.shape[:2], dtype='uint8')

    h = imgrey.shape[0]
    w = imgrey.shape[1]
    for y in range(1, h-1):
        for x in range(1, w-1):
            dissimilarity_edge_struct = 0
            bestfittingedgestructure = edge_structures[0]
            best_index = 0
            for es_index in range(0, len(edge_structures)):
                dissresult = calcDissimilarity(imgrey, x, y, roi_one[es_index], roi_two[es_index])
                if dissimilarity_edge_struct < dissresult:
                    best_index = es_index
                    dissimilarity_edge_struct = dissresult
                    bestfittingedgestructure = edge_structures[es_index]

            dissimilarity_start_gen[y, x] = dissimilarity_edge_struct
            if best_index <= 7:
                x_up = x
                y_up = y - 1
                x_down = x
                y_down = y + 1
                x_left = x - 1
                y_left = y
                x_right = x + 1
                y_right = y
                up_diss = calcDissimilarity(imgrey, x_up, y_up, roi_one[best_index], roi_two[best_index])
                down_diss = calcDissimilarity(imgrey, x_down, y_down, roi_one[best_index], roi_two[best_index])
                left_diss = calcDissimilarity(imgrey, x_left, y_left, roi_one[best_index], roi_two[best_index])
                right_diss = calcDissimilarity(imgrey, x_right, y_right, roi_one[best_index], roi_two[best_index])

                if max([dissimilarity_edge_struct, up_diss, down_diss, left_diss,
                        right_diss]) != dissimilarity_edge_struct:
                    delta = int(dissimilarity_edge_struct / 3)
                    pixelsites= np.where(edge_structures[best_index]==255)
                    dissimilarity_start_gen[y-1+pixelsites[0][0],x-1+pixelsites[1][0]] = int(max(0, dissimilarity_start_gen[y-1+pixelsites[0][0],x-1+pixelsites[1][0]]-delta))
                    dissimilarity_start_gen[y-1+pixelsites[0][1],x-1+pixelsites[1][1]] = int(max(0,dissimilarity_start_gen[y-1+pixelsites[0][1],x-1+pixelsites[1][1]]-delta))
                    dissimilarity_start_gen[y-1+pixelsites[0][2],x-1+pixelsites[1][2]] = int(max(0,dissimilarity_start_gen[y-1+pixelsites[0][2],x-1+pixelsites[1][2]]-delta))

            if 8 == best_index:
                x_diagonal_up = x - 1
                y_diagonal_up = y - 1
                x_diagonal_down = x + 1
                y_diagonal_down = y + 1
                diss_diag_up = calcDissimilarity(imgrey, x_diagonal_up, y_diagonal_up, roi_one[best_index],
                                                 roi_two[best_index])
                diss_diag_down = calcDissimilarity(imgrey, x_diagonal_down, y_diagonal_down, roi_one[best_index],
                                                   roi_two[best_index])

                if max([dissimilarity_edge_struct, diss_diag_up, diss_diag_down]) != dissimilarity_edge_struct:
                    delta = int(dissimilarity_edge_struct / 3)
                    pixelsites= np.where(edge_structures[best_index]==255)
                    dissimilarity_start_gen[y-1+pixelsites[0][0],x-1+pixelsites[1][0]] = int(max(0, dissimilarity_start_gen[y-1+pixelsites[0][0],x-1+pixelsites[1][0]]-delta))
                    dissimilarity_start_gen[y-1+pixelsites[0][1],x-1+pixelsites[1][1]] = int(max(0,dissimilarity_start_gen[y-1+pixelsites[0][1],x-1+pixelsites[1][1]]-delta))
                    dissimilarity_start_gen[y-1+pixelsites[0][2],x-1+pixelsites[1][2]] = int(max(0,dissimilarity_start_gen[y-1+pixelsites[0][2],x-1+pixelsites[1][2]]-delta))
            if 9 == best_index:
                x_diagonal_up = x + 1
                y_diagonal_up = y - 1
                x_diagonal_down = x - 1
                y_diagonal_down = y + 1
                diss_diag_up = calcDissimilarity(imgrey, x_diagonal_up, y_diagonal_up, roi_one[best_index],
                                                 roi_two[best_index])
                diss_diag_down = calcDissimilarity(imgrey, x_diagonal_down, y_diagonal_down, roi_one[best_index],
                                                   roi_two[best_index])

                if max([dissimilarity_edge_struct, diss_diag_up, diss_diag_down]) != dissimilarity_edge_struct:
                    delta = int(dissimilarity_edge_struct / 3)
                    pixelsites= np.where(edge_structures[best_index]==255)
                    dissimilarity_start_gen[y-1+pixelsites[0][0],x-1+pixelsites[1][0]] = int(max(0, dissimilarity_start_gen[y-1+pixelsites[0][0],x-1+pixelsites[1][0]]-delta))
                    dissimilarity_start_gen[y-1+pixelsites[0][1],x-1+pixelsites[1][1]] = int(max(0,dissimilarity_start_gen[y-1+pixelsites[0][1],x-1+pixelsites[1][1]]-delta))
                    dissimilarity_start_gen[y-1+pixelsites[0][2],x-1+pixelsites[1][2]] = int(max(0,dissimilarity_start_gen[y-1+pixelsites[0][2],x-1+pixelsites[1][2]]-delta))
            if 10 == best_index:
                x_left = x - 1
                y_left = y
                x_right = x + 1
                y_right = y
                diss_left = calcDissimilarity(imgrey, x_left, y_left, roi_one[best_index],
                                              roi_two[best_index])
                diss_right = calcDissimilarity(imgrey, x_right, y_right, roi_one[best_index],
                                               roi_two[best_index])

                if max([dissimilarity_edge_struct, diss_left, diss_right]) != dissimilarity_edge_struct:
                    delta = int(dissimilarity_edge_struct / 3)
                    pixelsites= np.where(edge_structures[best_index]==255)
                    dissimilarity_start_gen[y-1+pixelsites[0][0],x-1+pixelsites[1][0]] = int(max(0, dissimilarity_start_gen[y-1+pixelsites[0][0],x-1+pixelsites[1][0]]-delta))
                    dissimilarity_start_gen[y-1+pixelsites[0][1],x-1+pixelsites[1][1]] = int(max(0,dissimilarity_start_gen[y-1+pixelsites[0][1],x-1+pixelsites[1][1]]-delta))
                    dissimilarity_start_gen[y-1+pixelsites[0][2],x-1+pixelsites[1][2]] = int(max(0,dissimilarity_start_gen[y-1+pixelsites[0][2],x-1+pixelsites[1][2]]-delta))

            if 11 == best_index:
                x_up = x - 1
                y_up = y
                x_down = x + 1
                y_down = y
                up_diss = calcDissimilarity(imgrey, x_up, y_up, roi_one[best_index], roi_two[best_index])
                down_diss = calcDissimilarity(imgrey, x_down, y_down, roi_one[best_index], roi_two[best_index])

                if max([dissimilarity_edge_struct, up_diss, down_diss]) != dissimilarity_edge_struct:
                    delta = int(dissimilarity_edge_struct / 3)
                    pixelsites= np.where(edge_structures[best_index]==255)
                    dissimilarity_start_gen[y-1+pixelsites[0][0],x-1+pixelsites[1][0]] = int(max(0, dissimilarity_start_gen[y-1+pixelsites[0][0],x-1+pixelsites[1][0]]-delta))
                    dissimilarity_start_gen[y-1+pixelsites[0][1],x-1+pixelsites[1][1]] = int(max(0,dissimilarity_start_gen[y-1+pixelsites[0][1],x-1+pixelsites[1][1]]-delta))
                    dissimilarity_start_gen[y-1+pixelsites[0][2],x-1+pixelsites[1][2]] = int(max(0,dissimilarity_start_gen[y-1+pixelsites[0][2],x-1+pixelsites[1][2]]-delta))

    return dissimilarity_start_gen


def truncate_to_one(inputImg):
    img = np.float32(inputImg)
    h = img.shape[0]
    w = img.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            img[y, x] = img[y, x] / 255.0

    return img

def createRandomChromosome(inputImg):
    enhancedImg = inputImg.copy()
    h = enhancedImg.shape[0]
    w = enhancedImg.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            enhancedImg[y, x] = np.random.randint(0, 2)
    return np.float32(enhancedImg)


def createRandomPop(popsize,inputIm):
    pop = []
    for i in range(0,popsize):
        individual = createRandomChromosome(inputIm)
        pop.append(individual)
    return pop


@nb.njit(nogil=True)
def img2chromosome(img_arr):
    return img_arr.flatten()


@nb.njit(nogil=True)
def chromosome2img(chromosome, img_shape):
    return np.reshape(chromosome, img_shape)


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


@nb.njit(nogil=True)
def aedgestructureMutation(parent):
    kid = parent.copy()
    randsitex = random.randint(1, kid.shape[1] - 2)
    randsitey = random.randint(1, kid.shape[0] - 2)
    randedgestructure = random.randint(0, len(edgestructs) - 1)
    mutation = edgestructs[randedgestructure]
    kid[randsitey - 1:randsitey + 2, randsitex - 1:randsitex + 2] = mutation

    return kid

################################################################################################
#####################################fitness-function###########################################
################################################################################################

@nb.njit(nogil=True)
def calc_pop_fitness(pop):

    popfitness=[]
    for individual in range(0,len(pop)):
        fitness = decisionTreeCostWholeImage(pop[individual])
        popfitness.append(fitness)
    return popfitness


@nb.njit(nogil=True, parallel=True)
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





def stackImages(imgArray,scale,lables=[]):
    imgArray= np.float32(imgArray)
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list) or isinstance(imgArray[0],np.ndarray)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    imageArrBlank = np.zeros((imgArray.shape[0],imgArray.shape[1],int(height*scale), int(width*scale)), np.float32)
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                resized = cv2.resize(imgArray[x][y], (0,0), fx=scale, fy=scale)
                imageArrBlank[x][y] = resized*255
        imageBlank = np.zeros((height*scale, width*scale), np.float32)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imageArrBlank[x])
            hor_con[x] = np.concatenate(imageArrBlank[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imageArrBlank[x] = cv2.resize(imageArrBlank[x], (0,0), fx=scale, fy=scale)
        hor= np.hstack(imageArrBlank)
        hor_con= np.concatenate(imageArrBlank)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

def to_matrix(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]


if __name__ == '__main__':
    image = cv2.imread("kreis.png")
    cv2.GaussianBlur(image, (3, 3), 3, image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = truncate_to_one(generateEdgeEnhanced(image))
    cv2.imshow("enhanced", enhanced)

    init_gen = createRandomChromosome(enhanced)
    cv2.imshow("init", init_gen)
    optim = localsearch(decisionTreeCostVariableWindow, init_gen, acceptSimulatedAnnealing, edgestructureMutation,lambda gen: gen > 20000)

    init_pop = createRandomPop(popsize,enhanced)
    #init_pop_show = to_matrix(init_pop,int(np.sqrt(popsize)))
    #init_pop_show = stackImages((init_pop),1)
    #cv2.imshow("stackedIm",init_pop_show)
    #optim = geneticAlgorithm(calc_pop_fitness,init_pop,lambda gen: gen > 20000,tournamentselect,twoPointcrossover,asinglePixelChange)
    cv2.imwrite("Optimdog.jpg", optim * 255)
    print("finish")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

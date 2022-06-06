import functools
import math
import operator

import cv2
import numpy as np
import random

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


def calcDissimilarity(imgrey, x, y, darkregion, lightregion):
    darkregionvalues = []
    lightregionvalues = []
    for pos in range(len(darkregion)):
        greyvalueofdarkposx = x + darkregion[pos][0]
        greyvalueofdarkposy = y + darkregion[pos][1]
        greyvalueoflightposx = x + lightregion[pos][0]
        greyvalueoflightposy = y + lightregion[pos][1]
        try:
            darkregionvalues.append(imgrey[greyvalueofdarkposx, greyvalueofdarkposy])
            lightregionvalues.append(imgrey[greyvalueoflightposx, greyvalueoflightposy])
        except:
            continue

    return abs((sum(darkregionvalues) / len(darkregionvalues)) - (sum(lightregionvalues) / len(lightregionvalues)))


def generateEdgeEnhanced(imgrey):
    dissimilarity_start_gen = np.zeros(imgrey.shape[:2], dtype='uint8')

    h = imgrey.shape[0]
    w = imgrey.shape[1]
    for y in range(1, h - 2):
        for x in range(1, w - 2):
            dissimilarity_edge_struct = 0
            bestfittingedgestructure = edge_structures[0]
            best_index = 0
            for es_index in range(0, len(edge_structures)):
                dissresult = calcDissimilarity(imgrey, x, y, roi_one[es_index], roi_two[es_index])
                if dissimilarity_edge_struct < dissresult:
                    best_index = es_index
                    dissimilarity_edge_struct = dissresult
                    bestfittingedgestructure = edge_structures[es_index]

            dissimilarity_start_gen[x, y] = dissimilarity_edge_struct
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
                right_diss = calcDissimilarity(imgrey, x_left, y_left, roi_one[best_index], roi_two[best_index])

                if max([dissimilarity_edge_struct, up_diss, down_diss, left_diss,
                        right_diss]) == dissimilarity_edge_struct:
                    dissimilarity_start_gen[x - 1:x + 2, y - 1: y + 2] += int(dissimilarity_edge_struct / 3)

            if 8 == best_index:
                x_diagonal_up = x - 1
                y_diagonal_up = y - 1
                x_diagonal_down = x + 1
                y_diagonal_down = y + 1
                diss_diag_up = calcDissimilarity(imgrey, x_diagonal_up, y_diagonal_up, roi_one[best_index],
                                                 roi_two[best_index])
                diss_diag_down = calcDissimilarity(imgrey, x_diagonal_down, y_diagonal_down, roi_one[best_index],
                                                   roi_two[best_index])

                if max([dissimilarity_edge_struct, diss_diag_up, diss_diag_down]) == dissimilarity_edge_struct:
                    dissimilarity_start_gen[x - 1:x + 2, y - 1: y + 2] += int(dissimilarity_edge_struct / 3)
            if 9 == best_index:
                x_diagonal_up = x + 1
                y_diagonal_up = y - 1
                x_diagonal_down = x - 1
                y_diagonal_down = y + 1
                diss_diag_up = calcDissimilarity(imgrey, x_diagonal_up, y_diagonal_up, roi_one[best_index],
                                                 roi_two[best_index])
                diss_diag_down = calcDissimilarity(imgrey, x_diagonal_down, y_diagonal_down, roi_one[best_index],
                                                   roi_two[best_index])

                if max([dissimilarity_edge_struct, diss_diag_up, diss_diag_down]) == dissimilarity_edge_struct:
                    dissimilarity_start_gen[x - 1:x + 2, y - 1: y + 2] += int(dissimilarity_edge_struct / 3)
            if 10 == best_index:
                x_left = x - 1
                y_left = y
                x_right = x + 1
                y_right = y
                diss_left = calcDissimilarity(imgrey, x_left, y_left, roi_one[best_index],
                                              roi_two[best_index])
                diss_right = calcDissimilarity(imgrey, x_right, y_right, roi_one[best_index],
                                               roi_two[best_index])

                if max([dissimilarity_edge_struct, diss_left, diss_right]) == dissimilarity_edge_struct:
                    dissimilarity_start_gen[x - 1:x + 2, y - 1: y + 2] += int(dissimilarity_edge_struct / 3)
            if 11 == best_index:
                x_up = x - 1
                y_up = y
                x_down = x + 1
                y_down = y
                up_diss = calcDissimilarity(imgrey, x_up, y_up, roi_one[best_index], roi_two[best_index])
                down_diss = calcDissimilarity(imgrey, x_down, y_down, roi_one[best_index], roi_two[best_index])

                if max([dissimilarity_edge_struct, up_diss, down_diss]) == dissimilarity_edge_struct:
                    dissimilarity_start_gen[x - 1:x + 2, y - 1: y + 2] += int(dissimilarity_edge_struct / 3)

    return dissimilarity_start_gen


def truncate_to_one(inputImg):
    img = inputImg
    img = img.astype(float)
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
            enhancedImg[y, x] = np.random.randint(0, 2) * int(enhancedImg[y, x] + random.random())

    return enhancedImg


def dissimilarity(edgeImage, pixelposition, diss):
    if (edgeImage[pixelposition[0], pixelposition[1]] == 1):
        return 0
    else:
        return diss[pixelposition[0], pixelposition[1]]


def curvature(edgeImage, pixelposition):
    pixelsite = edgeImage[pixelposition[1] - 1:pixelposition[1] + 2:, pixelposition[0] - 1:pixelposition[0] + 2:]
    if pixelsite[1, 1] == 1:
        neighbourCounter = 0
        for y in range(0, 3):
            for x in range(0, 3):
                if y == 1 and x == 1:
                    continue
                else:
                    if pixelsite[y, x] == 1:
                        neighbourCounter += 1
        if neighbourCounter > 1:
            angle = 0
            for y in range(0, 3):
                for x in range(0, 3):
                    if y == 1 and x == 1:
                        continue
                    else:
                        if pixelsite[y, x] == 1:
                            a = math.atan2(y - 1, x - 1)
                            if a > angle:
                                angle = a
            if angle == 0:
                return 0
            if angle == 45:
                return 0.5
            if angle >= 90:
                return 1
    return 1


def fragmentation(edgeImage, pixelposition):
    pixelsite = edgeImage[pixelposition[1] - 1:pixelposition[1] + 2:, pixelposition[0] - 1:pixelposition[0] + 2:]
    if pixelsite[1, 1] == 1:
        neighbourCounter = 0
        for y in range(0, 3):
            for x in range(0, 3):
                if y == 1 and x == 1:
                    continue
                else:
                    if pixelsite[y, x] == 1:
                        neighbourCounter += 1

        if neighbourCounter == 0:
            return 1
        if neighbourCounter == 1:
            return 0.5
        if neighbourCounter > 1:
            return 0
    else:
        return 0

    pass


def numberofEdgePixels(edgeImage, pixelposition):
    pixelsite = edgeImage[pixelposition[1] - 1:pixelposition[1] + 2:, pixelposition[0] - 1:pixelposition[0] + 2:]
    edgePixelCounter = 0
    for y in range(0, 3):
        for x in range(0, 3):
            if edgePixelCounter == 2:
                return 0
            if pixelsite[y, x] == 1:
                edgePixelCounter += 1

    if edgePixelCounter == 1:
        return 1
    else:
        return 1


def thickness(edgeImage, pixelposition):
    pixelsite = edgeImage[pixelposition[1] - 1:pixelposition[1] + 2:, pixelposition[0] - 1:pixelposition[0] + 2:]
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
    return 1

def edgecostminimization(edgeConfiguration, enhanced, w_c=25, w_d=2.0, w_e=1.0, w_f=2.0, w_t=4.76):
    h = edgeConfiguration.shape[0]
    w = edgeConfiguration.shape[1]
    fitness = 0
    for y in range(2, h - 2):
        for x in range(2, w - 2):
            fitness += w_c * curvature(edgeConfiguration, (y, x)) + w_d * dissimilarity(edgeConfiguration, (y, x),
                                                                                        enhanced) + w_e * numberofEdgePixels(
                edgeConfiguration, (y, x)) + w_f * fragmentation(edgeConfiguration, (y, x)) + w_t * thickness(
                edgeConfiguration, (y, x))
    return fitness


def crossover(ParentA, ParentB):
    kidC = np.empty(ParentA.shape)
    kidD = np.empty(ParentA.shape)
    crossoverpoint = random.randint(0, len(ParentA))
    for i in range(0, crossoverpoint):
        kidC[i] = ParentA[i]
        kidD[i] = ParentB[i]
    for j in range(crossoverpoint, len(ParentA)):
        kidC[j] = ParentB[j]
        kidD[j] = ParentA[j]

    return kidC, kidD


def img2chromosome(img_arr):
    chromosome = np.reshape(a=img_arr, newshape=(functools.reduce(operator.mul, img_arr.shape)))

    return chromosome


def chromosome2img(chromosome, img_shape):
    img_arr = np.reshape(a=chromosome, newshape=img_shape)

    return img_arr

def mutiere(a):
    a=a.copy()
    index = random.randint(0, (len(a)-1))
    if a[index] == 0:
        a[index] = 1
    elif a[index] == 1:
        a[index] = 0
    return a


def binmutiere(x):
    x= x.copy()
    probMut = 1.0 / (len(x))
    for i in range(0, len(x)):
        u = random.uniform(0.0, 1.0)
        if u <= probMut:
            if x[i] == 0:
                x[i] = 1
            elif x[i] == 1:
                x[i] = 0
    return x


def singlePixelChange(x):
    x= x.copy()
    randsite = random.randint(0,len(x))
    if x[randsite] ==1:
        x[randsite] =0
    else:
        x[randsite] = 1
    return x

def doublePixelChange(x):
    x= x.copy()
    randsite = random.randint(0,len(x))
    if x[randsite] ==1:
        x[randsite] =0
        if x[randsite+1] == 1:
            x[randsite+1] = 0
        else:
            x[randsite+1] = 1
    else:
        x[randsite] = 1

        if x[randsite+1] == 1:
            x[randsite+1] = 0
        else:
            x[randsite+1] = 1
    return x

def initial_population(img_shape,enhanced, n_individuals=8):
    init_population = np.empty(shape=(n_individuals,
                                         functools.reduce(operator.mul, img_shape)),
                                  dtype=np.uint8)

    for indv_num in range(n_individuals):
        # Randomly generating initial population chromosomes genes values.
        init_population[indv_num, :] = createRandomChromosome(enhanced)

    return init_population

def hillclimbing(bewertungsfunc,enhanced,erzeugeKandidat,maxGen):
    a = erzeugeKandidat
    genCounter = 0
    while genCounter < maxGen:
        bewertunga = bewertungsfunc(a,enhanced=enhanced)
        b = doublePixelChange(img2chromosome(a))
        b=chromosome2img(b,a.shape)
        bewertungB = bewertungsfunc(b,enhanced=enhanced)
        if bewertungB < bewertunga:
            a = b
            print(bewertungB)
        genCounter+=1
    return a


def cal_pop_fitness(target_chrom, pop,fitness_fun):
    qualities = np.zeros(pop.shape[0])
    for indv_num in range(pop.shape[0]):
        qualities[indv_num] = fitness_fun(target_chrom, pop[indv_num, :])

    return qualities



def decisionTreeCostFunction(edgeImage,pixelsite,enhanced,w_c=25, w_d=2.0, w_e=1.0, w_f=2.0, w_t=4.76):
    costCurvature=1
    costFragment=1
    costNumberEdges=1
    costThickness=1
    costDiss=1
    if edgeImage[pixelsite[1],pixelsite[0]] == 0:
        costDiss = enhanced[pixelsite[1],pixelsite[0]]
        costCurvature = 0
        costFragment = 0
        costNumberEdges = 0
        costThickness = 0
    else:
        costDiss = 0
        costNumberEdges = 1
        pixelwindow = edgeImage[pixelsite[1] - 1:pixelsite[1] + 2:, pixelsite[0] - 1:pixelsite[0] + 2:]
        edgePixelCounter = 0
        for y in range(0, 3):
            for x in range(0, 3):
                if pixelwindow[y, x] == 1:
                    edgePixelCounter += 1
        if edgePixelCounter ==0:
            costCurvature = 0
            costFragment = 1
            costThickness = 0
        if edgePixelCounter ==1:
            costCurvature = 0
            costFragment = 0.5
            costThickness = 0
        if edgePixelCounter ==2:
            costFragment = 0
            if thickness(edgeImage,pixelsite)==1:
                costThickness = 1
                costCurvature = 1
            else:
                costThickness = 0
                costCurvature = curvature(edgeImage,pixelsite)
    return w_c*costCurvature+w_d*costDiss+w_e*costNumberEdges+w_f*costFragment+w_t*costThickness

def decisionTreeCostWholeImage(edgeConfiguration, enhanced):
    h = edgeConfiguration.shape[0]
    w = edgeConfiguration.shape[1]
    fitness = 0
    for y in range(2, h - 2):
        for x in range(2, w - 2):
            fitness += decisionTreeCostFunction(edgeConfiguration,[y,x],enhanced)
    return fitness


def generateEnhancedSobelImage(image):

    # generation of enhanced image
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(image, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(image, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return grad

# eventuell austesten ob was bringt???
def convert_gray(binary):
  binary = int(binary, 2)
  binary ^= (binary >> 1)
  return bin(binary)[2:]


def graytob(n):
    n = int(n, 2)  # convert to int

    mask = n
    while mask != 0:
        mask >>= 1
        n ^= mask

    return bin(n)[2:]

if __name__ == '__main__':
    image = cv2.imread("cat.jpeg")
    image = cv2.GaussianBlur(image,(5,5),5)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhancedImage = truncate_to_one(generateEnhancedSobelImage(image))

    init_gen = createRandomChromosome(enhancedImage)
    optim= hillclimbing(decisionTreeCostWholeImage,enhancedImage,createRandomChromosome(enhancedImage),200)
    cv2.imshow("init",init_gen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

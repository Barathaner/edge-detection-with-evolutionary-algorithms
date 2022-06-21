import numpy as np
import cv2
################################################################################################
#####################################Parameters#################################################
################################################################################################

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

roi_one = [r1_1, r2_1, r3_1, r4_1, r5_1, r6_1, r7_1, r8_1, r9_1, r10_1, r11_1, r12_1]
roi_two = [r1_2, r2_2, r3_2, r4_2, r5_2, r6_2, r7_2, r8_2, r9_2, r10_2, r11_2, r12_2]

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
edge_structures = [e_s1, e_s2, e_s3, e_s4, e_s5, e_s6, e_s7, e_s8, e_s9, e_s10, e_s11, e_s12]

################################################################################################
#####################################Sobel-method###############################################
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


################################################################################################
#####################################Region-Dissimilarity#######################################
################################################################################################


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

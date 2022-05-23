import cv2
import numpy as np

#basis edge set
#left right
region1_1 = np.array([[255, 255, 0, 0],[255, 255, 0, 0],[255, 255, 0, 0],[255, 255, 0, 0]])
region2_1 = np.array([[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255]])


#top down
region1_2= np.array([[255, 255, 255, 255],[255, 255, 255, 255],[0, 0, 0, 0],[0, 0, 0, 0]])
region2_2 = np.array([[0, 0, 0, 0],[0, 0, 0, 0],[255, 255, 255, 255],[255, 255, 255, 255]])

#diagonal
region1_3 = np.array([[255, 255, 255, 255],[255, 255, 0, 0],[255, 0, 0, 0],[0, 0, 0, 0]])
region2_3 = np.array([[0, 0, 0, 0],[0, 0, 255, 255],[0, 255, 255, 255],[255, 255, 255, 255]])

#diagonal reverse
region1_4 = np.array([[255, 255, 255, 255],[0, 255, 255, 255],[0, 0, 255, 255],[0, 0, 0, 0]])
region2_4 = np.array([[0, 0, 0, 0],[0, 0, 0, 255],[255, 255, 0, 0],[255, 255, 255, 255]])


regions=[(region1_1,region1_2),(region1_2,region2_2),(region1_3,region2_3),(region1_4,region2_4)]
# w means weight for the Cost factors Curvature, Dissimilarity, number of edge points, fragmentation, thickness
def curvature(edgeConfiguration):
    pass


def dissimilarity(edgeConfiguration):
    pass


def numberofpoints(edgeConfiguration):
    pass


def fragmentation(edgeConfiguration):
    pass


def thickness(edgeConfiguration):
    pass


def edgecostminimization(edgeConfiguration, w_c, w_d, w_e, w_f, w_t):
    fitness = w_c * curvature(edgeConfiguration) + w_d * dissimilarity(edgeConfiguration) + w_e * numberofpoints(edgeConfiguration) + w_f * fragmentation(edgeConfiguration) + w_t * thickness(edgeConfiguration)
    return fitness

def generatefirstgeneration(gray):
    dissimilarity_start_gen = np.zeros(gray.shape, dtype=np.uint8)
    for x in range(gray.shape[0]):
        for y in range(gray.shape[1]):
            fitt=0
            for pattern in regions:
                fittededgestructure = edgeStructure(gray[x - 2:x + 2, y - 2:y + 2],pattern[0],pattern[1])
                newfitt=dissimilaritymeasure(fittededgestructure)

                if newfitt>fitt:
                    fitt=newfitt
            dissimilarity_start_gen[x, y]= fitt
    return dissimilarity_start_gen

# S is a edgeStructure
def dissimilaritymeasure(S):
    return abs(S.R2.mean() - S.R1.mean())


class edgeStructure:
    l0=[]
    l1=[]
    l2=[]
    R1= np.zeros([4,4],dtype=np.uint8)
    R2= np.zeros([4,4],dtype=np.uint8)
    def __init__(self, edgeneighbourhood,region1,region2):
        if edgeneighbourhood.shape[0]==4 and edgeneighbourhood.shape[1]==4:
            self.R1 = edgeneighbourhood.copy()
            self.R1[region1==0] =0
            self.R1[region1!=0]=edgeneighbourhood[region1!=0]
            self.R2 = edgeneighbourhood.copy()
            self.R2[region2==0] =0
            self.R2[region2!=0]=edgeneighbourhood[region2!=0]



if __name__ == '__main__':

    image = cv2.imread("cat.jpeg")
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    fg = generatefirstgeneration(image)
    cv2.imshow("input",fg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
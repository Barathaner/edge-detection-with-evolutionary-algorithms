from generateedgeenhanced import *
from edgedetectionspecificoperator import *
from geneticalgorithms import *

if __name__ == '__main__':
    cv2.imshow("input", image)
    enhancedres=resizeImage(enhanced,400)
    cv2.imshow("enhanced", enhancedres)
    cv2.imwrite("outputimages/enhanced.jpg", enhancedres * 255)


    """LocalSearch"""
    init_gen = createRandomChromosome(enhanced)
    init_genres=resizeImage(init_gen,400)

    cv2.imshow("init", init_genres)
    optim = localsearch(decisionTreeCostVariableWindow, init_gen, acceptSimulatedAnnealing, edgestructureMutation,lambda gen: gen > 9000)
    cv2.imwrite("outputimages/Optimlocal.jpg", optim * 255)


    """Canonical Genetic Algorithm"""
    init_pop = createRandomPop(popsize,enhanced)
    init_pop_show = to_matrix(init_pop, int(np.sqrt(len(init_pop))))
    init_pop_show = stackImages(init_pop_show,1)
    init_popres=resizeImage(init_pop_show,100)
    cv2.imshow("stackedImCanonicalinit",init_popres)
    optim = cangeneticAlgorithm(decisionTreeCostWholeImage,init_pop,lambda gen: gen > 15000,tournamentselect,twoPointcrossover,aedgestructureMutation)
    cv2.imwrite("outputimages/OptimCanonicalgen.jpg", optim * 255)


    """Steady State Genetic Algorithm"""
    init_pop = createRandomPop(popsize,enhanced)
    init_pop_show = to_matrix(init_pop, int(np.sqrt(len(init_pop))))
    init_pop_show = stackImages(init_pop_show,1)
    init_popres=resizeImage(init_pop_show,100)
    cv2.imshow("stackedImSteadyStateinit",init_popres)
    optim = geneticAlgorithm(decisionTreeCostWholeImage,init_pop,lambda gen: gen > 15000,tournamentselect,twoPointcrossover,aedgestructureMutation)
    cv2.imwrite("outputimages/OptimSteadyStategen.jpg", optim * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

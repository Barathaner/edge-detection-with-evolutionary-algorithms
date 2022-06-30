![banner](https://user-images.githubusercontent.com/40422666/176719202-85e8fcbe-df35-4b65-990e-ce845c0d236b.png)

[![GitHub license](https://img.shields.io/github/license/Barathaner/edge-detection-with-evolutionary-alogirthms)](https://github.com/Barathaner/edge-detection-with-evolutionary-alogirthms/blob/main/LICENSE)



Educational Python implementation of evolutionary-algorithms such as Local-Search (Hillclimbing,Simulated Annealing), Genetic Algorithms (Canonical Genetic Algorithm, Steady State) to approximate edges in a small (e.g. 128X128px) picture.

##Installation

![carbon (1)](https://user-images.githubusercontent.com/40422666/176724165-e5698551-9a0b-47b8-a05b-ee1eb1a66d4b.png)



##Basic Question in Edge Detection- Developing fitness function
- 
- Looking for a neihgbourhood with strong signs of change
- Size of the neighbourhood 
- What metrics represent a change?

## SObel Idee
- erst grau dann gauss
- kernelprozess --- das kann ja bewertung sein? random kernel am anfang, dann bewertung ist 
- optimalen edge filter erstellen
- als cost minimization problem- allgemein genug um alle edge typen zu haben
- eine kante ist : teilt zwei regionen die unterschiedliche charakteristiken haben

## Cost Minimization function
- evaluates the quality of edge configurations
  - linear sum of weighted cost factors:
    - accuracy
    - thinness
    - continuity
- Advantage local edge continuity
- Search space large : 2^P P is Amount Pixel
- two steps : dissimilarity enhancement -> enhance good points of imgrey (user based measure)
### What is an edge?
- boundary that seperates two regions that have dissimilar characteristics
- should accurately partition dissimilar regions, thin, continous, sufficient length
  - e.g. geometry, surface reflectance characteristics, viewpoint, illumination

#### What is an imgrey (Input)
- 2D Array of pixels
  - G = {g(i,j;1<=i,j<=N)}
    - each pixel g(i,j) is a gray level
    - N is length of picture
#### What is an edge configuration(Output)
- 2D Array of pixels
  - S = {s(i,j);1<=i,j<=N}
    - each pixel s(i,j) can be 1 if it is an edge pixel or 0 if it is not an exge pixel
    - Set of possible edges
    - Def 1: Edge E is a component of the set of edge pixels in an edge configuration S
    - Def 2: A Sedment of an edge E is a subset of E that is connected
    - Def 3: An edge pixel that is not contained in any cycle of length 3 is called a thin edge pixel, otherwise it 
    is called a thick edge pixel-An edge that contains only thin edge pixels is called 
    a thin edge
    - Def 4: Let L be the NXN pixel site and L_ij be the 3x3 pixel site at position ij
#### Cost function for edges
- cost factors:
  - curvature, dissimilarity, number of edge points, fragmentation, thickness
##### Determining region Dissimilarity
- assigning values to regions that have large dissimilarity
  - dissimilarity imgrey D= {d(i,j);1<=i,j<N}
    - pixel value between 0<=d(i,j)<=1 (1 is a good candidate for edge)
      - basis set of 12 selected edge structures
        - dissimilar Region 1 and 2 are measured by f(R1,R2) = difference of average gray level (can be other approach) 
### How to get enhanced imgrey? (first gen)
1) all pixel d(l) = 0
2) for pixel site (NxN region)
   1) Each edge structure from basis set is fitted by centering it on the location l in the imgrey G
      1) determine regions R1 and R2 for each structure and value of f(R1,R2) is computed
      2) Structure with max f(R1,R2) is chosen as best fitted edge structure for l(at _ij)
         1) denote 3 sites of 3 edge pixels l, l1 , l2
   2) Nonmaximal suppression by shifting location of chosen best fitted edge in a perpendicular directiion determined by the edge structure
      1) determine new f(R1,R2)
         1) Either case: 
            1) no larger -> we set delta=f(R1,R2)/3 with f(R1,R2) fo best fitted structure. We then increment the value of each pixels d(l), d(l1), d(l2) by delta
            2) if larger no alter
   3) all values of the pixels d(l) are truncated to a max of 1

#### Cost Factors
nonendpoint edge pixel -> #neighbouredges >1 
## Curvature
- assigns cost to nonendpoint edge pixel based on a local measure of curvature.
- smooths out or remove curvy edges

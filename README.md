# edge-detection-with-evolutionary-alogirthms
Python implementation using Genetic Algorithms,
Tabu Search and, Evolutionary Tabu Search Algorithm to 
approximate / detect edges in a small picture.
##What is the Chromosome? What to be optimized
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
- two steps : dissimilarity enhancement -> enhance good points of image (user based measure)
### What is an edge?
- boundary that seperates two regions that have dissimilar characteristics
- should accurately partition dissimilar regions, thin, continous, sufficient length
  - e.g. geometry, surface reflectance characteristics, viewpoint, illumination

#### What is an Image (Input)
- 2D Array of pixels
  - G = {g(i,j;1<=i,j<=N)}
    - each pixel g(i,j) is a gray level
    - N is length of picture
#### What is an edge configuration(Output)
- 2D Array of pixels
  - S = {s(i,j);1<=i,j<=N}
    - each pixel s(i,j) can be 1 if it is an edge pixel or 0 if it is not an exge pixel
    - 
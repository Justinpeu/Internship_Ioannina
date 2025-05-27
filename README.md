# Internship_Ioannina
Here you can find all my results and scripts from my internship at the University of Ioannina.

The aim of the internship is to generate synthetic data in the medical field.

## State Of Art Synthetic Data
Initially, I worked on the state of the art concerning the generation of synthetic data, researching the methods, the aim and what synthetic data are.
"SoTA Synthetic Data V2"
## First generation
I then carried out my first tests on the Adults dataset from the Irvine database.
Using my script : "adults_dataset.py"

## Second Generation
The goal is to build a density plot from multiple generator for each feature from the real dataset and the synthetic dataset :
### First try 
Test only with VAE generator : ![Image](https://github.com/user-attachments/assets/15cf8b93-04cc-49ad-9415-079ff6ee1f15)
### Second try : multiple generation with more generator
First generation with multiple generator (CopulaGANSynthesizer, TVAESynthesizer, CTGANSynthesizer, GaussianCopulaSynthesizer) :![density_plot_multiple_generator4X4](https://github.com/user-attachments/assets/45e29f2d-fa19-46d1-b581-32787a4a26aa)
Second generation :   Third generation :

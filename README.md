# Internship_Ioannina
Here you can find all my results and scripts from my internship at the University of Ioannina.

The aim of the internship is to generate synthetic data in the medical field.

## State Of Art Synthetic Data
Initially, I worked on the state of the art concerning the generation of synthetic data, researching the methods, the aim and what synthetic data are.
"SoTA Synthetic Data V2"
## First Generation synthetic data
I then carried out my first tests on the Adults dataset from the Irvine database with multiple generations with multiples generator (CopulaGANSynthesizer, TVAESynthesizer, CTGANSynthesizer, GaussianCopulaSynthesizer).
Using my script : "adults_dataset.py"  
Results and evaluation :  
To analyze the synthetic data and evaluate them, I use different metrics : KL divergence, Jensen Shannon entropy, Hellinger distance and Goodness of fit.  
1st Generation :  
  
![image](https://github.com/user-attachments/assets/8f0f826a-4bb9-44a3-96e1-079b4dd92c67) ![image](https://github.com/user-attachments/assets/665acb52-ebe8-402a-b493-61ab956e7633)  
  
2nd Generation :  
  
![image](https://github.com/user-attachments/assets/5f864385-f31a-4e1b-9990-756c0d01d249) ![image](https://github.com/user-attachments/assets/b23da55c-9573-4d69-9d8c-822bfeaeba4b)  
  
3rd Generation :  
  
![image](https://github.com/user-attachments/assets/3ab90247-a059-4c3e-a3f1-56838c8ccca5) ![image](https://github.com/user-attachments/assets/533f52e3-0e68-45fe-893b-00497b116036)  

## Second Generation synthetic data
The goal is to build a density plot from multiple generator for each feature from the real dataset and the synthetic dataset on the same dataset ("Adults") with the function kdeplot :
### First try 
Test only with VAE generator :  
  
![Image](https://github.com/user-attachments/assets/15cf8b93-04cc-49ad-9415-079ff6ee1f15)
  
### Second try : multiple generation with more generator 
Script : 'Multiple_generator_density_test.py'
First generation with multiple generator (CopulaGANSynthesizer, TVAESynthesizer, CTGANSynthesizer, GaussianCopulaSynthesizer) :  
  
![density_plot_multiple_generator4X4](https://github.com/user-attachments/assets/45e29f2d-fa19-46d1-b581-32787a4a26aa)
  
Second generation :
  
![density_plot_multiple_generator4X4_2nd_gen](https://github.com/user-attachments/assets/ead37614-5104-44b1-ab2e-7088be22e375)
  
Third generation :  

![density_plot_multiple_generator4X4_3rd_gen](https://github.com/user-attachments/assets/7a9241c1-ff80-4b58-8f17-3a8c421bba38)

Generation with augmented size of real samples (1000 on the 3 first generation and 5000 on this one) :  
![density_plot_multiple_generator4X4_5000_realsamples](https://github.com/user-attachments/assets/44637e06-3e31-432b-ac61-eefb77c33503)

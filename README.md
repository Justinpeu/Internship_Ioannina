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
1st Generation :  
  
![image](https://github.com/user-attachments/assets/8f0f826a-4bb9-44a3-96e1-079b4dd92c67) ![image](https://github.com/user-attachments/assets/665acb52-ebe8-402a-b493-61ab956e7633)  
  
2nd Generation :  
  
![image](https://github.com/user-attachments/assets/5f864385-f31a-4e1b-9990-756c0d01d249) ![image](https://github.com/user-attachments/assets/b23da55c-9573-4d69-9d8c-822bfeaeba4b)  
  
3rd Generation :  
  
![image](https://github.com/user-attachments/assets/3ab90247-a059-4c3e-a3f1-56838c8ccca5) ![image](https://github.com/user-attachments/assets/533f52e3-0e68-45fe-893b-00497b116036)  

## Second Generation synthetic data
The goal is to build a density plot from multiple generator for each feature from the real dataset and the synthetic dataset on the same dataset ("Adults"):
### First try 
Test only with VAE generator :  
  
![Image](https://github.com/user-attachments/assets/15cf8b93-04cc-49ad-9415-079ff6ee1f15)
  
### Second try : multiple generation with more generator 

[Upload### Adult Dataset

# Importation dataset

import pandas as pd

data = pd.read_csv('C://Users//justi//Desktop//Stage_Ioannina//.venv//adult.data')
column_name= ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
               'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

data.columns = column_name
data = data.sample(n=1000)
print(data)

#Metadata creation 

from sdv.metadata import Metadata

metadata = Metadata.detect_from_dataframe(data)


##Data generation

from sdv.single_table import CopulaGANSynthesizer, TVAESynthesizer, CTGANSynthesizer, GaussianCopulaSynthesizer

#Gaussian Copula

GPS = GaussianCopulaSynthesizer(metadata)
GPS.fit(data)
GPS_synthetic_data = GPS.sample(num_rows=100)

#CTGAN

CTG = CTGANSynthesizer(metadata, epochs=500, discriminator_steps=3, embedding_dim=256)
CTG.fit(data)
CTG_synthetic_data = CTG.sample(num_rows=100)

#VAE
VAE = TVAESynthesizer(metadata, epochs=500, embedding_dim=256)
VAE.fit(data)
VAE_synthetic_data = VAE.sample(num_rows=100)

#Copula GAN
cop_GAN = CopulaGANSynthesizer(metadata, epochs=500, embedding_dim=256)
cop_GAN.fit(data)
cop_GAN_synthetic_data = cop_GAN.sample(num_rows=100)

#Plot


import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
for col in column_name :
    if not pd.api.types.is_numeric_dtype(data[col]):
        data[col]= le.fit_transform(data[col])
    if not pd.api.types.is_numeric_dtype(GPS_synthetic_data[col]):
        GPS_synthetic_data[col]=le.fit_transform(GPS_synthetic_data[col])
    if not pd.api.types.is_numeric_dtype(CTG_synthetic_data[col]):
        CTG_synthetic_data[col]=le.fit_transform(CTG_synthetic_data[col])
    if not pd.api.types.is_numeric_dtype(VAE_synthetic_data[col]):
        VAE_synthetic_data[col]=le.fit_transform(VAE_synthetic_data[col])
    if not pd.api.types.is_numeric_dtype(cop_GAN_synthetic_data[col]):
        cop_GAN_synthetic_data[col]=le.fit_transform(cop_GAN_synthetic_data[col])

fig, axes = plt.subplots(4,4, figsize=(16,16))
for i, colonne in enumerate(column_name):
    row = i // 4
    colo = i % 4
    ax = axes[row, colo]
    sb.kdeplot(data[colonne], ax=ax, color='blue')
    sb.kdeplot(GPS_synthetic_data[colonne], ax=ax, color='orange')
    sb.kdeplot(CTG_synthetic_data[colonne], ax=ax, color='green')
    sb.kdeplot(VAE_synthetic_data[colonne], ax=ax, color='yellow')
    sb.kdeplot(cop_GAN_synthetic_data[colonne], ax=ax, color='red')
    ax.legend()

fig.legend(['valeur réelle', 'GPS', 'CTG', 'VAE', 'cop GAN'], loc='lower right')
plt.tight_layout()
plt.show()ing Multiple_generator_density_test.py…]()

First generation with multiple generator (CopulaGANSynthesizer, TVAESynthesizer, CTGANSynthesizer, GaussianCopulaSynthesizer) :  
  
![density_plot_multiple_generator4X4](https://github.com/user-attachments/assets/45e29f2d-fa19-46d1-b581-32787a4a26aa)
  
Second generation :
  
![density_plot_multiple_generator4X4_2nd_gen](https://github.com/user-attachments/assets/ead37614-5104-44b1-ab2e-7088be22e375)
  
Third generation :  

![density_plot_multiple_generator4X4_3rd_gen](https://github.com/user-attachments/assets/7a9241c1-ff80-4b58-8f17-3a8c421bba38)

### Adult Dataset

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
plt.show()
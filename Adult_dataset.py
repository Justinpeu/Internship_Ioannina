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

##Evaluation

#Metrics

from sdv.evaluation.single_table import evaluate_quality, run_diagnostic

rd_gps = run_diagnostic(real_data=data, synthetic_data=GPS_synthetic_data, metadata=metadata)
eq_gps = evaluate_quality(real_data=data, synthetic_data=GPS_synthetic_data, metadata=metadata)

rd_ctg = run_diagnostic(real_data=data, synthetic_data=CTG_synthetic_data, metadata=metadata)
eq_ctg = evaluate_quality(real_data=data, synthetic_data=CTG_synthetic_data, metadata=metadata)

rd_vae = run_diagnostic(real_data=data, synthetic_data=VAE_synthetic_data, metadata=metadata)
eq_vae = evaluate_quality(real_data=data, synthetic_data=VAE_synthetic_data, metadata=metadata)

rd_cop = run_diagnostic(real_data=data, synthetic_data=cop_GAN_synthetic_data, metadata=metadata)
eq_cop = evaluate_quality(real_data=data, synthetic_data=cop_GAN_synthetic_data, metadata=metadata)

# KL divergence, the Jenssen shannon entropy, the Hellinger distance, and the goodness of fit

real_col = data['hours-per-week'].dropna()

gps_col = GPS_synthetic_data['hours-per-week'].dropna()
ctg_col =CTG_synthetic_data['hours-per-week'].dropna()
vae_col = VAE_synthetic_data['hours-per-week'].dropna()
copGAN_col = cop_GAN_synthetic_data['hours-per-week'].dropna()

import numpy as np

bins = np.histogram_bin_edges(real_col, bins='auto')

real_hist, _ = np.histogram(real_col, bins=bins, density=True)

gps_hist, _ = np.histogram(gps_col, bins=bins, density=True)
ctg_hist, _ = np.histogram(ctg_col, bins=bins, density=True)
vae_hist, _ = np.histogram(vae_col, bins=bins, density=True)
copGAN_hist, _ = np.histogram(copGAN_col, bins=bins, density=True)

epsilon = 1e-10
real_hist += epsilon
gps_hist += epsilon
ctg_hist += epsilon
vae_hist += epsilon
copGAN_hist += epsilon

real_hist /= real_hist.sum()
gps_hist /= gps_hist.sum()
ctg_hist /= ctg_hist.sum()
vae_hist /= vae_hist.sum()
copGAN_hist /= copGAN_hist.sum()


def KL_div(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def JS_div(p, q):
    M = 0.5*(p + q)
    JS = 0.5*KL_div(p, M)+ 0.5*(KL_div(q, M))
    return JS


def hellinger(p, q):
    return np.sqrt(0.5 * np.sum((np.sqrt(p)-np.sqrt(q))**2))



from scipy.stats import chisquare

chi2_stat_gps, p_value_gps = chisquare(f_obs=real_hist, f_exp=gps_hist)
chi2_stat_ctg, p_value_ctg = chisquare(f_obs=real_hist, f_exp=ctg_hist)
chi2_stat_vae, p_value_vae = chisquare(f_obs=real_hist, f_exp=vae_hist)
chi2_stat_copGAN, p_value_copGAN = chisquare(f_obs=real_hist, f_exp=copGAN_hist)


results={'Method':['GPS', 'CTG', 'VAE', 'COP GAN'], 'KL DIvergence': [KL_div(real_hist, gps_hist), KL_div(real_hist, ctg_hist), KL_div(real_hist, vae_hist), KL_div(real_hist, copGAN_hist)],
         'JS Divergence':[JS_div(real_hist, gps_hist), JS_div(real_hist, ctg_hist), JS_div(real_hist, vae_hist), JS_div(real_hist, copGAN_hist)],
         'Hellinger distance':[hellinger(real_hist, gps_hist), hellinger(real_hist, ctg_hist), hellinger(real_hist, vae_hist), hellinger(real_hist, copGAN_hist)],
         'Chi Square':[chi2_stat_gps, chi2_stat_ctg, chi2_stat_vae, chi2_stat_copGAN],
         'Chi Square P-value':[p_value_gps, p_value_ctg, p_value_vae, p_value_copGAN]}

df_results = pd.DataFrame(results)

print(df_results)

#2X2 Grid comparison

from sdv.evaluation.single_table import get_column_plot

GPS_plot = get_column_plot(real_data= data, synthetic_data=GPS_synthetic_data, metadata=metadata, column_name='hours-per-week')
CTG_plot = get_column_plot(real_data= data, synthetic_data=CTG_synthetic_data, metadata=metadata, column_name='hours-per-week')
VAE_plot = get_column_plot(real_data= data, synthetic_data=VAE_synthetic_data, metadata=metadata, column_name='hours-per-week')
cop_GAN_plot = get_column_plot(real_data= data, synthetic_data=cop_GAN_synthetic_data, metadata=metadata, column_name='hours-per-week')


from plotly.subplots import make_subplots

fig = make_subplots(rows =2, cols=2, subplot_titles=('GPS', 'CTG', 'VAE', 'COPGAN'))

for trace in GPS_plot.data:
    fig.add_trace(trace, row=1, col=1)

for trace in CTG_plot.data:
    fig.add_trace(trace, row=1, col=2)

for trace in VAE_plot.data:
    fig.add_trace(trace, row=2, col=1)

for trace in cop_GAN_plot.data:
    fig.add_trace(trace, row=2, col=2)

fig.update_layout(height=800, width=1000, title_text="Comparaison des modèles")

fig.show()
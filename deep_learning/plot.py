import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results_run.csv")
abscissa = df['abcsissa'].values
prec = df['precision_mean'].values
rec = df['recall_mean'].values
prec_std = df['precision_std'].values
rec_std = df['recall_std'].values



plt.rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)


plt.plot(abscissa, prec,color='green',label='Neural Net')
plt.fill_between(abscissa, prec-prec_std, prec+prec_std,  
    alpha=0.5,color='#e0e0e0')

plt.xlabel(r'\textbf{Abscissa}',fontsize=12)
plt.ylabel(r'\textbf{Precision} [\%]',fontsize=16)
plt.ylim(-0.005,1.05)
plt.legend()

plt.show()

plt.rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)


plt.plot(abscissa, rec,color='green',label='Neural Net')
plt.fill_between(abscissa, rec-rec_std, rec+rec_std,  # fills area according to standard deviation
    alpha=0.5,color='#e0e0e0')

plt.xlabel(r'\textbf{Abscissa}',fontsize=12)
plt.ylabel(r'\textbf{Recall} [\%]',fontsize=16)
plt.ylim(-0.005,1.05)
plt.legend()

plt.show()
exit(0)
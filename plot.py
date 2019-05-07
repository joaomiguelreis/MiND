
##################################

## PLOTTING RESULTS ##

plt.rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)

plt.plot(abscissa, prec_error_3,color='green',label='Tree depth 3')
plt.plot(abscissa, prec_error_6,color='red',label='Tree depth 6')
plt.plot(abscissa, prec_error_9,color='blue',label='Tree depth 9')
plt.fill_between(abscissa, prec_error_3-prec_errstd_3, prec_error_3+prec_errstd_3,
        alpha=0.5,color='#e0e0e0')

plt.xlabel(r'\textbf{Abscissa}',fontsize=12)
plt.ylabel(r'\textbf{Precision} [\%]',fontsize=16)
plt.ylim(-0.5,105)
plt.legend()

plt.show()

plt.rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)

plt.plot(abscissa, rec_error_3,color='green',label='Tree depth 3')
plt.plot(abscissa, rec_error_6,color='red',label='Tree depth 6')
plt.plot(abscissa, rec_error_9,color='blue',label='Tree depth 9')
plt.fill_between(abscissa, rec_error_3-rec_errstd_3, rec_error_3+rec_errstd_3,
        alpha=0.5,color='#e0e0e0')

plt.xlabel(r'\textbf{Abscissa}',fontsize=12)
plt.ylabel(r'\textbf{Recall} [\%]',fontsize=16)
plt.ylim(-0.5,105)
plt.legend()

plt.show()




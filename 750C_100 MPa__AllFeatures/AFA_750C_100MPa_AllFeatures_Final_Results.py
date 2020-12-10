import AFA_750C_100MPa_AllFeatures_KNN as KNN
import AFA_750C_100MPa_AllFeatures_RF as RF
import matplotlib.pyplot as plt

KNN_acc = []
KNN_acc.append(KNN.acc_features_2)
KNN_acc.append(KNN.acc_score_2)
KNN_acc.append(KNN.acc_score_r2_2)


RF_acc = []
RF_acc.append(RF.acc_features_2)
RF_acc.append(RF.acc_score_2)
RF_acc.append(RF.acc_score_r2_2)



plt.xlabel('# of features')
plt.ylabel('Accuracy (PCC)')
plt.title('Accuracy vs # of features')
plt.plot(RF_acc[0], RF_acc[1], 'g*-', label = 'RF')
plt.plot(KNN_acc[0], KNN_acc[1],'bo-', label = 'KNN')
plt.ylim([-25,100])
plt.legend()
plt.figure()

plt.xlabel('# of features')
plt.ylabel('Accuracy (R2)')
plt.title('Accuracy vs # of features')
plt.plot(RF_acc[0], RF_acc[2], 'g*-', label = 'RF')
plt.plot(KNN_acc[0], KNN_acc[2],'bo-', label = 'KNN')
plt.ylim([-25,100])
plt.legend()
plt.figure()


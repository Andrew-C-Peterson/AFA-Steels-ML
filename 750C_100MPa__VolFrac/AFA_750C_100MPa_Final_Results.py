import AFA_750C_100MPa_KNN as KNN
import AFA_750C_100MPa_LR as LM
import AFA_750C_100MPa_RF as RF
import matplotlib.pyplot as plt

KNN_acc = []
KNN_acc.append(KNN.acc_features)
KNN_acc.append(KNN.acc_score)
KNN_acc.append(KNN.acc_score_r2)

LM_acc = []
LM_acc.append(LM.acc_features)
LM_acc.append(LM.acc_score)
LM_acc.append(LM.acc_score_r2)

RF_acc = []
RF_acc.append(RF.acc_features)
RF_acc.append(RF.acc_score)
RF_acc.append(RF.acc_score_r2)



plt.xlabel('# of features')
plt.ylabel('Accuracy (PCC)')
plt.title('Accuracy vs # of features')
plt.plot(RF_acc[0], RF_acc[1], 'g*-', label = 'RF')
plt.plot(KNN_acc[0], KNN_acc[1],'bo-', label = 'KNN')
plt.plot(LM_acc[0], LM_acc[1],'r^-', label = 'LR')
plt.ylim([-80,100])
plt.legend()
plt.figure()

plt.xlabel('# of features')
plt.ylabel('Accuracy (R2)')
plt.title('Accuracy vs # of features')
plt.plot(RF_acc[0], RF_acc[2], 'g*-', label = 'RF')
plt.plot(KNN_acc[0], KNN_acc[2],'bo-', label = 'KNN')
plt.plot(LM_acc[0], LM_acc[2],'r^-', label = 'LR')
plt.ylim([-60,100])
plt.legend()
plt.figure()
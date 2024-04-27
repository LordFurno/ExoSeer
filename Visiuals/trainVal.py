
import matplotlib.pyplot as plt
#Training accuracy over epochs

epoch1Train=[0.7817,0.8020,0.8049,0.8048,0.8063,0.8047,0.8072,0.8073,0.8070, 0.8084,0.8080,0.8075,0.8086,0.8086,0.8082]
epoch2Train=[0.7826,0.8031,0.8039,0.8054,0.8049,0.8063,0.8057,0.8068,0.8062,0.8068,0.8069,0.8082,0.8070,0.8080,0.8073]
epoch3Train=[0.7864,0.8052,0.8078,0.8102,0.8096,0.8092,0.8103,0.8103,0.8098,0.8094,0.8106,0.8091,0.8096,0.8093,0.8101]
epoch4Train=[0.7814,0.8022,0.8058,0.8071,0.8076,0.8082,0.8076,0.8070,0.8081,0.8091,0.8099,0.8095,0.8097,0.8089,0.8103]
epoch5Train=[0.7838,0.8036,0.8083,0.8083,0.8097,0.8106,0.8105,0.8112,0.8100,0.8108,0.8098,0.8112,0.8095,0.8101,0.8111]

epoch1Val=[0.8189,0.8171,0.8176,0.8152,0.8181,0.8181,0.8179,0.8175,0.8170,0.8181,0.8175,0.8182,0.8067,0.8107,0.8152]
epoch2Val=[0.8147,0.8164,0.8172,0.8179,0.8162,0.8168,0.8168,0.7452,0.8169,0.7686,0.8168,0.8148,0.8170,0.8157,0.8159]
epoch3Val=[0.8150,0.8135,0.8118,0.8147,0.8136,0.8137,0.8137,0.8076,0.8143,0.8068,0.8135,0.8138,0.8144,0.8140,0.8131]
epoch4Val=[0.7460,0.8154,0.8150,0.8181,0.8186,0.8172,0.8178,0.8191,0.8173,0.8178,0.8044,0.8150,0.8178,0.8150,0.8150]
epoch5Val=[0.8174,0.8182,0.8162,0.8166,0.8166,0.8182,0.8174,0.7606,0.8175,0.8107,0.8166,0.8163,0.8160,0.8169,0.8171]

plt.figure(0,figsize=(15,15))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.rcParams.update({'font.size': 15})
plt.title("Training Accuracy Over Epochs for Each Fold")
plt.plot(epoch1Train,label="1st validation fold")
plt.plot(epoch2Train,label="2nd validation fold")
plt.plot(epoch3Train,label="3rd validation fold")
plt.plot(epoch4Train,label="4th validation fold")
plt.plot(epoch5Train,label="5th validation fold")
plt.legend()

plt.xlabel("Training epochs",fontsize=15)
plt.ylabel("Training accuracy",fontsize=15)
plt.show(block=True)


plt.figure(1,figsize=(15,15))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.rcParams.update({'font.size': 15})
plt.title("Validation Accuracy Over Epochs for Each Fold")
plt.plot(epoch1Val,label="1st validation fold")
plt.plot(epoch2Val,label="2nd validation fold")
plt.plot(epoch3Val,label="3rd validation fold")
plt.plot(epoch4Val,label="4th validation fold")
plt.plot(epoch5Val,label="5th validation fold")
plt.legend()

plt.xlabel("Training epochs",fontsize=15)
plt.ylabel("Validation accuracy",fontsize=15)
plt.show(block=True)
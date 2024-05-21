import matplotlib.pyplot as plt
epoch1Train=[1.4299,0.3885,0.3564,0.3445,0.3313,0.3296,0.3252,0.3167,0.3117,0.3085,0.3091,0.3053,0.3053,0.3045,0.3039]
epoch2Train=[1.4502,0.3864,0.3606,0.3439,0.3370,0.3265,0.3280,0.3157,0.3187,0.3154,0.3110,0.3114,0.3065,0.3051,0.3053]
epoch3Train=[1.4061,0.3841,0.3537,0.3381,0.3282,0.3202,0.3176,0.3115,0.3064,0.3097,0.3050,0.3079,0.3032,0.3022,0.3005]
epoch4Train=[1.4411,0.3863,0.3564,0.3428,0.3311,0.3209,0.3223,0.3141,0.3079,0.3082,0.3031,0.3055,0.3023,0.2987]
epoch5Train=[1.4785,0.3838,0.3515,0.3393,0.3287,0.3261,0.3169,0.3118,0.3143,0.3092,0.3036,0.3039,0.3062,0.3016,0.2987]


epoch1Val=[0.3616,0.3266,0.3158,0.3168,0.2992,0.2953,0.3041,0.2999,0.2880,0.2884,0.2786,0.2810,0.3123,0.3050,0.3001]
epoch2Val=[0.3719,0.3326,0.3297,0.3058,0.3032,0.3038,0.2939,0.4845,0.2888,0.7069,0.2886,0.2868,0.2778,0.2863,0.2829]
epoch3Val=[0.3859,0.3486,0.3661,0.3041,0.2997,0.2984,0.2903,0.3699,0.2845,0.3046,0.2807,0.2835,0.2856,0.2842,0.2913]
epoch4Val=[0.4680,0.3363,0.3180,0.3171,0.3004,0.2976,0.2900,0.2847,0.2963,0.2814,0.3085,0.2822,0.2811,0.2761,0.2757]
epoch5Val=[0.3767,0.3282,0.3090,0.3039,0.2989,0.2933,0.2959,0.5570,0.2851,0.3108,0.2795,0.2906,0.2828,0.3071,0.2760]



plt.figure(0,figsize=(10,2.81))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.rcParams.update({'font.size': 15})
plt.title("Training Loss Over Epochs for Each Fold")
plt.plot(epoch1Train,label="1st validation fold",marker='o')
plt.plot(epoch2Train,label="2nd validation fold", marker='s')
plt.plot(epoch3Train,label="3rd validation fold", marker='^')
plt.plot(epoch4Train,label="4th validation fold",marker='d')
plt.plot(epoch5Train,label="5th validation fold",marker='x')
plt.legend()
plt.savefig("TrainLoss.png")
plt.show()


plt.figure(1,figsize=(10,2.81))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.rcParams.update({'font.size': 15})
plt.title("Validation Loss Over Epochs for Each Fold")
plt.plot(epoch1Val,label="1st validation fold",marker='o')
plt.plot(epoch2Val,label="2nd validation fold", marker='s')
plt.plot(epoch3Val,label="3rd validation fold", marker='^')
plt.plot(epoch4Val,label="4th validation fold",marker='d')
plt.plot(epoch5Val,label="5th validation fold",marker='x')
plt.legend()
plt.savefig("ValLoss.png")

plt.show()


import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.simplefilter('ignore')
import operator

plt.style.use('ggplot')
#---PARAMETERS---
tick_min = -100
tick_max = 80
tick = 15

#---DATA---
x = np.arange(1, 101)
participants = {"p1":[-8,-7,-2,-23,8,-7,-2,-35,21,-13,7,4,-17,3,-5,-2,4,1,-8,4,5,-12,4,-35,-17,-4,12,3,-6,2,-47,-16,-10,-31,-48,-33,-31,-15,-28,-44,-26,-34,-29,-38,-21,-8,-11,-18,-34,-10,-28,-26,-23,-46,-9,-2,-31,-26,-8,-17,-28,-25,-16,-21,-26,-32,-15,-14,-22,-25,19,22,-32,1,12,4,8,20,-13,13,-13,1,2,13,-3,42,35,5,-5,26,30,2,-7,9,19,21,24,25,18,-10],
                "p2":[-15,2,-9,-2,-22,-17,2,-5,-4,-12,-15,-17,5,-27,-11,11,14,2,-3,-5,-7,-11,18,-18,1,7,-13,-9,7,-14,-76,-56,-6,-45,-24,-44,-41,-20,-14,-24,-37,-19,-23,-15,-13,-22,-23,-20,-17,-15,9,-6,-12,-11,-16,-8,-5,-21,-15,2,-9,3,-4, 3,-11,-8,-11,-10,-19,-2,-2,7,1,4,-2,-3,-4,6,-3,-4,-4,-3,2,-7,1,38,38,20,29,-4,-7,12,11,-7,22,6,8,-12,7,10],
                "p3":[-9, -7, -6, 4, -10, -20, 5, -3, 4, 0, -6, -2, 3, 2, -17, 0, -5, 10, 9, 8, -11, 0, 6, 18, -9, 15, 4, -5, -11, 0, -51, -11, -43, -44, -43, -34, -24, -28, -17, -27, -15, -14, -35, -25, -22, -24, -14, -27, -3, -4, -12, -7, -16, 2, 1, -8, -19, 4, -10, -11, -17, -5, -33, -12, -10, -11, -5, 1, -2, -18, 16, 7, 6, 0, -1, 0, -5, -13, 3, -6, 4, 10, -7, 11, 28, 20, 17, 13, 18, 1, 7, 11, 7, 1, -3, 11, 4, 2, 17, 12],
                "p4":[5, 0, -6, 3, 5, 1, -17, 1, -4, 10, -2, 15, 1, -5, -5, -12, 3, 3, 11, 4, -8, 14, -7, -2, -9, -5, 1, 6, 3, -58, -44, -56, -51, -41 ,-42, -38, -28, -4, -20, -14, -26, -10, -13, -10, -21, -27, -24, -10, -7, -15, -7, -10, -9, -17, -15, -6, -14, -19, -3, -12, 1, -15, -30, 0, 8, -17, -14, -11, -9, -6, 5 ,13, 12, 13, 14, 0, 5, 15, 9, -5, -4, -3, 4, -4, 43, 25, 58, 24, 24, 27, 5, 18, 17, 30, -5, -2, 9, 3, 13, 28]}

for i in range(1,5):
    plt.subplot(3, 2, i)
    y = participants[f'p{i}']
    j=['blue','red','green','brown','orange','purple']
    plt.plot(x, y, 'o',c=j[i-1])
    for l in [14, 29, 69, 84]:
        plt.axvline(x=l, linestyle='--')
    plt.title(f'Participant {i}')
    plt.yticks(np.arange(tick_min, tick_max, tick))
    plt.ylim(-100, 50)

# avg = [v+w+x+y for v,w,x,y in zip(participants['p1'],participants['p2'],participants['p3'],participants['p4'])]
# #avg = list(map(operator.add,participants['p1'],participants['p2']))
# #avg = np.add(participants['p1'],participants['p2'])
# plt.subplot(3,2,6)
# plt.plot(x,avg,'o')
# for l in [14, 29, 69, 84]:
#     plt.axvline(x=l, linestyle='--')
# plt.title('Average')
# tick_min = -200
# tick_max = 80
# tick = 15
# plt.yticks(np.arange(tick_min, tick_max, tick))
# plt.ylim(-200,50)
plt.tight_layout()
plt.show()






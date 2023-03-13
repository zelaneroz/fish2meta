import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

plt.style.use('ggplot')


def exp_model(x, a, b, c):
    return a - (b * np.exp(-x / c))

file = "Fish2Meta Data - Ex 3 - 1_27_23.csv"
#file="Fish2Meta Data - Ex 2 - 11_30_22.csv"
with open(file) as f:
    data = f.readlines()

n_participants = data[0].count(',')+1
#for experiment 3
stage = ['before_u', 'before_o', 'prism_u', 'prism_o', 'after_u', 'after_o']
#for experimetn 2
#stage = ['before_u', 'before_o', 'prism', 'after_u', 'after_o'] #exp2
exp_data = [{x: [] for x in stage} for _ in range(n_participants)]
duration = [15, 15, 40, 40, 15, 15] #exp3
#duration = [15,15,40,15,15]

times = [1]
for d in duration:
    times.append(times[-1] + d)
x_values = {x: list(range(t, t + d)) for x, t, d in zip(stage, times, duration)}

for x, d in enumerate(data):
    values = list(map(float, d.strip().split(',')))
    for v, p in zip(values, exp_data):
        for k, t in x_values.items():
            if x + 1 in t:
                p[k].append(v)
subject = 2
#avg=[-7,-3,-4,-3,-3,3,-4,-7,2,3,3,3,-11,-1,5,-7,-10,-8,2,-8,-7,-5,-3,-8,-6,13,-8,-16,3,-4,-56,-46,-26,-33,-19,-23,-18,-22,-12,-15,-17,-16,-13,-6,-12,5,-21,-11,-24,-11,-12,-13,-13,-18,-11,2,-13,-8,-16,-8,-21,-13,-14,-15,-8,-10,-9,2,-3,-31,-48,-45,-18,-43,-27,-14,-25,-12,-24,-29,-11,-27,-18,-11,-25,-24,-18,-2,-28,-17,-10,-16,-24,-32,-1,-18,-21,-7,-13,-14,-7,-7,-21,-11,-4,-4,-12,-5,-10,-18,19,29,28,10,12,20,-5,11,2,2,3,-6,7,2,8,23,22,22,13,14,6,2,1,3,6,14,3,6,-4,5]
#exp_data[subject]={'before_u': avg[0:15], 'before_o': avg[15:30], 'prism': avg[30:70], 'after_u':avg[70:85], 'after_o': avg[85:100]}
print(exp_data[subject])
avg=[-11,-5,4,4,-1,4,-1,-13,1,-6,2,-13,-8,-6,2,-2,-3,-10,-11,0,-12,-7,-6,-2,2,9,-7,-13,-10,3,-63,-46,-42,-40,-31,-20,-23,-29,-19,-27,-28,-21,-30,-14,-23,-14,-25,-23,-34,-22,-36,-26,-24,-31,-27,-8,-27,-14,-25,-28,-30,-26,-26,-22,-13,-29,-18,-10,-21,-38,-37,-47,-36,-49,-26,-28,-33,-20,-29,-45,-22,-31,-29,-30,-27,-20,-28,-20,-28,-26,-8,-21,-26,-39,-8,-28,-33,-21,-22,-33,-11,-33,-42,-31,-27,-18,-35,-14,-11,-28,29,40,27,9,19,24,2,20,8,3,2,-10,5,2,2,17,17,18,2,15,7,-1,0,3,6,14,7,-8,-7,9]
exp_data[subject] = {'before_u':avg[0:15],'before_o':avg[15:30],'prism_u':avg[30:70],'prism_o':avg[70:110],'after_u':avg[110:125],'after_o':avg[125:140]}
print(exp_data[subject])
my_colors = {'before_u': 'b', 'before_o': 'g', 'prism_u': 'r','prism_o': 'r', 'after_u': 'b', 'after_o': 'g'}
#my_colors = {'before_u': 'b', 'before_o': 'g', 'prism': 'r', 'after_u': 'b', 'after_o': 'g'}#exp 2
#print(exp_data[subject])
plt.figure(figsize=(14, 9))

x = 1  # start of the graph
model = {} # decay constants
for st in stage:
    y_values = exp_data[subject][st]
    x_model = []
    for n in y_values:
        x_model.append(x)
        plt.plot(x, n, 'o', markeredgecolor=my_colors[st], markersize=15, fillstyle='none')
        x += 2

    x += 5 #space between stages
    plt.vlines(x, -100, 100, colors='gray')
    x += 5  # space between stages
    if st in ['prism_u','prism_o','after_u', 'after_o']: #exp3
    #if st in ['prism','after_u','after_o'] #exp2:
        x_norm = (np.array(x_model) - x_model[0]) / (x_model[-1] - x_model[0])  # normalized
        a_0 = np.mean(y_values[-5:-1]) # final value exponential function moves towards
        print("a_o: ",a_0)
        c_0 = 1 # the decay constant
        b_0 = (y_values[0] - a_0) # the magnitude of adaptation
        print("b_o: ",b_0)
        popt, pcov = curve_fit(exp_model, x_norm, y_values, p0=(a_0, b_0, c_0))
        plt.plot(x_model, exp_model(x_norm, *popt), color='black',linestyle="dashed")
        print()
        c_ori = (x_model[-1] - x_model[0]) * popt[2]
        print("c_ori: ", c_ori, "\n")
        model[st] = (popt[0], popt[1], c_ori)
        print(f"{popt[0]:.1f}{'-+'[popt[1]>0]}{abs(popt[1]):.1f}e(-t/{c_ori:.1f})",)


#
#
plt.axhline(y=0, color='gray')
plt.ylim(-100, 100)
plt.ylabel('HORIZONTAL DISPLACEMENT (cm)', fontsize=22)
xlabel = '$TIME \longrightarrow$\n'
models =['prism_u','prism_o','after_u', 'after_o'] #exp3
#models =['prism','after_u', 'after_o'] #Exp 2
for s in models:
    m = model[s]
    xlabel+=f"${m[0]:.1f}{'-+'[m[1]>0]}{abs(m[1]):.1f}e(-t/{m[2]:.1f})$  "
#Experiment 3
plt.xlabel(xlabel, fontsize=22)
plt.yticks(fontsize=20)
plt.xticks([])
plt.text(-5, -90, '$BEFORE_u$', fontsize=20)
plt.text(37, -90, '$BEFORE_o$', fontsize=20)
plt.text(95, -90, '$PRISMS_u$', fontsize=20)
plt.text(185, -90, '$PRISMS_o$', fontsize=20)
plt.text(260, -90, '$AFTER_u$', fontsize=20)
plt.text(300, -90, '$AFTER_o$', fontsize=20)
#plt.title(f'SUBJECT {subject + 1} {file}', fontsize=22)
plt.title('MEAN', fontsize=22)

#PC standard deviation of the 8 throws before googles
PC_1 = np.std(exp_data[subject][stage[0]])
PC_2 = np.std(exp_data[subject][stage[1]])
plt.text(-5, 90, f'$PC = {PC_1:.1f}$', fontsize=20)
plt.text(37, 90, f'$PC = {PC_2:.1f}$', fontsize=20)
plt.text(95, 90, f'$AC = {model[stage[2]][2]:.1f}$', fontsize=20)
plt.text(185, 90, f'$AC = {model[stage[3]][2]:.1f}$', fontsize=20)
plt.text(260, 90, f'$AC = {model[stage[4]][2]:.1f}$', fontsize=20)
plt.text(300, 90, f'$AC = {model[stage[5]][2]:.1f}$', fontsize=20)
plt.savefig('media/final_final_plots/exp_3/avg.png')
plt.show()
print(model)

print(f"PCu = {PC_1:.1f}")
print(f"PCo = {PC_2:.1f}")
print(f"ACpu = {model[stage[2]][2]:.1f}")
print(f"ACpo = {model[stage[3]][2]:.1f}")
print(f"ACu = {model[stage[4]][2]:.1f}")
print(f"ACo = {model[stage[5]][2]:.1f}")

#Exp2
# plt.xlabel(xlabel, fontsize=22)
# plt.yticks(fontsize=20)
# plt.xticks([])
# plt.text(-5, -90, '$BEFORE_u$', fontsize=20)
# plt.text(37, -90, '$BEFORE_o$', fontsize=20)
# plt.text(105, -90, '$PRISMS$', fontsize=20)
# plt.text(170, -90, '$AFTER_u$', fontsize=20)
# plt.text(210, -90, '$AFTER_o$', fontsize=20)
# #plt.title(f'SUBJECT {subject + 1}', fontsize=22)
# plt.title(f'AVERAGE', fontsize=22)

# PC standard deviation of the 8 throws before googles
# PC_1 = np.std(exp_data[subject][stage[0]])
# PC_2 = np.std(exp_data[subject][stage[1]])
# plt.text(-5, 90, f'$PC = {PC_1:.1f}$', fontsize=20)
# plt.text(37, 90, f'$PC = {PC_2:.1f}$', fontsize=20)
# plt.text(110, 90, f'$AC = {model[stage[2]][2]:.1f}$', fontsize=20)
# plt.text(170, 90, f'$AC = {model[stage[3]][2]:.1f}$', fontsize=20)
# plt.text(210, 90, f'$AC = {model[stage[4]][2]:.1f}$', fontsize=20)
# plt.show()

# print(f"PCu = {PC_1:.1f}")
# print(f"PCo = {PC_2:.1f}")
# print(f"ACpo = {model[stage[2]][2]:.1f}")
# print(f"ACu = {model[stage[3]][2]:.1f}")
# print(f"ACo = {model[stage[4]][2]:.1f}")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

plt.style.use('ggplot')


def exp_model(x, a, b, c):
    return a - (b * np.exp(-x / c))

file = "Fish2Meta Data - Ex 4 - 3_4_23.csv"
with open(file) as f:
    data = f.readlines()

n_participants = data[0].count(',')+1
#for experiment 3
stage = ['before_u', 'before_o', 'prism_o', 'prism_u', 'after_o', 'after_u']
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
subject = 5
#avg = [2,7,-9,-7,-1,1,-6,-3,1,8,12,-6,-5,4,-4,4,-4,-15,4,-1,-3,-1,-3,2,-6,-7,-2,-9,3,2,-78,-51,-41,-36,-35,-26,-35,-22,-29,-33,-24,-26,-22,-15,-18,-25,-20,-21,-12,-18,-21,-15,-17,-13,-17,-22,-6,-19,-13,-15,-19,-5,-3,-6,-16,-15,3,-11,-15,-7,-56,-48,-37,-17,-10,-22,-28,-22,-28,-12,-11,-24,-16,-20,-13,-7,-5,-22,-3,-12,-14,-9,-4,-10,-9,-13,-2,-8,-9,-11,-13,-9,-5,-9,-14,-4,-17,-13,-14,-3,49,38,20,19,21,17,17,9,9,18,6,5,13,15,4,39,36,23,21,14,11,20,6,11,16,4,11,5,13,5]
#actually the median
avg=[-6,-3.1,-10,-6.3,2,2,-2,-2,0,3,3,-2,-4,1,-3,0,-4,-8,1,-5,-5.5,-5.1,-4,2,2,3,-4,-6,6,3,-70,-43,-35,-32.5,-19.3,-19,-24,-12,-12,-15.3,-16.7,-16,-12.9,-9,-15,-4,-20,-11,-16.1,-13,-15,-12,-12.7,-13,-11.1,-7,-10,-10.5,-11,-8,-21,-5.6,-9,-8,-7,-8,1,1.6,-8,-14,-52,-45,-33,-30.6,-14,-21,-25,-12,-24,-13,-11,-21,-11,-11,-17.4,-12,-11,-13,-9,-14,-13,-12,-9.5,-16,-5,-15,-12,-7,-13,-9,-8,-5,-9,-11,-11,-3.7,-12,-4.6,-9.7,-7,42.5,40.5,24.5,20,12,16,12,11.85,3,12.5,8,1,6.9,2,2.55,37,26,22,20,13,10.2,14,1,6.5,16,3,9,5.6,4.5,6]
#exp_data[subject]={'before_u': avg[0:15], 'before_o': avg[15:30], 'prism': avg[30:70], 'after_u':avg[70:85], 'after_o': avg[85:100]}
print(exp_data[subject])
exp_data[subject] = {'before_u':avg[0:15],'before_o':avg[15:30],'prism_o':avg[30:70],'prism_u':avg[70:110],'after_o':avg[110:125],'after_u':avg[125:140]}
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
    if st in ['prism_o','prism_u','after_o', 'after_u']: #exp3
    #if st in ['prism','after_u','after_o'] #exp2:
        x_norm = (np.array(x_model) - x_model[0]) / (x_model[-1] - x_model[0])  # normalized
        a_0 = np.mean(y_values[-5:-1]) # final value exponential function moves towards
        c_0 = 1 # the decay constant
        b_0 = (y_values[0] - a_0) # the magnitude of adaptation
        popt, pcov = curve_fit(exp_model, x_norm, y_values, p0=(a_0, b_0, c_0))
        plt.plot(x_model, exp_model(x_norm, *popt), color='black',linestyle="dashed")

        c_ori = (x_model[-1] - x_model[0]) * popt[2]
        model[st] = (popt[0], popt[1], c_ori)
        print(f"{popt[0]:.1f}{'-+'[popt[1]>0]}{abs(popt[1]):.1f}e(-t/{c_ori:.1f})",)
#

#
#
plt.axhline(y=0, color='gray')
plt.ylim(-100, 100)
plt.ylabel('HORIZONTAL DISPLACEMENT (cm)', fontsize=22)
xlabel = '$TIME \longrightarrow$\n'
models =['prism_o','prism_u','after_o', 'after_u'] #exp4
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
plt.text(95, -90, '$PRISMS_o$', fontsize=20)
plt.text(185, -90, '$PRISMS_u$', fontsize=20)
plt.text(260, -90, '$AFTER_o$', fontsize=20)
plt.text(300, -90, '$AFTER_u$', fontsize=20)
#plt.title(f'SUBJECT {subject + 2} {file}', fontsize=22)
plt.title('MEDIAN', fontsize=22)


#PC standard deviation of the 8 throws before googles
PC_1 = np.std(exp_data[subject][stage[0]])
PC_2 = np.std(exp_data[subject][stage[1]])
plt.text(-5, 90, f'$PC = {PC_1:.1f}$', fontsize=20)
plt.text(37, 90, f'$PC = {PC_2:.1f}$', fontsize=20)
plt.text(95, 90, f'$AC = {model[stage[2]][2]:.1f}$', fontsize=20)
plt.text(185, 90, f'$AC = {model[stage[3]][2]:.1f}$', fontsize=20)
plt.text(260, 90, f'$AC = {model[stage[4]][2]:.1f}$', fontsize=20)
plt.text(300, 90, f'$AC = {model[stage[5]][2]:.1f}$', fontsize=20)
plt.savefig('media/final_final_plots/median(3-4).png')
plt.show()
print(model)

print(f"PCu = {PC_1:.1f}")
print(f"PCo = {PC_2:.1f}")
print(f"ACpo = {model[stage[2]][2]:.1f}")
print(f"ACpu = {model[stage[3]][2]:.1f}")
print(f"ACo = {model[stage[4]][2]:.1f}")
print(f"ACu = {model[stage[5]][2]:.1f}")

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

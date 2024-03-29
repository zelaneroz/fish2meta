import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

plt.style.use('ggplot')


def exp_model(x, a, b, c):
    return a - (b * np.exp(-x / c))

#file = "Fish2Meta Data - Ex 4 - 3_4_23.csv"
file="Fish2Meta Data - Ex 3 - 1_27_23.csv"
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
subject = 3

#exp_data[subject]={'before_u': avg[0:15], 'before_o': avg[15:30], 'prism': avg[30:70], 'after_u':avg[70:85], 'after_o': avg[85:100]}
print(exp_data[subject])
avg=[-9,-6,-7,4,-3,3,-2,-9,3,-6,6,-12,-4,-4,2,1,-4,-10,-7,3,-12,-12,3,7,-1,0,-3,-16,5,-1,-76,-49,-41,-36,-22,-21,-23,-17,-12,-29,-16,-18,-17,-6,-16,-18,-21,-15,-28,-14,-23,-16,-20,-23,-17,-5,-11,-13,-8,-15,-19,-13,-14,-13,-6,-19,-3,-1,-20,-20,-53,-49,-53,-41,-18,-28,-32,-21,-28,-27,-18,-31,-20,-17,-21,-15,-15,-18,-11,-16,-15,-11,-23,-26,0,-16,-19,-15,-16,-23,-9,-22,-15,-13,-18,-2,-20,-5,-13,-6,55,46,28,20,30,22,13,15,6,9,0,-3,6,2,7,34,29,22,18,14,8,11,4,4,15,0,11,-2,-1,9]
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
#plt.title(f'SUBJECT {subject + 1} {file}', fontsize=22)
plt.title('MEDIAN (Experiment 3&4)', fontsize=22)


#PC standard deviation of the 8 throws before googles
PC_1 = np.std(exp_data[subject][stage[0]])
PC_2 = np.std(exp_data[subject][stage[1]])
plt.text(-5, 90, f'$PC = {PC_1:.1f}$', fontsize=20)
plt.text(37, 90, f'$PC = {PC_2:.1f}$', fontsize=20)
plt.text(95, 90, f'$AC = {model[stage[2]][2]:.1f}$', fontsize=20)
plt.text(185, 90, f'$AC = {model[stage[3]][2]:.1f}$', fontsize=20)
plt.text(260, 90, f'$AC = {model[stage[4]][2]:.1f}$', fontsize=20)
plt.text(300, 90, f'$AC = {model[stage[5]][2]:.1f}$', fontsize=20)
#plt.savefig(f'media/definite_plots/exp_3/{subject+1}.png')
plt.savefig(f'media/definite_plots/median(3-4).png')
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

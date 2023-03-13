import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

y = [19.1,25,29.4,22.7,11.5,20.1,33.6,10.1,-7.6,16.4,19.9,5.9,7.8,-4.4,8.1]
plt.style.use('ggplot')
def exp_model(x, a, b, c):
    return a - (b * np.exp(-x / c))
x_model=[]
x = 1  # start of the graph
model = {} # decay constants
for n in y:
    x_model.append(x)
    plt.plot(x, n, 'o', markeredgecolor='g', markersize=15, fillstyle='none')
    x += 1

a_0 = np.mean(y[:5]) # final value exponential function moves towards
print("a_o: ",a_0)
c_0 = 1 # the decay constant
b_0 = (y[0] - a_0) # the magnitude of adaptation
print("b_o: ",b_0)
x_norm = (np.array(x_model) - x_model[0]) / (x_model[-1] - x_model[0])
popt, pcov = curve_fit(exp_model, x_norm, y, p0=(a_0, b_0, c_0))
c_ori = (x_model[-1] - x_model[0]) * popt[2]
print("c_ori: ",c_ori)
plt.plot(x_model, exp_model(x_norm, *popt), color='black', linestyle="dashed")



#     print("X_model: ",x_model)
#     x += 5 #space between stages
#     plt.vlines(x, -100, 100, colors='gray')
#     x += 5  # space between stages
#     if st in ['prism_u','prism_o','after_u', 'after_o']: #exp3
#     #if st in ['prism','after_u','after_o'] #exp2:
#         x_norm = (np.array(x_model) - x_model[0]) / (x_model[-1] - x_model[0])  # normalized
#         a_0 = np.mean(y_values[-5:-1]) # final value exponential function moves towards
#         print("a_o: ",a_0)
#         c_0 = 1 # the decay constant
#         b_0 = (y_values[0] - a_0) # the magnitude of adaptation
#         print("b_o: ",b_0)
#         print("Exp model: ", x_model)
#         popt, pcov = curve_fit(exp_model, move_to_zero(120,x_norm), y_values, p0=(a_0, b_0, c_0))
#         plt.plot(x_model, exp_model(x_norm, *popt), color='black',linestyle="dashed")
#         print()
#         c_ori = (x_model[-1] - x_model[0]) * popt[2]
#         print("c_ori: ", c_ori, "\n")
#         model[st] = (popt[0], popt[1], c_ori)
#         print(f"{popt[0]:.1f}{'-+'[popt[1]>0]}{abs(popt[1]):.1f}e(-t/{c_ori:.1f})",)
#
#
# #
# #
# plt.axhline(y=0, color='gray')
# plt.ylim(-100, 100)
# plt.ylabel('HORIZONTAL DISPLACEMENT (cm)', fontsize=22)
# xlabel = '$TIME \longrightarrow$\n'
# models =['prism_u','prism_o','after_u', 'after_o'] #exp3
# #models =['prism','after_u', 'after_o'] #Exp 2
# for s in models:
#     m = model[s]
#     xlabel+=f"${m[0]:.1f}{'-+'[m[1]>0]}{abs(m[1]):.1f}e(-t/{m[2]:.1f})$  "
# #Experiment 3
# plt.xlabel(xlabel, fontsize=22)
# plt.yticks(fontsize=20)
# plt.xticks([])
# plt.text(-5, -90, '$BEFORE_u$', fontsize=20)
# plt.text(37, -90, '$BEFORE_o$', fontsize=20)
# plt.text(95, -90, '$PRISMS_u$', fontsize=20)
# plt.text(185, -90, '$PRISMS_o$', fontsize=20)
# plt.text(260, -90, '$AFTER_u$', fontsize=20)
# plt.text(300, -90, '$AFTER_o$', fontsize=20)
# plt.title(f'SUBJECT {subject + 1} {file}', fontsize=22)
# #plt.title('MEDIAN', fontsize=22)
#
# #PC standard deviation of the 8 throws before googles
# PC_1 = np.std(exp_data[subject][stage[0]])
# PC_2 = np.std(exp_data[subject][stage[1]])
# plt.text(-5, 90, f'$PC = {PC_1:.1f}$', fontsize=20)
# plt.text(37, 90, f'$PC = {PC_2:.1f}$', fontsize=20)
# plt.text(95, 90, f'$AC = {model[stage[2]][2]:.1f}$', fontsize=20)
# plt.text(185, 90, f'$AC = {model[stage[3]][2]:.1f}$', fontsize=20)
# plt.text(260, 90, f'$AC = {model[stage[4]][2]:.1f}$', fontsize=20)
# plt.text(300, 90, f'$AC = {model[stage[5]][2]:.1f}$', fontsize=20)
# plt.show()
# print(model)
#
# print(f"PCu = {PC_1:.1f}")
# print(f"PCo = {PC_2:.1f}")
# print(f"ACpu = {model[stage[2]][2]:.1f}")
# print(f"ACpo = {model[stage[3]][2]:.1f}")
# print(f"ACu = {model[stage[4]][2]:.1f}")
# print(f"ACo = {model[stage[5]][2]:.1f}")
plt.show()
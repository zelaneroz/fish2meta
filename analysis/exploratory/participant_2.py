import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import warnings
warnings.simplefilter('ignore')

plt.style.use('ggplot')
#---PARAMETERS---
tick_min = -100
tick_max = 80
tick = 15

#---DATA---
x = np.arange(1, 101)
participants = {"p1":[-8,-7,-2,-23,8,-7,-2,-35,21,-13,7,4,-17,3,-5,-2,4,1,-8,4,5,-12,4,-35,-17,-4,12,3,-6,2,-47,-16,-10,-31,-48,-33,-31,-15,-28,-44,-26,-34,-29,-38,-21,-8,-11,-18,-34,-10,-28,-26,-23,-46,-9,-2,-31,-26,-8,-17,-28,-25,-16,-21,-26,-32,-15,-14,-22,-25,19,22,-32,1,12,4,8,20,-13,13,-13,1,2,13,-3,42,35,5,-5,26,30,2,-7,9,19,21,24,25,18,-10]}

#-----PLOTS-----
fig, ax = plt.subplots(figsize=(18, 5))
fig = plt.figure(figsize = ([12,6]))
gs = gridspec.GridSpec(4,7)
gs.update(wspace=1.5,hspace=0.7)

for i in range(1,6):
   n=i-3
   y = participants[f'p{i}']
   j = ['blue', 'red', 'green', 'brown', 'orange', 'purple']
   ax[i]=plt.plot(x, y, 'o', c=j[i - 1])
   for l in [14, 29, 69, 84]:
       ax[i]=plt.axvline(x=l, linestyle='--')
   ax[i]=plt.subplot(gs[n,3:6])
   ax[i].set_yticks(np.arange(tick_min, tick_max, tick))
   ax[i].title.set_text(f'Participant {i}')
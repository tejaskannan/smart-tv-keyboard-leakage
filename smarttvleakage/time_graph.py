import json
import matplotlib.pyplot as plt

times_dict = {}
with open('times.json') as f:
	times_dict = json.load(f)

times_list = [0,0,0]
for i in times_dict:
	times_list[len(i)-3] += times_dict[i][0]

times_list = [i/10 for i in times_list]
print(times_list)
length = [3,4,5]
plt.plot(length,times_list)
plt.show()
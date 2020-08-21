import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("adiabaticity.csv") 

for pulse_length in [10, 40, 200, 450]:
    time_orig = list(df[df['pulse_length'] == pulse_length]['time'])
    time = [t/pulse_length for t in time_orig]
    adiabaticity = list(df[df['pulse_length'] == pulse_length]['adiabaticity'])
    plt.plot(time,adiabaticity, label = '{} ps'.format(pulse_length))

plt.xlabel('t/pulse_length')
plt.ylabel('adiabaticity')
plt.legend()
plt.show()
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import control as ct

def parallel(Z1,Z2):
    return (Z1*Z2/(Z1 + Z2))

# Define Laplace variable
s = sp.Symbol('s')

RC = .001           #Cap ESR
RL =.080            #Inductor ESR
RLOAD = 5           #Load Resistance
Cout = 47*(10**-6)  #Output Filter Capacitance
Lout = 20*(10**-6)  #Output Filter Inductor
Vin = 12            #Input Voltage
Vp = 2.5            #Pulse Width Modulator Peak Voltage
Vout = 5            #Desired Output Voltage

D =((RLOAD + RL)/(RLOAD))*(Vout/Vin)
H0_ctrl = (Vin/Vp)*(RLOAD / (RLOAD + RL))
H0_vin = D*(RLOAD)/(RLOAD + RL)
Z0 = parallel(RL, RLOAD)
w0 = ((Lout*Cout)**-(1/2)) * ((RL + RLOAD)**(1/2)) * ((RC + RLOAD)**(-1/2)) 
wzc = 1/(RC*Cout)
wzl = (RL/Lout)
QF = Lout*Cout*w0*(RC + RLOAD) / (Lout + Cout*(RL*RC + RL*RLOAD + RC*RLOAD))


print('D = ', D)
print('H0_ctrl = ', H0_ctrl)
print('H0_vin = ', H0_vin)
print('Z0 = ', Z0)
print('w0 = ', w0)
print('wzc = ', wzc)
print('wzl = ', wzl)
print('QF = ', QF)

# -----------------------------
# Step 1: Define Transfer Functions
# -----------------------------
C2O_OL = ct.TransferFunction([H0_ctrl/wzc,H0_ctrl],[1/(w0**2),1/(w0*QF),1])
print("\nOpen Loop Control to Output:\n",C2O_OL)

V2O_OL = ct.TransferFunction([H0_vin/wzc,H0_vin],[1/(w0**2),1/(w0*QF),1])
print("\nOpen Loop Vin to Output:\n",V2O_OL)

ZOUT_OL = ct.TransferFunction([Z0/wzc,Z0],[1]) * ct.TransferFunction([1/wzl,1],[1]) * ct.TransferFunction([1],[1/(w0**2),1/(w0*QF),1])
print("\nOpen Loop Zout:\n",ZOUT_OL)

CompGainDC = ct.TransferFunction([1],[1])       #example 10/s
CompGainPoleZeroPair = ct.TransferFunction([1,0],[100])   # (s/3000 + 1)
CompGain = CompGainDC*CompGainPoleZeroPair
print("\nCompensator Gain:]n",CompGain)

LoopGain = C2O_OL * CompGain
print("\nLoop Gain:\n",LoopGain)

C2O_CL = ct.feedback(C2O_OL,1,sign=-1)
print("\nClosed Loop Control:]n",C2O_CL)

V2O_CL = V2O_OL / (1 + LoopGain)
print("\nlosed Loop Vin:\n")

ZOUT_CL = ZOUT_OL / (1 + LoopGain)
print("\nClosed Loop Zout:\n", ZOUT_CL)


# -----------------------------
# Generate Bode Plots
# -----------------------------
fig, axs = plt.subplots(4, 4, figsize=(24, 12))
C2O_OL.bode_plot(dB=True, Hz=False, omega_limits=(10, 1000000),ax=(axs[0, 0], axs[1, 0]), label='Control OL', color='b')
axs[0, 0].set_title(f'Control OL')
V2O_OL.bode_plot(dB=True, Hz=False, omega_limits=(10, 1000000),ax=(axs[0, 1], axs[1, 1]), label='Vin OL', color='b')
axs[0, 1].set_title(f'Vin OL')
ZOUT_OL.bode_plot(dB=True, Hz=False, omega_limits=(10, 1000000),ax=(axs[0, 2], axs[1, 2]), label='Zout OL', color='b')
axs[0, 2].set_title(f'Zout OL')
CompGain.bode_plot(dB=True, Hz=False, omega_limits=(10, 1000000),ax=(axs[0, 3], axs[1, 3]), label='Compensator', color='b')
axs[0, 3].set_title(f'Compensator')
C2O_CL.bode_plot(dB=True, Hz=False, omega_limits=(10, 1000000),ax=(axs[2, 0], axs[3, 0]), label='Control CL', color='b')
axs[2, 0].set_title(f'Control CL')
V2O_CL.bode_plot(dB=True, Hz=False, omega_limits=(10, 1000000),ax=(axs[2, 1], axs[3, 1]), label='Vin CL', color='b')
axs[2, 1].set_title(f'Vin CL')
ZOUT_CL.bode_plot(dB=True, Hz=False, omega_limits=(10, 1000000),ax=(axs[2, 2], axs[3, 2]), label='Zout CL', color='b')
axs[2, 2].set_title(f'Zout CL')
LoopGain.bode_plot(dB=True, Hz=False, omega_limits=(10, 1000000),ax=(axs[2, 3], axs[3, 3]), label='Loop Gain', color='b')
axs[2, 3].set_title(f'Loop Gain')
plt.tight_layout()

#Root Locus?

# -----------------------------
# Step 4: Time-Domain Step Response
# -----------------------------

plt.figure(figsize=(16,12))

t, y = ct.impulse_response(C2O_OL)
plt.subplot(2,3,1)
plt.plot(t, y, 'b', linewidth=2)
plt.title('OL Ctrl Impulse Response')
plt.xlabel('Time (s)')
plt.ylabel('Voltage')
plt.grid(True)

#plt.show()

t, y = ct.step_response(V2O_OL*ct.TransferFunction([Vin],[1]))
plt.subplot(2,3,2)
plt.plot(t, y, 'b', linewidth=2)
plt.title('OL Vin Step Response')
plt.xlabel('Time (s)')
plt.ylabel('Voltage')
plt.grid(True)

t, y = ct.impulse_response(ZOUT_OL)
plt.subplot(2,3,3)
plt.plot(t, y, 'b', linewidth=2)
plt.title('OL Load Impulse Response')
plt.xlabel('Time (s)')
plt.ylabel('Voltage')
plt.grid(True)

t, y = ct.impulse_response(C2O_CL)
plt.subplot(2,3,4)
plt.plot(t, y, 'b', linewidth=2)
plt.title('CL Ctrl Impulse Response')
plt.xlabel('Time (s)')
plt.ylabel('Voltage')
plt.grid(True)

t, y = ct.step_response(V2O_CL*ct.TransferFunction([Vin],[1]))
plt.subplot(2,3,5)
plt.plot(t, y, 'b', linewidth=2)
plt.title('CL Vin Step Response')
plt.xlabel('Time (s)')
plt.ylabel('Voltage')
plt.grid(True)

t, y = ct.impulse_response(ZOUT_CL)
plt.subplot(2,3,6)
plt.plot(t, y, 'b', linewidth=2)
plt.title('CL Load Impulse Response')
plt.xlabel('Time (s)')
plt.ylabel('Voltage')
plt.grid(True)

plt.tight_layout()
plt.show()
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
# Step 1: Define and simplify transfer function with SymPy
# -----------------------------
C2O_OL = H0_ctrl*(1 + s/wzc) / (1 + s/(QF*w0) + (s/w0)**2)
C2O_OL_Numer = sp.numer(C2O_OL)
C2O_OL_Denom = sp.denom(C2O_OL)
print("\nOpen Loop Control to Output:")
sp.pprint(C2O_OL)

V2O_OL = H0_vin*(1 + s/wzc) / (1 + s/(QF*w0) + (s/w0)**2)
V2O_OL_Numer = sp.numer(V2O_OL)
V2O_OL_Denom = sp.denom(V2O_OL)
print("\nOpen Loop Vin to Output:")
sp.pprint(V2O_OL)

ZOUT_OL = Z0 * (1 + s/wzc)*(1 + s/wzl) / (1 + s/(QF*w0) +(s/w0)**2)
ZOUT_OL_Numer = sp.numer(ZOUT_OL)
ZOUT_OL_Denom = sp.denom(ZOUT_OL)
print("\nOpen Loop Zout:")
sp.pprint(ZOUT_OL)

CompGain = (1 + 0*s) / (1 + 0*s)
CompGain_Numer = sp.numer(CompGain)
CompGain_Denom = sp.denom(CompGain)
print("\nCompensator Gain: ")
sp.pprint(CompGain)


LoopGain = C2O_OL * CompGain
LoopGain_Numer = sp.numer(LoopGain)
LoopGain_Denom = sp.denom(LoopGain)
print("\nLoop Gain: ")
sp.pprint(LoopGain)

C2O_CL = LoopGain / (1 + LoopGain)
C2O_CL_Numer = sp.numer(C2O_CL)
C2O_CL_Denom = sp.denom(C2O_CL)
print("\nClosed Loop Control: ")
sp.pprint(C2O_CL)

V2O_CL = V2O_OL / (1 + LoopGain)
V2O_CL_Numer = sp.numer(V2O_CL)
V2O_CL_Denom = sp.denom(V2O_CL)
print("\nlosed Loop Vin: ")
sp.pprint(V2O_CL)

ZOUT_CL = ZOUT_OL / (1 + LoopGain)
ZOUT_CL_Numer = sp.numer(ZOUT_CL)
ZOUT_CL_Denom = sp.denom(ZOUT_CL)
print("\nClosed Loop Zout: ")
sp.pprint(ZOUT_CL)

# -----------------------------
# Step 2: Convert symbolic TF to numerical form for plotting
# -----------------------------
# Get coefficients and Create transfer function system (SciPy)

C20_OL_num_poly = sp.Poly(sp.simplify(sp.factor(C2O_OL_Numer)), s)
C20_OL_den_poly = sp.Poly(sp.simplify(sp.factor(C2O_OL_Denom)), s)
C20_OL_num_coeffs = [float(c) for c in C20_OL_num_poly.all_coeffs()]
C20_OL_den_coeffs = [float(c) for c in C20_OL_den_poly.all_coeffs()]
ControlToOutput= signal.TransferFunction(C20_OL_num_coeffs, C20_OL_den_coeffs)

C20_CL_num_poly = sp.Poly(sp.simplify(sp.factor(C2O_CL_Numer)), s)
C20_CL_den_poly = sp.Poly(sp.simplify(sp.factor(C2O_CL_Denom)), s)
C20_CL_num_coeffs = [float(c) for c in C20_CL_num_poly.all_coeffs()]
C20_CL_den_coeffs = [float(c) for c in C20_CL_den_poly.all_coeffs()]
ControlToOutputCL= signal.TransferFunction(C20_CL_num_coeffs, C20_CL_den_coeffs)


V20_OL_num_poly = sp.Poly(sp.simplify(sp.factor(V2O_OL_Numer)), s)
V20_OL_den_poly = sp.Poly(sp.simplify(sp.factor(V2O_OL_Denom)), s)
V20_OL_num_coeffs = [float(c) for c in V20_OL_num_poly.all_coeffs()]
V20_OL_den_coeffs = [float(c) for c in V20_OL_den_poly.all_coeffs()]
InputToOutput = signal.TransferFunction(V20_OL_num_coeffs, V20_OL_den_coeffs)

#InputToOutputCL = ct.feedback(InputToOutput,1)
V20_CL_num_poly = sp.Poly(sp.simplify(sp.factor(V2O_CL_Numer)), s)
V20_CL_den_poly = sp.Poly(sp.simplify(sp.factor(V2O_CL_Denom)), s)
V20_CL_num_coeffs = [float(c) for c in V20_CL_num_poly.all_coeffs()]
V20_CL_den_coeffs = [float(c) for c in V20_CL_den_poly.all_coeffs()]
InputToOutputCL = signal.TransferFunction(V20_CL_num_coeffs, V20_CL_den_coeffs)


ZOUT_OL_num_poly = sp.Poly(sp.simplify(sp.factor(ZOUT_OL_Numer)), s)
ZOUT_OL_den_poly = sp.Poly(sp.simplify(sp.factor(ZOUT_OL_Denom)), s)
ZOUT_OL_num_coeffs = [float(c) for c in ZOUT_OL_num_poly.all_coeffs()]
ZOUT_OL_den_coeffs = [float(c) for c in ZOUT_OL_den_poly.all_coeffs()]
OutputImpedance = signal.TransferFunction(ZOUT_OL_num_coeffs, ZOUT_OL_den_coeffs)

ZOUT_CL_num_poly = sp.Poly(sp.simplify(sp.factor(ZOUT_CL_Numer)), s)
ZOUT_CL_den_poly = sp.Poly(sp.simplify(sp.factor(ZOUT_CL_Denom)), s)
ZOUT_CL_num_coeffs = [float(c) for c in ZOUT_CL_num_poly.all_coeffs()]
ZOUT_CL_den_coeffs = [float(c) for c in ZOUT_CL_den_poly.all_coeffs()]
OutputImpedanceCL = signal.TransferFunction(ZOUT_CL_num_coeffs, ZOUT_CL_den_coeffs)

CompGain_num_poly = sp.Poly(sp.simplify(sp.factor(CompGain_Numer)), s)
CompGain_den_poly = sp.Poly(sp.simplify(sp.factor(CompGain_Denom)), s)
CompGain_num_coeffs = [float(c) for c in CompGain_num_poly.all_coeffs()]
CompGain_den_coeffs = [float(c) for c in CompGain_den_poly.all_coeffs()]
CompGainOL = signal.TransferFunction(CompGain_num_coeffs, CompGain_den_coeffs)

LoopGain_num_poly = sp.Poly(sp.simplify(sp.factor(LoopGain_Numer)), s)
LoopGain_den_poly = sp.Poly(sp.simplify(sp.factor(LoopGain_Denom)), s)
LoopGain_num_coeffs = [float(c) for c in LoopGain_num_poly.all_coeffs()]
LoopGain_den_coeffs = [float(c) for c in LoopGain_den_poly.all_coeffs()]
LoopGainOL = signal.TransferFunction(LoopGain_num_coeffs, LoopGain_den_coeffs)

# -----------------------------
# Step 3: Generate Bode Plot
# -----------------------------
w_ctrl, mag_ctrl, phase_ctrl = signal.bode(ControlToOutput)
w_vin, mag_vin, phase_vin = signal.bode(InputToOutput)
w_zout, mag_zout, phase_zout = signal.bode(OutputImpedance)
w_comp, mag_comp, phase_comp = signal.bode(CompGainOL)
w_loop, mag_loop, phase_loop = signal.bode(LoopGainOL)
w_ctrlCL, mag_ctrlCL, phase_ctrlCL = signal.bode(ControlToOutputCL)
w_vinCL, mag_vinCL, phase_vinCL = signal.bode(InputToOutputCL)
w_zoutCL, mag_zoutCL, phase_zoutCL = signal.bode(OutputImpedanceCL)

plt.figure(figsize=(20,10))

plt.subplot(4,4,1)
plt.semilogx(w_ctrl, mag_ctrl)
plt.title('Control to Output')
plt.ylabel('Magnitude (dB)')
plt.grid(which='both', axis='both')

plt.subplot(4,4,5)
plt.semilogx(w_ctrl, phase_ctrl)
plt.ylabel('Phase (deg)')
plt.xlabel('Frequency (rad/s)')
plt.grid(which='both', axis='both')

plt.subplot(4,4,9)
plt.semilogx(w_ctrlCL, mag_ctrlCL)
plt.title('Control to Output CL')
plt.ylabel('Magnitude (dB)')
plt.grid(which='both', axis='both')

plt.subplot(4,4,13)
plt.semilogx(w_ctrlCL, phase_ctrlCL)
plt.ylabel('Phase (deg)')
plt.xlabel('Frequency (rad/s)')
plt.grid(which='both', axis='both')

plt.subplot(4,4,2)
plt.semilogx(w_vin, mag_vin)
plt.title('Input to Output')
plt.ylabel('Magnitude (dB)')
plt.grid(which='both', axis='both')

plt.subplot(4,4,6)
plt.semilogx(w_vin, phase_vin)
plt.ylabel('Phase (deg)')
plt.xlabel('Frequency (rad/s)')
plt.grid(which='both', axis='both')

plt.subplot(4,4,10)
plt.semilogx(w_vinCL, mag_vinCL)
plt.title('Input to Output CL')
plt.ylabel('Magnitude (dB)')
plt.grid(which='both', axis='both')

plt.subplot(4,4,14)
plt.semilogx(w_vinCL, phase_vinCL)
plt.ylabel('Phase (deg)')
plt.xlabel('Frequency (rad/s)')
plt.grid(which='both', axis='both')

plt.subplot(4,4,3)
plt.semilogx(w_zout, mag_zout)
plt.title('Output Impedance')
plt.ylabel('Magnitude (dB)')
plt.grid(which='both', axis='both')

plt.subplot(4,4,7)
plt.semilogx(w_zout, phase_zout)
plt.ylabel('Phase (deg)')
plt.xlabel('Frequency (rad/s)')
plt.grid(which='both', axis='both')

plt.subplot(4,4,11)
plt.semilogx(w_zoutCL, mag_zoutCL)
plt.title('Output Impedance CL')
plt.ylabel('Magnitude (dB)')
plt.grid(which='both', axis='both')

plt.subplot(4,4,15)
plt.semilogx(w_zoutCL, phase_zoutCL)
plt.ylabel('Phase (deg)')
plt.xlabel('Frequency (rad/s)')
plt.grid(which='both', axis='both')

plt.subplot(4,4,4)
plt.semilogx(w_comp, mag_comp)
plt.title('Compensator Gain')
plt.ylabel('Magnitude (dB)')
plt.grid(which='both', axis='both')

plt.subplot(4,4,8)
plt.semilogx(w_comp, phase_comp)
plt.ylabel('Phase (deg)')
plt.xlabel('Frequency (rad/s)')
plt.grid(which='both', axis='both')

plt.subplot(4,4,12)
plt.semilogx(w_loop, mag_loop)
plt.title('Loop Gain')
plt.ylabel('Magnitude (dB)')
plt.grid(which='both', axis='both')

plt.subplot(4,4,16)
plt.semilogx(w_loop, phase_loop)
plt.ylabel('Phase (deg)')
plt.xlabel('Frequency (rad/s)')
plt.grid(which='both', axis='both')

plt.tight_layout()
plt.show()

#Root Locus?

# -----------------------------
# Step 4: Time-Domain Step Response
# -----------------------------

plt.figure(figsize=(16,12))

t, y = signal.impulse(ControlToOutput)
plt.subplot(2,3,1)
plt.plot(t, y, 'b', linewidth=2)
plt.title('OL Ctrl Impulse Response')
plt.xlabel('Time (s)')
plt.ylabel('Voltage')
plt.grid(True)

t, y = signal.step(InputToOutput)
y = y*Vin
plt.subplot(2,3,2)
plt.plot(t, y, 'b', linewidth=2)
plt.title('OL Vin Step Response')
plt.xlabel('Time (s)')
plt.ylabel('Voltage')
plt.grid(True)

t, y = signal.step(OutputImpedance)
y = y*(Vout/RLOAD)
plt.subplot(2,3,3)
plt.plot(t, y, 'b', linewidth=2)
plt.title('OL Load Step Response')
plt.xlabel('Time (s)')
plt.ylabel('Voltage')
plt.grid(True)

t, y = signal.impulse(ControlToOutputCL)
plt.subplot(2,3,4)
plt.plot(t, y, 'b', linewidth=2)
plt.title('CL Ctrl Impulse Response')
plt.xlabel('Time (s)')
plt.ylabel('Voltage')
plt.grid(True)

t, y = signal.step(InputToOutputCL)
y = y*Vin
plt.subplot(2,3,5)
plt.plot(t, y, 'b', linewidth=2)
plt.title('CL Vin Step Response')
plt.xlabel('Time (s)')
plt.ylabel('Voltage')
plt.grid(True)

t, y = signal.step(OutputImpedanceCL)
y = y*(Vout/RLOAD)
plt.subplot(2,3,6)
plt.plot(t, y, 'b', linewidth=2)
plt.title('CL Load Step Response')
plt.xlabel('Time (s)')
plt.ylabel('Voltage')
plt.grid(True)

plt.tight_layout()
plt.show()
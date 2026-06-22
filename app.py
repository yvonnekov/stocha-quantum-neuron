import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt 
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator 

def execute_quantum_neuron(inputs, weights, bias):
    #sammeln von signalen
    theta = sum(x* w for x, w in zip(inputs, weights))+bias
    #"abflachen" auf werte zwischen -1 und 1 
    # bildet das neuronale spektrum ab: von gehemmt (-1) über neutral (0) bis erregt (1)
    a = np.tanh(theta)

    #zustände in wahrscheinlichkeiten für die quantenmessung übersetzen
    target_angle = (a+1.0)*(np.pi/2) 
    qc = QuantumCircuit(1,1)
    qc.ry(target_angle, 0) 
    #ket(phi) = alpha ket(0)+ beta ket(1) mit alpha und beta als wharscheinlichkeitsamplituden
    
    qc.measure(0,0) 
    
    simulator = AerSimulator()
    result = simulator.run(qc, shots=1).result() #neuronen in echtzeit
    counts = result.get_counts()

    if '1' in counts:
        return 1
    return 0


st.set_page_config(page_title = "Stochastisches Quanten-LIF_Neuron", layout="centered")
st.title("Quanten-LIF Neuron")
st.write("Auf Grundlage des biologischen LIF-Modells wird hier ein Neuron und dessen Aktivität simuliert. Um das natürliche, stochastische Rauschen des menschlichen Neurons realitätsnah darzustellen, nutzen wir Quantenzufall. Die klassischen Input-Signale und Gewichte drehen dafür ein Qbit in eine Quantensuperposition, deren stochastischer Kollaps entscheidet, ob das Neuron feuert oder nicht. Das Gewicht verstärkt oder schwächt das eingehende Signal. Der Bias bestimmt wie stark aktuelle Inputs gewogen werden. Der Leaky faktor bestimmt wie viel Spannung mit der Zeit abfällt und die schwelle, wann das Neuron feuert.")
st.sidebar.header("Simulations-Parameter")
#bioparameter
v_thresh= st.sidebar.slider("Schwellenwert (V_thresh)", 0.5,2.0,1.0, step=0.1)
leak_factor = st.sidebar.slider("Leaky-Faktor (Wiederstand)", 0.01, 0.2, 0.05, step=0.01)
#quanten parameter
input_signal = st.sidebar.selectbox("Input-Signal (x)", [0, 1], index=1)
weight = st.sidebar.slider("Gewicht (w)", -2.0, 2.0, 1.3, step=0.1)
bias = st.sidebar.slider("Bias (b)", -1.0, 1.0,-0.2, step=0.1)

time_steps= 101


#Neuron simulation
v_membrane = 0.0
v_history = []
spike_history = []

for t in range(time_steps):
    v_membrane -= leak_factor * v_membrane
    #Quanten-neuron entscheidet stochastisch über stromeingang
    
    
    quantum_activation = execute_quantum_neuron([input_signal], [weight], bias)
    
    if quantum_activation == 1:
        v_membrane += 0.4 #Stromstoß
    
    if v_membrane >= v_thresh:
        v_history.append(v_membrane)
        spike_history.append(t)
        v_membrane = 0.0
    else:
        v_history.append(v_membrane)





st.subheader("Dynamischer Spannungsverlauf über die Zeit")
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(v_history, label="Membranspannung ($V_m$)", color="#50C878", linewidth = 2)
ax.axhline (y=v_thresh, color="red", linestyle="--", label="Schwellenwert ($V_{thresh}$)")

for spike_time in spike_history:
    ax.axvline(x=spike_time, color="red", linestyle="--", alpha=.05)

ax.set_xlabel("Zeit ($t$)")
ax.set_ylabel("Spannung ($V$)")
ax.set_title("Quanten-gesteuerte LIF-Neurosimulation")
ax.legend(loc="upper right")
ax.set_ylim(-0.1, v_thresh +0.5)

st.pyplot(fig)

st.subheader("Analyse metrik")
col1, col2 = st.columns(2)
with col1:
    st.metric("Anzahl registrierter Spikes", f"{len(spike_history)} mal gefeuret")
with col2:
    st.metric("Durschnittliche spannung", f"{np.mean(v_history):.2f} V")




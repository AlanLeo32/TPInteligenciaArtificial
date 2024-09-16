import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Definición de las variables difusas
temperatura_exterior = ctrl.Antecedent(np.arange(0, 51, 1), 'temperatura_exterior')
temperatura_interior = ctrl.Antecedent(np.arange(50, 101, 1), 'temperatura_interior')
llama = ctrl.Consequent(np.arange(0, 101, 1), 'llama')

# Funciones de pertenencia para la temperatura exterior
temperatura_exterior['baja'] = fuzz.trimf(temperatura_exterior.universe, [0, 0, 30])
temperatura_exterior['media'] = fuzz.trimf(temperatura_exterior.universe, [10, 30, 50])
temperatura_exterior['alta'] = fuzz.trimf(temperatura_exterior.universe, [30, 50, 50])

# Funciones de pertenencia para la temperatura interior
temperatura_interior['normal'] = fuzz.trimf(temperatura_interior.universe, [50, 50, 75])
temperatura_interior['alta'] = fuzz.trimf(temperatura_interior.universe, [70, 80, 90])
temperatura_interior['critica'] = fuzz.trimf(temperatura_interior.universe, [85, 100, 100])

# Funciones de pertenencia para el tamaño de la llama (combustión)
llama['piloto'] = fuzz.trimf(llama.universe, [0, 0, 40])
llama['moderada'] = fuzz.trimf(llama.universe, [20, 50, 80])
llama['alta'] = fuzz.trimf(llama.universe, [60, 100, 100])

# Reglas difusas
rule1 = ctrl.Rule(temperatura_exterior['baja'], llama['alta'])
rule2 = ctrl.Rule(temperatura_exterior['media'], llama['moderada'])
rule3 = ctrl.Rule(temperatura_exterior['alta'], llama['piloto'])
rule4 = ctrl.Rule(temperatura_interior['alta'], llama['moderada'])
rule5 = ctrl.Rule(temperatura_interior['critica'], llama['piloto'])

# Sistema de control difuso
control_llama = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
simulador_llama = ctrl.ControlSystemSimulation(control_llama)

# Ejemplo de simulación
simulador_llama.input['temperatura_exterior'] = 30
simulador_llama.input['temperatura_interior'] = 120

# Calcular el tamaño de la llama
simulador_llama.compute()
print(f"Tamaño de la llama: {simulador_llama.output['llama']:.2f}")

# Visualizar el resultado
temperatura_exterior.view(sim=simulador_llama)
temperatura_interior.view(sim=simulador_llama)
llama.view(sim=simulador_llama)
plt.show()
import cv2
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl

# Fuzzy variables
brightness = ctrl.Antecedent(np.arange(0, 256, 1), 'brightness')
edge = ctrl.Antecedent(np.arange(0, 256, 1), 'edge')
output = ctrl.Consequent(np.arange(0, 101, 1), 'output')

# Membership functions
brightness.automf(3)   # poor, average, good
edge.automf(3)

output['low'] = fuzz.trimf(output.universe, [0, 25, 50])
output['med'] = fuzz.trimf(output.universe, [25, 50, 75])
output['high'] = fuzz.trimf(output.universe, [50, 75, 100])

# Rules
rules = [
    ctrl.Rule(brightness['poor'] | edge['poor'], output['low']),
    ctrl.Rule(brightness['average'] | edge['average'], output['med']),
    ctrl.Rule(brightness['good'] | edge['good'], output['high'])
]

system = ctrl.ControlSystem(rules)
sim = ctrl.ControlSystemSimulation(system)

# Real-time classification
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    b_val = np.mean(gray)
    e_val = np.mean(cv2.Canny(gray, 100, 200))

    sim.input['brightness'] = b_val
    sim.input['edge'] = e_val
    sim.compute()

    result = sim.output['output']

    cv2.putText(frame, f'{result:.2f}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Fuzzy Classification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
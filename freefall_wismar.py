# --------------------------------------------
# Eulerverfahren für den freien Fall
# Variante 1: ohne Luftwiderstand
# Variante 2: mit Luftwiderstand (proportional zur Geschwindigkeit)
# --------------------------------------------

import numpy as np
import matplotlib.pyplot as plt



# Speicher für Auswertung
times = []
vel_euler = []
pos_euler = []
vel_rk = []
pos_rk = []
vel_analy = []
pos_analy = []

#Fehlerberechnung
d_err_rk_vel = []
d_err_euler_vel = []
de_err_rk_pos = []
de_err_euler_pos = []

class Parameters:
      def __init__(self):
         self.g = 9.81           # Erdbeschleunigung (m/s^2)
         self.cw = 1.1           # Luftwiderstandskoeffizient (1/s) – setze auf 0 für idealen freien Fall
         self.A = 1.0            # Querschnittsfläche (m^2)
         self.rho = 1.225        # Luftdichte (kg/m^3)
         self.m = 80.0           # Masse des fallenden Objekts (kg)
         self.dt = 1.5           # Zeitschritt (s)
         self.t_max = 15.0       # Gesamtdauer der Simulation (s)
         self.v0 = 0.0           # Anfangsgeschwindigkeit (m/s)
         self.x0 = 1000.0        # Anfangshöhe (m)

# Anfangsbedingungen
p = Parameters()
v0_eu = p.v0           # Anfangsgeschwindigkeit (m/s)
x0_eu = p.x0           # Anfangshöhe (m)
v0_rk = p.v0         # Anfangsgeschwindigkeit für Runge-Kutta (m/s)
x0_rk = p.x0         # Anfangshöhe für Runge
v0_ana = p.v0         # Anfangsgeschwindigkeit analytisch (m/s)
x0_ana = p.x0         # Anfangshöhe analytisch (m)
t = 0.0
a = 0.0

def analytical_solution_vel(t, params=Parameters()):
   """
   Analytische Lösung der Geschwindigkeit im freien Fall mit Luftwiderstand
   Parameter: 
      t: Zeitpunkt (t)
      params: Parameter-Objekt mit physikalischen Konstanten
   Returns:
      Geschwindigkeit (v) zum Zeitpunkt t
   """ 
   #if t < params.dt:
   #   return p.v0
   c2 = params.g
   c1 = 0.5 * params.rho * params.A * params.cw / params.m

   v = - (np.sqrt(c2) * np.tanh(c1 * np.sqrt(c1 * c2) +np.sqrt(c1 * c2) *t ))/np.sqrt(c1) + p.v0
   return v

def analytical_solution_pos(t, params=Parameters()):
   """
   Analytische Lösung der Position im freien Fall mit Luftwiderstand
   Parameter: 
      t: Zeitpunkt (t)
      params: Parameter-Objekt mit physikalischen Konstanten
   Returns:
      Position (x) zum Zeitpunkt t
   """   
   #if t < params.dt:
   #   return p.x0

   c2 = params.g
   c1 = 0.5 * params.rho * params.A * params.cw / params.m

   x = (c2) - np.log(np.cosh(np.sqrt(c1 * c2) *(c1 +  t)))/c1 + p.x0
   return x

def euler_step(v_0, x_0, params=Parameters()):
   """
   Einzelschritt des Euler-Verfahrens für den freien Fall mit Luftwiderstand
   Parameter:
       v_0 : Aktuelle Geschwindigkeit (m/s)
       x_0 : Aktuelle Position (m)
       params: Parameter-Objekt mit physikalischen Konstanten
   Returns:
       v_k1: Neue Geschwindigkeit (m/s)
       x_k1: Neue Position (m)
   """
   dt = params.dt
   # Berechnung der neuen Werte mit dem Euler-Verfahren
   ar = 0.5*params.rho*params.A*params.cw*v_0*v_0/params.m 
   a =  ar - params.g               # Beschleunigung mit Luftwiderstand
   v_k1 = v_0 + dt * a              # Update Geschwindigkeit
   x_k1 = x_0 + dt * v_k1           # Update Position
   return v_k1, x_k1

def rk4_step(f, x0, y0, params=Parameters()):
   """
   4th‑order Runge–Kutta solver for dx/dt = f(t, x0, y0)
   
   Parameter:
       f     : Funktion f(t, x) returning derivative
       x0    : Initialwert 
       y0    : Initialwert der Ableitung
       
   Returns:
       x: Lösung für den nächsten Zeitschritt
   """
   x = x0
   h = params.dt

       
   k1 = f(x, y0, params)
   k2 = f(x + 0.5*h*k1, y0, params)
   k3 = f(x + 0.5*h*k2, y0, params)
   k4 = f(x + h*k3, y0, params)

   kg = (k1 + 2*k2 + 2*k3 + k4)/6.0

   x = x + h * kg

   return x

def freefall_firstderiv(x,y, params=Parameters()):
   """
   Erste Ableitung der Geschwindigkeit im freien Fall mit Luftwiderstand
   Parameter:
      x: Geschwindigkeit (m/s)
      y: Nicht benutzt
      params: Parameter-Objekt mit physikalischen Konstanten
   Returns:
      Beschleunigung (a) zum Zeitpunkt t
   """
   v = x
   ar = 0.5*params.rho*params.A*params.cw*v*v/params.m 
   a =  ar - params.g  # Beschleunigung mit Luftwiderstand
   return a

def freefall_secondderiv(x,y, params = Parameters()):
   """
   Zweite Ableitung der Position im freien Fall mit Luftwiderstand
   Parameter:
      x: Position (m)
      y: Geschwindigkeit (m/s)
      params: Parameter-Objekt mit physikalischen Konstanten
   Returns:
      Geschwindigkeit (v) zum Zeitpunkt t
   """
   v =  y  #Geschwindigkeit
   return v

###############################
# Main Program
################################

# Parameter
t_max = p.t_max     # Gesamtdauer der Simulation (s)

###################################
# Simulationsschleife

while t <= t_max and x0_eu >= 0:
   # Speichern
   times.append(t)
   vel_euler.append(v0_eu)
   pos_euler.append(x0_eu)

   vel_rk.append(v0_rk)  
   pos_rk.append(x0_rk)      

   # Fehlerberechnung
   d_err_euler_vel.append(abs(v0_eu - v0_rk))
   de_err_euler_pos.append(abs(x0_eu - x0_rk))

   
   # Euler-Schritt: Berechnung der neuen Werte
   v0_eu, x0_eu = euler_step(v0_eu, x0_eu, p)
   # Runge-Kutta-Schritt: Berechnung der neuen Werte
   v0_rk = rk4_step(freefall_firstderiv, v0_rk,a, p)
   x0_rk = rk4_step(freefall_secondderiv, x0_rk,v0_rk,p)
   
   t += p.dt

   


# Ausgabe
print("Simulation abgeschlossen.")
print(f"Letzte Position Euler: {x0_eu:.2f} m")
print(f"Letzte Geschwindigkeit Euler: {v0_eu:.2f} m/s")
print(f"Letzte Position Runge-Kutta: {x0_rk:.2f} m")
print(f"Letzte Geschwindigkeit Runge-Kutta: {v0_rk:.2f} m/s")
print(f"Letzte Position Analytisch: {x0_ana:.2f} m")
print(f"Letzte Geschwindigkeit Analytisch: {v0_ana:.2f} m/s")
print(f"Simulationsschritte: {len(times)}")

# Optional: kurze Textausgabe der ersten Werte
#plot in matplotlib
fig, axes = plt.subplots(4, 1,)  # 2 rows, 1 column
axes[0].plot(times, vel_rk, '-o', label='Geschwindigkeit der Runge-Kutta-Lösung',color='orange')
axes[0].plot(times, vel_euler, 'x', label='Geschwindigkeit der Eulerlösung',color='blue')
axes[0].set_title('Freier Fall mit Luftwiderstand - Geschwindigkeit')
axes[0].set_ylabel('Geschwindigkeit (m/s)')
axes[0].set_xlabel('Zeit (s)')
axes[0].set_xlim(0, t_max)
axes[0].grid()
axes[0].legend()
axes[1].plot(times, d_err_euler_vel, '-x', label='Fehler Euler',color='blue')
axes[1].set_title('Fehlervergleich der Methoden - Geschwindigkeit')
axes[1].set_xlabel('Zeit (s)')
axes[1].set_ylabel('Absoluter Fehler (m/s)')
axes[1].set_xlim(0, t_max)
axes[1].grid()
axes[2].plot(times, pos_rk, '-o', label='Position der Runge-Kutta-Lösung',color='orange')
axes[2].plot(times, pos_euler, 'x', label='Position der Eulerlösung',color='blue')
axes[2].legend()
axes[2].set_title('Freier Fall mit Luftwiderstand - Position')
axes[2].set_ylabel('Position (m)')
axes[2].set_xlabel('Zeit (s)')
axes[2].set_xlim(0, t_max)
axes[2].grid()
axes[3].plot(times, de_err_euler_pos, '-x', label='Fehler Euler', color='blue')
axes[3].set_title('Fehlervergleich der Methoden - Position')
axes[3].set_xlabel('Zeit (s)')
axes[3].set_ylabel('Absoluter Fehler (m)')
axes[3].set_xlim(0, t_max)
axes[3].grid()
plt.tight_layout()

# Darstellung des Einflusses des Zeitschritts auf die Genauigkeit der Simulation
fig1,ax = plt.subplots(2,1)

for dt_test in [0.1,0.5, 1.0, 1.5]:
   # Anfangsbedingungen
   p.dt = dt_test
   v0_eu = p.v0           # Anfangsgeschwindigkeit (m/s)
   x0_eu = p.x0           # Anfangshöhe (m)
   v0_rk = p.v0         # Anfangsgeschwindigkeit für Runge-Kutta (m/s)
   x0_rk = p.x0         # Anfangshöhe für Runge
   t = 0.0
   a = 0.0
   ###################################
   # Simulationsschleife

   times = []
   d_err_euler_vel = []
   de_err_euler_pos = []
   while t <= t_max and x0_eu >= 0:
      # Speichern
      times.append(t)
      vel_euler.append(v0_eu)
      pos_euler.append(x0_eu)

      # Fehlerberechnung
      d_err_euler_vel.append(abs(v0_eu - v0_rk))
      de_err_euler_pos.append(abs(x0_eu - x0_rk))

      # Euler-Schritt: Berechnung der neuen Werte
      v0_eu, x0_eu = euler_step(v0_eu, x0_eu, p)

      #calc analytical solution one step ahead
      # Runge-Kutta-Schritt: Berechnung der neuen Werte
      v0_rk = rk4_step(freefall_firstderiv, v0_rk,a, p)
      x0_rk = rk4_step(freefall_secondderiv, x0_rk,v0_rk,p)

      t += p.dt

   ax[0].plot(times, d_err_euler_vel, '-o', label=f'dt={dt_test}s')
   ax[0].set_title('Einfluss des Zeitschritts auf den Fehler der Geschwindigkeit (Euler)')
   ax[0].set_xlabel('Zeit (s)')
   ax[0].set_ylabel('Absoluter Fehler (m/s)')
   ax[0].grid()
   ax[0].legend()
   ax[1].plot(times, de_err_euler_pos, '-o', label=f'dt={dt_test}s')
   ax[1].set_title('Einfluss des Zeitschritts auf den Fehler der Position (Euler)')
   ax[1].set_xlabel('Zeit (s)')
   ax[1].set_ylabel('Absoluter Fehler (m)')
   ax[1].grid()
   ax[1].legend()
plt.tight_layout()
plt.show()      

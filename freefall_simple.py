# -------------------------------------------------------------------------
# Die nachfolgende Datei implementiert
# das Eulerverfahren und das Runge-Kutta-Verfahren
# zur Simulation des freien Falls mit Luftwiderstand.
# Datei: freefall_simple.py
# -------------------------------------------------------------------------


import matplotlib.pyplot as plt

# Speicher für die Auswertung
times = []
vel_euler = []
pos_euler = []
vel_rk = []
pos_rk = []

#Fehlerberechnung
vel_euler_err = []
pos_euler_err = []

class Parameters:
      def __init__(self):
         self.g = 9.81           # Erdbeschleunigung (m/s^2)
         self.cw = 1.1           # Luftwiderstandskoeffizient – setze auf 0 für idealen freien Fall
         self.A = .35            # Querschnittsfläche (m^2)
         self.rho = 1.20        # Luftdichte (kg/m^3)
         self.m = 0.40           # Masse des fallenden Objekts (kg)
         self.dt = 0.1           # Zeitschritt (s)
         self.t_max = 25.0        # Gesamtdauer der Simulation (s)
         self.v0 = 0.0           # Anfangsgeschwindigkeit (m/s)
         self.x0 = 100.0         # Anfangshöhe (m)

# Anfangsbedingungen
p = Parameters()
v0_eu = p.v0         # Anfangsgeschwindigkeit (m/s)
x0_eu = p.x0         # Anfangshöhe (m)
v0_rk = p.v0         # Anfangsgeschwindigkeit für Runge-Kutta (m/s)
x0_rk = p.x0         # Anfangshöhe für Runge
t = 0.0


def euler_step(f, x_k, dx_k, params=Parameters()):
   """
   Einzelschritt des Euler-Verfahrens für dx/dt = f(t, x_k)

   Parameter:
       f     : Funktionsreferenz f(x, dx, params) returning derivative
       x_k   : Ausgangswert
       dx_k  : Ableitung am Ausgangswert

   Returns:
       x_k1: Lösung für den nächsten Zeitschritt
   """
   dt = params.dt
   x_k1 = x_k + dt * f(x_k,dx_k, params)
   return x_k1

def rk4_step(f, x_k, dx_k, params=Parameters()):
   """
   4th‑order Runge–Kutta solver for dx/dt = f(t, x0, y0)
   
   Parameter:
       f     : Funktion f(t, x) returning derivative
       x_k   : Ausgangswert
       dx_k  : Initialwert der Ableitung
       
   Returns:
       x: Lösung für den nächsten Zeitschritt
   """
   x = x_k
   dt = params.dt

       
   k1 = f(x, dx_k, params)
   k2 = f(x + 0.5 * dt * k1, dx_k + 0.5 * (k1 - dx_k), params)
   k3 = f(x + 0.5 * dt * k2, dx_k + 0.5 * (k2 - k1), params)
   k4 = f(x + dt * k3, dx_k +  (k3 - dx_k) , params)

   kg = (k1 + 2*k2 + 2*k3 + k4)/6.0

   x = x + dt * kg

   return x

def freefall_firder(x, dx, params):
   """
   Erste Ableitung der Geschwindigkeit im freien Fall mit Luftwiderstand
   Parameter:
      x: Geschwindigkeit (m/s)
      dx: Nicht benutzt
      params: Parameter-Objekt mit physikalischen Konstanten
   Returns:
      Beschleunigung (a) zum Zeitpunkt t
   """
   v = x
   g = params.g
   rho = params.rho
   Af = params.A
   cw = params.cw
   m = params.m
   dv_k = rho * cw* Af * v * v /(2* m) - g
   return dv_k

def freefall_secder(x, dx, params):
   """
   Zweite Ableitung der Position im freien Fall mit Luftwiderstand
   Parameter:
      x: Position (m)
      dx: Geschwindigkeit (m/s)
      params: Parameter-Objekt mit physikalischen Konstanten
   Returns:
      Geschwindigkeit (v) zum Zeitpunkt t
   """
   v =  dx  #Geschwindigkeit
   return v

###############################
# Main Program
################################

# Parameter
t_max = p.t_max     # Gesamtdauer der Simulation (s)
tmp = 0.0           # temporäre Variable für Zwischenspeicherung
###################################
# Simulationsschleife

while t <= t_max and x0_rk >= 0:
   # Speichern
   times.append(t)
   vel_euler.append(v0_eu)
   pos_euler.append(x0_eu)

   vel_rk.append(v0_rk)
   pos_rk.append(x0_rk)


   # Euler-Schritt: Berechnung der neuen Werte
   v0_eu = euler_step(freefall_firder, v0_eu, tmp, p)
   x0_eu = euler_step(freefall_secder, x0_eu, v0_eu, p)
   # Runge-Kutta-Schritt: Berechnung der neuen Werte
   v0_rk = rk4_step(freefall_firder, v0_rk, tmp, p)
   x0_rk = rk4_step(freefall_secder, x0_rk, v0_rk, p)
   #inkrementiere Zeit
   t += p.dt

   


# Ausgabe
print("Simulation abgeschlossen.")
print(f"Letzte Position Euler: {x0_eu:.2f} m")
print(f"Letzte Geschwindigkeit Euler: {v0_eu:.2f} m/s")
print(f"Letzte Position Runge-Kutta: {x0_rk:.2f} m")
print(f"Letzte Geschwindigkeit Runge-Kutta: {v0_rk:.2f} m/s")
print(f"Simulationsschritte: {len(times)} and t_sim={p.dt * len(times)} s")


fig2,ax2 = plt.subplots(1,1)
ax2.plot(times, vel_euler, 'x', label='Geschwindigkeit der Eulerlösung',color='blue')
ax2.set_ylabel('Geschwindigkeit (m/s)')
ax2.set_xlabel('Zeit (s)')
ax2.set_xlim(0, t_max)
ax2.grid()
ax2.legend()

fig3,ax3 = plt.subplots(1,1)
ax3.plot(times, pos_rk, '-', label='Geschwindigkeit der Runge-Kutta-Lösung',color='orange')
ax3.plot(times, pos_euler, '-', label='Weg der Eulerlösung',color='blue')
ax3.set_ylabel('Weg (m)')
ax3.set_xlabel('Zeit (s)')
ax3.set_xlim(0, t_max)
ax3.grid()
ax3.legend()

plt.tight_layout()
plt.show()      

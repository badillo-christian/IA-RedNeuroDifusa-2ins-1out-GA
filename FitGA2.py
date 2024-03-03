import numpy as np
import math
import time
import skfuzzy as fuzz

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2)
fig.set_size_inches(12,8)

def function_xy(x,y): 
    return (x + y)*0.5 

def drawSurface( X, Y, Z ):
    ref = np.array( [[ 0.,1.11111111,2.22222222,3.33333333,4.44444444,5.55555556,6.66666667,7.77777778,8.88888889,10.],
                    [ 1.11111111,3.07692308,3.84615385,4.61538462,5.45454545,7.27272727,8.46153846,9.23076923,10.,11.11111111],
                    [ 2.22222222,3.84615385,4.70588235,5.33333333,6.36363636,8.18181818,9.33333333,10.,10.76923077,12.22222222],
                    [ 3.33333333,4.61538462,5.33333333,6.,7.27272727,9.09090909,10.,10.66666667,11.53846154,13.33333333],
                    [ 4.44444444,5.45454545,6.36363636,7.27272727,8.18181818,10.,10.90909091,11.81818182,12.72727273,14.44444444],
                    [ 5.55555556,7.27272727,8.18181818,9.09090909,10.,11.81818182,12.72727273,13.63636364,14.54545455,15.55555556],
                    [ 6.66666667,8.46153846,9.33333333,10.,10.90909091,12.72727273,14.,14.66666667,15.38461538,16.66666667],
                    [ 7.77777778,9.23076923,10.,10.66666667,11.81818182,13.63636364,14.66666667,15.29411765,16.15384615,17.77777778],
                    [ 8.88888889,10.,10.76923077,11.53846154,12.72727273,14.54545455,15.38461538,16.15384615,16.92307692,18.88888889],
                    [ 10.,11.11111111,12.22222222,13.33333333,14.44444444,15.55555556,16.66666667,17.77777778,18.88888889,20.]])

    
    surf = ax.plot_surface(X, Y, Z,  color = 'lightgreen', lw=0.5, rstride=1, cstride=1,alpha=0.8, linewidth = 0.3, edgecolor = 'black' )
    surfReferencia = ax.plot_surface(X, Y, ref, color= 'red', edgecolor='royalblue', linewidth=0, antialiased=False, alpha=0.3 )
    ax.set(xlim=(0, 10), ylim=(0, 10), zlim=(0, 20), xlabel='calidad servicio', ylabel='calidad comida', zlabel='% Propina')
    plt.draw()

def plot_fmGaussianaAlimentos():
    x = np.arange(0, 10, 0.10)
    fp1 = fuzz.gaussmf(x, 0, 5)
    fp2 = fuzz.gaussmf(x, 10, 5)
    fig_scale = 0.8
    plt.figure(figsize=(4.8 * fig_scale, 4.8 * fig_scale))
    plt.subplot(111)
    plt.plot(x, fp1, label="comida mala")
    plt.plot(x, fp2, label="comida buena")

    plt.legend(loc="lower right")
    plt.ylabel("Membresía")
    plt.xlabel("Universo de Discurso Comida")

def plot_fmGaussianaServicio():
    x = np.arange(0, 10, 0.10)
    fp1 = fuzz.gaussmf(x, 0, 5)
    fp2 = fuzz.gaussmf(x, 10, 5)
    fig_scale = 0.8
    plt.figure(figsize=(4.8 * fig_scale, 4.8 * fig_scale))
    plt.subplot(111)
    plt.plot(x, fp1, label="servicio malo")
    plt.plot(x, fp2, label="servicio bueno")

    plt.legend(loc="upper right")
    plt.ylabel("Membresía")
    plt.xlabel("Universo de Discurso Servicio ")   

def plot_2dError(results: np.array, expected = [], labels=[''], title='', block=False):

    ax[1].clear()

    ax[1].plot(results.T[0], results.T[1], '-b', label=labels[0])
    if len(expected):
        ax[1].plot(expected.T[0], expected.T[1], '-r', label=labels[1])

    ax[1].set_ylabel('Error')
    ax[1].set_xlabel('Epoca')
    ax[1].set_title(title)

    plt.show(block=block)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.25)

def exp_cuadratica(x):
    return  math.exp(-1/2*math.pow(x,2))

class FuzzyTsk:

    def shuffle_surface(self):
        self.surface_shuffled = np.random.permutation(self.Z)    

    def __init__(self, function, step=0.005, epocas=150, min_x=-2, max_x=2, alpha=0.01):

        self.f = function
        self.step = step
        self.epocas = epocas
        self.min_x = min_x
        self.max_x = max_x
        self.alpha = alpha

        self.points_x = np.linspace(0, 10, 2000)
        self.points_y = np.linspace(0, 10, 2000)

        X,Y = np.meshgrid(self.points_x, self.points_y)
        self.Z = np.zeros_like(X)
        i=0
        j=0
        for x in self.points_x:
            for y in self.points_y:
                self.Z[i,j] = self.f(x,y)
                j=j+1
            j=0        
            i=i+1    
        self.shuffle_surface()

    def estimar_valor_z(self, x1m, x1s, p1x, q1x, x2m, x2s, p2x, q2x, x, y1m, y1s, p1y, q1y, y2m, y2s, p2y, q2y, y,  return_all=True):

        w1x = exp_cuadratica((x-x1m)/x1s)
        w2x = exp_cuadratica((x-x2m)/x2s)

        w1y = exp_cuadratica((y-y1m)/y1s)
        w2y = exp_cuadratica((y-y2m)/y2s)

        w1nx = w1x/(w1x + w2x)
        w2nx = w2x/(w1x + w2x)

        w1ny = w1y/(w1y + w2y)
        w2ny = w2y/(w1y + w2y)

        zx1 = p1x*x + q1x 
        zx2 = p2x*x + q2x

        zy1 = p1y*y + q1y 
        zy2 = p2y*y + q2y

        z = (w1nx*zx1 + w2nx*zx2 + w1ny*zy1 + w2ny*zy2) * 0.5

        if return_all:
            return w1x, w2x, w1nx, w2nx, zx1, zx2, w1y, w2y, w1ny, w2ny, zy1, zy2, z
        return z


    def execute(self):

        x1m = 0
        x1s = 5
        p1x = 0
        q1x = 0

        x2m = 10
        x2s = 5
        p2x = 10
        q2x = 0

        y1m = 0
        y1s = 5
        p1y = 0
        q1y = 0

        y2m = 10
        y2s = 5
        p2y = 10
        q2y = 0

        plot_fmGaussianaAlimentos()
        plot_fmGaussianaServicio()
        sumatoriaError = 0
        listaErrores = []
        for epoca in range(0, self.epocas):

            if not epoca or epoca % (self.epocas//150) == 0 or epoca == self.epocas - 1:
                self.Z = self.estimar_valor_z(x1m, x1s, p1x,q1x, x2m, x2s, p2x, q2x, x, y1m, y1s, p1y, q1y, y2m, y2s, p2y, q2y, y, return_all=False)
                drawSurface(self.points_x, self.points_y, self.Z)

            sumatoriaError = 0

            # Inferencia de los puntos generados

            i=0
            j=0
            for i,x,y in enumerate(self.points_x.T[0]):
                for j,x,y in enumerate(self.points_y.T[0]):
                    w1x, w2x, w1nx, w2nx, zx1, zx2, w1y, w2y, w1ny, w2ny, zy1, zy2, z = self.estimar_valor_z(x1m, x1s, p1x,q1x, x2m, x2s, p2x, q2x, x, y1m, y1s, p1y, q1y, y2m, y2s, p2y, q2y, y)
                    zd = self.Z[i][j]
                    e = z - zd

                    # calcular de los parametros
                    p1x = p1x - self.alpha*e*w1nx*x
                    p2x = p2x - self.alpha*e*w2nx*x

                    q1x = q1x - self.alpha*e*w1nx
                    q2x = q2x - self.alpha*e*w2nx

                    x1m = x1m - self.alpha*e*w2x*(zx1-zx2)/(pow(w1x+w2x, 2))*(x-x1m)/pow(x1s,2)*exp_cuadratica((x-x1m)/x1s)
                    x2m = x2m - self.alpha*e*w1x*(zx2-zx1)/(pow(w1x+w2x, 2))*(x-x2m)/pow(x2s,2)*exp_cuadratica((x-x2m)/x2s)

                    x1s = x1s - self.alpha*e*w2x*(zx1-zx2)/(pow(w1x+w2x, 2))*pow(x-x1m,2)/pow(x1s,3)*exp_cuadratica((x-x1m)/x1s)
                    x2s = x2s - self.alpha*e*w1x*(zx2-zx1)/(pow(w1x+w2x, 2))*pow(x-x2m,2)/pow(x2s,3)*exp_cuadratica((x-x2m)/x2s)

                    p1y = p1y - self.alpha*e*w1ny*y
                    p2y = p2y - self.alpha*e*w2ny*y

                    q1y = q1y - self.alpha*e*w1ny
                    q2y = q2y - self.alpha*e*w2ny

                    y1m = y1m - self.alpha*e*w2y*(zy1-zy2)/(pow(w1y+w2y, 2))*(y-y1m)/pow(y1s,2)*exp_cuadratica((y-y1m)/y1s)
                    y2m = y2m - self.alpha*e*w1y*(zy2-zy1)/(pow(w1y+w2y, 2))*(y-y2m)/pow(y2s,2)*exp_cuadratica((y-y2m)/y2s)

                    y1s = y1s - self.alpha*e*w2y*(zy1-zy2)/(pow(w1y+w2y, 2))*pow(y-y1m,2)/pow(y1s,3)*exp_cuadratica((y-y1m)/y1s)
                    y2s = y2s - self.alpha*e*w1y*(zy2-zy1)/(pow(w1y+w2y, 2))*pow(y-y2m,2)/pow(y2s,3)*exp_cuadratica((y-y2m)/y2s)

                    sumatoriaError = sumatoriaError + pow(e, 2)                
                    j=j+1
                j=0        
                i=i+1    

            self.shuffle_surface()
            listaErrores.append((int(epoca), sumatoriaError))
            plot_2dError(np.array(listaErrores), labels=['Error'], title='Error por epoca', block=False)

if __name__ == "__main__":

    ft = FuzzyTsk(function_xy, alpha=0.01, epocas=150)
    ft.execute()

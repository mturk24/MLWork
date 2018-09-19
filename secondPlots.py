import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def main():
	part_a()
	part_b()
	part_c()
	part_d()
	part_e()
	
def part_a():
	change = 0.025
	x = np.arange(-10.0, 10.0, change)
	y = np.arange(-10.0, 10.0, change)
	X, Y = np.meshgrid(x, y)
	Gauss = mlab.bivariate_normal(X, Y, 1.0, 2.0, 1.0, 1.0)
	plt.figure()
	CS = plt.contour(X,Y, Gauss)
	plt.show()	

def part_b():
	change = 0.025
	x = np.arange(-10.0, 10.0, change)
	y = np.arange(-10.0, 10.0, change)
	X, Y = np.meshgrid(x, y)
	Gauss = mlab.bivariate_normal(X, Y, 2.0, 3.0, -1.0, 2.0, 1)
	plt.figure()
	CS = plt.contour(X,Y, Gauss)
	plt.show()	

def part_c():
	change = 0.025
	x = np.arange(-10.0, 10.0, change)
	y = np.arange(-10.0, 10.0, change)
	X, Y = np.meshgrid(x, y)
	G1 = mlab.bivariate_normal(X, Y, 2.0, 1.0, 0.0, 2.0, 1)
	G2 = mlab.bivariate_normal(X, Y, 2.0, 1.0, 2.0, 0.0, 1)
	Gauss = G1 - G2
	plt.figure()
	CS = plt.contour(X,Y, Gauss)
	plt.show()	

def part_d():
	change = 0.025
	x = np.arange(-10.0, 10.0, change)
	y = np.arange(-10.0, 10.0, change)
	X, Y = np.meshgrid(x, y)
	G1 = mlab.bivariate_normal(X, Y, 2.0, 1.0, 0.0, 2.0, 1)
	G2 = mlab.bivariate_normal(X, Y, 2.0, 3.0, 2.0, 0.0, 1)
	Gauss = G1 - G2
	plt.figure()
	CS = plt.contour(X,Y, Gauss)
	plt.show()	

def part_e():
	change = 0.025
	x = np.arange(-10.0, 10.0, change)
	y = np.arange(-10.0, 10.0, change)
	X, Y = np.meshgrid(x, y)
	G1 = mlab.bivariate_normal(X, Y, 2.0, 1.0, 1.0, 1.0, 0)
	G2 = mlab.bivariate_normal(X, Y, 2.0, 2.0, -1.0, -1.0, 0)
	Gauss = G1 - G2
	plt.figure()
	CS = plt.contour(X,Y, Gauss)
	plt.show()	

if __name__ == "__main__":
	main()
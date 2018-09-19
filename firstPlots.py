import matplotlib.pyplot as plt
import numpy as np

def main():
	x1 = np.array(np.random.normal(4, 4, 100))
	x2 = np.array(0.5*x1 + np.random.normal(3, 9, 100))
	covarMatrix = np.cov(x1, x2)
	mu = np.array([np.mean(x1), np.mean(x2)])	
	eigenvalues, eigenvectors = np.linalg.eigh(covarMatrix)	

	print "Mean of x1: ", np.mean(x1)
	print "Mean of x2: ", np.mean(x2)
	print "Covariance Matrix: ", covarMatrix
	print "Mu constant", mu
	print "Eigenvalues: ", eigenvalues
	print "Eigenvectors: \n", eigenvectors, "\n"

	concatenation = np.concatenate(([x1], [x2]))
	scaled = eigenvectors.transpose()
	
	for row in concatenation.transpose():
		row[0], row[1] = row[0] - mu[0], row[1] - mu[1]
		row = scaled.dot(row)	

	fig, ax = plt.subplots()
	ax.scatter(x1, x2)
	for vector in eigenvectors.transpose():
		plt.arrow(mu[0], mu[1], vector[0], vector[1], head_width=0.5, head_length=.5)
	plt.axis([-15, 15, -15, 15])
	plt.show()	
	plt.scatter(concatenation[0], concatenation[1])
	plt.axis([-15, 15, -15, 15])
	plt.show()		
	
if __name__ == "__main__":
	main()
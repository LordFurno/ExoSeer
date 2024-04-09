**Introduction and Background**
For as long as humans have existed, we have looked up to the stars and wondered if there was anything else out there, if there was any other life in the universe. 
Exoplanets are planets that orbit stars that are outside our solar system. Of the nearly infinite exoplanets in the universe, we have detected around 5,600. Of these 5,600 exoplanets, only around 59 of them are thought to be likely habitable for us. 
Searching for these exoplanets provides an opportunity to find planets that are suitable for us, and maybe even be suitable for alien life. 
With the data available, trying to manually detect exoplanets, and determining the atmospheric makeup of said planets is very time-consuming and inefficient. This is why many algorithms have been developed to try and quickly perform these tasks.


**Purpose**
The purpose of this project is to create an algorithm that will contain all the tools to accurately determine the presence of exoplanets and accurately determine the molecular makeup of these exoplanet’s atmospheres. 


**Design Criteria**
This algorithm should be capable of accurately determining the presence and size of exoplanets, calculating the orbital period, and detecting the presence of certain molecules in the atmosphere of the exoplanet.

**Procedure**
	The first step in this project was to get the data that would be analyzed. There were 3 different types of data. Molecule absorption lines, for detecting the presence of molecules, spectral data of planets, to analyze, and data to train the machine learning model on. 
Afterwards, I used the Box-Least-Squares (BLS) method to detect transiting exoplanets in light curves. From this, I could calculate the orbital period of the exoplanet and its size relative to its star. 
	Next was performing a line-analysis on exoplanet spectra to determine the atmospheric makeup. I overlayed the molecule absorption lines on the spectral data for different exoplanets. From there, I calculated the probability that the molecule was present from how many lines matched up with the dips in the spectral data and the intensity of the line. 
After that, I had to call an API to get the data required for the machine-learning model. However, after getting the data and training my model, I realized the data I got was not what I needed. I had to then repeat this process. 
	I tested the detection scheme on various light curves to ensure that the outputs were correct. I then checked that the molecular line detection program output results lined up with known atmospheres.  For the machine learning model, I calculated its testing and validation accuracy and got an F-1 score. 


**Results and conclusion**
	The detecting portion of this algorithm worked very well. It could accurately determine the orbital period and relative size of the exoplanet. I compared the values calculated by the algorithm to the observed values from different Kepler missions, and they resemble each other. 
	The line-analysis component of this algorithm has demonstrated its potential by successfully identifying the presence of certain molecules when compared to data that has been analyzed by experts. Although there are cases where it may not be as effective, its ability to detect molecules signifies a significant step forward in the automated analysis process.
	The machine learning model demonstrates an accuracy of 40% and achieves an F-1 score of 0.83. 
In conclusion, this algorithm displays its practicality in analyzing exoplanets. This algorithm can help in detecting and determining the atmospheric makeup of exoplanets. 

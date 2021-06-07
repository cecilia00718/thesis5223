import numpy as np
import matplotlib.pyplot as plt
import math
import random
from sympy import *
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16
plt.rcParams["font.weight"] = "normal"
plt.rcParams.update({'figure.max_open_warning': 0})


class Beam():
    
    def __init__(self, **kwargs):
        # Parse Input Parameters
        for key, value in kwargs.items():
            if key == "k":
                self.k = value
            elif key == "delta0":
                self.delta0 = value
            elif key == "beta":
                self.beta = value
            elif key == "T0":
                self.T0 = value
            elif key == "T1":
                self.T1 = value
            elif key == "delta_T1":
                self.delta_T1 = value
            elif key == "alpha":
                self.alpha = value
            elif key == "R0": 
                self.R0 = value
            elif key == "T":
                self.T = value
            elif key == "R_T":
                self.R_T = value
            elif key == "phi":
                self.phi = value
            elif key == "B":
                self.B = value
            elif key == "H":
                self.H = value
            elif key == "mean_deadLoad":
                self.mean_deadLoad = value
            elif key == "variance_deadLoad":
                self.variance_deadLoad = value
            elif key == "mean_sustainedLoad":
                self.mean_sustainedLoad = value
            elif key == "variance_sustaineLoad":
                self.variance_sustaineLoad = value
            elif key == "beta_sustaintedLoad":
                self.beta_sustaintedLoad = value
            elif key == "mean_extroLoad":
                self.mean_extroLoad = value
            elif key == "variance_extroLoad":
                self.variance_extroLoad = value
            elif key == "beta_extroLoad":
                self.beta_extroLoad = value
            elif key == "load_duration":
                self.load_duration = value
            elif key == "L":
                self.L = value
            elif key == "b_t":
                self.b_t = value
            elif key == "alphaA":
                self.alphaA = value
            elif key == "alphaB":
                self.alphaB = value
        
        self.delta_T0 = self.k * self.delta0
        
    """
        PART 1 - Crack Depth (mm)
    """
    def calculateLambda(self):
        lam = ( (self.delta_T1 - self.delta_T0) * self.T0 ) / ( self.k * self.delta0 * (self.T1 - self.T0)**self.beta )
        self.lam = lam
        return lam
    
    def calculateDeltaT_year(self, duration_year=1000):
        deltaT = []
        duration_year = int(duration_year)
        if duration_year < self.T0:
            for year in range(duration_year):
                deltaT.append( self.k * self.delta0 * year / self.T0 )
                
        else:
            for year in range(duration_year):
                if year <= self.T0:
                    deltaT.append( self.k * self.delta0 * year / self.T0 )
                else:
                    deltaT.append( self.k * self.delta0 * (1 + self.lam * (year - self.T0)**self.beta / self.T0) )
            
        return deltaT
    
        
    def calculateDeltaT_day(self, duration_year=1000):
        deltaT_day = []
        deltaT_year = self.calculateDeltaT_year(duration_year)
        for delta in deltaT_year:
            deltaT_day += [delta] * 365
        return deltaT_day
    
    """
        PART 2 - Strength Degradation (Mpa)
    """
    def calculateA(self):
        a = ( 1 - self.R_T / self.R0 ) / ( self.T ** self.alpha )
        self.a = a
        return a
    
    def calculateR_year(self, duration_year=1000):
        R_year = []
        for year in range(duration_year):
            R_year.append( self.R0 * (1 - self.a * year ** self.alpha) )
        return R_year
    
    def calculateR_day(self, duration_year=1000):
        R_day = []
        R_year = self.calculateR_year(duration_year)
        for R in R_year:
            R_day += [R] * 365
        return R_day
    
    """
        PART 3 - Sigma S (Mpa)
    """
    def calculateSigmaS_year(self, duration_year=1000):
        sigmaS_year = []
        deltaT_year = self.calculateDeltaT_year(duration_year)
        R_year = self.calculateR_year(duration_year)
        for index in range(duration_year):
            w_eff = ((self.B - 2 * deltaT_year[index]/1000) * (self.H - 2 * deltaT_year[index]/1000) ** 2) / 6
            sigmaS_year.append(self.phi * R_year[index] * w_eff)
        return sigmaS_year
    
    def calculateSigmaS_day(self, duration_year=1000):
        sigmaS_day = []
        deltaT_day = self.calculateDeltaT_day(duration_year)
        R_day = self.calculateR_day(duration_year)
        for index in range(len(delta_T_day)):
            w_eff = ((self.B - 2 * deltaT_day[index]/1000) * (self.H - 2 * deltaT_day[index]/1000) ** 2) / 6
            sigmaS_day.append(self.phi * R_day[index] * w_eff)
        return sigmaS_day
    
    """
        PART 4 - Simulate Loads (kPa)
    """
    def calculateDeadLoadSingle(self):
        mu = self.mean_deadLoad
        sigma = math.sqrt(self.variance_deadLoad)
        deadLoadSingle = np.random.normal(mu, sigma, 1)[0]
        return deadLoadSingle
        
    def calculateSustainedLiveLoad_day(self, duration_year=1000):
        theta = self.variance_sustaineLoad / self.mean_sustainedLoad
        k = self.mean_sustainedLoad / theta
        timeLimit = 365 * duration_year
        currentTime = 0
        value = []
        sustainedTime = []
        sustainedLiveLoad_day = []
        while currentTime < timeLimit:
            value.append(np.random.gamma(theta, k, 1)[0])
            duration = int(np.random.exponential(self.beta_sustaintedLoad, 1)[0] * 365)
            sustainedTime.append(duration)
            currentTime += duration
        for index in range(len(sustainedTime)):
            sustainedLiveLoad_day += [value[index]] * sustainedTime[index]
        return sustainedLiveLoad_day[:duration_year*365]
    
    def calculateSustainedLiveLoad_year(self, duration_year=1000):
        sustainedLiveLoad_day = self.calculateSustainedLiveLoad_day(duration_year)
        sustainedLiveLoad_year = []
        for year in range(duration_year):
            sustainedLiveLoad_year.append(np.mean(sustainedLiveLoad_day[year*365: (year+1)*365]))
        return sustainedLiveLoad_year
    
    def calculateExtroLoad_day(self, duration_year=1000):
        theta = self.variance_extroLoad / self.mean_extroLoad
        k = self.mean_extroLoad / theta
        extroLiveLoad_day = [0] * 365 * 1000
        for i in range(duration_year):
            extroLive = np.random.gamma(theta, k, 1)[0]
            duration = random.randint(1, self.load_duration)
            occurance = int(np.random.exponential(self.beta_extroLoad, 1)[0])
            time = (np.random.random(occurance) * 365).astype(int)
            for startTime in time:
                for j in range(startTime, startTime + duration):
                    extroLiveLoad_day[i*365 + j] += extroLive
        extroLiveLoad_day = extroLiveLoad_day[:365 * duration_year]
        return extroLiveLoad_day
    
    def calculateExtroLoad_year(self, duration_year=1000):
        extroLiveLoad_day = self.calculateExtroLoad_day(duration_year)
        extroLiveLoad_year = []
        for year in range(duration_year):
            extroLiveLoad_year.append(np.mean(extroLiveLoad_day[year*365: (year+1)*365]))
        return extroLiveLoad_year
    
    """
        PART 5 - Sigma T (Mpa)
    """
    def calculateSigmaT_day(self, duration_year=1000):
        deadLoadSingle = self.calculateDeadLoadSingle() / 1000
        sustainedLive_day = self.calculateSustainedLiveLoad_day(duration_year) / 1000
        w = (np.array(sustainedLive_day) + deadLoadSingle)
        sigmaT_day = w * self.b_t * self.L**2 / 8
        return sigmaT_day
    
    def calculateSigmaT_year(self, duration_year=1000):
        deadLoadSingle = self.calculateDeadLoadSingle() / 1000
        sustainedLive_year = self.calculateSustainedLiveLoad_year(duration_year) / 1000
        w = (np.array(sustainedLive_year) + deadLoadSingle)
        sigmaT_year = w * self.b_t * self.L**2 / 8
        return sigmaT_year
    
    """
        PART 6 - Damage Accumulation Alpha
    """
    def calculateAlpha(self, sigmaT, sigmaS):
        a = self.alphaA * np.log(10) / self.alphaB
        b = np.log(10) / self.alphaB
        alphaRate = []
        Alpha = []
        if len(list(sigmaT)) == len(list(sigmaS)):
            sigmaT = list(sigmaT)
            sigmaS = list(sigmaS)
            for index in range(len(list(sigmaT))):
                alphaRate.append( np.exp(-a + b * sigmaT[index] / sigmaS[index]) )
            
            Alpha = np.cumsum(alphaRate)
            return alphaRate, Alpha
        else:
            print("LENGTH OF SIGMAT AND SIGMAS SHOULD BE THE SAME!")
        return alphaRate, Alpha
    

if __name__ == '__main__':
    # INPUT PARAMETERS
    params_beam = {
        "k" : 0.25,                       # ========= DeltaT
        "delta0" : 4*1.06, 
        "beta": 2, 
        "T0" : 267, 
        "T1": 954, 
        "delta_T1": 16,           
        "alpha": 2,                       # ========= Strength Degradation
        "R0": 111.11, 
        "T": 921, 
        "R_T": 94.6,            
        "phi": 0.7,                       # ========= SigmaS
        "B": 0.11,
        "H": 0.2,           
        "mean_deadLoad": 1.63,            # ========= Loads
        "variance_deadLoad": 0.1,
        "mean_sustainedLoad": 0.6, 
        "variance_sustaineLoad": 0.13, 
        "beta_sustaintedLoad": 10, 
        "mean_extroLoad": 0.5,
        "variance_extroLoad": 0.7478, 
        "beta_extroLoad": 10/3, 
        "load_duration": 5,
        "L": 6.9,                         # ========= Sigma T
        "b_t": 2.26, 
        "alphaA": 0.9,                    # ======== Alpha
        "alphaB": 0.0495
    }
    
    # Crack Depth
    rectangular_beam = Beam(**params_beam)
    lam = rectangular_beam.calculateLambda()
    deltaT_year = rectangular_beam.calculateDeltaT_year(duration_year=1000)
    deltaT_day = rectangular_beam.calculateDeltaT_day(duration_year=1000)

    # Strength Degradation
    a = rectangular_beam.calculateA()
    R_year = rectangular_beam.calculateR_year(duration_year=1000)
    R_day = rectangular_beam.calculateR_day(duration_year=1000)

    # Sigma S
    sigmaS_year = rectangular_beam.calculateSigmaS_year(duration_year=1000)
    sigmaS_day = rectangular_beam.calculateSigmaS_day(duration_year=1000)

    # Load
    deadLoadSingle = rectangular_beam.calculateDeadLoadSingle()
    sustainedLiveLoad_day = rectangular_beam.calculateSustainedLiveLoad_day(duration_year=1000)
    sustainedLiveLoad_year = rectangular_beam.calculateSustainedLiveLoad_year(duration_year=1000)
    extroLiveLoad_day = rectangular_beam.calculateExtroLoad_day(duration_year=1000)
    extroLiveLoad_year = rectangular_beam.calculateExtroLoad_year(duration_year=1000)

    # Sigma T
    sigmaT_day = rectangular_beam.calculateSigmaT_day(duration_year=1000)
    sigmaT_year = rectangular_beam.calculateSigmaT_year(duration_year=1000)

    # AlphaRate and Alpha
    alphaRate, Alpha = rectangular_beam.calculateAlpha(sigmaT=sigmaT_day, sigmaS=sigmaS_day)
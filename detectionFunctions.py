import lightkurve as lk
import numpy as np
import math
import matplotlib.pyplot as plt 
def getLightcurveData(observation,cadence):
    search_result=lk.search_lightcurve(observation,author="Kepler",exptime=cadence)

    if list(search_result)==[]:#Some parameter is wrong
        return "Error"
    lc_collection=search_result.download_all()
    #.stitch removes the offset between the different observations
    #.flatten() applies the Savitzky-Golay filter
    #.remove_outliers() 
    # lc_collection.plot();
    # plt.show(block=True)
    lc = lc_collection.stitch().flatten().remove_outliers()
    return lc

def getBLSData(periodSearch,lightcurve):
    bls=lightcurve.to_periodogram(method='bls',period=periodSearch,frequency_factor=500)
    return bls
def extractPlanetData(blsData):
    planetPeriod = blsData.period_at_max_power
    planetT0 = blsData.transit_time_at_max_power
    planetDur = blsData.duration_at_max_power
    planetModel=blsData.get_transit_model(period=planetPeriod, transit_time=planetT0,duration=planetDur)
    return (planetPeriod.value,planetT0.value,planetDur.value,planetModel)
def createCadenceMask(blsData,period,t0,dur):
    cadenceMask=blsData.get_transit_mask(period=period,transit_time=t0,duration=dur)
    return cadenceMask
def applyCadenceMask(lightcurve,mask):
    maskedLc=lightcurve[~mask]
    return maskedLc
def findPlanets(period,lc):
    counter=0
    planets={}
    while counter <= 15:
        bls=getBLSData(period, lc)
        
        # Calculate the percentile of bls.power.max() in the entire distribution of bls.power
        medianRank=np.median(bls.power)
        
        data=extractPlanetData(bls)
        
        # Assuming you have a significance threshold for detection
        if bls.power.max() > medianRank*2.5:
            planetModel=data[3]
            transitModel=set(map(lambda x:x.value,list(planetModel["flux"])))
            relativeRad=math.sqrt(max(transitModel)-min(transitModel))

            planetPeriod=data[0]

            planetT0=data[1]

            planetDuration=data[2]



            #Planet period, PlanetT0, PlanetDuration, relative radius, bls power
            planets[counter] = [planetPeriod, planetT0,planetDuration,relativeRad,bls.power.max().value]
            mask = createCadenceMask(bls, planets[counter][0], planets[counter][1], planets[counter][2])
            lc = applyCadenceMask(lc, mask)
        else:
            break
        counter+=1
    return planets

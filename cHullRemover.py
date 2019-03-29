# From henceforth we could do this recursively - that is to use the output of a function as an input to itself:
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
class cHullRemover():
    """A class of static functions operating on an ndarray[[w wavelengths],[r reflectances]]"""
    # pSample is in a format: nparray[[wavelengths],[reflectances]]
    def __init__(self):
        pass
    
    @staticmethod
    def removeNoData(pSample: 'ndarray[[w],[r]]') -> 'ndarray[[w],[r]]':
        """Removes nodata (nan) values from spectrum sample"""
        return np.asarray([_ for _ in pSample.T if not np.isnan(_[1])]).T
    
    @staticmethod
    def smooth(pSample: 'ndarray[[wl],[refl]]') -> 'ndarray[[w],[r]]':
        """Computes a 3-element moving average, discarding first and last values"""
        return np.stack([pSample[0][1:-1],(pSample[1][0:-2]+pSample[1][1:-1]+pSample[1][2:])/3])
    
    @staticmethod
    def removeContinuum(pSample: 'ndarray[[w],[r]]') -> 'ndarray[[w],[r]]':
        """Removes continuum"""
        pSampleLineX=[pSample[0][0],pSample[0][-1]]
        pSampleLineY=[pSample[1][0],pSample[1][-1]]
        pSampleLine=[pSampleLineX,pSampleLineY]
        finterp = interp1d(pSampleLine[0],pSampleLine[1])#create interploation function
        return np.asarray([pSample[0],pSample[1]-finterp(pSample[0])])

    @staticmethod
    def __getMaxima(pSample: 'ndarray[[w],[r]]') -> 'ndarray[w]':
        """Hidden function returning list of wavelenghts with local maxima reflectances"""
        def getMaximaInner(innerSample):
            contRem=cHullRemover.removeContinuum(innerSample)[1]
            maxIndex=np.argmax(contRem)
            maxVal=contRem[maxIndex]
            maxLoc=innerSample[0][maxIndex]
            if len(contRem)>2 and maxVal>contRem[0] and maxVal>contRem[-1]: # check that the maximum is more than edges
                maxLocArray.append(maxLoc)
                subsetLeft=[innerSample[0][:maxIndex+1],innerSample[1][:maxIndex+1]]
                subsetRight=[innerSample[0][maxIndex:],innerSample[1][maxIndex:]]
                getMaximaInner(subsetLeft)
                getMaximaInner(subsetRight)
        maxLocArray=[] #initialize array to store a list of points on a convex hull
        getMaximaInner(pSample)
        maxLocArray.sort()
        return [pSample[0][0]]+maxLocArray+[pSample[0][-1]]

    @staticmethod
    def getHull(pSample: 'ndarray[[w],[r]]') -> 'ndarray[[w],[r]]':
        """Compute convex hull using a quick hull algorithm"""
        maxList=cHullRemover.__getMaxima(pSample)
        return [maxList,[_[1] for _ in pSample.swapaxes(0,1) if _[0] in maxList]]
    
    @staticmethod
    def subtractSpectra(pSample: 'ndarray[[w],[r]]',pToSubtract: 'ndarray[[w],[r]]') -> 'ndarray[[w],[r]]':
        """Subtract second spectra from the first, use this to remove convex hull."""
        finterp = interp1d(pToSubtract[0],pToSubtract[1])#create interploation function
        return np.asarray([pSample[0],pSample[1]-finterp(pSample[0])])
    
    @staticmethod
    def splitSample(pSample: 'ndarray[[w],[r]]',pHull: 'ndarray[[w],[r]]') -> '[ndarray[[w],[r]],...]':
        """Subtact hull and split sample based in intersection with the hull"""
        splitInd=[list(pSample[0]).index(_) for _ in pHull[0]]
        return [[pSample[0][splitInd[_]:splitInd[_+1]+1],
                 cHullRemover.subtractSpectra(pSample,pHull)[1][splitInd[_]:splitInd[_+1]+1]] 
                     for _ in range(len(splitInd)-1) if splitInd[_+1]-splitInd[_]>2]
        # Finding local minima is then straightforward:
    
    @staticmethod
    def listMinimaWrtSample(pSample: 'ndarray[[w],[r]]') -> 'ndarray[[w],[r]]':
        """Get a list of local minima on a sample spectrum"""
        pHull=cHullRemover.getHull(pSample)
        splitSample=cHullRemover.splitSample(pSample, pHull)
        listMinimaX=[_[0][np.argmin(_[1])] for _ in np.asarray(splitSample)]
        return [listMinimaX,
                [pSample[1][list(pSample[0]).index(_)] for _ in listMinimaX]]
    
    @staticmethod
    def listMinimaWrtHull(pSample: 'ndarray[[w],[r]]',pHull: 'ndarray[[w],[r]]') -> 'ndarray[[w],[r]]':
        """Get a list of local minima on a sample spectrum with hull subtracted"""
        splitSample=cHullRemover.splitSample(pSample, pHull)
        return [[_[0][np.argmin(_[1])] for _ in np.asarray(splitSample)],
                [_[1][np.argmin(_[1])] for _ in np.asarray(splitSample)]]
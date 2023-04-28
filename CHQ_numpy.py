# USAGE:
# python3 CHQ_numpy.py source_cube.tif source_cube_clean.tif source_cube_chq.tif

# This script will perform Convex Hull Quotent
# Note that the input spectrum is smoothed by 3-element-wide median filter
# Quotent is calculated by dividing unsmoothed source data by this hull
# The script also outputs a copy of the source image with first two bands removed.

import numpy as np
#from scipy.spatial import ConvexHull
import sys
from scipy import signal

path_source_cube = sys.argv[1]
# TODO: ADD NUMBER OF ITERATIONS
path_source_cube_clean = sys.argv[2]
path_CHQ_cube = sys.argv[3]


import rasterio
import geopandas as gpd
from shapely.geometry import LineString, Point
import numpy as np
from matplotlib import pyplot as plt
wavelengths_general = [540.8400, 580.7600, 620.6900, 660.6100, 700.5400, 730.4800, 750.4400, 770.4000, 790.3700, 810.3300, 830.2900, 850.2500, 870.2100, 890.1700, 910.1400, 930.1000, 950.0600, 970.0200, 989.9800, 1009.950, 1029.910, 1049.870, 1069.830, 1089.790, 1109.760, 1129.720, 1149.680, 1169.640, 1189.600, 1209.570, 1229.530, 1249.490, 1269.450, 1289.410, 1309.380, 1329.340, 1349.300, 1369.260, 1389.220, 1409.190, 1429.150, 1449.110, 1469.070, 1489.030, 1508.990, 1528.960, 1548.920, 1578.860, 1618.790, 1658.710, 1698.630, 1738.560, 1778.480, 1818.400, 1858.330, 1898.250, 1938.180, 1978.100, 2018.020, 2057.950, 2097.870, 2137.800, 2177.720, 2217.640, 2257.570, 2297.490, 2337.420, 2377.340, 2417.260, 2457.190, 2497.110, 2537.030, 2576.960, 2616.880, 2656.810, 2696.730, 2736.650, 2776.580, 2816.500, 2856.430, 2896.350, 2936.270, 2976.200]

wavelengths_targeted = [ 446.0200, 456.0000, 465.9800, 475.9600, 485.9500, 495.9300, 505.9100, 515.8900, 525.8700, 535.8500, 545.8300, 555.8100, 565.7900, 575.7700, 585.7600, 595.7400, 605.7200, 615.7000, 625.6800, 635.6600, 645.6400, 655.6200, 665.6000, 675.5800, 685.5600, 695.5500, 705.5300, 715.5100, 725.4900, 735.4700, 745.4500, 755.4300, 765.4100, 775.3900, 785.3700, 795.3600, 805.3400, 815.3200, 825.3000, 835.2800, 845.2600, 855.2400, 865.2200, 875.2000, 885.1800, 895.1700, 905.1500, 915.1300, 925.1100, 935.0900, 945.0700, 955.0500, 965.0300, 975.0100, 984.9900, 994.9700, 1004.960, 1014.940, 1024.920, 1034.900, 1044.880, 1054.860, 1064.840, 1074.820, 1084.800, 1094.780, 1104.770, 1114.750, 1124.730, 1134.710, 1144.690, 1154.670, 1164.650, 1174.630, 1184.610, 1194.590, 1204.580, 1214.560, 1224.540, 1234.520, 1244.500, 1254.480, 1264.460, 1274.440, 1284.420, 1294.400, 1304.380, 1314.370, 1324.350, 1334.330, 1344.310, 1354.290, 1364.270, 1374.250, 1384.230, 1394.210, 1404.190, 1414.180, 1424.160, 1434.140, 1444.120, 1454.100, 1464.080, 1474.060, 1484.040, 1494.020, 1504.000, 1513.990, 1523.970, 1533.950, 1543.930, 1553.910, 1563.890, 1573.870, 1583.850, 1593.830, 1603.810, 1613.800, 1623.780, 1633.760, 1643.740, 1653.720, 1663.700, 1673.680, 1683.660, 1693.640, 1703.620, 1713.600, 1723.590, 1733.570, 1743.550, 1753.530, 1763.510, 1773.490, 1783.470, 1793.450, 1803.430, 1813.410, 1823.400, 1833.380, 1843.360, 1853.340, 1863.320, 1873.300, 1883.280, 1893.260, 1903.240, 1913.220, 1923.210, 1933.190, 1943.170, 1953.150, 1963.130, 1973.110, 1983.090, 1993.070, 2003.050, 2013.030, 2023.010, 2033.000, 2042.980, 2052.960, 2062.940, 2072.920, 2082.900, 2092.880, 2102.860, 2112.840, 2122.820, 2132.810, 2142.790, 2152.770, 2162.750, 2172.730, 2182.710, 2192.690, 2202.670, 2212.650, 2222.630, 2232.620, 2242.600, 2252.580, 2262.560, 2272.540, 2282.520, 2292.500, 2302.480, 2312.460, 2322.440, 2332.420, 2342.410, 2352.390, 2362.370, 2372.350, 2382.330, 2392.310, 2402.290, 2412.270, 2422.250, 2432.230, 2442.220, 2452.200, 2462.180, 2472.160, 2482.140, 2492.120, 2502.100, 2512.080, 2522.060, 2532.040, 2542.030, 2552.010, 2561.990, 2571.970, 2581.950, 2591.930, 2601.910, 2611.890, 2621.870, 2631.850, 2641.830, 2651.820, 2661.800, 2671.780, 2681.760, 2691.740, 2701.720, 2711.700, 2721.680, 2731.660, 2741.640, 2751.630, 2761.610, 2771.590, 2781.570, 2791.550, 2801.530, 2811.510, 2821.490, 2831.470, 2841.450, 2851.440, 2861.420, 2871.400, 2881.380, 2891.360, 2901.340, 2911.320, 2921.300, 2931.280, 2941.260, 2951.250, 2961.230, 2971.210, 2981.190, 2991.170]


class masked_interp_class2d():
        # mask contains array on 1's padded with 0's
        # describing which part of array to interpolate
    def __init__(self, wl):
        wl = np.asarray(wl)
        self.wl = np.pad(wl, (1,1))
        self.nomask = self.wl*0+1
        

    def get_mask_edge_L2d(self, mask2d): # Given this mask, get the first element as one, else zero
        mask_rolled2d=np.roll(mask2d, (1,0))
        mask_rolled2d[:,-1]=0
        mask_delta2d = mask2d - mask_rolled2d
        return mask_delta2d>0

    def get_mask_edge_R2d(self, mask2d): # Given this mask, get the last element as one, else zero
        mask_rolled2d=np.roll(mask2d, (-1,0))
        mask_rolled2d[:,0]=0
        mask_delta2d = mask2d - mask_rolled2d
        return mask_delta2d>0
    
    def get_nomask2d(self, arr2d):
        mask2d = np.tile(self.nomask, (arr2d.shape[0], 1))
        return mask2d
    
    def get_wl2d(self, arr2d):
        wl2d = np.tile(self.wl, (arr2d.shape[0], 1))
        return wl2d
    
    def masked_interp2d(self, arr2d, mask2d): 
        arr2d=np.pad(arr2d,((0,0),(1,1)))
        mask2d=np.pad(mask2d,((0,0),(1,1)))
        # EVALUATE IF sum(mask)>1
        #IF TRUE:
        mask_edge_R2d=self.get_mask_edge_R2d(mask2d)
        mask_edge_L2d=self.get_mask_edge_L2d(mask2d)
        
        arr_sum_R2d=np.sum(arr2d*mask_edge_R2d, axis=1)
        arr_sum_L2d=np.sum(arr2d*mask_edge_L2d, axis=1)
        arr_range2d=(arr_sum_R2d-arr_sum_L2d)[:,None]
        
        wl2d=self.get_wl2d(arr2d)
        
        wl_sum_R2d=np.sum(wl2d*mask_edge_R2d, axis=1)
        wl_sum_L2d=np.sum(wl2d*mask_edge_L2d, axis=1)
        
        wl_range2d=(wl_sum_R2d-wl_sum_L2d)[:,None]
        
        #wl_norm=(wl2d - np.sum(wl2d*mask_edge_L2d, axis=1)[:,None])/wl_range2d
        wl_diff=wl2d - np.sum(wl2d*mask_edge_L2d, axis=1)[:,None]
        wl_norm=np.divide(wl_diff, wl_range2d, out=np.zeros_like(wl_diff), where=wl_range2d!=0)
        
        interp_sum_mask2d_mt_1=wl_norm*arr_range2d + \
                                np.sum(arr2d*mask_edge_L2d, axis=1)[:,None]
        np.nan_to_num(interp_sum_mask2d_mt_1, copy=False) # Replace nan with 0 inplace.
        #IF FALSE
        interp_sum_mask2d_le_1=arr2d
        
        mask_condition=np.sum(mask2d, axis=1)[:,None]
        # EVAL
        interp =(mask_condition>1)*interp_sum_mask2d_mt_1+\
                (mask_condition<=1)*interp_sum_mask2d_le_1
        # RETURN result
        result=interp*mask2d
        return result[:,1:-1]

    def segment_multimask2d(self, multimask2d):
        return (1-multimask2d).cumsum(axis=1)*2-(1-multimask2d)
    
    def multi_interp2d(self, arr2d, multimask2d):
        segmented_multimask2d=self.segment_multimask2d(multimask2d)
        interp_compilation2d=multimask2d*0
        interps=[]
        # GOOD UP TO HERE
        for i in range((segmented_multimask2d.max()/2).astype(int)+1):
            low_bound=i*2-1
            high_bound=i*2+1
            submask_low_bound2d = segmented_multimask2d>=low_bound 
            submask_high_bound2d = segmented_multimask2d<=high_bound
            submask2d = (submask_low_bound2d*submask_high_bound2d).astype(np.float64)
            interp_step2d = self.masked_interp2d(arr2d, submask2d)
            interp_compilation2d += interp_step2d.astype(np.float64)
        result = interp_compilation2d-interp_compilation2d*(1-multimask2d)/2
        result[:,-1]=arr2d[:,-1]
        return result
    
mi_obj2d = masked_interp_class2d(wavelengths_general)

# Load M3 data cube
with rasterio.open(path_source_cube) as data_cube:
    data_cube_array = data_cube.read()

data_cube_array_clean=data_cube_array[2:,:,:]
#data_cube_array_clean=data_cube_array ## USE THIS IF YOUR IMAGE ALREADY HAS FIRST TWO BANDS REMOVED

#Clean data from outliers using median filter # You might want to comment this out if you want 'classic' result
data_cube_array_clean_med = signal.medfilt(data_cube_array_clean,(3,1,1))
print(data_cube_array_clean_med.shape)
AOI_reshaped=data_cube_array_clean_med.reshape((83, -1)).T

# RUN IT LIKE THIS:
all_spectra=AOI_reshaped.copy()
mask_current=all_spectra*0+1
print('nomask created')
difference_E0 = all_spectra-mi_obj2d.multi_interp2d(all_spectra, mask_current)
mask_E1=1.0-1*(difference_E0==difference_E0.max(axis=1)[:,None])
mask_current=mask_current*mask_E1

for i in range(23):
#for i in range(1):
    print(i)
    difference_E1 = all_spectra-mi_obj2d.multi_interp2d(all_spectra, mask_current) # This step eats most time. Can this run on a subset instead?
    max_E1=difference_E1.max(axis=1)[:,None]
    #print(max_E1)
    print(max_E1.max(),i)
    if max_E1.max()<=0:
        print('DONE on max',max_E1.max(),max_E1.sum(),i)
        break
    mask_E2=1.0-1*(difference_E1==max_E1)
    mask_old = mask_current.copy() # NEW
    mask_current=mask_current*mask_E2
    if (mask_old == mask_current).all(): # NEw
        print('DONE on repeat',max_E1.max(),i) # NEW
        break # NEW

ConvexHull=mi_obj2d.multi_interp2d(AOI_reshaped, mask_current).T.reshape(data_cube_array_clean.shape)

#CHQ=AOI_reshaped/mi_obj2d.multi_interp2d(AOI_reshaped, mask_current)
#AOI_CHQ=CHQ.T.reshape(data_cube_array_clean.shape)

AOI_CHQ=data_cube_array_clean/ConvexHull


with rasterio.open(path_source_cube) as data_cube:
    profile = data_cube.profile.copy() 
    profile.update(count=AOI_CHQ.shape[0])
    with rasterio.open(path_source_cube_clean, 'w', **profile) as out_cube:
        out_cube.write(data_cube_array_clean)
#        out_cube.write(data_cube.read()[:,:,:]) ## USE THIS IF YOUR IMAGE ALREADY HAS FIRST TWO BANDS REMOVED
    with rasterio.open(path_CHQ_cube, 'w', **profile) as out_cube:
        out_cube.write(AOI_CHQ)


#with rasterio.open(path_CHQ_cube) as data_cube:
#    profile = data_cube.profile.copy() 
#    profile.update(count=AOI_CHQ.shape[0])

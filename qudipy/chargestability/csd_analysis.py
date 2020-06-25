'''
File used to analyze charge stability diagrams in order to produce capacitance matrices, virtual gating, lever arms, etc.
For a good review of the Hough transform, check https://alyssaq.github.io/2014/understanding-hough-transform/.
Hough transfrom code based off of https://github.com/alyssaq/hough_transform.
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster

class CSDAnalysis:
    '''
    Initialize the charge stability diagram analysis class which analyzes charge stability diagrams to extract parameters. 

    '''
    def __init__(self, csd):
        '''
         
        Parameters
        ----------
        csd : CSD object from csd_gen

        Returns
        -------
        None

        '''
        self.csd = csd

    def generate_bitmap(self, threshold):
        '''
        Transforms the charge stability diagram into a bitmap. Threshold determines whether bit is considered 'on' or 'off'

        Parameters
        ----------
        threshold: threshold which determines whether bit is considered 'on' or 'off' 

        Returns
        -------
        None

        '''
        self.csd_bitmap = self.csd.mask(abs(self.csd) > threshold, other=1).mask(abs(self.csd) <= threshold, other=0)


    def hough_transform(self, num_thetas=180, rho_num=100):
        '''
        Performs the Hough transform on the charge stability diagram bitmap stored in the object

        Parameters
        ----------
        None

        Keyword Arguments
        -----------------
        num_thetas: number of angle points to sweep over (default 180) 
        rho_num: number of distance numbers to start with. Number of distnce points in the end will be roughly 2^(3/2) times the original amount (default 100)

        Returns
        -------
        Accumulator: 2D array with counts of each theta and 
        thetas: values of theta for which accumulator swept over
        rhos: values of distance for which accumulator swept over

        '''
        # Get charge stability diagram bitmap
        img = self.csd_bitmap
        
        # Rho and Theta ranges
        thetas = np.deg2rad(np.linspace(-90.0, 90.0, num=num_thetas))
        width = img.columns[-1]
        height = img.index[-1]
        diag_len = np.sqrt(width ** 2 + height ** 2)   # max_dist

        index_height, index_width = img.shape
        index_diag_len = int(round(np.sqrt(index_width ** 2 + index_height ** 2)))
        rhos = np.linspace(-diag_len, diag_len, num=2*index_diag_len)

        # Cache some resuable values
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)  

        # Hough accumulator array of theta vs rho
        accumulator = np.zeros((2 * index_diag_len, num_thetas), dtype=np.uint64)
        y_idxs, x_idxs = np.nonzero(img.to_numpy())  # (row, col) indexes to edges

        # Vote in the hough accumulator
        for i in range(len(x_idxs)):
            x = x_idxs[i]
            y = y_idxs[i]

            for t_idx in range(num_thetas):
                # Calculate rho. diag_len is added for a positive index
                rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]) + index_diag_len)
                accumulator[rho, t_idx] += 1

        self.accumulator = accumulator
        self.thetas = thetas
        self.rhos = rhos

        return accumulator, thetas, rhos

    def threshold_hough_accumulator(self, threshold, threshold_type='percentile'):
        '''
        Transforms the Hough transform accumulator stored in the objects into a binary accumulator using a threshold.
        Threshold determines whether accumulator value is set to 1 or 0. The threshold flag determines how the thrsehold is interpretted

        Parameters
        ----------
        threshold: number which specifies the threshold. Behaves differently depending on threshold_type

        Keyword Arguments
        -----------------
        threshold_type: String flag for which type of thresholding to do (default 'percentile')
            - 'percentile': will set all elements in the array above the set percentile to 1 and all those below to 0 
                    e.g with threshold=99, only elements above the 99th percentile will be set to 1
            - 'absolute': will set all elements in the array above the set percentile to 1 and all those below to 0 
                    e.g with threshold=20, only elements whos value exceeds 20 will be set to 1

        Returns
        -------
        accumulator_threshold: 2D array with counts 

        '''
        if threshold_type == 'percentile':
            percentile = np.percentile(self.accumulator, threshold)
            accumulator_threshold = np.zeros(self.accumulator.shape)
            
            for index, value in np.ndenumerate(self.accumulator):
                if value >= percentile:
                    accumulator_threshold[index] = 1

            self.accumulator_threshold = accumulator_threshold

        elif threshold_type == 'absolute':
            accumulator_threshold = np.zeros(self.accumulator.shape)
            
            for index, value in np.ndenumerate(self.accumulator):
                if value >= threshold:
                    accumulator_threshold[index] = 1

            self.accumulator_threshold = accumulator_threshold

        else:
            raise ValueError('Unrecognized threshold type: ' + str(threshold_type))

        return accumulator_threshold

    def hough_cluster(self):
        x = self.accumulator_threshold
        plt.imshow(x)
        plt.gca().invert_yaxis()
        plt.show()
        a = np.array(x.nonzero())
        points = []
        for i in range(len(a[0])):
            points.append([a[0][i], a[1][i]])
        points = np.array(points)
        
        db = cluster.DBSCAN(eps=3, min_samples=3).fit(points)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)

        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = points[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=14)

            xy = points[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()

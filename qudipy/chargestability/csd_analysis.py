'''
File used to analyze charge stability diagrams in order to produce capacitance matrices, virtual gating, lever arms, etc.
For a good review of the Hough transform, check https://alyssaq.github.io/2014/understanding-hough-transform/.
Hough transfrom code based off of https://github.com/alyssaq/hough_transform.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import cluster
from sklearn.neighbors import NearestCentroid

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

    def generate_bitmap(self, threshold, plotting=False):
        '''
        Transforms the charge stability diagram into a bitmap. Threshold determines whether bit is considered 'on' or 'off'

        Parameters
        ----------
        threshold: threshold which determines whether bit is considered 'on' or 'off' 

        Returns
        -------
        None

        '''
        self.csd_bitmap = self.csd.csd_der.mask(abs(self.csd.csd_der) > threshold, other=1).mask(abs(self.csd.csd_der) <= threshold, other=0)
        if plotting is True:
            self._plot_heatmap(self.csd_bitmap, self.csd.v_1_values, self.csd.v_2_values, r'V$_1$', r'V$_2$')


    def hough_transform(self, num_thetas=180, rho_num=100, plotting=False):
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

        if plotting is True:
            # Round data to avoid ridiculously long tick markers
            rhos = np.round(rhos, 3)
            thetas = np.round(thetas, 3)
            # Call heatmap plotting function
            self._plot_heatmap(accumulator, thetas, rhos, r'$\theta$ (rad)', r'$\rho$ (V)')

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
            - 'absolute': will set all elements in the array above the specified value to 1 and all those below to 0 
                    e.g with threshold=20, only elements whos is value greater or equal 20 will be set to 1

        Returns
        -------
        accumulator_threshold: 2D array with counts 

        '''
        if threshold_type == 'percentile':
            # Converts percentile type threshold into absolute type to use same for loop
            threshold = np.percentile(self.accumulator, threshold)

        elif threshold_type == 'absolute':
            # Do nothing to the threshold, but avoid raising an error
            pass

        else:
            raise ValueError('Unrecognized threshold type: ' + str(threshold_type))

        # Go through all the elements in the accumulator, setting all the elements above the threshold to 1
        accumulator_threshold = np.zeros(self.accumulator.shape)
        for index, value in np.ndenumerate(self.accumulator):
            if value >= threshold:
                accumulator_threshold[index] = 1
        self.accumulator_threshold = accumulator_threshold

        return accumulator_threshold

    def hough_cluster(self, eps, min_samples, plotting=False):
        '''
        Clusters the points in the thresholded Hough transform accumulator.

        Parameters
        ----------
        eps: maximum distance for points to be considered in the local neighbourhood of each other
        min_samples: minimum number of samples within the local neighbourhood in order for the point to be considered a core part of the cluster.

        Keyword Arguments
        -----------------
        potting: boolean flag which sets whether or not to plot the results of clustering (default False)
        
        Returns
        -------
        centroids: numpy array of pairs [theta, rhos] corresponding to valid charge transitions

        '''
        # Get the accumulator threshold parameters and reorder as a list of pairs of indices instead of a 2D array
        a = np.array(self.accumulator_threshold.nonzero())
        points_index = []
        for i in range(len(a[0])):
            points_index.append([a[1][i], a[0][i]])

        # Function that does the clusterting. Save the labels for later
        self.db = cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(points_index)
        labels = self.db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        # Convert from indices to pairs of the for [theta, rho] and convert into a numpy array
        points = []
        for pair in points_index:
            points.append([self.thetas[pair[0]], self.rhos[pair[1]]])
        points = np.array(points)

        # Plotting subroutine which plots all the point in different clusters in different colors
        if plotting is True:
            # take the transpose of points for ease of plotting 
            temp_points = points.transpose()
            # Create uniques colors for each cluster
            unique_labels = set(labels)
            colors = [plt.cm.viridis(each) for each in np.linspace(0, 1, len(unique_labels))]
            # Loop over each
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]

                points_to_plot_x = []
                points_to_plot_y = []
                for i in range(len(labels)):
                    if k == labels[i]:
                        points_to_plot_x.append(temp_points[0][i])
                        points_to_plot_y.append(temp_points[1][i])

                sb.scatterplot(x=points_to_plot_x, y=points_to_plot_y, color=tuple(col), s=100)

            plt.title('Estimated number of clusters: %d' % n_clusters_)
            plt.show()

        # get centroids of each cluster and save for later
        clf = NearestCentroid()
        clf.fit(points, labels)
        centroid = clf.centroids_
        # drop invalid centroids for charge stability diagram (which correcpond to positive slope)
        valid_centroids = np.array([i for i in centroid if i[0]>0])
        self.centroids = valid_centroids

        return valid_centroids

    def _plot_heatmap(self, data, x_values, y_values, x_label, y_label):
        df1 = pd.DataFrame(data, index=y_values, columns=x_values)
        s = sb.heatmap(df1, cbar=True, xticklabels=int(self.csd.num/5), yticklabels=int(self.csd.num/5))
        s.axes.invert_yaxis()
        s.axes.set_xlabel(x_label)
        s.axes.set_ylabel(y_label)
        plt.show()

    def plot_csd_with_lines(self):
        num = self.csd.csd.shape[0]

        f, ax = plt.subplots()
        sb.heatmap(self.csd.csd, cbar=False, xticklabels=int(num/5), yticklabels=int(num/5))
        ax.axes.invert_yaxis()
        ax2 = ax.twinx().twiny()

        for centroid in self.centroids:
            theta = centroid[0]
            rho = centroid[1]
            m = -np.cos(theta)/np.sin(theta)
            b = rho/np.sin(theta)
            x = np.linspace(self.csd.v_g1_min, self.csd.v_g1_max, num=num)
            y = m * x + b
            sb.lineplot(x=x, y=y, ax=ax2)

        ax2.set_xlim([self.csd.v_g1_min,self.csd.v_g1_max])
        ax2.set_ylim([self.csd.v_g2_min,self.csd.v_g2_max])
        ax2.get_yaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)
        plt.show()
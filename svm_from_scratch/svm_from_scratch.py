import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')


class SupportVectorMachine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # fit means training
    def fit(self, data):
        self.data = data
        optDict = {}
        transforms = [[1, 1],
                      [- 1, 1],
                      [-1, -1],
                      [1, -1]]
        # transforms applied to vectors w

        allData = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    allData.append(feature)

        self.maxFeatureValue = max(allData)
        self.minFeatureValue = min(allData)
        allData = None

        stepSizes = [self.maxFeatureValue * 0.1,
                     self.maxFeatureValue * 0.01,
                     self.maxFeatureValue * 0.001]

        bRangeMultiple = 5
        bMultiple = 5
        latestOptimum = self.maxFeatureValue * 10

        for step in stepSizes:
            w = np.array([latestOptimum, latestOptimum])
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.maxFeatureValue*bRangeMultiple),
                                   self.maxFeatureValue*bRangeMultiple,
                                   step*bMultiple):
                    for transformation in transforms:
                        wt = w * transformation
                        foundOption = True
                        # yi(xi.w + b) >=1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(wt, xi)+b) >= 1:
                                    foundOption = False
                                    # add a break here
                        if foundOption:
                            optDict[np.linalg.norm(wt)] = [wt, b]
                if w[0] < 0:  # because we are transforming
                    optimized = True
                    print("Optimized a step.")
                else:
                    w = w - step
            norms = sorted([n for n in optDict])
            # ||w|| : [w,b]
            optChoice = optDict[norms[0]]
            self.w = optChoice[0]
            self.b = optChoice[1]

            latestOptimum = optChoice[0][0]+step*2

    def predict(self, features):
        # sign(x.w + b)
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker="*", c=self.colors[classification])
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i])
          for x in data_dict[i]] for i in data_dict]

        def hyperplane(x, w, b, v):
            # hyperplane = x.w+b
            # v = x.w+b
            # pav = 1
            # nav  = -1
            # dec = 0
            return (-w[0]*x-b+v)/w[1]
        
        datarange = (self.minFeatureValue*0.9, self.maxFeatureValue*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        #(w.x + b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'y')

        #(w.x + b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'y')

        #(w.x + b) = 0
        # 
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'w--')

        plt.show()


data_dict = {-1: np.array([[-5, 10], [4, 14], [6, 8],]),
             1: np.array([[2, 1], [12, -1], [7, 5],])}

svm = SupportVectorMachine()
svm.fit(data = data_dict)

predict_us = [[0,10],
              [1,3],
              [-5,8],
              [6,2],
              [-2,-4],
              [5,6],
              [10,4]]

for p in predict_us:
    svm.predict(p)

svm.visualize()
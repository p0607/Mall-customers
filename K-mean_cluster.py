{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from sklearn import metrics\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.metrics import confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CustomerID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Annual Income (k$)</th>\n",
       "      <th>Spending Score (1-100)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Male</td>\n",
       "      <td>19</td>\n",
       "      <td>15</td>\n",
       "      <td>39</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>Male</td>\n",
       "      <td>21</td>\n",
       "      <td>15</td>\n",
       "      <td>81</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>Female</td>\n",
       "      <td>20</td>\n",
       "      <td>16</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>Female</td>\n",
       "      <td>23</td>\n",
       "      <td>16</td>\n",
       "      <td>77</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>Female</td>\n",
       "      <td>31</td>\n",
       "      <td>17</td>\n",
       "      <td>40</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   CustomerID  Gender  Age  Annual Income (k$)  Spending Score (1-100)\n",
       "0           1    Male   19                  15                      39\n",
       "1           2    Male   21                  15                      81\n",
       "2           3  Female   20                  16                       6\n",
       "3           4  Female   23                  16                      77\n",
       "4           5  Female   31                  17                      40"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df =pd.read_csv(\"Mall_Customers.csv\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "x =df.iloc[:, [3, 4]].values  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# finding optimul number of clstures using elbow method\n",
    "from sklearn.cluster import KMeans  \n",
    "wcss_list= []  #Initializing the list for the values of WCSS  \n",
    "  \n",
    "#Using for loop for iterations from 1 to 10.  \n",
    "for i in range(1, 11):  \n",
    "    kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42)  \n",
    "    kmeans.fit(x)   \n",
    "    wcss_list.append(kmeans.inertia_)  \n",
    "plt.plot(range(1, 11), wcss_list)  \n",
    "plt.title('The Elobw Method Graph')  \n",
    "plt.xlabel('Number of clusters(k)')  \n",
    "plt.ylabel('wcss_list')  \n",
    "plt.show()  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[1673624.5000000007,\n",
       " 440146.70193894394,\n",
       " 333387.2626572081,\n",
       " 251784.26317424315,\n",
       " 189924.58972877782,\n",
       " 157456.86162913853,\n",
       " 128233.34513242339,\n",
       " 99080.96695587024,\n",
       " 78123.16798641863,\n",
       " 68376.72415422738]"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "wcss_list"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    " # from this elbow plot we get to know that there will be formation of 2 clusters\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "#training the K-means model on a dataset  \n",
    "kmeans = KMeans(n_clusters=4, init='k-means++', random_state= 42)  \n",
    "y_predict= kmeans.fit_predict(x)  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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R4qa5itWrHh9wIABWQ8bcVNi77+BbwBbgK3A140xP8m2IHkjUeytF8X78MM2/3GeA09vvhl6907dJlpoyI94Wu2n+NDqPMVBYT+n98qtM5TfqBUDrNgP3neVRzhVmb3/+q//OmjMmDGfGTFixOirrrrqEIDXXnut1+GHH1533nnnHTZs2LC6z3/+88O3b98uiUrrDRo06IiZM2cePHbs2JG//e1v97/99tv7jxgxYvTw4cPrLrvssk8q/wwaNOiIaN7jq6+++lNDhw4dc/zxx49Yv369Z+s6Fre/Vl7GVv+5D/hYRIakbl5kxOYl7uqCPXu8lb676aZ9cxunyoMcALW1cO+9qQ3shx6ya6x+xNNqP8WHVucpDgr7Od16AGmK/KRHjO0nPcnK7C1btqzm9ddfr2xubl6zZs2a1S+//HJVU1NTNcCGDRsqr7jiig9ff/31ln79+nUuXLhw//jSetXV1QagsrKy64UXXnjti1/84vbrrrtu0FNPPbVu9erVLS+99FKfRYsW7Rd7zb/85S9V9913X/9XX3119cMPP/z6K6+80ieTUXBTz/Y7wAfAY8DDwHLntecR62rrJf64AMy5SMQqU5HuFmh1dXYMbK32U3xodZ7ioLCf06u99/U69srukO3HP3/84x9rnn766ZrRo0ePrqurGx2JRCrXrl1bCTBo0KDdxx9//E6Ao48+uu2tt95KaoFedNFFWwD++te/9jnuuONaDznkkI5wOMy55567OT5H8pNPPlk9fvz4rX379u3q379/1xe/+MWtmdyDG8v2SmCkMabOGFNvjDnCGFOfyUULknhXWy/k2ZyLFX3Hjr37w2FrqJ9xRuYGtlb7KT60Ok9xUNjP6V+eyshl2k+yMnvGGL773e9ujJbG27Bhw6qrrrrqI4BevXp9YhmVlZWZjo6OpD8OoiX03CZzEi8znGlwo2zfAXp2bqFMkgvDXnMuD0GoufISzme1H43t9YdW5ykOCvs51WQpG5S7fpKV2aupqelctGjRgG3btoUA3nzzzfB7772XUoHHltaL58QTT9zx3HPP9d24cWN5R0cH99xzT/+TTz65m9V06qmnbl++fPl+27dvly1btoQee+yx/dzcQzLcKNs3gKdE5BoR+V50y+SiBYcbV9tkRM25PAWh5spLOJXjthevZq9obK9/avvXsvScpVSFq/axnMKh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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#visulaizing the clusters  \n",
    "plt.scatter(x[y_predict == 0, 0], x[y_predict == 0, 1], s = 100, c = 'blue', label = 'Cluster 1') #for first cluster  \n",
    "plt.scatter(x[y_predict == 1, 0], x[y_predict == 1, 1], s = 100, c = 'green', label = 'Cluster 2') #for second cluster  \n",
    "plt.scatter(x[y_predict== 2, 0], x[y_predict == 2, 1], s = 100, c = 'red', label = 'Cluster 3') #for third cluster  \n",
    "plt.scatter(x[y_predict == 3, 0], x[y_predict == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4') #for fourth cluster  \n",
    "plt.scatter(x[y_predict == 4, 0], x[y_predict == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5') #for fifth cluster  \n",
    "plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroid')   \n",
    "plt.title('Clusters of customers')  \n",
    "plt.xlabel('Annual Income (k$)')  \n",
    "plt.ylabel('Spending Score (1-100)')  \n",
    "plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  \n",
    "plt.show()  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
       "       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
       "       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
       "       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
       "       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
       "       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 0, 3, 0, 3, 0, 3, 0, 3,\n",
       "       0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,\n",
       "       0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,\n",
       "       0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,\n",
       "       0, 3, 0, 3, 0, 3, 0, 3, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1])"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_predict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

# Import required libraries:
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from TE_features import XMEAS, XMV
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation


def load_te_data(data_file_name="all.csv'"):
    print(' Loading dataset ....')
    # Load the features names
    names = (XMEAS + XMV)
    # Add a columns for labels (classes)
    names.append("Process Faults")
    # Read the dataset file
    df = pd.read_csv(data_file_name, delimiter='\t', header=None, names=names)
    # Remove feature with same value for all the samples
    df = df.drop("Agitator speed",1)
    # Convert string labels to numbers
    df[df.columns[-1]] = df[df.columns[-1]].map(lambda element: '0'+ element[1:])
    df[df.columns[-1]] = df[df.columns[-1]].map(lambda element:np.argmax(np.array(list(element[:]))))
    # Create the features matrix and standardize data though mean removal and variance scaling
    observations = StandardScaler().fit_transform(df.values[:, 0: -1])
    # Create the target matrix (the column with classes)
    classes = np.ravel(pd.DataFrame(df.iloc[:, -1]).values)
    print('\t Dataset loaded ....')
    return observations, classes

# initialization function: plot the background of each frame
def init():
    ax.scatter([], [], [], marker=markers[0], c=colors[0], s=10)
    return ax,

# animation function.  This is called sequentially
def animate(time,fault_1, fault_2, fault_4, fault_6, markers, colors):
    ax.scatter(fault_1.values[10*time][0], fault_1.values[10*time][1], time, marker=markers[0], c=colors[fault_1_rfe.values[10*time][2]], s=10)
    ax.scatter(fault_2.values[10*time][0], fault_2.values[10*time][1], time, marker=markers[1], c=colors[fault_2_rfe.values[10*time][2]], s=10)
    ax.scatter(fault_4.values[10*time][0], fault_4.values[10*time][1], time, marker=markers[2], c=colors[fault_4_rfe.values[10*time][2]], s=10)
    ax.scatter(fault_6.values[10*time][0], fault_6.values[10*time][1], time, marker=markers[3], c=colors[fault_6_rfe.values[10*time][2]], s=10)
    ax.legend(("Normal","Fault 1", "Fault 2", "Fault 4", "Fault 6"))
    leg = ax.get_legend()
    leg.legendHandles[0].set_color(colors[0])
    leg.legendHandles[1].set_color(colors[1])
    leg.legendHandles[2].set_color(colors[2])
    leg.legendHandles[3].set_color(colors[4])
    leg.legendHandles[4].set_color(colors[6])
    return ax, leg

def add_plot_3d_timeseries(ax,data,data_target,t,feature1,feature2,targets, colors,marker):
    for target, color in zip(targets, colors):
        indicesToKeep = data_target == target
        ax.scatter(data.loc[indicesToKeep, feature1], data.loc[indicesToKeep, feature2]
                   , t[indicesToKeep], marker=marker, c=color, s=10)


if __name__ == "__main__":
    print("Executing __main__...")

    # 1) Load the dataset
    X1, y1 = load_te_data("fault_1.csv")
    X2, y2 = load_te_data("fault_2.csv")
    X4, y4 = load_te_data("fault_4.csv")
    X6, y6 = load_te_data("fault_6.csv")

    X = np.vstack((X1,X2,X4,X6))
    y = np.concatenate((y1, y2, y4, y6))
    n_samples = int(X.shape[0]/4)

    # Variables for ploting
    t = np.arange(0, X1.shape[0])
    targets = [0, 1, 2, 4, 6]
    colors = {0: 'tab:green', 1: 'tab:cyan', 2: 'tab:red', 4: 'tab:blue', 6: 'tab:orange'}
    markers = ['o', '^', 's', 'D']
    features_names = (XMEAS + XMV)

    # 2) Features selection using Recursive feature elimination (RFE)
    # Create the RFE object and rank each feature:
    print("Executing Features selection using Recursive feature elimination (RFE)")
    estimator = SVC(kernel="linear", C=1)
    selector = RFE(estimator=estimator, n_features_to_select=2, step=1)
    selector.fit(X, y)
    features_ranking = selector.ranking_
    print("The features ranking is =\n", features_ranking)
    # Discover what are de "n" best features
    n = 2
    selected_features = np.ravel(np.where(features_ranking < 3))
    print("Selected features for ploting are =", selected_features)
    # Take the data from seleted features
    data = np.zeros( (X.shape[0],2))
    data[:, 0] = X[:, selected_features[0]]
    data[:, 1] = X[:, selected_features[1]]
    rfeDf = pd.DataFrame(data=data, columns=['Selected feature 1', 'Selected feature 2'])
    rfeDf = pd.concat([rfeDf, pd.DataFrame(y, columns=['target'])], axis=1)
    fault_1_rfe = pd.DataFrame(rfeDf.loc[: n_samples-1])
    fault_2_rfe = pd.DataFrame(rfeDf.loc[n_samples: 2*n_samples-1])
    fault_4_rfe = pd.DataFrame(rfeDf.loc[2*n_samples: 3*n_samples-1])
    fault_6_rfe = pd.DataFrame(rfeDf.loc[3*n_samples: 4*n_samples-1])
    # Animate the timeseries
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(1, figsize=(10, 8))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=35, azim=107)
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=80,
                                   fargs=(fault_1_rfe, fault_2_rfe, fault_4_rfe, fault_6_rfe, markers, colors),
                                   interval=1,
                                   blit=True)
    # Create de axis captions
    ax.set_xlabel(features_names[selected_features[0]])
    ax.set_ylabel(features_names[selected_features[1]])
    ax.set_zlabel('t')
    ax.set_title('Selected Features with RFE', fontsize=10)
    # Show the animation
    plt.show()

    # 3) Dimensionality Reduction with PCA
    print("Executing Dimensionality Reduction with PCA")
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    pcaDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2'])
    pcaDf = pd.concat([pcaDf, pd.DataFrame(y, columns=['target'])], axis=1)
    fault_1_pca = pd.DataFrame(pcaDf.loc[: n_samples-1])
    fault_2_pca = pd.DataFrame(pcaDf.loc[n_samples: 2*n_samples-1])
    fault_4_pca = pd.DataFrame(pcaDf.loc[2*n_samples: 3*n_samples-1])
    fault_6_pca = pd.DataFrame(pcaDf.loc[3*n_samples: 4*n_samples-1])

    # Animate the timeseries
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(1, figsize=(10, 8))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=35, azim=107)

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=80,
                                   fargs=(fault_1_pca, fault_2_pca, fault_4_pca, fault_6_pca, markers, colors),
                                   interval=1,
                                   blit=True)
        # Create de axis captions
    ax.set_xlabel('Principal Component 1', fontsize=10)
    ax.set_ylabel('Principal Component 2', fontsize=10)
    ax.set_zlabel('t')
    ax.set_title('2 componente PCA', fontsize=10)
    # Show the animation
    plt.show()

    # 4) Dimensionality Reduction with t-SNE
    print("Executing Dimensionality Reduction with t-SNE...")
    tsne = TSNE(n_components= 2, n_iter=300).fit_transform(X)
    print("t-SNE Completed")
    tsneDf = pd.DataFrame(data=tsne , columns=['t-SNE component 1', 't-SNE component 2'])
    tsneDf = pd.concat([tsneDf, pd.DataFrame(y, columns=['target'])], axis=1)
    fault_1_tsne = pd.DataFrame(tsneDf.loc[: n_samples - 1])
    fault_2_tsne = pd.DataFrame(tsneDf.loc[n_samples: 2 * n_samples - 1])
    fault_4_tsne = pd.DataFrame(tsneDf.loc[2 * n_samples: 3 * n_samples - 1])
    fault_6_tsne = pd.DataFrame(tsneDf.loc[3 * n_samples: 4 * n_samples - 1])

    # Animate the timeseries
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(1, figsize=(10, 8))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=35, azim=107)
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=80,
                                   fargs=(fault_1_tsne, fault_2_tsne, fault_4_tsne, fault_6_tsne, markers, colors),
                                   interval=1,
                                   blit=True)
    # Create de axis captions
    ax.set_xlabel('t-SNE Component 1', fontsize=10)
    ax.set_ylabel('t-SNE Component 2', fontsize=10)
    ax.set_zlabel('t')
    ax.set_title('2 componente t-SNE', fontsize=10)
    # Show the animation
    plt.show()

    # 5) Radviz  multidimensional plot in 2D
    features_names.pop()
    df = pd.DataFrame(data=X, columns=features_names)
    df = pd.concat([df, pd.DataFrame(y, columns=['target'])], axis=1)
    rad_viz = pd.plotting.radviz(df, 'target')
    plt.show()

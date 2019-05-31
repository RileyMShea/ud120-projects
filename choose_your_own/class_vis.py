#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt


def pretty_picture(clf,
                   x_test,
                   y_test):
    """ Write a plot of a classifier and it's test data to file
    Parameters
    ----------
    clf : RandomForestClassifier or similar classifier object
        A fitted classification object
    x_test : numpy array or pandas series/dataframe
        The features test data
    y_test :  numpy array or pandas series/dataframe
        The labels test data

    Returns
    -------
    None

    """
    # initialize variables for plot
    x_min, x_max, y_min, y_max = 0., 1., 0., 1.

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    step_size = .01
    total_steps = int((x_max - x_min) / step_size + 1)
    x_mesh, y_mesh = np.meshgrid(np.linspace(x_min, x_max, total_steps),
                                 np.linspace(y_min, y_max, total_steps))
    # ravel - sort of like unzip
    Z = clf.predict(np.c_[x_mesh.ravel(), y_mesh.ravel()])



    # Put the result into a color plot
    Z = Z.reshape(x_mesh.shape)
    plt.xlim(x_mesh.min(), x_mesh.max())
    plt.ylim(y_mesh.min(), y_mesh.max())

    # plt.pcolormesh(x_mesh, y_mesh, Z, cmap=plt.get_cmap('PiYG'))
    plt.contourf(x_mesh, y_mesh, Z, alpha=1
                 )

    # Plot also the test points
    grade_sig = [x_test[ii][0] for ii in range(0, len(x_test)) if y_test[ii] == 0]
    bumpy_sig = [x_test[ii][1] for ii in range(0, len(x_test)) if y_test[ii] == 0]
    grade_bkg = [x_test[ii][0] for ii in range(0, len(x_test)) if y_test[ii] == 1]
    bumpy_bkg = [x_test[ii][1] for ii in range(0, len(x_test)) if y_test[ii] == 1]

    plt.scatter(grade_sig,
                bumpy_sig,
                # color="b",
                label="fast")

    plt.scatter(grade_bkg,
                bumpy_bkg,
                # color="r",
                label="slow")

    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")

    plt.savefig("test.png")
    plt.show()

from Tests.test_continuous import (
    test_continuous_3d_elbow,
    test_continuous_sideways_saddle,
    test_continuous_curve,
    test_continuous_unbounded_variation,
    test_continuous_periodic,
    test_continuous_polynomial
)

from Tests.test_autoencoder import (
    test_figlet_autoencoder
)

from Tests.test_dataset import (
    test_iris,
    test_boston,
    test_pums_multisource,
    test_pums
)

plot = True

if __name__ == '__main__':

    test_continuous_3d_elbow(plot=plot)
    # test_continuous_sideways_saddle(plot=plot)
    # test_continuous_curve(plot=plot)
    # test_continuous_unbounded_variation(plot=plot)
    # test_continuous_polynomial(plot=plot)
    # test_continuous_periodic(plot=plot)

    # test_pums(plot=plot)
    # test_pums_multisource(plot=plot)
    # test_boston(plot=plot)
    # test_iris(plot=plot)

    # test_figlet_autoencoder(plot=plot)

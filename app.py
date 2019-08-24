from Tests.test_continuous import (
    test_continuous_sideways_saddle,
    test_continuous_3d_elbow,
    test_continuous_periodic
)

from Tests.test_dataset import (
    test_pums,
    test_pums_multisource,
    test_boston
)

import numpy as np
np.set_printoptions(suppress=True, linewidth=10000)

# test_continuous_3d_elbow(plot=True)
# test_continuous_sideways_saddle(plot=True)
# test_continuous_periodic(plot=True)
# test_pums(plot=True)
# test_pums_multisource(plot=True)

test_boston(plot=True)

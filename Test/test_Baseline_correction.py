import pytest
import numpy as np

from Basic_Functions.Baseline_correction import (
    wavenumber,
    wavenumber_to_index,
    slicing_all_BD,
    baseline_correction,
    batch_baseline_correction,
    plot_baseline_correction
)

#----------------------------Testing baseline corrcection functions---------------------------------

@pytest.mark.parametrize("wavenumber, goal_wavenumber, expected_index", [(np.linspace(400, 4000, 1764), 400, 0), 
                                                                          (np.linspace(400, 4000, 1764), 4000, 1763), 
                                                                          (np.linspace(400, 4000, 1764), 2200, 882)])
def test_wavenumber_to_index(wavenumber, goal_wavenumber, expected_index):
    assert wavenumber_to_index(wavenumber, goal_wavenumber) == expected_index
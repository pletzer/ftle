#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm> // for std::min, std::max, std::floor

namespace py = pybind11;

// function: compute lower time index and parametric coordinate
py::tuple get_lower_time_index_and_param_coord(double time_val,
                                               py::array_t<double, py::array::c_style | py::array::forcecast> t_axis)
{
    auto t = t_axis.unchecked<1>();  // 1D array
    double t_axis_min = t(0);
    double dt = t(1) - t(0);
    ssize_t nt = t_axis.shape(0);

    // compute time index
    ssize_t t_index = static_cast<ssize_t>(std::floor((time_val - t_axis_min) / dt));
    t_index = std::max(ssize_t(0), std::min(t_index, nt - 2));

    // compute parametric coordinate
    double mu = (time_val - t(t_index)) / dt;
    mu = std::max(0.0, std::min(mu, 1.0));

    return py::make_tuple(t_index, mu);
}

// Bind to Python
PYBIND11_MODULE(time_index, m) {
    m.def("get_lower_time_index_and_param_coord", &get_lower_time_index_and_param_coord,
          py::arg("time_val"), py::arg("t_axis"));
}


#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>           // mandatory for myPyObject.cast<std::vector<T>>()
#include <pybind11/functional.h>    // mandatory for py::cast( std::function )

// test dlib
// #include <dlib/optimization.h>
// #include <dlib/global_optimization.h>

namespace py = pybind11;


int main()
{    


    return 0;
    py::scoped_interpreter guard{};

    py::module np = py::module::import("numpy");
    py::module scipy = py::module::import("scipy.optimize");
    py::module plt = py::module::import("matplotlib.pyplot");
    py::module customfunc = py::module::import("module");

    py::function curve_fit = scipy.attr("curve_fit");
    py::function func_exp_F = customfunc.attr("func_exp_F");


    py::array_t<double> x = np.attr("arange")(-10, 10, 0.1);    
    py::array_t<double> y = func_exp_F(x, 0.1, 2);
    py::array_t<double> noise = py::float_(5*1e5) * np.attr("random").attr("normal")( 0, 1, x.attr("size"));
    y += noise;
    
        
    plt.attr("plot")(x, y);

    py::object retVals = curve_fit(func_exp_F, x, y);
    py::object optVals = retVals.attr("__getitem__")(0);

    std::vector<double> retValsStd = optVals.cast<std::vector<double>>();
    std::cout << "Fitted parameter a = " << retValsStd[0] <<  ", b=" << retValsStd[1] << std::endl;
    py::array_t<double> yp = func_exp_F(x, retValsStd[0], retValsStd[1] );

    plt.attr("plot")(x, yp);
    plt.attr("show")();
    return 0;
}
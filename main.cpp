#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>           // mandatory for myPyObject.cast<std::vector<T>>()
#include <pybind11/functional.h>    // mandatory for py::cast( std::function )


#include "curve_fit.hpp"
#include <random>


// test dlib
// #include <dlib/optimization.h>
// #include <dlib/global_optimization.h>

namespace py = pybind11;


template <typename Container>
auto linspace(typename Container::value_type a, typename Container::value_type b, size_t n)
{
    assert(b > a);
    assert(n > 1);

    Container res(n);
    const auto step = (b - a) / (n - 1);
    auto val = a;
    for(auto& e: res)
    {
        e = val;
        val += step;
    }
    return res;
}


double gaussian(double x, double a, double b, double c)
{
    const double z = (x - b) / c;
    return a * std::exp(-0.5 * z * z);
}

double exp_f(double x, double a, double b){
    return a * std::exp(b * x) - a;
}

int main()
{
    py::scoped_interpreter guard{};

    py::module np = py::module::import("numpy");
    py::module scipy = py::module::import("scipy.optimize");
    py::module plt = py::module::import("matplotlib.pyplot");
    py::module customfunc = py::module::import("module");

    py::function curve_fit_python = scipy.attr("curve_fit");
    py::function func_exp_F = customfunc.attr("func_exp_F");


    std::cout << "Try GSL method" << std::endl;

    auto device = std::random_device();
    auto gen    = std::mt19937(device());

    py::array_t<double> x = np.attr("arange")(-10, 10, 0.1);    
    auto xs = x.cast<std::vector<double>>();
    auto ys = std::vector<double>(xs.size());

    // Fill ys using gaussian function
    double a = 0.1, b = 2.0, c = 0.15;
    for(size_t i = 0; i < xs.size(); i++)
    {
        auto y =  exp_f(xs[i], a, b);
        auto dist  = std::normal_distribution(0.0, 0.1 * y);
        ys[i] = y + dist(gen);
    }

    // py::array_t<double> x = py::cast(xs);
    py::array_t<double> y = py::cast(ys);

    plt.attr("plot")(x, y);
    auto r = curve_fit(exp_f, {0.1, 5.0}, xs, ys);

    std::cout << "result: " << r[0] << ' ' << r[1] << std::endl;
    std::cout << "error : " << r[0] - a << ' ' << r[1] - b  << std::endl;

    // auto ypv = std::vector<double>(xs.size());
    // for(size_t i = 0; i < xs.size(); i++)
    // {
    //     auto y =  gaussian(xs[i], r[0], r[1], r[2]);
        
    //     ypv[i] = y;
    // }
    // auto yp = py::cast(ypv);

    // plt.attr("plot")(x, yp);

    plt.attr("show")();
    
    return 0;


    
    // py::array_t<double> y = func_exp_F(x, 0.1, 2);
    // py::array_t<double> noise = py::float_(5*1e5) * np.attr("random").attr("normal")( 0, 1, x.attr("size"));
    // y += noise;
    
        
    // plt.attr("plot")(x, y);

    // py::object retVals = curve_fit(func_exp_F, x, y);
    // py::object optVals = retVals.attr("__getitem__")(0);

    // std::vector<double> retValsStd = optVals.cast<std::vector<double>>();
    // std::cout << "Fitted parameter a = " << retValsStd[0] <<  ", b=" << retValsStd[1] << std::endl;
    // py::array_t<double> yp = func_exp_F(x, retValsStd[0], retValsStd[1] );

    // plt.attr("plot")(x, yp);
    // plt.attr("show")();
    return 0;
}
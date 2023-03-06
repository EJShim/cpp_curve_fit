#include <iostream>
#include <vector>
#include <ceres/ceres.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>           // mandatory for myPyObject.cast<std::vector<T>>()
#include <pybind11/functional.h>    // mandatory for py::cast( std::function )          // mandatory for myPyObject.cast<std::vector<T>>()

const int kNumObservations = 67;

std::vector<double> dataX = {
	0.000000e+00,
	7.500000e-02,
	1.500000e-01,
	2.250000e-01,
	3.000000e-01,
	3.750000e-01,
	4.500000e-01,
	5.250000e-01,
	6.000000e-01,
	6.750000e-01,
	7.500000e-01,
	8.250000e-01,
	9.000000e-01,
	9.750000e-01,
	1.050000e+00,
	1.125000e+00,
	1.200000e+00,
	1.275000e+00,
	1.350000e+00,
	1.425000e+00,
	1.500000e+00,
	1.575000e+00,
	1.650000e+00,
	1.725000e+00,
	1.800000e+00,
	1.875000e+00,
	1.950000e+00,
	2.025000e+00,
	2.100000e+00,
	2.175000e+00,
	2.250000e+00,
	2.325000e+00,
	2.400000e+00,
	2.475000e+00,
	2.550000e+00,
	2.625000e+00,
	2.700000e+00,
	2.775000e+00,
	2.850000e+00,
	2.925000e+00,
	3.000000e+00,
	3.075000e+00,
	3.150000e+00,
	3.225000e+00,
	3.300000e+00,
	3.375000e+00,
	3.450000e+00,
	3.525000e+00,
	3.600000e+00,
	3.675000e+00,
	3.750000e+00,
	3.825000e+00,
	3.900000e+00,
	3.975000e+00,
	4.050000e+00,
	4.125000e+00,
	4.200000e+00,
	4.275000e+00,
	4.350000e+00,
	4.425000e+00,
	4.500000e+00,
	4.575000e+00,
	4.650000e+00,
	4.725000e+00,
	4.800000e+00,
	4.875000e+00,
	4.950000e+00
};

std::vector<double> dataY = {
	1.133898e+00,
	1.334902e+00,
	1.213546e+00,
	1.252016e+00,
	1.392265e+00,
	1.314458e+00,
	1.472541e+00,
	1.536218e+00,
	1.355679e+00,
	1.463566e+00,
	1.490201e+00,
	1.658699e+00,
	1.067574e+00,
	1.464629e+00,
	1.402653e+00,
	1.713141e+00,
	1.527021e+00,
	1.702632e+00,
	1.423899e+00,
	1.543078e+00,
	1.664015e+00,
	1.732484e+00,
	1.543296e+00,
	1.959523e+00,
	1.685132e+00,
	1.951791e+00,
	2.095346e+00,
	2.361460e+00,
	2.169119e+00,
	2.061745e+00,
	2.178641e+00,
	2.104346e+00,
	2.584470e+00,
	1.914158e+00,
	2.368375e+00,
	2.686125e+00,
	2.712395e+00,
	2.499511e+00,
	2.558897e+00,
	2.309154e+00,
	2.869503e+00,
	3.116645e+00,
	3.094907e+00,
	2.471759e+00,
	3.017131e+00,
	3.232381e+00,
	2.944596e+00,
	3.385343e+00,
	3.199826e+00,
	3.423039e+00,
	3.621552e+00,
	3.559255e+00,
	3.530713e+00,
	3.561766e+00,
	3.544574e+00,
	3.867945e+00,
	4.049776e+00,
	3.885601e+00,
	4.110505e+00,
	4.345320e+00,
	4.161241e+00,
	4.363407e+00,
	4.161576e+00,
	4.619728e+00,
	4.737410e+00,
	4.727863e+00,
	4.669206e+00
};

template <typename T>
T innerFunction(double x, const T a, const T b ){
	return exp(a * x + b);
}


struct ExponentialResidual {
	ExponentialResidual(double x, double y) : x_(x), y_(y) {}

	template <typename T>
	bool operator()(const T* const a, const T* const b, T* residual) const {
		// residual[0] = y_ - exp(a[0] * x_ + b[0]);
		residual[0] = y_ - innerFunction(x_, a[0], b[0]);
		
		// residual[0] = innerFunction(x_, y_, a, b);
		return true;
	}

	private:
	// Observations for a sample.
	const double x_;
	const double y_;
};

namespace py = pybind11;


int main(){
	py::scoped_interpreter guard{};

    py::module np = py::module::import("numpy");
    py::module plt = py::module::import("matplotlib.pyplot");

	py::array_t<double> xx = py::cast(dataX);
	py::array_t<double> yy = py::cast(dataY);

	plt.attr("plot")(xx, yy);	

	std::cout << "Data conversion" << std::endl;


		
	double m = 0.0;
	double c = 0.0;


	std::cout << "hello world" << std::endl;

	ceres::Problem problem;
	
	for (int i = 0; i < dataX.size(); ++i) {
		ceres::CostFunction* cost_function =new ceres::AutoDiffCostFunction<ExponentialResidual, 1, 1, 1>(new ExponentialResidual(dataX[i], dataY[i]));		
		problem.AddResidualBlock(cost_function, nullptr, &m, &c);
	}

	ceres::Solver::Options options;
	options.max_num_iterations = 25;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	std::cout << summary.BriefReport() << "\n";
	std::cout << "Initial m: " << 0.0 << " c: " << 0.0 << "\n";
	std::cout << "Final   m: " << m << " c: " << c << "\n";


	std::cout << "hello world" << std::endl;


	std::vector<double> yp;
	for (int i = 0; i < dataX.size(); ++i) {
		yp.push_back( innerFunction(dataX[i], m, c));		
	}
	py::array_t<double> yyp = py::cast(yp);
	plt.attr("plot")(xx, yyp);
	plt.attr("show")();

	return 0;
}
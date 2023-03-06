#include <iostream>
#include <vector>
#include <ceres/ceres.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>           // mandatory for myPyObject.cast<std::vector<T>>()
#include <pybind11/functional.h>    // mandatory for py::cast( std::function )          // mandatory for myPyObject.cast<std::vector<T>>()


template <typename T>
T curveFunc(double x, const T a, const T b ){
	return exp(a * x + b);
}



template <typename T>
T exp_curveFunc(double x, const T a, const T b ){
	return a * exp(b * x) - a;
}

struct ExponentialResidual {
	ExponentialResidual(double x, double y) : x_(x), y_(y) {}

	template <typename T>
	bool operator()(const T* const a, const T* const b, T* residual) const {		
		residual[0] = y_ - exp_curveFunc(x_, a[0], b[0]);
		
		return true;
	}

	private:
	// Observations for a sample.
	const double x_;
	const double y_;
};

namespace py = pybind11;


int main(){

	double gt_a = 0.1;
	double gt_c = 2.0;
	py::scoped_interpreter guard{};

    py::module np = py::module::import("numpy");
    py::module plt = py::module::import("matplotlib.pyplot");

	py::array_t<double> xx = np.attr("arange")(-10, 10, 0.1);    
	std::vector<double> dataX = xx.cast<std::vector<double>>();
	std::vector<double> dataY(dataX.size());
	for(int i=0 ; i<dataX.size() ; i++){
		dataY[i] = exp_curveFunc(dataX[i], gt_a, gt_c);
	}	
	py::array_t<double> yy = py::cast(dataY);
	py::array_t<double> noise = py::float_(5.0 * 1e5) * np.attr("random").attr("normal")( 0, 1, yy.attr("size"));
	yy = yy + noise;

	dataY = yy.cast<std::vector<double>>();

	plt.attr("plot")(xx, yy);	

	std::cout << "Data conversion" << std::endl;
	// plt.attr("show")();
	// return 0;

		
	double m = 1e-6;
	double c = 1e-6;


	std::cout << "hello world" << std::endl;

	ceres::Problem problem;
	
	for (int i = 0; i < dataX.size(); ++i) {
		ceres::CostFunction* cost_function =new ceres::AutoDiffCostFunction<ExponentialResidual, 1, 1, 1>(new ExponentialResidual(dataX[i], dataY[i]));		
		problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(0.5), &m, &c);
	}

	ceres::Solver::Options options;
	options.max_num_iterations = 25;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	std::cout << summary.BriefReport() << "\n";
	std::cout << "GT m: " << gt_a << " c: " << gt_c << std::endl;
	std::cout << "Final   m: " << m << " c: " << c << std::endl;


	std::cout << "hello world" << std::endl;


	std::vector<double> yp;
	for (int i = 0; i < dataX.size(); ++i) {
		yp.push_back( exp_curveFunc(dataX[i], m, c));		
	}
	py::array_t<double> yyp = py::cast(yp);
	plt.attr("plot")(xx, yyp);
	plt.attr("show")();

	return 0;
}
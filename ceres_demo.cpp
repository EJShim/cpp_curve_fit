#include <iostream>


struct ExponentialResidual {
    ExponentialResidual(double x, double y) : x_(x), y_(y) {}
  
    template <typename T>
    bool operator()(const T* const m, const T* const c, T* residual) const {
      residual[0] = y_ - exp(m[0] * x_ + c[0]);
      return true;
    }
  
   private:
    // Observations for a sample.
    const double x_;
    const double y_;
  };

int main(){
    double m = 0.0;
    double c = 0.0;
    
    // Problem problem;


    std::cout << "hello world" << std::endl;
    return 0;
}
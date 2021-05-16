//
//  main.cpp
//  AutoDiff
//
//  Created by Don Kushan Nilanga Athalage on 14/5/21.
//

#include <iostream>
#include "tensor.hpp"
#include <vector>

int main(int argc, const char * argv[]) {
    // a
    std::vector<float> data_a = {1.0};
    tensor *a = new tensor(&data_a, true);
    
    // x
    std::vector<float> data_x = {2.0};
    tensor *x = new tensor(&data_x, true);
    
    // b
    std::vector<float> data_b = {3.0};
    tensor *b = new tensor(&data_b, true);
    
    // operation
    tensor *y = *(*a * *x) + *b;
    
    // grad
    std::vector<float> data_d = {5.0};
    tensor *d = new tensor(&data_d, false);
    
    y->backward(d);
    return 0;
}


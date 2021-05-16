//
//  tensor.cpp
//  AutoDiff
//
//  Created by Don Kushan Nilanga Athalage on 16/5/21.
//

#include <stdio.h>
#include "tensor.hpp"


/*
 Addition Operation
 
 y = ( a + b ) * f(x)
 (a) dy/da = 1 * f(x)
 (b) dy/db = 1 * f(x)
 */
tensor* tensor::operator+(tensor &b){
    
    // calculation
    std::vector<float> *result_data = new std::vector<float>();
    for(int i = 0; i < this->data.size(); i++){ result_data->push_back(this->data.at(i) + b.data.at(i));}

    // setting up output tensor
    auto dependencies = new std::vector<tensor*>();
    dependencies->push_back(this);
    dependencies->push_back(&b);
    return new tensor(result_data, this->requires_grad | b.requires_grad, dependencies, ADD);
}



/*
 Multiplication operation
 
 y = a * b * (fx)
 
 (a) dy/da = b * f(x)
 (b) dy/db = a * f(x)
 
 */
tensor* tensor::operator*(tensor &b){
    
    // calculation
    std::vector<float> *result_data = new std::vector<float>();
    for(int i = 0; i < this->data.size(); i++){ result_data->push_back(this->data.at(i) * b.data.at(i));}
    
    // setting up output tensor
    auto dependencies = new std::vector<tensor*>();
    dependencies->push_back(this);
    dependencies->push_back(&b);
    return new tensor(result_data, this->requires_grad | b.requires_grad, dependencies, MUL);
}

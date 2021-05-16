//
//  tensor.hpp
//  AutoDiff
//
//  Created by Don Kushan Nilanga Athalage on 14/5/21.
//

#ifndef tensor_hpp
#define tensor_hpp

#include <ostream>
#include <iostream>
#include <vector>

enum GRAD_FN{
    NAN = -1,
    ADD,
    SUB,
    MUL,
    DIV
};

// Tensor data structure
class tensor{
public:
    std::vector<float> &data;
    std::vector<tensor*> *dependencies;
    bool requires_grad;
    GRAD_FN grad_fn;
    tensor *grad;
    
    tensor(std::vector<float> *data, bool requires_grad) : data(*data), requires_grad(requires_grad){
        grad_fn = NAN;
        if(requires_grad){
            grad = new tensor(new std::vector<float>(), false);
            
            // populate data
            for(int i=0; i < data->size(); i++){
                grad->data.push_back(0);
            }
        }
    }
    
    tensor(std::vector<float> *data, bool requires_grad, std::vector<tensor*> *dependencies, GRAD_FN grad_func):
    data(*data),
    requires_grad(requires_grad),
    dependencies(dependencies),
    grad_fn(grad_func)
    {
        grad_fn = NAN;
        if(requires_grad){
            grad = new tensor(new std::vector<float>(), false);
            
            // populate data
            for(int i=0; i < data->size(); i++){
                grad->data.push_back(0);
            }
        }
    }
    
    // operators +
    tensor* operator+(tensor &b);
    
    // operators *
    tensor* operator*(tensor &b);
    
    
    void backward(tensor* grad){
        if(this->grad_fn==NAN){return;}
        switch (this->grad_fn) {
            case ADD:
                
                // assigning gradient
                if(this->dependencies->at(0)->requires_grad){
                    for(int i = 0; i < grad->data.size(); i++){
                        this->dependencies->at(0)->grad->data.at(i) = grad->data.at(i);
                    }
                }
                
                if(this->dependencies->at(1)->requires_grad){
                    for(int i = 0; i < grad->data.size(); i++){
                        this->dependencies->at(1)->grad->data.at(i) = grad->data.at(i);
                    }
                }
                break;
                
            case MUL:
                if(this->dependencies->at(0)->requires_grad){
                    for(int i = 0; i < grad->data.size(); i++){
                        this->dependencies->at(0)->grad->data.at(i) = grad->data.at(i) * this->dependencies->at(1)->data.at(i);
                    }
                }
                
                if(this->dependencies->at(1)->requires_grad){
                    for(int i = 0; i < grad->data.size(); i++){
                        this->dependencies->at(1)->grad->data.at(i) = grad->data.at(i) * this->dependencies->at(0)->data.at(i);
                    }
                }
                break;
            default:
                break;
        }
        
        for(int i = 0; i < this->dependencies->size(); i++){
            if(this->dependencies->at(i)->requires_grad)
                this->dependencies->at(i)->backward(this);
        }
        
    }
};


#endif

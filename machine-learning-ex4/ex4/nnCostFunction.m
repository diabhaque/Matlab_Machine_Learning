function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X=[ones(m,1) X];

a2=sigmoid(Theta1*X');

a2=[ones(m,1) a2'];

ypred=sigmoid(Theta2*a2');

ynew=zeros(m, num_labels);

for i=1:m
    for j=1:num_labels
        ynew(i, j)=0;
        if j==y(i)
            ynew(i,j)=1;
        end    
    end
end

ynew2=ynew';
err= ypred-ynew2;

for i=1:m
    for j=1:num_labels
        if(ynew2(j,i)==1)
            err(j,i)=-log(ypred(j,i));
        end
        
        if(ynew2(j,i)==0)
            err(j,i)=-log(1-ypred(j,i));
        end
        
    end
end

x= ones(size(err,1), 1);
innerSum=x'*err;
J=(1/m)*(ones(size(err,2), 1)'*innerSum');

T1=Theta1(:, 2:end).^2;
T2=Theta2(:, 2:end).^2;

regCostT1= ones(1,hidden_layer_size)*(T1*ones(input_layer_size, 1));
regCostT2= ones(1,num_labels)*(T2*ones(hidden_layer_size, 1));

J=J+(lambda/(2*m))*(regCostT1+regCostT2);
% -------------------------------------------------------------
delta2=zeros(num_labels, hidden_layer_size+1);
delta1=zeros(hidden_layer_size, input_layer_size+1);

for i=1:m
    z2=Theta1 * X(i, :)';
    l2=sigmoid(z2);
    l2new=[1; l2];
    l3=sigmoid(Theta2 * l2new);
    del3=l3-ynew(i,:)';
    del2=(Theta2'*del3).*sigmoidGradient([1;z2]);
    del2new=del2(2: end, 1);
    delta2=delta2+del3*l2new';
    delta1=delta1+del2new*X(i,:);
end

regTheta1=(lambda/m)*Theta1;
regTheta1=[zeros(size(Theta1, 1), 1) regTheta1(:,2:end)];

regTheta2=(lambda/m)*Theta2;
regTheta2=[zeros(size(Theta2, 1), 1) regTheta2(:,2:end)];

Theta2_grad=(1/m)*delta2+regTheta2;
Theta1_grad=(1/m)*delta1+regTheta1;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

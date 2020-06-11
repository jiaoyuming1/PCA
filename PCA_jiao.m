clc;
clear all;
close all;
%>>>>>>-------------------1st.手动输入数据和维数---------------------<<<<<<<

% input为输入样本，k为降到k维
input = [4.9 4.7 4.6 5.0 5.4;3.0 3.2 3.1 3.6 3.9;
    1.4 1.3 1.5 1.4 1.7;0.2 0.2 0.2 0.2 0.4]';
k = 3;

%-------------------------------------------------------------------<<<<<<<


%>>>>>>-------------------2nd.实验准备-------------------------------<<<<<<<

W = [];
X = input;
[n, d] = size(X);
% 检错
if k>d
    error(['欲降维度大于初始维度']);
end
% epsilon = 1E-10; 修正算法
%-------------------------------------------------------------------<<<<<<<


%>>>>>>-------------------3rd.PCA-L1迭代求出基函数W------------------<<<<<<<
tic;%计时
%% PCA-L1算法复现 更新v，添加到W。
for i = 1:k
    
    if i ~= 1
        X = X - X * W(:,i-1) * W(:,i-1)';  % X - WV
    end
    % 初始化w0(v)  w(0)/||w(0)||2
    % 对x的第j行求2范数,得到x_norm数组中存着4个维度的二范数,也就是A*A'的最大特征值的平方
    for j = 1:n
        x_norm2(j) = norm(X(j,:));
    end
    [x_norm2sort, index] = sort(x_norm2);  % 对特征值进行从小到大排序
    v = X(index(end),:)' / x_norm2sort(end); % v=对应x的第end行'除以最大特征值
    v_prev = zeros(d,1);   
    % 更新迭代 v，直到全都相等且没有0
    % while ( norm(v-v_prev) > epsilon)  %修正的PCA-L1算法 设定阈值
    while ( v ~= v_prev)
        
        v_prev = v;
   
        % 极性检测 翻转和最大化
        v_t = zeros(1,d);
        for j = 1:n
            if X(j,:) * v < 0
                p(j) = -1;
            else
                p(j) = 1;
            end
            v_t = v_t + p(j)*X(j,:);     
        end
        v_t2 = sqrt(sum(v_t.*v_t));   % 和的平方，再求和，再根号，以二范
        % v = v_t'+randi([0,1],1,1)/v_t2+randi([0,1],1,1);
        v = v_t'/v_t2;          % 五个样本每个样本维度的和 x'v x = 1求出v   
    end
    W = [W, v];    %把新求出的v 添加到最终W中    
end

%-------------------------------------------------------------------<<<<<<<


%>>>>>>-------------------4th.输出结果-------------------------------<<<<<<<
V = W'* input'; % 为了计时对比 放到了前面
toc;

%% 输出基函数和输入的数据
disp('-----------------------PCA-L1-----------------------------')
t_pcal1 = toc;
disp('输入数据 X：')
disp(input)
disp(['降低到的超平面维数：',num2str(k)])
disp('基函数 w：')
disp(W)

disp('-----------------------未中心化-----------------------------')

%% 非中心化的数据处理降维
% 误差求解  推导方法：||X-WV||= ||X-WW'X||

disp('降维后的数据V：')

disp(V)
X_k = W * V;
B = X_k - input';
error_jiao = sum(sum(abs(B))); %自创误差 所有数相加误差
error = norm(B,1);
disp('PCA-L1 ||x-x^||1误差：')
disp(error)
disp('PCA-L1焦氏误差')   %---为了和L2-PCA方法对比误差
disp(error_jiao)
% 降维数据 公式推导方法：W*V = W*W'*X' ----V = W'*X'



disp('-----------------------中心化后-----------------------------')

%% 中心化后的数据处理降维
meanX=mean(input,1); 
tempX= input - repmat(meanX,n,1);
% 投影到基向量空间
X__k = W * W' *tempX';
% Y(:,[1])=[];  %去掉第一列的0向量
% disp('误差')  %中心化误差跟不中心化一样 就不显示了
A = X__k - tempX';
% error1_jiao = sum(sum(abs(A))); 所有数相加的误差
error1 = norm(A,1);
% disp(error1)
disp('中心化降维后的数据V：')
V_nor = W'* tempX';
disp(V_nor)

%-------------------------------------------------------------------<<<<<<<


%>>>>>>-------------------5th.L2-PCA对照实验-------------------------<<<<<<<

disp('-----------------------L2-PCA-----------------------------')
%% 自己实现PCA的方法
tic;

covX=cov(input);                                    %求样本的协方差矩阵（散步矩阵除以(n-1)即为协方差矩阵）
[V D]=eigs(covX);                               %求协方差矩阵的特征值D和特征向量V
meanX_=mean(input);                                  %样本均值m
%所有样本X减去样本均值m，再乘以协方差矩阵（散步矩阵）的特征向量V，即为样本的主成份SCORE
tempX_= input - repmat(meanX_,n,1);
all2=(tempX_)*V;                              %主成份：SCORE
pcaData2=all2(:,1:k);
toc;
t_pcal2 = toc;
disp('PCA降维后的数据V：')
disp(pcaData2')
C = tempX_' - V(:,1:k)*pcaData2';
error_L2jiao = sum(sum(abs(C))); 
error_L2 = norm(C);
disp('L2-PCA ||x-x^||2误差:')
disp(error_L2)
disp('L2-PCA焦氏误差：')
disp(error_L2jiao)

%-------------------------------------------------------------------<<<<<<<


%% 备用代码草稿
%x = pcaData2(:,1);
%y = pcaData2(:,2);
%z = pcaData2(:,3);
%plot3(x,y,z)
%pX = [0.0807 0.1942 -0.3015;0.0491 -0.1002 -0.2953;0.1584 -0.0202 -0.3841;
%    0.0779 0.1382 0.2074;-0.0494 0.0645 0.7736]';
%  error =  input    -    input * W * W'      XWW'=WV
% X = WV V是求得向量  W * W’ * input'    v = w'*input'
%x = V_nor(1,:);
%y = V_nor(2,:);
%z = V_nor(3,:);
%plot(x,y)
%plot3(x,y,z)
%P = [0.4896 0.6230 0.6016;-0.863 -0.6579 0.7478 ; -0.8361 0.3827 0.2308; -0.2320 0.1804 0.1600];

%C = tempX' - P*pX;
%error_L2jiao = sum(sum(abs(C))); 
%error_L2 = norm(C);
%disp('L2-PCA ||x-x^||2误差:')
%disp(error_L2)
%disp('L2-PCA焦氏误差：')
%disp(error_L2jiao)









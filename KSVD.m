function [Dictionary,output] = KSVD(...
    Data,... % an nXN matrix that contins N signals (Y), each of dimension n.
    param)
% =========================================================================
%                          K-SVD algorithm
% =========================================================================
% The K-SVD algorithm finds a dictionary for linear representation of
% signals. Given a set of signals, it searches for the best dictionary that
% can sparsely represent each signal. Detailed discussion on the algorithm
% and possible applications can be found in "The K-SVD: An Algorithm for 
% Designing of Overcomplete Dictionaries for Sparse Representation", written
% by M. Aharon, M. Elad, and A.M. Bruckstein and appeared in the IEEE Trans. 
% On Signal Processing, Vol. 54, no. 11, pp. 4311-4322, November 2006. 
% 给一组信号，它将搜索可以稀疏表示每个信号的最佳字典
% =========================================================================
% INPUT ARGUMENTS:
% Data                         an nXN matrix that contins N signals (Y), each of dimension n. 
% n乘N的矩阵，N个信号，每个信号长度为n
% param                        structure that includes all required
%                                 parameters for the K-SVD execution.
%                                 Required fields are:
%    K, ...                    the number of dictionary elements to train要训练的字典元素的数量
%    numIteration,...          number of iterations to perform迭代次数
%    errorFlag...              if =0, a fix number of coefficients is
%                                 used for representation of each signal. If so, param.L must be
%                                 specified as the number of representing atom. if =1, arbitrary number
%                                 of atoms represent each signal, until a specific representation error
%                                 is reached. If so, param.errorGoal must be specified as the allowed
%                                 error.如果= 0，则使用固定数量的系数表示每个信号。 如果是这样，则必须将param.L指定为表示原子的数目。 如果= 1，则任意数量的原子表示每个信号，直到达到特定的表示误差为止。 如果是这样，必须将param.errorGoal指定为允许的错误。
%    preserveDCAtom...         if =1 then the first atom in the dictionary
%                                 is set to be constant, and does not ever change. This
%                                 might be useful for working with natural
%                                 images (in this case, only param.K-1
%                                 atoms are trained).如果= 1，则字典中的第一个原子设置为常数，并且永远不变。 这对于处理自然图像可能很有用（在这种情况下，仅训练K-1参数原子）。
%    (optional, see errorFlag) L,...                 % maximum coefficients to use in OMP coefficient calculations.OMP系数计算中使用的最大系数。
%    (optional, see errorFlag) errorGoal, ...        % allowed representation error in representing each signal.表示每个信号时允许出现表示错误。
%    InitializationMethod,...  mehtod to initialize the dictionary, can
%                                 be one of the following arguments: 
%                                 * 'DataElements' (initialization by the signals themselves), or: 
%                                 * 'GivenMatrix' (initialization by a given matrix param.initialDictionary).
%    (optional, see InitializationMethod) initialDictionary,...      % if the initialization method 
%                                 is 'GivenMatrix', this is the matrix that will be used.
%    (optional) TrueDictionary, ...        % if specified, in each
%                                 iteration the difference between this dictionary and the trained one
%                                 is measured and displayed.
%    displayProgress, ...      if =1 progress information is displyed. If param.errorFlag==0, 
%                                 the average repersentation error (RMSE) is displayed, while if 
%                                 param.errorFlag==1, the average number of required coefficients for 
%                                 representation of each signal is displayed.
% =========================================================================
% OUTPUT ARGUMENTS:
%  Dictionary                  The extracted dictionary of size nX(param.K).提取的大小为nX（param.K）的字典。
%  output                      Struct that contains information about the current run. It may include the following fields:
%    CoefMatrix                  The final coefficients matrix (it should hold that Data equals approximately Dictionary*output.CoefMatrix.最终系数矩阵（应该保持Data大约等于Dictionary * output.CoefMatrix。
%    ratio                       If the true dictionary was defined (in
%                                synthetic experiments), this parameter holds a vector of length
%                                param.numIteration that includes the detection ratios in each
%                                iteration).如果定义了真正的字典（在合成实验中），则此参数将保存一个长度为param.numIteration的向量，其中包括每次迭代中的检测率。
%    totalerr                    The total representation error after each
%                                iteration (defined only if
%                                param.displayProgress=1 and
%                                param.errorFlag = 0)每次迭代后的总表示错误（仅在param.displayProgress = 1和param.errorFlag = 0时定义）
%    numCoef                     A vector of length param.numIteration that
%                                include the average number of coefficients required for representation
%                                of each signal (in each iteration) (defined only if
%                                param.displayProgress=1 and
%                                param.errorFlag = 1)一个长度为param.numIteration的向量，其中包括表示每个信号（在每个迭代中）所需的平均系数数（仅在param.displayProgress = 1和param.errorFlag = 1时定义）
% =========================================================================

% isfield(param,'displayProgress'):表示的是param中是否含有displayPrograess，如果含有则返回1，没有则返回0
% 原来的程序中含有param.displayProgress = displayFlag;%displayFlag = 1;  所以此句也不会执行
if (~isfield(param,'displayProgress'))
    param.displayProgress = 0;
end
totalerr(1) = 99999;%代表的累积误差
% param.errorFlag = 1;   此句也不会执行
if (isfield(param,'errorFlag')==0)
    param.errorFlag = 0;
end

% param中没有TrueDictionary
if (isfield(param,'TrueDictionary'))
    displayErrorWithTrueDictionary = 1;
    ErrorBetweenDictionaries = zeros(param.numIteration+1,1);
    ratio = zeros(param.numIteration+1,1);
else
    displayErrorWithTrueDictionary = 0;
	ratio = 0;%参数包含一个长度为param.numIteration的矢量，包含每个中的检测比率
end
if (param.preserveDCAtom>0)%字典中的第一个原子被设定为恒定的，并且不会改变。这个可能对于处理自然图像有用（在这种情况下，只有param.K-1原子被训练）。
    FixedDictionaryElement(1:size(Data,1),1) = 1/sqrt(size(Data,1));
% 最后param.preserveDCAtom=0
else
    FixedDictionaryElement = [];%固定字典元素
end
% coefficient calculation method is OMP with fixed number of coefficients 系数计算方法是固定数目的系数的OMP

if (size(Data,2) < param.K)
    disp('Size of data is smaller than the dictionary size. Trivial solution...');
    Dictionary = Data(:,1:size(Data,2));
    return;
    %若参数K大于信号的个数 则将数据集作为字典集
elseif (strcmp(param.InitializationMethod,'DataElements'))%比较两个字符串是否相等
    Dictionary(:,1:param.K-param.preserveDCAtom) = Data(:,1:param.K-param.preserveDCAtom);
% 最后用的是这个，代码是在denoiseimageKSVD实现的构建好了的字典，param.preserveDCAtom是0
elseif (strcmp(param.InitializationMethod,'GivenMatrix'))
    % 一模一样的变过来
    Dictionary(:,1:param.K-param.preserveDCAtom) = param.initialDictionary(:,1:param.K-param.preserveDCAtom);
end

% reduce the components in Dictionary that are spanned by the fixed
% elements
% param.preserveDCAtom是0
if (param.preserveDCAtom)
    tmpMat = FixedDictionaryElement \ Dictionary;
    Dictionary = Dictionary - FixedDictionaryElement*tmpMat;
end


%normalize the dictionary.归一化字典
Dictionary = Dictionary*diag(1./sqrt(sum(Dictionary.*Dictionary)));%把D的每一列数据除以该列数据的平方和，从而进行归一化
%diag生成一个对角矩阵 ，sqrt开平方根
Dictionary = Dictionary.*repmat(sign(Dictionary(1,:)),size(Dictionary,1),1); % multiply in the sign of the first element.
%B = repmat(A,m,n)，将矩阵 A 复制 m×n 块，即把 A 作为 B 的元素，B 由 m×n 个 A 平铺而成。
%sign 取符号，size返回行数
%字典集中的每个元素的化为正数
totalErr = zeros(1,param.numIteration);%返回m*n的零矩阵

% the K-SVD algorithm starts here.KSVD开始
% 这里首先通过根据字典求稀疏系数，然后再通过稀疏系数求字典
for iterNum = 1:param.numIteration%迭代次数
    % find the coefficients
    %得到稀疏矩阵
    if (param.errorFlag==0)%固定表达系数的个数
        %CoefMatrix = mexOMPIterative2(Data, [FixedDictionaryElement,Dictionary],param.L);
        CoefMatrix = OMP([FixedDictionaryElement,Dictionary],Data, param.L);
    else 
        % 走的这条线
        %CoefMatrix = mexOMPerrIterative(Data, [FixedDictionaryElement,Dictionary],param.errorGoal);
        CoefMatrix = OMPerr([FixedDictionaryElement,Dictionary],Data, param.errorGoal);
        param.L = 1;
    end

    replacedVectorCounter = 0;
    
    % rPerm相当于是先知道有多少列，然后把这些列打乱，乱更新
	rPerm = randperm(size(Dictionary,2));%生成一个1到size(Dictionary,2)的随机的向量 y = randperm(n)，y是把1到n这些数随机打乱得到的一个数字序列。
    %随机更新某一列，即K-means思想
    for j = rPerm
        % 初步估计是每次只更新一列
        [betterDictionaryElement,CoefMatrix,addedNewVector] = I_findBetterDictionaryElement(Data,...
            [FixedDictionaryElement,Dictionary],j+size(FixedDictionaryElement,2),...
            CoefMatrix ,param.L);
        
        % 只更新了第j列
        Dictionary(:,j) = betterDictionaryElement;
        
        % param.preserveDCAtom暂时=0，作用有待考证，这段代码不运行
        if (param.preserveDCAtom)
            tmpCoef = FixedDictionaryElement\betterDictionaryElement;
            Dictionary(:,j) = betterDictionaryElement - FixedDictionaryElement*tmpCoef;
            Dictionary(:,j) = Dictionary(:,j)./sqrt(Dictionary(:,j)'*Dictionary(:,j));%去掉一列以后重新归一化
        end
        
        % 简单的计数器
        replacedVectorCounter = replacedVectorCounter+addedNewVector;%实验证明（针对w.jpg图像），值累加了一次
    end
    
    % 把内容展示出来的函数，比较没有用
    if (iterNum>1 & param.displayProgress)
        if (param.errorFlag==0)%不执行
            output.totalerr(iterNum-1) = sqrt(sum(sum((Data-[FixedDictionaryElement,Dictionary]*CoefMatrix).^2))/prod(size(Data)));
            disp(['Iteration   ',num2str(iterNum),'   Total error is: ',num2str(output.totalerr(iterNum-1))]);% disp将内容输出在MATLAB命令行窗口中
        else%执行
            %显示迭代过程
            output.numCoef(iterNum-1) = length(find(CoefMatrix))/size(Data,2);%%CoefMatrix中所有非0元素的长度/数据的列数=平均每列非零系数的个数
            disp(['Iteration   ',num2str(iterNum),'   Average number of coefficients: ',num2str(output.numCoef(iterNum-1))]);
        end
    end
    
    % displayErrorWithTrueDictionary=0，暂不运行
    if (displayErrorWithTrueDictionary ) 
        [ratio(iterNum+1),ErrorBetweenDictionaries(iterNum+1)] = I_findDistanseBetweenDictionaries(param.TrueDictionary,Dictionary);
        disp(strcat(['Iteration  ', num2str(iterNum),' ratio of restored elements: ',num2str(ratio(iterNum+1))]));
        output.ratio = ratio;
    end
    
    % 这里又有一个更新
    Dictionary = I_clearDictionary(Dictionary,CoefMatrix(size(FixedDictionaryElement,2)+1:end,:),Data);
    
    % 和进度条有点关系貌似
    if (isfield(param,'waitBarHandle'))
        waitbar(iterNum/param.counterForWaitBar);
    end
end

output.CoefMatrix = CoefMatrix;
Dictionary = [FixedDictionaryElement,Dictionary];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  findBetterDictionaryElement找到更好的字典元素，
%  这个函数是随机的更新字典中的每个向量，其中I_findDistanseBetweenDictionaries()函数是这样的：
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%得到非零元在第j行的系数矩阵中的位置 
function [betterDictionaryElement,CoefMatrix,NewVectorAdded] = I_findBetterDictionaryElement(Data,Dictionary,j,CoefMatrix,numCoefUsed)
%CoefMatrix为字典最终的系数
if (length(who('numCoefUsed'))==0)
    numCoefUsed = 1;
end
relevantDataIndices = find(CoefMatrix(j,:)); % 非零元在第j行的系数矩阵中的位置the data indices that uses the j'th dictionary element.
%某一列全为0时怎么做？？？？
if (length(relevantDataIndices)<1) %(length(relevantDataIndices)==0)如果系数矩阵的第j列全为零
    ErrorMat = Data-Dictionary*CoefMatrix;%在已有的字典集下和系数下对data项的估计误差
    ErrorNormVec = sum(ErrorMat.^2);%对误差每项平方
    [d,i] = max(ErrorNormVec);%d为所有列中最大项，i为其第几列
    betterDictionaryElement = Data(:,i);%ErrorMat(:,i); %数据项的i列赋给betterDictionaryElement
    betterDictionaryElement = betterDictionaryElement./sqrt(betterDictionaryElement'*betterDictionaryElement);%归一化betterDictionaryElement
    betterDictionaryElement = betterDictionaryElement.*sign(betterDictionaryElement(1));%将betterDictionaryElement中负的元素化为正的
    CoefMatrix(j,:) = 0;%将系数矩阵的第j行赋值为0
    NewVectorAdded = 1;
    return;
end
%执行除掉某一行后得到的误差,只计算非零项
NewVectorAdded = 0;
tmpCoefMatrix = CoefMatrix(:,relevantDataIndices); %将系数矩阵的第j行的非零项所在的列赋给tmpCoefMatrix，记录非零项的位置
tmpCoefMatrix(j,:) = 0;% the coeffitients of the element we now improve are not relevant.将tmpCoefMatrix第j行赋0
errors =(Data(:,relevantDataIndices) - Dictionary*tmpCoefMatrix); %在除去字典中第j个的元素后数据集与预测数据之间的误差% vector of errors that we want to minimize with the new element
% % the better dictionary element and the values of beta are found using svd.
% % This is because we would like to minimize || errors - beta*element ||_F^2. 
% % that is, to approximate the matrix 'errors' with a one-rank matrix. This
% % is done using the largest singular value.
%SVD计算得到特征值
[betterDictionaryElement,singularValue,betaVector] = svds(errors,1);%取出第一主分量
%betterDictionaryElement为右特征向量 singularValue为最大特征值 betaVector左特征向量
CoefMatrix(j,relevantDataIndices) = singularValue*betaVector';% *signOfFirstElem 系数矩阵的第j行的非零元的位置换为singularValue*betaVector的值

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  findDistanseBetweenDictionaries查找字典之间的距离
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ratio,totalDistances] = I_findDistanseBetweenDictionaries(original,new)
% first, all the column in oiginal starts with positive values.
catchCounter = 0;
totalDistances = 0;
for i = 1:size(new,2)
    new(:,i) = sign(new(1,i))*new(:,i);
end
for i = 1:size(original,2)
    d = sign(original(1,i))*original(:,i);
    distances =sum ( (new-repmat(d,1,size(new,2))).^2);
    [minValue,index] = min(distances);
    errorOfElement = 1-abs(new(:,index)'*d);
    totalDistances = totalDistances+errorOfElement;
    catchCounter = catchCounter+(errorOfElement<0.01);
end
ratio = 100*catchCounter/size(original,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  I_clearDictionary清洗字典,归一化处理
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Dictionary = I_clearDictionary(Dictionary,CoefMatrix,Data)
T2 = 0.99;
T1 = 3;
% K是字典的列数
K=size(Dictionary,2);
% Er是平方和
Er=sum((Data-Dictionary*CoefMatrix).^2,1);
% G是D转置乘D
G=Dictionary'*Dictionary;
% 相当于G把对角线上的点都变为了0
G = G-diag(diag(G));

% 从1到K针对于dictionary的每列进行循环
for jj=1:1:K,
    % G中如果最大值大于1或者系数矩阵普遍为0
    if max(G(jj,:))>T2 | length(find(abs(CoefMatrix(jj,:))>1e-7))<=T1 ,
        
        % 每列最大的位置,Er是个数
        [val,pos]=max(Er);
        
        % Er变为0
        Er(pos(1))=0;
        
        Dictionary(:,jj)=Data(:,pos(1))/norm(Data(:,pos(1)));% norm(Data(:,pos(1))：求向量的模   此整句相当于归一化
        G=Dictionary'*Dictionary; G = G-diag(diag(G));
    end;
end;


----------------------------------------------------------------------
-- This script shows how to train different models on the MNIST 
-- dataset, using multiple optimization techniques (SGD, LBFGS)
--
-- This script demonstrates a classical example of training 
-- well-known models (convnet, MLP, logistic regression)
-- on a 10-class classification problem. 
--
-- It illustrates several points:
-- 1/ description of the model
-- 2/ choice of a loss function (criterion) to minimize
-- 3/ creation of a dataset as a simple Lua table
-- 4/ description of training and test procedures
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'
require 'nn'
-- require 'nnx'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'pl'
require 'paths'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -s,--save                                subdirectory to save logs
   -n,--network       (default "")          reload pretrained network
   -m,--model         (default "convnet")   type of model tor train: convnet | mlp | linear
   -f,--full                                use the full dataset
   -p,--plot                                plot while training
   -o,--optimization  (default "SGD")       optimization: SGD | LBFGS 
   -r,--learningRate  (default 0.05)        learning rate, for SGD only
   -b,--batchSize     (default 10)          batch size
   -m,--momentum      (default 0)           momentum, for SGD only
   -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
   --coefL1           (default 0)           L1 penalty on the weights
   --coefL2           (default 0)           L2 penalty on the weights
   -t,--threads       (default 4)           number of threads
   --type             (default float)       float or cuda
   --devid            (default 1)           device ID (if using CUDA)
   -v,--verbose                             print out progress statements or not
   --epochs           (default 10)          number of epochs to train for
   --seed             (default 1)           random seed
]]

-- fix seed
torch.manualSeed(opt.seed)

-- threads
torch.setnumthreads(opt.threads)
if opt.verbose then
    print('<torch> set nb of threads to ' .. torch.getnumthreads())
end
if opt.type == 'cuda' then
   require 'cunn'
   cutorch.setDevice(opt.devid)
   nn.SpatialConvolutionMM = nn.SpatialConvolution
   if opt.verbose then
      print(sys.COLORS.red ..  '==> switching to CUDA')
      print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
   end
end

-- use floats, for SGD
if opt.optimization == 'SGD' then
   torch.setdefaulttensortype('torch.FloatTensor')
end

-- batch size?
if opt.optimization == 'LBFGS' and opt.batchSize < 100 then
   error('LBFGS should not be used with small mini-batches; 1000 is a recommended')
end

----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
classes = {'1','2','3','4','5','6','7','8','9','10'}

-- geometry: width and height of input images
geometry = {32,32}

if opt.network == '' then
   -- define model to train
   model = nn.Sequential()

   if opt.model == 'convnet' then
      ------------------------------------------------------------
      -- convolutional network 
      ------------------------------------------------------------
      -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
      model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(3, 3, 3, 3))
      -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
      model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      -- stage 3 : standard 2-layer MLP:
      model:add(nn.Reshape(64*2*2))
      model:add(nn.Linear(64*2*2, 200))
      model:add(nn.Tanh())
      model:add(nn.Linear(200, #classes))
      ------------------------------------------------------------

   elseif opt.model == 'mlp' then
      ------------------------------------------------------------
      -- regular 2-layer MLP
      ------------------------------------------------------------
      model:add(nn.Reshape(1024))
      model:add(nn.Linear(1024, 2048))
      model:add(nn.Tanh())
      model:add(nn.Linear(2048,#classes))
      ------------------------------------------------------------

   elseif opt.model == 'linear' then
      ------------------------------------------------------------
      -- simple linear model: logistic regression
      ------------------------------------------------------------
      model:add(nn.Reshape(1024))
      model:add(nn.Linear(1024,#classes))
      ------------------------------------------------------------

   else
      print('Unknown model type')
      cmd:text()
      error()
   end
else
   if opt.verbose then
      print('<trainer> reloading previously trained network')
   end
   model = torch.load(opt.network)
end

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- verbose
if opt.verbose then
   print('<mnist> using model:')
   print(model)
end

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

if opt.type == 'cuda' then
  model = model:cuda()
  criterion = criterion:cuda()
end

----------------------------------------------------------------------
-- get/create dataset
--
if opt.full then
   nbTrainingPatches = 60000
   nbTestingPatches = 10000
else
   nbTrainingPatches = 2000
   nbTestingPatches = 1000
   if opt.verbose then
      print('<warning> only using 2000 samples to train quickly (use flag -full to use 60000 samples)')
   end
end

-- create training set and normalize
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

-- create test set and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)

-- allocate memory for minibatches
local inputs = torch.Tensor(
  opt.batchSize,
  1,
  geometry[1],
  geometry[2])
local targets = torch.Tensor(opt.batchSize)

-- cast processed data into CudaTensors
if opt.type == 'cuda' then
  inputs = inputs:cuda()
  targets = targets:cuda()
end

----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
if opt.save then
    trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
    testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
end

-- training function
function train(dataset)
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- do one epoch
   if opt.verbose then
      print('<trainer> on training set:')
      print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   end
   for t = 1,dataset:size(),opt.batchSize do
      -- shuffle at each epoch
      local shuffle = torch.randperm(dataset:size())

      -- create mini batch
      local idx = 1
      for i = t, math.min(t + opt.batchSize-1, dataset:size()) do
         -- load new sample
         inputs[idx] = dataset.data[shuffle[i]]
         targets[idx] = dataset.labels[shuffle[i]]
         idx = idx + 1
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- just in case:
         collectgarbage()

         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- evaluate function for complete mini batch
         local outputs = model:forward(inputs)
         local f = criterion:forward(outputs, targets)

         -- estimate df/dW
         local df_do = criterion:backward(outputs, targets)
         model:backward(inputs, df_do)

         -- penalties (L1 and L2):
         if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
            -- locals:
            local norm,sign= torch.norm,torch.sign

            -- Loss:
            f = f + opt.coefL1 * norm(parameters,1)
            f = f + opt.coefL2 * norm(parameters,2)^2/2

            -- Gradients:
            gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
         end

         -- update confusion
         for i = 1,opt.batchSize do
            confusion:add(outputs[i], targets[i])
         end

         -- return f and df/dX
         return f,gradParameters
      end

      -- optimize on current mini-batch
      if opt.optimization == 'LBFGS' then

         -- Perform LBFGS step:
         lbfgsState = lbfgsState or {
            maxIter = opt.maxIter,
            lineSearch = optim.lswolfe
         }
         optim.lbfgs(feval, parameters, lbfgsState)
       
         -- disp report:
         if opt.verbose then
            print('LBFGS step')
            print(' - progress in batch: ' .. t .. '/' .. dataset:size())
            print(' - nb of iterations: ' .. lbfgsState.nIter)
            print(' - nb of function evalutions: ' .. lbfgsState.funcEval)
         end

      elseif opt.optimization == 'SGD' then

         -- Perform SGD step:
         sgdState = sgdState or {
            learningRate = opt.learningRate,
            momentum = opt.momentum,
            learningRateDecay = 5e-7
         }
         optim.sgd(feval, parameters, sgdState)
      
         -- disp progress
         if opt.verbose then
            xlua.progress(t, dataset:size())
         end

      else
         error('unknown optimization method')
      end
   end
   
   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
   if opt.verbose then
      print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')
   end

   -- print confusion matrix
   if opt.verbose then
      print(confusion)
   end
   if opt.save then
      trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}

      -- save/log current net
      local filename = paths.concat(opt.save, 'mnist.net')
      os.execute('mkdir -p ' .. sys.dirname(filename))
      if paths.filep(filename) then
         os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
      end
      if opt.verbose then
         print('<trainer> saving network to '..filename)
      end
   end

   -- torch.save(filename, model)
   confusion:zero()

   -- next epoch
   epoch = epoch + 1
end

-- test function
function test(dataset)
   -- local vars
   local time = sys.clock()

   -- test over given dataset
   if opt.verbose then
      print('<trainer> on testing Set:')
   end
   for t = 1,dataset:size(),opt.batchSize do
      -- disp progress
      if opt.verbose then
         xlua.progress(t, dataset:size())
      end

      -- create mini batch
      local idx = 1
      for i = t, math.min(t + opt.batchSize-1, dataset:size()) do
        -- load new sample
        inputs[idx] = dataset.data[i]
        targets[idx] = dataset.labels[i]
        idx = idx + 1
      end

      -- test samples
      local preds = model:forward(inputs)

      -- confusion:
      for i = 1,opt.batchSize do
         confusion:add(preds[i], targets[i])
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   if opt.verbose then
      print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')
   end

   -- print confusion matrix
   if opt.verbose then
      print(confusion)
   else
      confusion:updateValids()
   end
   accuracy = confusion.totalValid * 100
   if opt.save then
      testLogger:add{['% mean class accuracy (test set)'] = accuracy}
   end
   confusion:zero()

   return accuracy
end

----------------------------------------------------------------------
-- and train!
--
for i = 1, opt.epochs do
   -- train/test
   train(trainData)
   acc = test(testData)

   -- plot errors
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      trainLogger:plot()
      testLogger:plot()
   end
end

print('<output> = ' .. acc)



--[[

 Long Short-Term Memory.

--]]

local LSTMDecoder, parent = torch.class('treelstm.LSTMDecoder', 'nn.Module')

function LSTMDecoder:__init(config)
  parent.__init(self)

  self.in_dim = config.mem_dim
  self.mem_dim = config.mem_dim

  self.master_cell = self:new_cell()
  self.depth = 0
  self.cells = {}  -- table of cells in a roll-out

  local ctable_init, ctable_grad, htable_init, htable_grad
  ctable_init = torch.zeros(self.mem_dim)
  htable_init = torch.zeros(self.mem_dim)
  ctable_grad = torch.zeros(self.mem_dim)
  htable_grad = torch.zeros(self.mem_dim)
  self.initial_values = {ctable_init, htable_init}
  self.gradInput = {
    torch.zeros(self.in_dim),
    ctable_grad,
    htable_grad
  }
end

function LSTMDecoder:new_cell()
  local input = nn.Identity()()
  local ctable_p = nn.Identity()()
  local htable_p = nn.Identity()()
  local htable, ctable = {}, {}

  local new_gate = function()
    local in_module = nn.Linear(self.mem_dim, self.mem_dim)(input)
    return nn.CAddTable(){
      in_module,
      nn.Linear(self.mem_dim, self.mem_dim)(htable_p)
    }
  end

  -- input, forget, and output gates
  local i = nn.Sigmoid()(new_gate())
  local f = nn.Sigmoid()(new_gate())
  local update = nn.Tanh()(new_gate())

  -- update the state of the LSTM cell
  ctable[1] = nn.CAddTable(){
    nn.CMulTable(){f, ctable_p},
    nn.CMulTable(){i, update}
  }

  local o = nn.Sigmoid()(new_gate())
  htable[1] = nn.CMulTable(){o, nn.Tanh()(ctable[1])}

  htable, ctable = nn.Identity()(htable), nn.Identity()(ctable)
  local cell = nn.gModule({input, ctable_p, htable_p}, {ctable, htable})

  --if self.master_cell then
    --share_params(cell, self.master_cell)
  --end
  return cell
end

function LSTMDecoder:forward(inputs)
  local out_encoder={}
  local size = inputs:size(1)
  for t = 1, size do
    local input = inputs[t]
    self.depth = self.depth + 1
    local cell = self.cells[self.depth]
    if cell == nil then
      cell = self:new_cell()
      self.cells[self.depth] = cell
    end
    local prev_output
    if self.depth > 1 then
      prev_output = self.cells[self.depth - 1].output
    else
      prev_output = self.initial_values
    end
    local outputs = cell:forward({input, prev_output[1], prev_output[2]})
    local ctable, htable = unpack(outputs)
    out_encoder[t] = htable
  end
  --self.output = torch.cat(out_encoder)
  self.output = out_encoder
  return self.output
end

function LSTMDecoder:backward(inputs, grad_outputs)
  local size = inputs:size(1)
  local input_grads = torch.Tensor(inputs:size())
  for t = size, 1, -1 do
    local input = inputs[t]
    local grad_output = grad_outputs[t]
    local cell = self.cells[self.depth]
    local grads = {self.gradInput[2], self.gradInput[3]}
    grads[2]:add(grad_output)
    local prev_output = (self.depth > 1) and self.cells[self.depth - 1].output
                                         or self.initial_values
    self.gradInput = cell:backward({input, prev_output[1], prev_output[2]}, grads)
    input_grads[t] = self.gradInput[1]
    self.depth = self.depth - 1
  end
  self:forget()
  return input_grads
end

function LSTMDecoder:zeroGradParameters()
  self.master_cell:zeroGradParameters()
end

function LSTMDecoder:parameters()
  return self.master_cell:parameters()
end

function LSTMDecoder:forget()
  for i = 1, #self.gradInput do
    local gradInput = self.gradInput[i]
    if type(gradInput) == 'table' then
      for _, t in pairs(gradInput) do t:zero() end
    else
      self.gradInput[i]:zero()
    end
  end
end

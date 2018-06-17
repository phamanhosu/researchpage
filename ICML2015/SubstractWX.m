function [max_out,wx_out]=SubstractWX(wx_in)
max_out=max(wx_in);
max_out_matrix=ones(size(wx_in,1),1)*max_out;
wx_out=wx_in-max_out_matrix;
end
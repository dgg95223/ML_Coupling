��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.9.12v2.9.0-18-gd8ce9f9c3018��
�
mlp_base_2/dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namemlp_base_2/dense_23/bias
�
,mlp_base_2/dense_23/bias/Read/ReadVariableOpReadVariableOpmlp_base_2/dense_23/bias*
_output_shapes
:*
dtype0
�
mlp_base_2/dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*+
shared_namemlp_base_2/dense_23/kernel
�
.mlp_base_2/dense_23/kernel/Read/ReadVariableOpReadVariableOpmlp_base_2/dense_23/kernel*
_output_shapes
:	�*
dtype0
�
mlp_base_2/dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_namemlp_base_2/dense_22/bias
�
,mlp_base_2/dense_22/bias/Read/ReadVariableOpReadVariableOpmlp_base_2/dense_22/bias*
_output_shapes	
:�*
dtype0
�
mlp_base_2/dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*+
shared_namemlp_base_2/dense_22/kernel
�
.mlp_base_2/dense_22/kernel/Read/ReadVariableOpReadVariableOpmlp_base_2/dense_22/kernel* 
_output_shapes
:
��*
dtype0
�
mlp_base_2/dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_namemlp_base_2/dense_21/bias
�
,mlp_base_2/dense_21/bias/Read/ReadVariableOpReadVariableOpmlp_base_2/dense_21/bias*
_output_shapes	
:�*
dtype0
�
mlp_base_2/dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*+
shared_namemlp_base_2/dense_21/kernel
�
.mlp_base_2/dense_21/kernel/Read/ReadVariableOpReadVariableOpmlp_base_2/dense_21/kernel* 
_output_shapes
:
��*
dtype0
�
mlp_base_2/dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_namemlp_base_2/dense_20/bias
�
,mlp_base_2/dense_20/bias/Read/ReadVariableOpReadVariableOpmlp_base_2/dense_20/bias*
_output_shapes	
:�*
dtype0
�
mlp_base_2/dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*+
shared_namemlp_base_2/dense_20/kernel
�
.mlp_base_2/dense_20/kernel/Read/ReadVariableOpReadVariableOpmlp_base_2/dense_20/kernel*
_output_shapes
:	�*
dtype0
�
1mlp_base_2/batch_normalization_17/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*B
shared_name31mlp_base_2/batch_normalization_17/moving_variance
�
Emlp_base_2/batch_normalization_17/moving_variance/Read/ReadVariableOpReadVariableOp1mlp_base_2/batch_normalization_17/moving_variance*
_output_shapes	
:�*
dtype0
�
-mlp_base_2/batch_normalization_17/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*>
shared_name/-mlp_base_2/batch_normalization_17/moving_mean
�
Amlp_base_2/batch_normalization_17/moving_mean/Read/ReadVariableOpReadVariableOp-mlp_base_2/batch_normalization_17/moving_mean*
_output_shapes	
:�*
dtype0
�
&mlp_base_2/batch_normalization_17/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&mlp_base_2/batch_normalization_17/beta
�
:mlp_base_2/batch_normalization_17/beta/Read/ReadVariableOpReadVariableOp&mlp_base_2/batch_normalization_17/beta*
_output_shapes	
:�*
dtype0
�
'mlp_base_2/batch_normalization_17/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*8
shared_name)'mlp_base_2/batch_normalization_17/gamma
�
;mlp_base_2/batch_normalization_17/gamma/Read/ReadVariableOpReadVariableOp'mlp_base_2/batch_normalization_17/gamma*
_output_shapes	
:�*
dtype0
�
1mlp_base_2/batch_normalization_16/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*B
shared_name31mlp_base_2/batch_normalization_16/moving_variance
�
Emlp_base_2/batch_normalization_16/moving_variance/Read/ReadVariableOpReadVariableOp1mlp_base_2/batch_normalization_16/moving_variance*
_output_shapes	
:�*
dtype0
�
-mlp_base_2/batch_normalization_16/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*>
shared_name/-mlp_base_2/batch_normalization_16/moving_mean
�
Amlp_base_2/batch_normalization_16/moving_mean/Read/ReadVariableOpReadVariableOp-mlp_base_2/batch_normalization_16/moving_mean*
_output_shapes	
:�*
dtype0
�
&mlp_base_2/batch_normalization_16/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&mlp_base_2/batch_normalization_16/beta
�
:mlp_base_2/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOp&mlp_base_2/batch_normalization_16/beta*
_output_shapes	
:�*
dtype0
�
'mlp_base_2/batch_normalization_16/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*8
shared_name)'mlp_base_2/batch_normalization_16/gamma
�
;mlp_base_2/batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOp'mlp_base_2/batch_normalization_16/gamma*
_output_shapes	
:�*
dtype0
�
1mlp_base_2/batch_normalization_15/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*B
shared_name31mlp_base_2/batch_normalization_15/moving_variance
�
Emlp_base_2/batch_normalization_15/moving_variance/Read/ReadVariableOpReadVariableOp1mlp_base_2/batch_normalization_15/moving_variance*
_output_shapes	
:�*
dtype0
�
-mlp_base_2/batch_normalization_15/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*>
shared_name/-mlp_base_2/batch_normalization_15/moving_mean
�
Amlp_base_2/batch_normalization_15/moving_mean/Read/ReadVariableOpReadVariableOp-mlp_base_2/batch_normalization_15/moving_mean*
_output_shapes	
:�*
dtype0
�
&mlp_base_2/batch_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&mlp_base_2/batch_normalization_15/beta
�
:mlp_base_2/batch_normalization_15/beta/Read/ReadVariableOpReadVariableOp&mlp_base_2/batch_normalization_15/beta*
_output_shapes	
:�*
dtype0
�
'mlp_base_2/batch_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*8
shared_name)'mlp_base_2/batch_normalization_15/gamma
�
;mlp_base_2/batch_normalization_15/gamma/Read/ReadVariableOpReadVariableOp'mlp_base_2/batch_normalization_15/gamma*
_output_shapes	
:�*
dtype0

NoOpNoOp
�G
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�F
value�FB�F B�F
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
setting

	input1


input2
concate
BN1
BN2
BN3

dense1

dense2

dense3

denseO

signatures*
�
0
1
2
3
4
5
6
7
8
9
10
11
 12
!13
"14
#15
$16
%17
&18
'19*
j
0
1
2
3
4
5
 6
!7
"8
#9
$10
%11
&12
'13*
* 
�
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
-trace_0
.trace_1
/trace_2
0trace_3* 
6
1trace_0
2trace_1
3trace_2
4trace_3* 
* 
* 
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses* 
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses* 
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses* 
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses
Maxis
	gamma
beta
moving_mean
moving_variance*
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
Taxis
	gamma
beta
moving_mean
moving_variance*
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses
[axis
	gamma
beta
moving_mean
moving_variance*
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

 kernel
!bias*
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

"kernel
#bias*
�
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses

$kernel
%bias*
�
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses

&kernel
'bias*

tserving_default* 
ga
VARIABLE_VALUE'mlp_base_2/batch_normalization_15/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&mlp_base_2/batch_normalization_15/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE-mlp_base_2/batch_normalization_15/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1mlp_base_2/batch_normalization_15/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'mlp_base_2/batch_normalization_16/gamma&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&mlp_base_2/batch_normalization_16/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE-mlp_base_2/batch_normalization_16/moving_mean&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1mlp_base_2/batch_normalization_16/moving_variance&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'mlp_base_2/batch_normalization_17/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&mlp_base_2/batch_normalization_17/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE-mlp_base_2/batch_normalization_17/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1mlp_base_2/batch_normalization_17/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEmlp_base_2/dense_20/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEmlp_base_2/dense_20/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEmlp_base_2/dense_21/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEmlp_base_2/dense_21/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEmlp_base_2/dense_22/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEmlp_base_2/dense_22/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEmlp_base_2/dense_23/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEmlp_base_2/dense_23/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
.
0
1
2
3
4
5*
J
	0

1
2
3
4
5
6
7
8
9*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses* 

ztrace_0* 

{trace_0* 
* 
* 
* 
�
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
�layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
 
0
1
2
3*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
 
0
1
2
3*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
 
0
1
2
3*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

 0
!1*

 0
!1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

"0
#1*

"0
#1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

$0
%1*

$0
%1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

&0
'1*

&0
'1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
serving_default_input_1Placeholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1mlp_base_2/dense_20/kernelmlp_base_2/dense_20/bias-mlp_base_2/batch_normalization_15/moving_mean1mlp_base_2/batch_normalization_15/moving_variance&mlp_base_2/batch_normalization_15/beta'mlp_base_2/batch_normalization_15/gammamlp_base_2/dense_21/kernelmlp_base_2/dense_21/bias-mlp_base_2/batch_normalization_16/moving_mean1mlp_base_2/batch_normalization_16/moving_variance&mlp_base_2/batch_normalization_16/beta'mlp_base_2/batch_normalization_16/gammamlp_base_2/dense_22/kernelmlp_base_2/dense_22/bias-mlp_base_2/batch_normalization_17/moving_mean1mlp_base_2/batch_normalization_17/moving_variance&mlp_base_2/batch_normalization_17/beta'mlp_base_2/batch_normalization_17/gammamlp_base_2/dense_23/kernelmlp_base_2/dense_23/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_3394196
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename;mlp_base_2/batch_normalization_15/gamma/Read/ReadVariableOp:mlp_base_2/batch_normalization_15/beta/Read/ReadVariableOpAmlp_base_2/batch_normalization_15/moving_mean/Read/ReadVariableOpEmlp_base_2/batch_normalization_15/moving_variance/Read/ReadVariableOp;mlp_base_2/batch_normalization_16/gamma/Read/ReadVariableOp:mlp_base_2/batch_normalization_16/beta/Read/ReadVariableOpAmlp_base_2/batch_normalization_16/moving_mean/Read/ReadVariableOpEmlp_base_2/batch_normalization_16/moving_variance/Read/ReadVariableOp;mlp_base_2/batch_normalization_17/gamma/Read/ReadVariableOp:mlp_base_2/batch_normalization_17/beta/Read/ReadVariableOpAmlp_base_2/batch_normalization_17/moving_mean/Read/ReadVariableOpEmlp_base_2/batch_normalization_17/moving_variance/Read/ReadVariableOp.mlp_base_2/dense_20/kernel/Read/ReadVariableOp,mlp_base_2/dense_20/bias/Read/ReadVariableOp.mlp_base_2/dense_21/kernel/Read/ReadVariableOp,mlp_base_2/dense_21/bias/Read/ReadVariableOp.mlp_base_2/dense_22/kernel/Read/ReadVariableOp,mlp_base_2/dense_22/bias/Read/ReadVariableOp.mlp_base_2/dense_23/kernel/Read/ReadVariableOp,mlp_base_2/dense_23/bias/Read/ReadVariableOpConst*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_3394951
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename'mlp_base_2/batch_normalization_15/gamma&mlp_base_2/batch_normalization_15/beta-mlp_base_2/batch_normalization_15/moving_mean1mlp_base_2/batch_normalization_15/moving_variance'mlp_base_2/batch_normalization_16/gamma&mlp_base_2/batch_normalization_16/beta-mlp_base_2/batch_normalization_16/moving_mean1mlp_base_2/batch_normalization_16/moving_variance'mlp_base_2/batch_normalization_17/gamma&mlp_base_2/batch_normalization_17/beta-mlp_base_2/batch_normalization_17/moving_mean1mlp_base_2/batch_normalization_17/moving_variancemlp_base_2/dense_20/kernelmlp_base_2/dense_20/biasmlp_base_2/dense_21/kernelmlp_base_2/dense_21/biasmlp_base_2/dense_22/kernelmlp_base_2/dense_22/biasmlp_base_2/dense_23/kernelmlp_base_2/dense_23/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_3395021��
�
�
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_3393451

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�-
cast_2_readvariableop_resource:	�-
cast_3_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpm
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:u
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������l
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�n
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_15_layer_call_fn_3394562

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3393369p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
E__inference_dense_23_layer_call_and_return_conditional_losses_3393719

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_dense_21_layer_call_fn_3394818

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_3393668p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�T
�
#__inference__traced_restore_3395021
file_prefixG
8assignvariableop_mlp_base_2_batch_normalization_15_gamma:	�H
9assignvariableop_1_mlp_base_2_batch_normalization_15_beta:	�O
@assignvariableop_2_mlp_base_2_batch_normalization_15_moving_mean:	�S
Dassignvariableop_3_mlp_base_2_batch_normalization_15_moving_variance:	�I
:assignvariableop_4_mlp_base_2_batch_normalization_16_gamma:	�H
9assignvariableop_5_mlp_base_2_batch_normalization_16_beta:	�O
@assignvariableop_6_mlp_base_2_batch_normalization_16_moving_mean:	�S
Dassignvariableop_7_mlp_base_2_batch_normalization_16_moving_variance:	�I
:assignvariableop_8_mlp_base_2_batch_normalization_17_gamma:	�H
9assignvariableop_9_mlp_base_2_batch_normalization_17_beta:	�P
Aassignvariableop_10_mlp_base_2_batch_normalization_17_moving_mean:	�T
Eassignvariableop_11_mlp_base_2_batch_normalization_17_moving_variance:	�A
.assignvariableop_12_mlp_base_2_dense_20_kernel:	�;
,assignvariableop_13_mlp_base_2_dense_20_bias:	�B
.assignvariableop_14_mlp_base_2_dense_21_kernel:
��;
,assignvariableop_15_mlp_base_2_dense_21_bias:	�B
.assignvariableop_16_mlp_base_2_dense_22_kernel:
��;
,assignvariableop_17_mlp_base_2_dense_22_bias:	�A
.assignvariableop_18_mlp_base_2_dense_23_kernel:	�:
,assignvariableop_19_mlp_base_2_dense_23_bias:
identity_21��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp8assignvariableop_mlp_base_2_batch_normalization_15_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp9assignvariableop_1_mlp_base_2_batch_normalization_15_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp@assignvariableop_2_mlp_base_2_batch_normalization_15_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpDassignvariableop_3_mlp_base_2_batch_normalization_15_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp:assignvariableop_4_mlp_base_2_batch_normalization_16_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp9assignvariableop_5_mlp_base_2_batch_normalization_16_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp@assignvariableop_6_mlp_base_2_batch_normalization_16_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpDassignvariableop_7_mlp_base_2_batch_normalization_16_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp:assignvariableop_8_mlp_base_2_batch_normalization_17_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp9assignvariableop_9_mlp_base_2_batch_normalization_17_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpAassignvariableop_10_mlp_base_2_batch_normalization_17_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpEassignvariableop_11_mlp_base_2_batch_normalization_17_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp.assignvariableop_12_mlp_base_2_dense_20_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp,assignvariableop_13_mlp_base_2_dense_20_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp.assignvariableop_14_mlp_base_2_dense_21_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp,assignvariableop_15_mlp_base_2_dense_21_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp.assignvariableop_16_mlp_base_2_dense_22_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp,assignvariableop_17_mlp_base_2_dense_22_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp.assignvariableop_18_mlp_base_2_dense_23_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp,assignvariableop_19_mlp_base_2_dense_23_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_21IdentityIdentity_20:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_21Identity_21:output:0*=
_input_shapes,
*: : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
,__inference_mlp_base_2_layer_call_fn_3394286

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_mlp_base_2_layer_call_and_return_conditional_losses_3393937o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�>
�	
G__inference_mlp_base_2_layer_call_and_return_conditional_losses_3393937

inputs#
dense_20_3393889:	�
dense_20_3393891:	�-
batch_normalization_15_3393894:	�-
batch_normalization_15_3393896:	�-
batch_normalization_15_3393898:	�-
batch_normalization_15_3393900:	�$
dense_21_3393903:
��
dense_21_3393905:	�-
batch_normalization_16_3393908:	�-
batch_normalization_16_3393910:	�-
batch_normalization_16_3393912:	�-
batch_normalization_16_3393914:	�$
dense_22_3393917:
��
dense_22_3393919:	�-
batch_normalization_17_3393922:	�-
batch_normalization_17_3393924:	�-
batch_normalization_17_3393926:	�-
batch_normalization_17_3393928:	�#
dense_23_3393931:	�
dense_23_3393933:
identity��.batch_normalization_15/StatefulPartitionedCall�.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCalld
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
flatten_10/PartitionedCallPartitionedCallstrided_slice:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_10_layer_call_and_return_conditional_losses_3393608f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
flatten_11/PartitionedCallPartitionedCallstrided_slice_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_11_layer_call_and_return_conditional_losses_3393620�
concatenate_5/PartitionedCallPartitionedCall#flatten_10/PartitionedCall:output:0#flatten_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_concatenate_5_layer_call_and_return_conditional_losses_3393629�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0dense_20_3393889dense_20_3393891*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_3393642�
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0batch_normalization_15_3393894batch_normalization_15_3393896batch_normalization_15_3393898batch_normalization_15_3393900*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3393416�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0dense_21_3393903dense_21_3393905*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_3393668�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0batch_normalization_16_3393908batch_normalization_16_3393910batch_normalization_16_3393912batch_normalization_16_3393914*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_3393498�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0dense_22_3393917dense_22_3393919*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_3393694�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0batch_normalization_17_3393922batch_normalization_17_3393924batch_normalization_17_3393926batch_normalization_17_3393928*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_3393580�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0dense_23_3393931dense_23_3393933*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_3393719x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
G__inference_flatten_11_layer_call_and_return_conditional_losses_3394536

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
G__inference_flatten_11_layer_call_and_return_conditional_losses_3393620

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
E__inference_dense_23_layer_call_and_return_conditional_losses_3394868

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�3
�
 __inference__traced_save_3394951
file_prefixF
Bsavev2_mlp_base_2_batch_normalization_15_gamma_read_readvariableopE
Asavev2_mlp_base_2_batch_normalization_15_beta_read_readvariableopL
Hsavev2_mlp_base_2_batch_normalization_15_moving_mean_read_readvariableopP
Lsavev2_mlp_base_2_batch_normalization_15_moving_variance_read_readvariableopF
Bsavev2_mlp_base_2_batch_normalization_16_gamma_read_readvariableopE
Asavev2_mlp_base_2_batch_normalization_16_beta_read_readvariableopL
Hsavev2_mlp_base_2_batch_normalization_16_moving_mean_read_readvariableopP
Lsavev2_mlp_base_2_batch_normalization_16_moving_variance_read_readvariableopF
Bsavev2_mlp_base_2_batch_normalization_17_gamma_read_readvariableopE
Asavev2_mlp_base_2_batch_normalization_17_beta_read_readvariableopL
Hsavev2_mlp_base_2_batch_normalization_17_moving_mean_read_readvariableopP
Lsavev2_mlp_base_2_batch_normalization_17_moving_variance_read_readvariableop9
5savev2_mlp_base_2_dense_20_kernel_read_readvariableop7
3savev2_mlp_base_2_dense_20_bias_read_readvariableop9
5savev2_mlp_base_2_dense_21_kernel_read_readvariableop7
3savev2_mlp_base_2_dense_21_bias_read_readvariableop9
5savev2_mlp_base_2_dense_22_kernel_read_readvariableop7
3savev2_mlp_base_2_dense_22_bias_read_readvariableop9
5savev2_mlp_base_2_dense_23_kernel_read_readvariableop7
3savev2_mlp_base_2_dense_23_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Bsavev2_mlp_base_2_batch_normalization_15_gamma_read_readvariableopAsavev2_mlp_base_2_batch_normalization_15_beta_read_readvariableopHsavev2_mlp_base_2_batch_normalization_15_moving_mean_read_readvariableopLsavev2_mlp_base_2_batch_normalization_15_moving_variance_read_readvariableopBsavev2_mlp_base_2_batch_normalization_16_gamma_read_readvariableopAsavev2_mlp_base_2_batch_normalization_16_beta_read_readvariableopHsavev2_mlp_base_2_batch_normalization_16_moving_mean_read_readvariableopLsavev2_mlp_base_2_batch_normalization_16_moving_variance_read_readvariableopBsavev2_mlp_base_2_batch_normalization_17_gamma_read_readvariableopAsavev2_mlp_base_2_batch_normalization_17_beta_read_readvariableopHsavev2_mlp_base_2_batch_normalization_17_moving_mean_read_readvariableopLsavev2_mlp_base_2_batch_normalization_17_moving_variance_read_readvariableop5savev2_mlp_base_2_dense_20_kernel_read_readvariableop3savev2_mlp_base_2_dense_20_bias_read_readvariableop5savev2_mlp_base_2_dense_21_kernel_read_readvariableop3savev2_mlp_base_2_dense_21_bias_read_readvariableop5savev2_mlp_base_2_dense_22_kernel_read_readvariableop3savev2_mlp_base_2_dense_22_bias_read_readvariableop5savev2_mlp_base_2_dense_23_kernel_read_readvariableop3savev2_mlp_base_2_dense_23_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *#
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :�:�:�:�:�:�:�:�:�:�:�:�:	�:�:
��:�:
��:�:	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!	

_output_shapes	
:�:!


_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:%!

_output_shapes
:	�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: 
�

�
E__inference_dense_22_layer_call_and_return_conditional_losses_3393694

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
[
/__inference_concatenate_5_layer_call_fn_3394542
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_concatenate_5_layer_call_and_return_conditional_losses_3393629`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�$
�
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_3393498

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0m
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�l
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_15_layer_call_fn_3394575

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3393416p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_dense_20_layer_call_and_return_conditional_losses_3394809

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_20_layer_call_and_return_conditional_losses_3393642

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_dense_20_layer_call_fn_3394798

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_3393642p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
,__inference_mlp_base_2_layer_call_fn_3393769
input_1
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_mlp_base_2_layer_call_and_return_conditional_losses_3393726o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
8__inference_batch_normalization_16_layer_call_fn_3394655

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_3393498p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�$
�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_3394789

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0m
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�l
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�>
�	
G__inference_mlp_base_2_layer_call_and_return_conditional_losses_3394149
input_1#
dense_20_3394101:	�
dense_20_3394103:	�-
batch_normalization_15_3394106:	�-
batch_normalization_15_3394108:	�-
batch_normalization_15_3394110:	�-
batch_normalization_15_3394112:	�$
dense_21_3394115:
��
dense_21_3394117:	�-
batch_normalization_16_3394120:	�-
batch_normalization_16_3394122:	�-
batch_normalization_16_3394124:	�-
batch_normalization_16_3394126:	�$
dense_22_3394129:
��
dense_22_3394131:	�-
batch_normalization_17_3394134:	�-
batch_normalization_17_3394136:	�-
batch_normalization_17_3394138:	�-
batch_normalization_17_3394140:	�#
dense_23_3394143:	�
dense_23_3394145:
identity��.batch_normalization_15/StatefulPartitionedCall�.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCalld
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
flatten_10/PartitionedCallPartitionedCallstrided_slice:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_10_layer_call_and_return_conditional_losses_3393608f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSliceinput_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
flatten_11/PartitionedCallPartitionedCallstrided_slice_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_11_layer_call_and_return_conditional_losses_3393620�
concatenate_5/PartitionedCallPartitionedCall#flatten_10/PartitionedCall:output:0#flatten_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_concatenate_5_layer_call_and_return_conditional_losses_3393629�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0dense_20_3394101dense_20_3394103*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_3393642�
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0batch_normalization_15_3394106batch_normalization_15_3394108batch_normalization_15_3394110batch_normalization_15_3394112*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3393416�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0dense_21_3394115dense_21_3394117*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_3393668�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0batch_normalization_16_3394120batch_normalization_16_3394122batch_normalization_16_3394124batch_normalization_16_3394126*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_3393498�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0dense_22_3394129dense_22_3394131*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_3393694�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0batch_normalization_17_3394134batch_normalization_17_3394136batch_normalization_17_3394138batch_normalization_17_3394140*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_3393580�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0dense_23_3394143dense_23_3394145*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_3393719x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
,__inference_mlp_base_2_layer_call_fn_3394241

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_mlp_base_2_layer_call_and_return_conditional_losses_3393726o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
G__inference_mlp_base_2_layer_call_and_return_conditional_losses_3394514

inputs:
'dense_20_matmul_readvariableop_resource:	�7
(dense_20_biasadd_readvariableop_resource:	�M
>batch_normalization_15_assignmovingavg_readvariableop_resource:	�O
@batch_normalization_15_assignmovingavg_1_readvariableop_resource:	�B
3batch_normalization_15_cast_readvariableop_resource:	�D
5batch_normalization_15_cast_1_readvariableop_resource:	�;
'dense_21_matmul_readvariableop_resource:
��7
(dense_21_biasadd_readvariableop_resource:	�M
>batch_normalization_16_assignmovingavg_readvariableop_resource:	�O
@batch_normalization_16_assignmovingavg_1_readvariableop_resource:	�B
3batch_normalization_16_cast_readvariableop_resource:	�D
5batch_normalization_16_cast_1_readvariableop_resource:	�;
'dense_22_matmul_readvariableop_resource:
��7
(dense_22_biasadd_readvariableop_resource:	�M
>batch_normalization_17_assignmovingavg_readvariableop_resource:	�O
@batch_normalization_17_assignmovingavg_1_readvariableop_resource:	�B
3batch_normalization_17_cast_readvariableop_resource:	�D
5batch_normalization_17_cast_1_readvariableop_resource:	�:
'dense_23_matmul_readvariableop_resource:	�6
(dense_23_biasadd_readvariableop_resource:
identity��&batch_normalization_15/AssignMovingAvg�5batch_normalization_15/AssignMovingAvg/ReadVariableOp�(batch_normalization_15/AssignMovingAvg_1�7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_15/Cast/ReadVariableOp�,batch_normalization_15/Cast_1/ReadVariableOp�&batch_normalization_16/AssignMovingAvg�5batch_normalization_16/AssignMovingAvg/ReadVariableOp�(batch_normalization_16/AssignMovingAvg_1�7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_16/Cast/ReadVariableOp�,batch_normalization_16/Cast_1/ReadVariableOp�&batch_normalization_17/AssignMovingAvg�5batch_normalization_17/AssignMovingAvg/ReadVariableOp�(batch_normalization_17/AssignMovingAvg_1�7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_17/Cast/ReadVariableOp�,batch_normalization_17/Cast_1/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOpd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_maska
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten_10/ReshapeReshapestrided_slice:output:0flatten_10/Const:output:0*
T0*'
_output_shapes
:���������f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_maska
flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten_11/ReshapeReshapestrided_slice_1:output:0flatten_11/Const:output:0*
T0*'
_output_shapes
:���������[
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_5/concatConcatV2flatten_10/Reshape:output:0flatten_11/Reshape:output:0"concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_20/MatMulMatMulconcatenate_5/concat:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_20/TanhTanhdense_20/BiasAdd:output:0*
T0*(
_output_shapes
:����������
5batch_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_15/moments/meanMeandense_20/Tanh:y:0>batch_normalization_15/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
+batch_normalization_15/moments/StopGradientStopGradient,batch_normalization_15/moments/mean:output:0*
T0*
_output_shapes
:	��
0batch_normalization_15/moments/SquaredDifferenceSquaredDifferencedense_20/Tanh:y:04batch_normalization_15/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
9batch_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_15/moments/varianceMean4batch_normalization_15/moments/SquaredDifference:z:0Bbatch_normalization_15/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
&batch_normalization_15/moments/SqueezeSqueeze,batch_normalization_15/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
(batch_normalization_15/moments/Squeeze_1Squeeze0batch_normalization_15/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 q
,batch_normalization_15/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_15/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_15_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*batch_normalization_15/AssignMovingAvg/subSub=batch_normalization_15/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_15/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
*batch_normalization_15/AssignMovingAvg/mulMul.batch_normalization_15/AssignMovingAvg/sub:z:05batch_normalization_15/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
&batch_normalization_15/AssignMovingAvgAssignSubVariableOp>batch_normalization_15_assignmovingavg_readvariableop_resource.batch_normalization_15/AssignMovingAvg/mul:z:06^batch_normalization_15/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_15/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_15_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_15/AssignMovingAvg_1/subSub?batch_normalization_15/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_15/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
,batch_normalization_15/AssignMovingAvg_1/mulMul0batch_normalization_15/AssignMovingAvg_1/sub:z:07batch_normalization_15/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
(batch_normalization_15/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_15_assignmovingavg_1_readvariableop_resource0batch_normalization_15/AssignMovingAvg_1/mul:z:08^batch_normalization_15/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
*batch_normalization_15/Cast/ReadVariableOpReadVariableOp3batch_normalization_15_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_15/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_15_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_15/batchnorm/addAddV21batch_normalization_15/moments/Squeeze_1:output:0/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_15/batchnorm/RsqrtRsqrt(batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_15/batchnorm/mulMul*batch_normalization_15/batchnorm/Rsqrt:y:04batch_normalization_15/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_15/batchnorm/mul_1Muldense_20/Tanh:y:0(batch_normalization_15/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
&batch_normalization_15/batchnorm/mul_2Mul/batch_normalization_15/moments/Squeeze:output:0(batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
$batch_normalization_15/batchnorm/subSub2batch_normalization_15/Cast/ReadVariableOp:value:0*batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
&batch_normalization_15/batchnorm/add_1AddV2*batch_normalization_15/batchnorm/mul_1:z:0(batch_normalization_15/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_21/MatMulMatMul*batch_normalization_15/batchnorm/add_1:z:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_21/TanhTanhdense_21/BiasAdd:output:0*
T0*(
_output_shapes
:����������
5batch_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_16/moments/meanMeandense_21/Tanh:y:0>batch_normalization_16/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
+batch_normalization_16/moments/StopGradientStopGradient,batch_normalization_16/moments/mean:output:0*
T0*
_output_shapes
:	��
0batch_normalization_16/moments/SquaredDifferenceSquaredDifferencedense_21/Tanh:y:04batch_normalization_16/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
9batch_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_16/moments/varianceMean4batch_normalization_16/moments/SquaredDifference:z:0Bbatch_normalization_16/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
&batch_normalization_16/moments/SqueezeSqueeze,batch_normalization_16/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
(batch_normalization_16/moments/Squeeze_1Squeeze0batch_normalization_16/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 q
,batch_normalization_16/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_16/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_16_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*batch_normalization_16/AssignMovingAvg/subSub=batch_normalization_16/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_16/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
*batch_normalization_16/AssignMovingAvg/mulMul.batch_normalization_16/AssignMovingAvg/sub:z:05batch_normalization_16/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
&batch_normalization_16/AssignMovingAvgAssignSubVariableOp>batch_normalization_16_assignmovingavg_readvariableop_resource.batch_normalization_16/AssignMovingAvg/mul:z:06^batch_normalization_16/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_16/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_16_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_16/AssignMovingAvg_1/subSub?batch_normalization_16/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_16/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
,batch_normalization_16/AssignMovingAvg_1/mulMul0batch_normalization_16/AssignMovingAvg_1/sub:z:07batch_normalization_16/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
(batch_normalization_16/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_16_assignmovingavg_1_readvariableop_resource0batch_normalization_16/AssignMovingAvg_1/mul:z:08^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
*batch_normalization_16/Cast/ReadVariableOpReadVariableOp3batch_normalization_16_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_16/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_16_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_16/batchnorm/addAddV21batch_normalization_16/moments/Squeeze_1:output:0/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_16/batchnorm/RsqrtRsqrt(batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_16/batchnorm/mulMul*batch_normalization_16/batchnorm/Rsqrt:y:04batch_normalization_16/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_16/batchnorm/mul_1Muldense_21/Tanh:y:0(batch_normalization_16/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
&batch_normalization_16/batchnorm/mul_2Mul/batch_normalization_16/moments/Squeeze:output:0(batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
$batch_normalization_16/batchnorm/subSub2batch_normalization_16/Cast/ReadVariableOp:value:0*batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
&batch_normalization_16/batchnorm/add_1AddV2*batch_normalization_16/batchnorm/mul_1:z:0(batch_normalization_16/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_22/MatMulMatMul*batch_normalization_16/batchnorm/add_1:z:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_22/TanhTanhdense_22/BiasAdd:output:0*
T0*(
_output_shapes
:����������
5batch_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_17/moments/meanMeandense_22/Tanh:y:0>batch_normalization_17/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
+batch_normalization_17/moments/StopGradientStopGradient,batch_normalization_17/moments/mean:output:0*
T0*
_output_shapes
:	��
0batch_normalization_17/moments/SquaredDifferenceSquaredDifferencedense_22/Tanh:y:04batch_normalization_17/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
9batch_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_17/moments/varianceMean4batch_normalization_17/moments/SquaredDifference:z:0Bbatch_normalization_17/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
&batch_normalization_17/moments/SqueezeSqueeze,batch_normalization_17/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
(batch_normalization_17/moments/Squeeze_1Squeeze0batch_normalization_17/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 q
,batch_normalization_17/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_17/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_17_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*batch_normalization_17/AssignMovingAvg/subSub=batch_normalization_17/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_17/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
*batch_normalization_17/AssignMovingAvg/mulMul.batch_normalization_17/AssignMovingAvg/sub:z:05batch_normalization_17/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
&batch_normalization_17/AssignMovingAvgAssignSubVariableOp>batch_normalization_17_assignmovingavg_readvariableop_resource.batch_normalization_17/AssignMovingAvg/mul:z:06^batch_normalization_17/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_17/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_17_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_17/AssignMovingAvg_1/subSub?batch_normalization_17/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_17/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
,batch_normalization_17/AssignMovingAvg_1/mulMul0batch_normalization_17/AssignMovingAvg_1/sub:z:07batch_normalization_17/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
(batch_normalization_17/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_17_assignmovingavg_1_readvariableop_resource0batch_normalization_17/AssignMovingAvg_1/mul:z:08^batch_normalization_17/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
*batch_normalization_17/Cast/ReadVariableOpReadVariableOp3batch_normalization_17_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_17/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_17_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_17/batchnorm/addAddV21batch_normalization_17/moments/Squeeze_1:output:0/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_17/batchnorm/RsqrtRsqrt(batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_17/batchnorm/mulMul*batch_normalization_17/batchnorm/Rsqrt:y:04batch_normalization_17/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_17/batchnorm/mul_1Muldense_22/Tanh:y:0(batch_normalization_17/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
&batch_normalization_17/batchnorm/mul_2Mul/batch_normalization_17/moments/Squeeze:output:0(batch_normalization_17/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
$batch_normalization_17/batchnorm/subSub2batch_normalization_17/Cast/ReadVariableOp:value:0*batch_normalization_17/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
&batch_normalization_17/batchnorm/add_1AddV2*batch_normalization_17/batchnorm/mul_1:z:0(batch_normalization_17/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_23/MatMulMatMul*batch_normalization_17/batchnorm/add_1:z:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_23/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������	
NoOpNoOp'^batch_normalization_15/AssignMovingAvg6^batch_normalization_15/AssignMovingAvg/ReadVariableOp)^batch_normalization_15/AssignMovingAvg_18^batch_normalization_15/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_15/Cast/ReadVariableOp-^batch_normalization_15/Cast_1/ReadVariableOp'^batch_normalization_16/AssignMovingAvg6^batch_normalization_16/AssignMovingAvg/ReadVariableOp)^batch_normalization_16/AssignMovingAvg_18^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_16/Cast/ReadVariableOp-^batch_normalization_16/Cast_1/ReadVariableOp'^batch_normalization_17/AssignMovingAvg6^batch_normalization_17/AssignMovingAvg/ReadVariableOp)^batch_normalization_17/AssignMovingAvg_18^batch_normalization_17/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_17/Cast/ReadVariableOp-^batch_normalization_17/Cast_1/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_15/AssignMovingAvg&batch_normalization_15/AssignMovingAvg2n
5batch_normalization_15/AssignMovingAvg/ReadVariableOp5batch_normalization_15/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_15/AssignMovingAvg_1(batch_normalization_15/AssignMovingAvg_12r
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_15/Cast/ReadVariableOp*batch_normalization_15/Cast/ReadVariableOp2\
,batch_normalization_15/Cast_1/ReadVariableOp,batch_normalization_15/Cast_1/ReadVariableOp2P
&batch_normalization_16/AssignMovingAvg&batch_normalization_16/AssignMovingAvg2n
5batch_normalization_16/AssignMovingAvg/ReadVariableOp5batch_normalization_16/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_16/AssignMovingAvg_1(batch_normalization_16/AssignMovingAvg_12r
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_16/Cast/ReadVariableOp*batch_normalization_16/Cast/ReadVariableOp2\
,batch_normalization_16/Cast_1/ReadVariableOp,batch_normalization_16/Cast_1/ReadVariableOp2P
&batch_normalization_17/AssignMovingAvg&batch_normalization_17/AssignMovingAvg2n
5batch_normalization_17/AssignMovingAvg/ReadVariableOp5batch_normalization_17/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_17/AssignMovingAvg_1(batch_normalization_17/AssignMovingAvg_12r
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_17/Cast/ReadVariableOp*batch_normalization_17/Cast/ReadVariableOp2\
,batch_normalization_17/Cast_1/ReadVariableOp,batch_normalization_17/Cast_1/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_3394675

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�-
cast_2_readvariableop_resource:	�-
cast_3_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpm
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:u
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������l
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�n
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�>
�	
G__inference_mlp_base_2_layer_call_and_return_conditional_losses_3394087
input_1#
dense_20_3394039:	�
dense_20_3394041:	�-
batch_normalization_15_3394044:	�-
batch_normalization_15_3394046:	�-
batch_normalization_15_3394048:	�-
batch_normalization_15_3394050:	�$
dense_21_3394053:
��
dense_21_3394055:	�-
batch_normalization_16_3394058:	�-
batch_normalization_16_3394060:	�-
batch_normalization_16_3394062:	�-
batch_normalization_16_3394064:	�$
dense_22_3394067:
��
dense_22_3394069:	�-
batch_normalization_17_3394072:	�-
batch_normalization_17_3394074:	�-
batch_normalization_17_3394076:	�-
batch_normalization_17_3394078:	�#
dense_23_3394081:	�
dense_23_3394083:
identity��.batch_normalization_15/StatefulPartitionedCall�.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCalld
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
flatten_10/PartitionedCallPartitionedCallstrided_slice:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_10_layer_call_and_return_conditional_losses_3393608f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSliceinput_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
flatten_11/PartitionedCallPartitionedCallstrided_slice_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_11_layer_call_and_return_conditional_losses_3393620�
concatenate_5/PartitionedCallPartitionedCall#flatten_10/PartitionedCall:output:0#flatten_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_concatenate_5_layer_call_and_return_conditional_losses_3393629�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0dense_20_3394039dense_20_3394041*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_3393642�
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0batch_normalization_15_3394044batch_normalization_15_3394046batch_normalization_15_3394048batch_normalization_15_3394050*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3393369�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0dense_21_3394053dense_21_3394055*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_3393668�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0batch_normalization_16_3394058batch_normalization_16_3394060batch_normalization_16_3394062batch_normalization_16_3394064*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_3393451�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0dense_22_3394067dense_22_3394069*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_3393694�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0batch_normalization_17_3394072batch_normalization_17_3394074batch_normalization_17_3394076batch_normalization_17_3394078*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_3393533�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0dense_23_3394081dense_23_3394083*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_3393719x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
t
J__inference_concatenate_5_layer_call_and_return_conditional_losses_3393629

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�$
�
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3394629

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0m
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�l
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_3394196
input_1
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_3393345o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�

�
E__inference_dense_21_layer_call_and_return_conditional_losses_3394829

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_dense_22_layer_call_fn_3394838

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_3393694p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
H
,__inference_flatten_10_layer_call_fn_3394519

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_10_layer_call_and_return_conditional_losses_3393608`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_17_layer_call_fn_3394735

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_3393580p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�$
�
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_3394709

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0m
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�l
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
,__inference_mlp_base_2_layer_call_fn_3394025
input_1
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_mlp_base_2_layer_call_and_return_conditional_losses_3393937o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_3393533

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�-
cast_2_readvariableop_resource:	�-
cast_3_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpm
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:u
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������l
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�n
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�$
�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_3393580

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0m
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�l
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_3394755

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�-
cast_2_readvariableop_resource:	�-
cast_3_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpm
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:u
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������l
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�n
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
G__inference_flatten_10_layer_call_and_return_conditional_losses_3394525

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�$
�
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3393416

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0m
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�l
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_dense_22_layer_call_and_return_conditional_losses_3394849

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�>
�	
G__inference_mlp_base_2_layer_call_and_return_conditional_losses_3393726

inputs#
dense_20_3393643:	�
dense_20_3393645:	�-
batch_normalization_15_3393648:	�-
batch_normalization_15_3393650:	�-
batch_normalization_15_3393652:	�-
batch_normalization_15_3393654:	�$
dense_21_3393669:
��
dense_21_3393671:	�-
batch_normalization_16_3393674:	�-
batch_normalization_16_3393676:	�-
batch_normalization_16_3393678:	�-
batch_normalization_16_3393680:	�$
dense_22_3393695:
��
dense_22_3393697:	�-
batch_normalization_17_3393700:	�-
batch_normalization_17_3393702:	�-
batch_normalization_17_3393704:	�-
batch_normalization_17_3393706:	�#
dense_23_3393720:	�
dense_23_3393722:
identity��.batch_normalization_15/StatefulPartitionedCall�.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCalld
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
flatten_10/PartitionedCallPartitionedCallstrided_slice:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_10_layer_call_and_return_conditional_losses_3393608f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
flatten_11/PartitionedCallPartitionedCallstrided_slice_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_11_layer_call_and_return_conditional_losses_3393620�
concatenate_5/PartitionedCallPartitionedCall#flatten_10/PartitionedCall:output:0#flatten_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_concatenate_5_layer_call_and_return_conditional_losses_3393629�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0dense_20_3393643dense_20_3393645*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_3393642�
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0batch_normalization_15_3393648batch_normalization_15_3393650batch_normalization_15_3393652batch_normalization_15_3393654*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3393369�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0dense_21_3393669dense_21_3393671*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_3393668�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0batch_normalization_16_3393674batch_normalization_16_3393676batch_normalization_16_3393678batch_normalization_16_3393680*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_3393451�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0dense_22_3393695dense_22_3393697*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_3393694�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0batch_normalization_17_3393700batch_normalization_17_3393702batch_normalization_17_3393704batch_normalization_17_3393706*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_3393533�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0dense_23_3393720dense_23_3393722*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_3393719x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3394595

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�-
cast_2_readvariableop_resource:	�-
cast_3_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpm
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:u
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������l
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�n
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�y
�
G__inference_mlp_base_2_layer_call_and_return_conditional_losses_3394379

inputs:
'dense_20_matmul_readvariableop_resource:	�7
(dense_20_biasadd_readvariableop_resource:	�B
3batch_normalization_15_cast_readvariableop_resource:	�D
5batch_normalization_15_cast_1_readvariableop_resource:	�D
5batch_normalization_15_cast_2_readvariableop_resource:	�D
5batch_normalization_15_cast_3_readvariableop_resource:	�;
'dense_21_matmul_readvariableop_resource:
��7
(dense_21_biasadd_readvariableop_resource:	�B
3batch_normalization_16_cast_readvariableop_resource:	�D
5batch_normalization_16_cast_1_readvariableop_resource:	�D
5batch_normalization_16_cast_2_readvariableop_resource:	�D
5batch_normalization_16_cast_3_readvariableop_resource:	�;
'dense_22_matmul_readvariableop_resource:
��7
(dense_22_biasadd_readvariableop_resource:	�B
3batch_normalization_17_cast_readvariableop_resource:	�D
5batch_normalization_17_cast_1_readvariableop_resource:	�D
5batch_normalization_17_cast_2_readvariableop_resource:	�D
5batch_normalization_17_cast_3_readvariableop_resource:	�:
'dense_23_matmul_readvariableop_resource:	�6
(dense_23_biasadd_readvariableop_resource:
identity��*batch_normalization_15/Cast/ReadVariableOp�,batch_normalization_15/Cast_1/ReadVariableOp�,batch_normalization_15/Cast_2/ReadVariableOp�,batch_normalization_15/Cast_3/ReadVariableOp�*batch_normalization_16/Cast/ReadVariableOp�,batch_normalization_16/Cast_1/ReadVariableOp�,batch_normalization_16/Cast_2/ReadVariableOp�,batch_normalization_16/Cast_3/ReadVariableOp�*batch_normalization_17/Cast/ReadVariableOp�,batch_normalization_17/Cast_1/ReadVariableOp�,batch_normalization_17/Cast_2/ReadVariableOp�,batch_normalization_17/Cast_3/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOpd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_maska
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten_10/ReshapeReshapestrided_slice:output:0flatten_10/Const:output:0*
T0*'
_output_shapes
:���������f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_maska
flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten_11/ReshapeReshapestrided_slice_1:output:0flatten_11/Const:output:0*
T0*'
_output_shapes
:���������[
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_5/concatConcatV2flatten_10/Reshape:output:0flatten_11/Reshape:output:0"concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_20/MatMulMatMulconcatenate_5/concat:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_20/TanhTanhdense_20/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
*batch_normalization_15/Cast/ReadVariableOpReadVariableOp3batch_normalization_15_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_15/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_15_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_15/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_15_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_15/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_15_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_15/batchnorm/addAddV24batch_normalization_15/Cast_1/ReadVariableOp:value:0/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_15/batchnorm/RsqrtRsqrt(batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_15/batchnorm/mulMul*batch_normalization_15/batchnorm/Rsqrt:y:04batch_normalization_15/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_15/batchnorm/mul_1Muldense_20/Tanh:y:0(batch_normalization_15/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
&batch_normalization_15/batchnorm/mul_2Mul2batch_normalization_15/Cast/ReadVariableOp:value:0(batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
$batch_normalization_15/batchnorm/subSub4batch_normalization_15/Cast_2/ReadVariableOp:value:0*batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
&batch_normalization_15/batchnorm/add_1AddV2*batch_normalization_15/batchnorm/mul_1:z:0(batch_normalization_15/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_21/MatMulMatMul*batch_normalization_15/batchnorm/add_1:z:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_21/TanhTanhdense_21/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
*batch_normalization_16/Cast/ReadVariableOpReadVariableOp3batch_normalization_16_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_16/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_16_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_16/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_16_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_16/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_16_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_16/batchnorm/addAddV24batch_normalization_16/Cast_1/ReadVariableOp:value:0/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_16/batchnorm/RsqrtRsqrt(batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_16/batchnorm/mulMul*batch_normalization_16/batchnorm/Rsqrt:y:04batch_normalization_16/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_16/batchnorm/mul_1Muldense_21/Tanh:y:0(batch_normalization_16/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
&batch_normalization_16/batchnorm/mul_2Mul2batch_normalization_16/Cast/ReadVariableOp:value:0(batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
$batch_normalization_16/batchnorm/subSub4batch_normalization_16/Cast_2/ReadVariableOp:value:0*batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
&batch_normalization_16/batchnorm/add_1AddV2*batch_normalization_16/batchnorm/mul_1:z:0(batch_normalization_16/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_22/MatMulMatMul*batch_normalization_16/batchnorm/add_1:z:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_22/TanhTanhdense_22/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
*batch_normalization_17/Cast/ReadVariableOpReadVariableOp3batch_normalization_17_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_17/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_17_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_17/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_17_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_17/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_17_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_17/batchnorm/addAddV24batch_normalization_17/Cast_1/ReadVariableOp:value:0/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_17/batchnorm/RsqrtRsqrt(batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_17/batchnorm/mulMul*batch_normalization_17/batchnorm/Rsqrt:y:04batch_normalization_17/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_17/batchnorm/mul_1Muldense_22/Tanh:y:0(batch_normalization_17/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
&batch_normalization_17/batchnorm/mul_2Mul2batch_normalization_17/Cast/ReadVariableOp:value:0(batch_normalization_17/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
$batch_normalization_17/batchnorm/subSub4batch_normalization_17/Cast_2/ReadVariableOp:value:0*batch_normalization_17/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
&batch_normalization_17/batchnorm/add_1AddV2*batch_normalization_17/batchnorm/mul_1:z:0(batch_normalization_17/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_23/MatMulMatMul*batch_normalization_17/batchnorm/add_1:z:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_23/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp+^batch_normalization_15/Cast/ReadVariableOp-^batch_normalization_15/Cast_1/ReadVariableOp-^batch_normalization_15/Cast_2/ReadVariableOp-^batch_normalization_15/Cast_3/ReadVariableOp+^batch_normalization_16/Cast/ReadVariableOp-^batch_normalization_16/Cast_1/ReadVariableOp-^batch_normalization_16/Cast_2/ReadVariableOp-^batch_normalization_16/Cast_3/ReadVariableOp+^batch_normalization_17/Cast/ReadVariableOp-^batch_normalization_17/Cast_1/ReadVariableOp-^batch_normalization_17/Cast_2/ReadVariableOp-^batch_normalization_17/Cast_3/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : 2X
*batch_normalization_15/Cast/ReadVariableOp*batch_normalization_15/Cast/ReadVariableOp2\
,batch_normalization_15/Cast_1/ReadVariableOp,batch_normalization_15/Cast_1/ReadVariableOp2\
,batch_normalization_15/Cast_2/ReadVariableOp,batch_normalization_15/Cast_2/ReadVariableOp2\
,batch_normalization_15/Cast_3/ReadVariableOp,batch_normalization_15/Cast_3/ReadVariableOp2X
*batch_normalization_16/Cast/ReadVariableOp*batch_normalization_16/Cast/ReadVariableOp2\
,batch_normalization_16/Cast_1/ReadVariableOp,batch_normalization_16/Cast_1/ReadVariableOp2\
,batch_normalization_16/Cast_2/ReadVariableOp,batch_normalization_16/Cast_2/ReadVariableOp2\
,batch_normalization_16/Cast_3/ReadVariableOp,batch_normalization_16/Cast_3/ReadVariableOp2X
*batch_normalization_17/Cast/ReadVariableOp*batch_normalization_17/Cast/ReadVariableOp2\
,batch_normalization_17/Cast_1/ReadVariableOp,batch_normalization_17/Cast_1/ReadVariableOp2\
,batch_normalization_17/Cast_2/ReadVariableOp,batch_normalization_17/Cast_2/ReadVariableOp2\
,batch_normalization_17/Cast_3/ReadVariableOp,batch_normalization_17/Cast_3/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_16_layer_call_fn_3394642

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_3393451p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
G__inference_flatten_10_layer_call_and_return_conditional_losses_3393608

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
v
J__inference_concatenate_5_layer_call_and_return_conditional_losses_3394549
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
Џ
�
"__inference__wrapped_model_3393345
input_1E
2mlp_base_2_dense_20_matmul_readvariableop_resource:	�B
3mlp_base_2_dense_20_biasadd_readvariableop_resource:	�M
>mlp_base_2_batch_normalization_15_cast_readvariableop_resource:	�O
@mlp_base_2_batch_normalization_15_cast_1_readvariableop_resource:	�O
@mlp_base_2_batch_normalization_15_cast_2_readvariableop_resource:	�O
@mlp_base_2_batch_normalization_15_cast_3_readvariableop_resource:	�F
2mlp_base_2_dense_21_matmul_readvariableop_resource:
��B
3mlp_base_2_dense_21_biasadd_readvariableop_resource:	�M
>mlp_base_2_batch_normalization_16_cast_readvariableop_resource:	�O
@mlp_base_2_batch_normalization_16_cast_1_readvariableop_resource:	�O
@mlp_base_2_batch_normalization_16_cast_2_readvariableop_resource:	�O
@mlp_base_2_batch_normalization_16_cast_3_readvariableop_resource:	�F
2mlp_base_2_dense_22_matmul_readvariableop_resource:
��B
3mlp_base_2_dense_22_biasadd_readvariableop_resource:	�M
>mlp_base_2_batch_normalization_17_cast_readvariableop_resource:	�O
@mlp_base_2_batch_normalization_17_cast_1_readvariableop_resource:	�O
@mlp_base_2_batch_normalization_17_cast_2_readvariableop_resource:	�O
@mlp_base_2_batch_normalization_17_cast_3_readvariableop_resource:	�E
2mlp_base_2_dense_23_matmul_readvariableop_resource:	�A
3mlp_base_2_dense_23_biasadd_readvariableop_resource:
identity��5mlp_base_2/batch_normalization_15/Cast/ReadVariableOp�7mlp_base_2/batch_normalization_15/Cast_1/ReadVariableOp�7mlp_base_2/batch_normalization_15/Cast_2/ReadVariableOp�7mlp_base_2/batch_normalization_15/Cast_3/ReadVariableOp�5mlp_base_2/batch_normalization_16/Cast/ReadVariableOp�7mlp_base_2/batch_normalization_16/Cast_1/ReadVariableOp�7mlp_base_2/batch_normalization_16/Cast_2/ReadVariableOp�7mlp_base_2/batch_normalization_16/Cast_3/ReadVariableOp�5mlp_base_2/batch_normalization_17/Cast/ReadVariableOp�7mlp_base_2/batch_normalization_17/Cast_1/ReadVariableOp�7mlp_base_2/batch_normalization_17/Cast_2/ReadVariableOp�7mlp_base_2/batch_normalization_17/Cast_3/ReadVariableOp�*mlp_base_2/dense_20/BiasAdd/ReadVariableOp�)mlp_base_2/dense_20/MatMul/ReadVariableOp�*mlp_base_2/dense_21/BiasAdd/ReadVariableOp�)mlp_base_2/dense_21/MatMul/ReadVariableOp�*mlp_base_2/dense_22/BiasAdd/ReadVariableOp�)mlp_base_2/dense_22/MatMul/ReadVariableOp�*mlp_base_2/dense_23/BiasAdd/ReadVariableOp�)mlp_base_2/dense_23/MatMul/ReadVariableOpo
mlp_base_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        q
 mlp_base_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       q
 mlp_base_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
mlp_base_2/strided_sliceStridedSliceinput_1'mlp_base_2/strided_slice/stack:output:0)mlp_base_2/strided_slice/stack_1:output:0)mlp_base_2/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_maskl
mlp_base_2/flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
mlp_base_2/flatten_10/ReshapeReshape!mlp_base_2/strided_slice:output:0$mlp_base_2/flatten_10/Const:output:0*
T0*'
_output_shapes
:���������q
 mlp_base_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"mlp_base_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"mlp_base_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
mlp_base_2/strided_slice_1StridedSliceinput_1)mlp_base_2/strided_slice_1/stack:output:0+mlp_base_2/strided_slice_1/stack_1:output:0+mlp_base_2/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_maskl
mlp_base_2/flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
mlp_base_2/flatten_11/ReshapeReshape#mlp_base_2/strided_slice_1:output:0$mlp_base_2/flatten_11/Const:output:0*
T0*'
_output_shapes
:���������f
$mlp_base_2/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
mlp_base_2/concatenate_5/concatConcatV2&mlp_base_2/flatten_10/Reshape:output:0&mlp_base_2/flatten_11/Reshape:output:0-mlp_base_2/concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
)mlp_base_2/dense_20/MatMul/ReadVariableOpReadVariableOp2mlp_base_2_dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
mlp_base_2/dense_20/MatMulMatMul(mlp_base_2/concatenate_5/concat:output:01mlp_base_2/dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*mlp_base_2/dense_20/BiasAdd/ReadVariableOpReadVariableOp3mlp_base_2_dense_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
mlp_base_2/dense_20/BiasAddBiasAdd$mlp_base_2/dense_20/MatMul:product:02mlp_base_2/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
mlp_base_2/dense_20/TanhTanh$mlp_base_2/dense_20/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
5mlp_base_2/batch_normalization_15/Cast/ReadVariableOpReadVariableOp>mlp_base_2_batch_normalization_15_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7mlp_base_2/batch_normalization_15/Cast_1/ReadVariableOpReadVariableOp@mlp_base_2_batch_normalization_15_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7mlp_base_2/batch_normalization_15/Cast_2/ReadVariableOpReadVariableOp@mlp_base_2_batch_normalization_15_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7mlp_base_2/batch_normalization_15/Cast_3/ReadVariableOpReadVariableOp@mlp_base_2_batch_normalization_15_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0v
1mlp_base_2/batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
/mlp_base_2/batch_normalization_15/batchnorm/addAddV2?mlp_base_2/batch_normalization_15/Cast_1/ReadVariableOp:value:0:mlp_base_2/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
1mlp_base_2/batch_normalization_15/batchnorm/RsqrtRsqrt3mlp_base_2/batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes	
:��
/mlp_base_2/batch_normalization_15/batchnorm/mulMul5mlp_base_2/batch_normalization_15/batchnorm/Rsqrt:y:0?mlp_base_2/batch_normalization_15/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
1mlp_base_2/batch_normalization_15/batchnorm/mul_1Mulmlp_base_2/dense_20/Tanh:y:03mlp_base_2/batch_normalization_15/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
1mlp_base_2/batch_normalization_15/batchnorm/mul_2Mul=mlp_base_2/batch_normalization_15/Cast/ReadVariableOp:value:03mlp_base_2/batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
/mlp_base_2/batch_normalization_15/batchnorm/subSub?mlp_base_2/batch_normalization_15/Cast_2/ReadVariableOp:value:05mlp_base_2/batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
1mlp_base_2/batch_normalization_15/batchnorm/add_1AddV25mlp_base_2/batch_normalization_15/batchnorm/mul_1:z:03mlp_base_2/batch_normalization_15/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
)mlp_base_2/dense_21/MatMul/ReadVariableOpReadVariableOp2mlp_base_2_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
mlp_base_2/dense_21/MatMulMatMul5mlp_base_2/batch_normalization_15/batchnorm/add_1:z:01mlp_base_2/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*mlp_base_2/dense_21/BiasAdd/ReadVariableOpReadVariableOp3mlp_base_2_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
mlp_base_2/dense_21/BiasAddBiasAdd$mlp_base_2/dense_21/MatMul:product:02mlp_base_2/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
mlp_base_2/dense_21/TanhTanh$mlp_base_2/dense_21/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
5mlp_base_2/batch_normalization_16/Cast/ReadVariableOpReadVariableOp>mlp_base_2_batch_normalization_16_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7mlp_base_2/batch_normalization_16/Cast_1/ReadVariableOpReadVariableOp@mlp_base_2_batch_normalization_16_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7mlp_base_2/batch_normalization_16/Cast_2/ReadVariableOpReadVariableOp@mlp_base_2_batch_normalization_16_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7mlp_base_2/batch_normalization_16/Cast_3/ReadVariableOpReadVariableOp@mlp_base_2_batch_normalization_16_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0v
1mlp_base_2/batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
/mlp_base_2/batch_normalization_16/batchnorm/addAddV2?mlp_base_2/batch_normalization_16/Cast_1/ReadVariableOp:value:0:mlp_base_2/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
1mlp_base_2/batch_normalization_16/batchnorm/RsqrtRsqrt3mlp_base_2/batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes	
:��
/mlp_base_2/batch_normalization_16/batchnorm/mulMul5mlp_base_2/batch_normalization_16/batchnorm/Rsqrt:y:0?mlp_base_2/batch_normalization_16/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
1mlp_base_2/batch_normalization_16/batchnorm/mul_1Mulmlp_base_2/dense_21/Tanh:y:03mlp_base_2/batch_normalization_16/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
1mlp_base_2/batch_normalization_16/batchnorm/mul_2Mul=mlp_base_2/batch_normalization_16/Cast/ReadVariableOp:value:03mlp_base_2/batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
/mlp_base_2/batch_normalization_16/batchnorm/subSub?mlp_base_2/batch_normalization_16/Cast_2/ReadVariableOp:value:05mlp_base_2/batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
1mlp_base_2/batch_normalization_16/batchnorm/add_1AddV25mlp_base_2/batch_normalization_16/batchnorm/mul_1:z:03mlp_base_2/batch_normalization_16/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
)mlp_base_2/dense_22/MatMul/ReadVariableOpReadVariableOp2mlp_base_2_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
mlp_base_2/dense_22/MatMulMatMul5mlp_base_2/batch_normalization_16/batchnorm/add_1:z:01mlp_base_2/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*mlp_base_2/dense_22/BiasAdd/ReadVariableOpReadVariableOp3mlp_base_2_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
mlp_base_2/dense_22/BiasAddBiasAdd$mlp_base_2/dense_22/MatMul:product:02mlp_base_2/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
mlp_base_2/dense_22/TanhTanh$mlp_base_2/dense_22/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
5mlp_base_2/batch_normalization_17/Cast/ReadVariableOpReadVariableOp>mlp_base_2_batch_normalization_17_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7mlp_base_2/batch_normalization_17/Cast_1/ReadVariableOpReadVariableOp@mlp_base_2_batch_normalization_17_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7mlp_base_2/batch_normalization_17/Cast_2/ReadVariableOpReadVariableOp@mlp_base_2_batch_normalization_17_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7mlp_base_2/batch_normalization_17/Cast_3/ReadVariableOpReadVariableOp@mlp_base_2_batch_normalization_17_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0v
1mlp_base_2/batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
/mlp_base_2/batch_normalization_17/batchnorm/addAddV2?mlp_base_2/batch_normalization_17/Cast_1/ReadVariableOp:value:0:mlp_base_2/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
1mlp_base_2/batch_normalization_17/batchnorm/RsqrtRsqrt3mlp_base_2/batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes	
:��
/mlp_base_2/batch_normalization_17/batchnorm/mulMul5mlp_base_2/batch_normalization_17/batchnorm/Rsqrt:y:0?mlp_base_2/batch_normalization_17/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
1mlp_base_2/batch_normalization_17/batchnorm/mul_1Mulmlp_base_2/dense_22/Tanh:y:03mlp_base_2/batch_normalization_17/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
1mlp_base_2/batch_normalization_17/batchnorm/mul_2Mul=mlp_base_2/batch_normalization_17/Cast/ReadVariableOp:value:03mlp_base_2/batch_normalization_17/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
/mlp_base_2/batch_normalization_17/batchnorm/subSub?mlp_base_2/batch_normalization_17/Cast_2/ReadVariableOp:value:05mlp_base_2/batch_normalization_17/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
1mlp_base_2/batch_normalization_17/batchnorm/add_1AddV25mlp_base_2/batch_normalization_17/batchnorm/mul_1:z:03mlp_base_2/batch_normalization_17/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
)mlp_base_2/dense_23/MatMul/ReadVariableOpReadVariableOp2mlp_base_2_dense_23_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
mlp_base_2/dense_23/MatMulMatMul5mlp_base_2/batch_normalization_17/batchnorm/add_1:z:01mlp_base_2/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
*mlp_base_2/dense_23/BiasAdd/ReadVariableOpReadVariableOp3mlp_base_2_dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
mlp_base_2/dense_23/BiasAddBiasAdd$mlp_base_2/dense_23/MatMul:product:02mlp_base_2/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������s
IdentityIdentity$mlp_base_2/dense_23/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp6^mlp_base_2/batch_normalization_15/Cast/ReadVariableOp8^mlp_base_2/batch_normalization_15/Cast_1/ReadVariableOp8^mlp_base_2/batch_normalization_15/Cast_2/ReadVariableOp8^mlp_base_2/batch_normalization_15/Cast_3/ReadVariableOp6^mlp_base_2/batch_normalization_16/Cast/ReadVariableOp8^mlp_base_2/batch_normalization_16/Cast_1/ReadVariableOp8^mlp_base_2/batch_normalization_16/Cast_2/ReadVariableOp8^mlp_base_2/batch_normalization_16/Cast_3/ReadVariableOp6^mlp_base_2/batch_normalization_17/Cast/ReadVariableOp8^mlp_base_2/batch_normalization_17/Cast_1/ReadVariableOp8^mlp_base_2/batch_normalization_17/Cast_2/ReadVariableOp8^mlp_base_2/batch_normalization_17/Cast_3/ReadVariableOp+^mlp_base_2/dense_20/BiasAdd/ReadVariableOp*^mlp_base_2/dense_20/MatMul/ReadVariableOp+^mlp_base_2/dense_21/BiasAdd/ReadVariableOp*^mlp_base_2/dense_21/MatMul/ReadVariableOp+^mlp_base_2/dense_22/BiasAdd/ReadVariableOp*^mlp_base_2/dense_22/MatMul/ReadVariableOp+^mlp_base_2/dense_23/BiasAdd/ReadVariableOp*^mlp_base_2/dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : 2n
5mlp_base_2/batch_normalization_15/Cast/ReadVariableOp5mlp_base_2/batch_normalization_15/Cast/ReadVariableOp2r
7mlp_base_2/batch_normalization_15/Cast_1/ReadVariableOp7mlp_base_2/batch_normalization_15/Cast_1/ReadVariableOp2r
7mlp_base_2/batch_normalization_15/Cast_2/ReadVariableOp7mlp_base_2/batch_normalization_15/Cast_2/ReadVariableOp2r
7mlp_base_2/batch_normalization_15/Cast_3/ReadVariableOp7mlp_base_2/batch_normalization_15/Cast_3/ReadVariableOp2n
5mlp_base_2/batch_normalization_16/Cast/ReadVariableOp5mlp_base_2/batch_normalization_16/Cast/ReadVariableOp2r
7mlp_base_2/batch_normalization_16/Cast_1/ReadVariableOp7mlp_base_2/batch_normalization_16/Cast_1/ReadVariableOp2r
7mlp_base_2/batch_normalization_16/Cast_2/ReadVariableOp7mlp_base_2/batch_normalization_16/Cast_2/ReadVariableOp2r
7mlp_base_2/batch_normalization_16/Cast_3/ReadVariableOp7mlp_base_2/batch_normalization_16/Cast_3/ReadVariableOp2n
5mlp_base_2/batch_normalization_17/Cast/ReadVariableOp5mlp_base_2/batch_normalization_17/Cast/ReadVariableOp2r
7mlp_base_2/batch_normalization_17/Cast_1/ReadVariableOp7mlp_base_2/batch_normalization_17/Cast_1/ReadVariableOp2r
7mlp_base_2/batch_normalization_17/Cast_2/ReadVariableOp7mlp_base_2/batch_normalization_17/Cast_2/ReadVariableOp2r
7mlp_base_2/batch_normalization_17/Cast_3/ReadVariableOp7mlp_base_2/batch_normalization_17/Cast_3/ReadVariableOp2X
*mlp_base_2/dense_20/BiasAdd/ReadVariableOp*mlp_base_2/dense_20/BiasAdd/ReadVariableOp2V
)mlp_base_2/dense_20/MatMul/ReadVariableOp)mlp_base_2/dense_20/MatMul/ReadVariableOp2X
*mlp_base_2/dense_21/BiasAdd/ReadVariableOp*mlp_base_2/dense_21/BiasAdd/ReadVariableOp2V
)mlp_base_2/dense_21/MatMul/ReadVariableOp)mlp_base_2/dense_21/MatMul/ReadVariableOp2X
*mlp_base_2/dense_22/BiasAdd/ReadVariableOp*mlp_base_2/dense_22/BiasAdd/ReadVariableOp2V
)mlp_base_2/dense_22/MatMul/ReadVariableOp)mlp_base_2/dense_22/MatMul/ReadVariableOp2X
*mlp_base_2/dense_23/BiasAdd/ReadVariableOp*mlp_base_2/dense_23/BiasAdd/ReadVariableOp2V
)mlp_base_2/dense_23/MatMul/ReadVariableOp)mlp_base_2/dense_23/MatMul/ReadVariableOp:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3393369

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�-
cast_2_readvariableop_resource:	�-
cast_3_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpm
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:u
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������l
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�n
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_dense_21_layer_call_and_return_conditional_losses_3393668

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_17_layer_call_fn_3394722

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_3393533p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_dense_23_layer_call_fn_3394858

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_3393719o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
H
,__inference_flatten_11_layer_call_fn_3394530

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_11_layer_call_and_return_conditional_losses_3393620`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
?
input_14
serving_default_input_1:0���������<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
setting

	input1


input2
concate
BN1
BN2
BN3

dense1

dense2

dense3

denseO

signatures"
_tf_keras_model
�
0
1
2
3
4
5
6
7
8
9
10
11
 12
!13
"14
#15
$16
%17
&18
'19"
trackable_list_wrapper
�
0
1
2
3
4
5
 6
!7
"8
#9
$10
%11
&12
'13"
trackable_list_wrapper
 "
trackable_list_wrapper
�
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
-trace_0
.trace_1
/trace_2
0trace_32�
,__inference_mlp_base_2_layer_call_fn_3393769
,__inference_mlp_base_2_layer_call_fn_3394241
,__inference_mlp_base_2_layer_call_fn_3394286
,__inference_mlp_base_2_layer_call_fn_3394025�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z-trace_0z.trace_1z/trace_2z0trace_3
�
1trace_0
2trace_1
3trace_2
4trace_32�
G__inference_mlp_base_2_layer_call_and_return_conditional_losses_3394379
G__inference_mlp_base_2_layer_call_and_return_conditional_losses_3394514
G__inference_mlp_base_2_layer_call_and_return_conditional_losses_3394087
G__inference_mlp_base_2_layer_call_and_return_conditional_losses_3394149�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z1trace_0z2trace_1z3trace_2z4trace_3
�B�
"__inference__wrapped_model_3393345input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_dict_wrapper
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses"
_tf_keras_layer
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses"
_tf_keras_layer
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses"
_tf_keras_layer
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses
Maxis
	gamma
beta
moving_mean
moving_variance"
_tf_keras_layer
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
Taxis
	gamma
beta
moving_mean
moving_variance"
_tf_keras_layer
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses
[axis
	gamma
beta
moving_mean
moving_variance"
_tf_keras_layer
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

 kernel
!bias"
_tf_keras_layer
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

"kernel
#bias"
_tf_keras_layer
�
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses

$kernel
%bias"
_tf_keras_layer
�
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses

&kernel
'bias"
_tf_keras_layer
,
tserving_default"
signature_map
6:4�2'mlp_base_2/batch_normalization_15/gamma
5:3�2&mlp_base_2/batch_normalization_15/beta
>:<� (2-mlp_base_2/batch_normalization_15/moving_mean
B:@� (21mlp_base_2/batch_normalization_15/moving_variance
6:4�2'mlp_base_2/batch_normalization_16/gamma
5:3�2&mlp_base_2/batch_normalization_16/beta
>:<� (2-mlp_base_2/batch_normalization_16/moving_mean
B:@� (21mlp_base_2/batch_normalization_16/moving_variance
6:4�2'mlp_base_2/batch_normalization_17/gamma
5:3�2&mlp_base_2/batch_normalization_17/beta
>:<� (2-mlp_base_2/batch_normalization_17/moving_mean
B:@� (21mlp_base_2/batch_normalization_17/moving_variance
-:+	�2mlp_base_2/dense_20/kernel
':%�2mlp_base_2/dense_20/bias
.:,
��2mlp_base_2/dense_21/kernel
':%�2mlp_base_2/dense_21/bias
.:,
��2mlp_base_2/dense_22/kernel
':%�2mlp_base_2/dense_22/bias
-:+	�2mlp_base_2/dense_23/kernel
&:$2mlp_base_2/dense_23/bias
J
0
1
2
3
4
5"
trackable_list_wrapper
f
	0

1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_mlp_base_2_layer_call_fn_3393769input_1"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_mlp_base_2_layer_call_fn_3394241inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_mlp_base_2_layer_call_fn_3394286inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_mlp_base_2_layer_call_fn_3394025input_1"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_mlp_base_2_layer_call_and_return_conditional_losses_3394379inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_mlp_base_2_layer_call_and_return_conditional_losses_3394514inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_mlp_base_2_layer_call_and_return_conditional_losses_3394087input_1"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_mlp_base_2_layer_call_and_return_conditional_losses_3394149input_1"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
�
ztrace_02�
,__inference_flatten_10_layer_call_fn_3394519�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zztrace_0
�
{trace_02�
G__inference_flatten_10_layer_call_and_return_conditional_losses_3394525�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z{trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
�layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_flatten_11_layer_call_fn_3394530�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_flatten_11_layer_call_and_return_conditional_losses_3394536�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_concatenate_5_layer_call_fn_3394542�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_concatenate_5_layer_call_and_return_conditional_losses_3394549�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
8__inference_batch_normalization_15_layer_call_fn_3394562
8__inference_batch_normalization_15_layer_call_fn_3394575�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3394595
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3394629�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
8__inference_batch_normalization_16_layer_call_fn_3394642
8__inference_batch_normalization_16_layer_call_fn_3394655�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_3394675
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_3394709�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
8__inference_batch_normalization_17_layer_call_fn_3394722
8__inference_batch_normalization_17_layer_call_fn_3394735�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_3394755
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_3394789�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_20_layer_call_fn_3394798�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_20_layer_call_and_return_conditional_losses_3394809�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_21_layer_call_fn_3394818�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_21_layer_call_and_return_conditional_losses_3394829�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_22_layer_call_fn_3394838�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_22_layer_call_and_return_conditional_losses_3394849�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_23_layer_call_fn_3394858�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_23_layer_call_and_return_conditional_losses_3394868�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�B�
%__inference_signature_wrapper_3394196input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_flatten_10_layer_call_fn_3394519inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_flatten_10_layer_call_and_return_conditional_losses_3394525inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_flatten_11_layer_call_fn_3394530inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_flatten_11_layer_call_and_return_conditional_losses_3394536inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_concatenate_5_layer_call_fn_3394542inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_concatenate_5_layer_call_and_return_conditional_losses_3394549inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
8__inference_batch_normalization_15_layer_call_fn_3394562inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
8__inference_batch_normalization_15_layer_call_fn_3394575inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3394595inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3394629inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
8__inference_batch_normalization_16_layer_call_fn_3394642inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
8__inference_batch_normalization_16_layer_call_fn_3394655inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_3394675inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_3394709inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
8__inference_batch_normalization_17_layer_call_fn_3394722inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
8__inference_batch_normalization_17_layer_call_fn_3394735inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_3394755inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_3394789inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dense_20_layer_call_fn_3394798inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_20_layer_call_and_return_conditional_losses_3394809inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dense_21_layer_call_fn_3394818inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_21_layer_call_and_return_conditional_losses_3394829inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dense_22_layer_call_fn_3394838inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_22_layer_call_and_return_conditional_losses_3394849inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dense_23_layer_call_fn_3394858inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_23_layer_call_and_return_conditional_losses_3394868inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
"__inference__wrapped_model_3393345� !"#$%&'4�1
*�'
%�"
input_1���������
� "3�0
.
output_1"�
output_1����������
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3394595d4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3394629d4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
8__inference_batch_normalization_15_layer_call_fn_3394562W4�1
*�'
!�
inputs����������
p 
� "������������
8__inference_batch_normalization_15_layer_call_fn_3394575W4�1
*�'
!�
inputs����������
p
� "������������
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_3394675d4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_3394709d4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
8__inference_batch_normalization_16_layer_call_fn_3394642W4�1
*�'
!�
inputs����������
p 
� "������������
8__inference_batch_normalization_16_layer_call_fn_3394655W4�1
*�'
!�
inputs����������
p
� "������������
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_3394755d4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_3394789d4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
8__inference_batch_normalization_17_layer_call_fn_3394722W4�1
*�'
!�
inputs����������
p 
� "������������
8__inference_batch_normalization_17_layer_call_fn_3394735W4�1
*�'
!�
inputs����������
p
� "������������
J__inference_concatenate_5_layer_call_and_return_conditional_losses_3394549�Z�W
P�M
K�H
"�
inputs/0���������
"�
inputs/1���������
� "%�"
�
0���������
� �
/__inference_concatenate_5_layer_call_fn_3394542vZ�W
P�M
K�H
"�
inputs/0���������
"�
inputs/1���������
� "�����������
E__inference_dense_20_layer_call_and_return_conditional_losses_3394809] !/�,
%�"
 �
inputs���������
� "&�#
�
0����������
� ~
*__inference_dense_20_layer_call_fn_3394798P !/�,
%�"
 �
inputs���������
� "������������
E__inference_dense_21_layer_call_and_return_conditional_losses_3394829^"#0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� 
*__inference_dense_21_layer_call_fn_3394818Q"#0�-
&�#
!�
inputs����������
� "������������
E__inference_dense_22_layer_call_and_return_conditional_losses_3394849^$%0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� 
*__inference_dense_22_layer_call_fn_3394838Q$%0�-
&�#
!�
inputs����������
� "������������
E__inference_dense_23_layer_call_and_return_conditional_losses_3394868]&'0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� ~
*__inference_dense_23_layer_call_fn_3394858P&'0�-
&�#
!�
inputs����������
� "�����������
G__inference_flatten_10_layer_call_and_return_conditional_losses_3394525X/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� {
,__inference_flatten_10_layer_call_fn_3394519K/�,
%�"
 �
inputs���������
� "�����������
G__inference_flatten_11_layer_call_and_return_conditional_losses_3394536X/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� {
,__inference_flatten_11_layer_call_fn_3394530K/�,
%�"
 �
inputs���������
� "�����������
G__inference_mlp_base_2_layer_call_and_return_conditional_losses_3394087w !"#$%&'8�5
.�+
%�"
input_1���������
p 
� "%�"
�
0���������
� �
G__inference_mlp_base_2_layer_call_and_return_conditional_losses_3394149w !"#$%&'8�5
.�+
%�"
input_1���������
p
� "%�"
�
0���������
� �
G__inference_mlp_base_2_layer_call_and_return_conditional_losses_3394379v !"#$%&'7�4
-�*
$�!
inputs���������
p 
� "%�"
�
0���������
� �
G__inference_mlp_base_2_layer_call_and_return_conditional_losses_3394514v !"#$%&'7�4
-�*
$�!
inputs���������
p
� "%�"
�
0���������
� �
,__inference_mlp_base_2_layer_call_fn_3393769j !"#$%&'8�5
.�+
%�"
input_1���������
p 
� "�����������
,__inference_mlp_base_2_layer_call_fn_3394025j !"#$%&'8�5
.�+
%�"
input_1���������
p
� "�����������
,__inference_mlp_base_2_layer_call_fn_3394241i !"#$%&'7�4
-�*
$�!
inputs���������
p 
� "�����������
,__inference_mlp_base_2_layer_call_fn_3394286i !"#$%&'7�4
-�*
$�!
inputs���������
p
� "�����������
%__inference_signature_wrapper_3394196� !"#$%&'?�<
� 
5�2
0
input_1%�"
input_1���������"3�0
.
output_1"�
output_1���������
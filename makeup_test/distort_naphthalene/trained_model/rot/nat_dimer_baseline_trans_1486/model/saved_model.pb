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
 �"serve*2.9.12v2.9.0-18-gd8ce9f9c3018�
�
mlp_base/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namemlp_base/dense_7/bias
{
)mlp_base/dense_7/bias/Read/ReadVariableOpReadVariableOpmlp_base/dense_7/bias*
_output_shapes
:*
dtype0
�
mlp_base/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*(
shared_namemlp_base/dense_7/kernel
�
+mlp_base/dense_7/kernel/Read/ReadVariableOpReadVariableOpmlp_base/dense_7/kernel*
_output_shapes
:	�*
dtype0
�
mlp_base/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_namemlp_base/dense_6/bias
|
)mlp_base/dense_6/bias/Read/ReadVariableOpReadVariableOpmlp_base/dense_6/bias*
_output_shapes	
:�*
dtype0
�
mlp_base/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*(
shared_namemlp_base/dense_6/kernel
�
+mlp_base/dense_6/kernel/Read/ReadVariableOpReadVariableOpmlp_base/dense_6/kernel* 
_output_shapes
:
��*
dtype0
�
mlp_base/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_namemlp_base/dense_5/bias
|
)mlp_base/dense_5/bias/Read/ReadVariableOpReadVariableOpmlp_base/dense_5/bias*
_output_shapes	
:�*
dtype0
�
mlp_base/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*(
shared_namemlp_base/dense_5/kernel
�
+mlp_base/dense_5/kernel/Read/ReadVariableOpReadVariableOpmlp_base/dense_5/kernel* 
_output_shapes
:
��*
dtype0
�
mlp_base/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_namemlp_base/dense_4/bias
|
)mlp_base/dense_4/bias/Read/ReadVariableOpReadVariableOpmlp_base/dense_4/bias*
_output_shapes	
:�*
dtype0
�
mlp_base/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*(
shared_namemlp_base/dense_4/kernel
�
+mlp_base/dense_4/kernel/Read/ReadVariableOpReadVariableOpmlp_base/dense_4/kernel*
_output_shapes
:	�*
dtype0
�
.mlp_base/batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*?
shared_name0.mlp_base/batch_normalization_5/moving_variance
�
Bmlp_base/batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp.mlp_base/batch_normalization_5/moving_variance*
_output_shapes	
:�*
dtype0
�
*mlp_base/batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*;
shared_name,*mlp_base/batch_normalization_5/moving_mean
�
>mlp_base/batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp*mlp_base/batch_normalization_5/moving_mean*
_output_shapes	
:�*
dtype0
�
#mlp_base/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#mlp_base/batch_normalization_5/beta
�
7mlp_base/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOp#mlp_base/batch_normalization_5/beta*
_output_shapes	
:�*
dtype0
�
$mlp_base/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*5
shared_name&$mlp_base/batch_normalization_5/gamma
�
8mlp_base/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOp$mlp_base/batch_normalization_5/gamma*
_output_shapes	
:�*
dtype0
�
.mlp_base/batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*?
shared_name0.mlp_base/batch_normalization_4/moving_variance
�
Bmlp_base/batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp.mlp_base/batch_normalization_4/moving_variance*
_output_shapes	
:�*
dtype0
�
*mlp_base/batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*;
shared_name,*mlp_base/batch_normalization_4/moving_mean
�
>mlp_base/batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp*mlp_base/batch_normalization_4/moving_mean*
_output_shapes	
:�*
dtype0
�
#mlp_base/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#mlp_base/batch_normalization_4/beta
�
7mlp_base/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp#mlp_base/batch_normalization_4/beta*
_output_shapes	
:�*
dtype0
�
$mlp_base/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*5
shared_name&$mlp_base/batch_normalization_4/gamma
�
8mlp_base/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp$mlp_base/batch_normalization_4/gamma*
_output_shapes	
:�*
dtype0
�
.mlp_base/batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*?
shared_name0.mlp_base/batch_normalization_3/moving_variance
�
Bmlp_base/batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp.mlp_base/batch_normalization_3/moving_variance*
_output_shapes	
:�*
dtype0
�
*mlp_base/batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*;
shared_name,*mlp_base/batch_normalization_3/moving_mean
�
>mlp_base/batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp*mlp_base/batch_normalization_3/moving_mean*
_output_shapes	
:�*
dtype0
�
#mlp_base/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#mlp_base/batch_normalization_3/beta
�
7mlp_base/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp#mlp_base/batch_normalization_3/beta*
_output_shapes	
:�*
dtype0
�
$mlp_base/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*5
shared_name&$mlp_base/batch_normalization_3/gamma
�
8mlp_base/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp$mlp_base/batch_normalization_3/gamma*
_output_shapes	
:�*
dtype0

NoOpNoOp
�F
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�F
value�FB�F B�E
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
d^
VARIABLE_VALUE$mlp_base/batch_normalization_3/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#mlp_base/batch_normalization_3/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE*mlp_base/batch_normalization_3/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE.mlp_base/batch_normalization_3/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$mlp_base/batch_normalization_4/gamma&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#mlp_base/batch_normalization_4/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE*mlp_base/batch_normalization_4/moving_mean&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE.mlp_base/batch_normalization_4/moving_variance&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$mlp_base/batch_normalization_5/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#mlp_base/batch_normalization_5/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*mlp_base/batch_normalization_5/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.mlp_base/batch_normalization_5/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEmlp_base/dense_4/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEmlp_base/dense_4/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEmlp_base/dense_5/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEmlp_base/dense_5/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEmlp_base/dense_6/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEmlp_base/dense_6/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEmlp_base/dense_7/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEmlp_base/dense_7/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1mlp_base/dense_4/kernelmlp_base/dense_4/bias*mlp_base/batch_normalization_3/moving_mean.mlp_base/batch_normalization_3/moving_variance#mlp_base/batch_normalization_3/beta$mlp_base/batch_normalization_3/gammamlp_base/dense_5/kernelmlp_base/dense_5/bias*mlp_base/batch_normalization_4/moving_mean.mlp_base/batch_normalization_4/moving_variance#mlp_base/batch_normalization_4/beta$mlp_base/batch_normalization_4/gammamlp_base/dense_6/kernelmlp_base/dense_6/bias*mlp_base/batch_normalization_5/moving_mean.mlp_base/batch_normalization_5/moving_variance#mlp_base/batch_normalization_5/beta$mlp_base/batch_normalization_5/gammamlp_base/dense_7/kernelmlp_base/dense_7/bias* 
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
%__inference_signature_wrapper_1130644
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename8mlp_base/batch_normalization_3/gamma/Read/ReadVariableOp7mlp_base/batch_normalization_3/beta/Read/ReadVariableOp>mlp_base/batch_normalization_3/moving_mean/Read/ReadVariableOpBmlp_base/batch_normalization_3/moving_variance/Read/ReadVariableOp8mlp_base/batch_normalization_4/gamma/Read/ReadVariableOp7mlp_base/batch_normalization_4/beta/Read/ReadVariableOp>mlp_base/batch_normalization_4/moving_mean/Read/ReadVariableOpBmlp_base/batch_normalization_4/moving_variance/Read/ReadVariableOp8mlp_base/batch_normalization_5/gamma/Read/ReadVariableOp7mlp_base/batch_normalization_5/beta/Read/ReadVariableOp>mlp_base/batch_normalization_5/moving_mean/Read/ReadVariableOpBmlp_base/batch_normalization_5/moving_variance/Read/ReadVariableOp+mlp_base/dense_4/kernel/Read/ReadVariableOp)mlp_base/dense_4/bias/Read/ReadVariableOp+mlp_base/dense_5/kernel/Read/ReadVariableOp)mlp_base/dense_5/bias/Read/ReadVariableOp+mlp_base/dense_6/kernel/Read/ReadVariableOp)mlp_base/dense_6/bias/Read/ReadVariableOp+mlp_base/dense_7/kernel/Read/ReadVariableOp)mlp_base/dense_7/bias/Read/ReadVariableOpConst*!
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
 __inference__traced_save_1131399
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename$mlp_base/batch_normalization_3/gamma#mlp_base/batch_normalization_3/beta*mlp_base/batch_normalization_3/moving_mean.mlp_base/batch_normalization_3/moving_variance$mlp_base/batch_normalization_4/gamma#mlp_base/batch_normalization_4/beta*mlp_base/batch_normalization_4/moving_mean.mlp_base/batch_normalization_4/moving_variance$mlp_base/batch_normalization_5/gamma#mlp_base/batch_normalization_5/beta*mlp_base/batch_normalization_5/moving_mean.mlp_base/batch_normalization_5/moving_variancemlp_base/dense_4/kernelmlp_base/dense_4/biasmlp_base/dense_5/kernelmlp_base/dense_5/biasmlp_base/dense_6/kernelmlp_base/dense_6/biasmlp_base/dense_7/kernelmlp_base/dense_7/bias* 
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
#__inference__traced_restore_1131469��
�$
�
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1130028

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
�
[
/__inference_concatenate_1_layer_call_fn_1130990
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
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1130077`
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
�

�
D__inference_dense_6_layer_call_and_return_conditional_losses_1130142

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
�=
�	
E__inference_mlp_base_layer_call_and_return_conditional_losses_1130385

inputs"
dense_4_1130337:	�
dense_4_1130339:	�,
batch_normalization_3_1130342:	�,
batch_normalization_3_1130344:	�,
batch_normalization_3_1130346:	�,
batch_normalization_3_1130348:	�#
dense_5_1130351:
��
dense_5_1130353:	�,
batch_normalization_4_1130356:	�,
batch_normalization_4_1130358:	�,
batch_normalization_4_1130360:	�,
batch_normalization_4_1130362:	�#
dense_6_1130365:
��
dense_6_1130367:	�,
batch_normalization_5_1130370:	�,
batch_normalization_5_1130372:	�,
batch_normalization_5_1130374:	�,
batch_normalization_5_1130376:	�"
dense_7_1130379:	�
dense_7_1130381:
identity��-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCalld
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
flatten_2/PartitionedCallPartitionedCallstrided_slice:output:0*
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
GPU 2J 8� *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_1130056f
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
flatten_3/PartitionedCallPartitionedCallstrided_slice_1:output:0*
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
GPU 2J 8� *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_1130068�
concatenate_1/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0*
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
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1130077�
dense_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_4_1130337dense_4_1130339*
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
GPU 2J 8� *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_1130090�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_3_1130342batch_normalization_3_1130344batch_normalization_3_1130346batch_normalization_3_1130348*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1129864�
dense_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_5_1130351dense_5_1130353*
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
GPU 2J 8� *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_1130116�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_4_1130356batch_normalization_4_1130358batch_normalization_4_1130360batch_normalization_4_1130362*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1129946�
dense_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_6_1130365dense_6_1130367*
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
GPU 2J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_1130142�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_5_1130370batch_normalization_5_1130372batch_normalization_5_1130374batch_normalization_5_1130376*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1130028�
dense_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_7_1130379dense_7_1130381*
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
GPU 2J 8� *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_1130167w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�=
�	
E__inference_mlp_base_layer_call_and_return_conditional_losses_1130174

inputs"
dense_4_1130091:	�
dense_4_1130093:	�,
batch_normalization_3_1130096:	�,
batch_normalization_3_1130098:	�,
batch_normalization_3_1130100:	�,
batch_normalization_3_1130102:	�#
dense_5_1130117:
��
dense_5_1130119:	�,
batch_normalization_4_1130122:	�,
batch_normalization_4_1130124:	�,
batch_normalization_4_1130126:	�,
batch_normalization_4_1130128:	�#
dense_6_1130143:
��
dense_6_1130145:	�,
batch_normalization_5_1130148:	�,
batch_normalization_5_1130150:	�,
batch_normalization_5_1130152:	�,
batch_normalization_5_1130154:	�"
dense_7_1130168:	�
dense_7_1130170:
identity��-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCalld
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
flatten_2/PartitionedCallPartitionedCallstrided_slice:output:0*
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
GPU 2J 8� *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_1130056f
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
flatten_3/PartitionedCallPartitionedCallstrided_slice_1:output:0*
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
GPU 2J 8� *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_1130068�
concatenate_1/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0*
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
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1130077�
dense_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_4_1130091dense_4_1130093*
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
GPU 2J 8� *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_1130090�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_3_1130096batch_normalization_3_1130098batch_normalization_3_1130100batch_normalization_3_1130102*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1129817�
dense_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_5_1130117dense_5_1130119*
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
GPU 2J 8� *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_1130116�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_4_1130122batch_normalization_4_1130124batch_normalization_4_1130126batch_normalization_4_1130128*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1129899�
dense_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_6_1130143dense_6_1130145*
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
GPU 2J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_1130142�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_5_1130148batch_normalization_5_1130150batch_normalization_5_1130152batch_normalization_5_1130154*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1129981�
dense_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_7_1130168dense_7_1130170*
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
GPU 2J 8� *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_1130167w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
D__inference_dense_6_layer_call_and_return_conditional_losses_1131297

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
�
b
F__inference_flatten_3_layer_call_and_return_conditional_losses_1130068

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
�2
�
 __inference__traced_save_1131399
file_prefixC
?savev2_mlp_base_batch_normalization_3_gamma_read_readvariableopB
>savev2_mlp_base_batch_normalization_3_beta_read_readvariableopI
Esavev2_mlp_base_batch_normalization_3_moving_mean_read_readvariableopM
Isavev2_mlp_base_batch_normalization_3_moving_variance_read_readvariableopC
?savev2_mlp_base_batch_normalization_4_gamma_read_readvariableopB
>savev2_mlp_base_batch_normalization_4_beta_read_readvariableopI
Esavev2_mlp_base_batch_normalization_4_moving_mean_read_readvariableopM
Isavev2_mlp_base_batch_normalization_4_moving_variance_read_readvariableopC
?savev2_mlp_base_batch_normalization_5_gamma_read_readvariableopB
>savev2_mlp_base_batch_normalization_5_beta_read_readvariableopI
Esavev2_mlp_base_batch_normalization_5_moving_mean_read_readvariableopM
Isavev2_mlp_base_batch_normalization_5_moving_variance_read_readvariableop6
2savev2_mlp_base_dense_4_kernel_read_readvariableop4
0savev2_mlp_base_dense_4_bias_read_readvariableop6
2savev2_mlp_base_dense_5_kernel_read_readvariableop4
0savev2_mlp_base_dense_5_bias_read_readvariableop6
2savev2_mlp_base_dense_6_kernel_read_readvariableop4
0savev2_mlp_base_dense_6_bias_read_readvariableop6
2savev2_mlp_base_dense_7_kernel_read_readvariableop4
0savev2_mlp_base_dense_7_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0?savev2_mlp_base_batch_normalization_3_gamma_read_readvariableop>savev2_mlp_base_batch_normalization_3_beta_read_readvariableopEsavev2_mlp_base_batch_normalization_3_moving_mean_read_readvariableopIsavev2_mlp_base_batch_normalization_3_moving_variance_read_readvariableop?savev2_mlp_base_batch_normalization_4_gamma_read_readvariableop>savev2_mlp_base_batch_normalization_4_beta_read_readvariableopEsavev2_mlp_base_batch_normalization_4_moving_mean_read_readvariableopIsavev2_mlp_base_batch_normalization_4_moving_variance_read_readvariableop?savev2_mlp_base_batch_normalization_5_gamma_read_readvariableop>savev2_mlp_base_batch_normalization_5_beta_read_readvariableopEsavev2_mlp_base_batch_normalization_5_moving_mean_read_readvariableopIsavev2_mlp_base_batch_normalization_5_moving_variance_read_readvariableop2savev2_mlp_base_dense_4_kernel_read_readvariableop0savev2_mlp_base_dense_4_bias_read_readvariableop2savev2_mlp_base_dense_5_kernel_read_readvariableop0savev2_mlp_base_dense_5_bias_read_readvariableop2savev2_mlp_base_dense_6_kernel_read_readvariableop0savev2_mlp_base_dense_6_bias_read_readvariableop2savev2_mlp_base_dense_7_kernel_read_readvariableop0savev2_mlp_base_dense_7_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
�
�
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1131123

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
�
*__inference_mlp_base_layer_call_fn_1130217
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
GPU 2J 8� *N
fIRG
E__inference_mlp_base_layer_call_and_return_conditional_losses_1130174o
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
7__inference_batch_normalization_5_layer_call_fn_1131183

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1130028p
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
Չ
�
"__inference__wrapped_model_1129793
input_1B
/mlp_base_dense_4_matmul_readvariableop_resource:	�?
0mlp_base_dense_4_biasadd_readvariableop_resource:	�J
;mlp_base_batch_normalization_3_cast_readvariableop_resource:	�L
=mlp_base_batch_normalization_3_cast_1_readvariableop_resource:	�L
=mlp_base_batch_normalization_3_cast_2_readvariableop_resource:	�L
=mlp_base_batch_normalization_3_cast_3_readvariableop_resource:	�C
/mlp_base_dense_5_matmul_readvariableop_resource:
��?
0mlp_base_dense_5_biasadd_readvariableop_resource:	�J
;mlp_base_batch_normalization_4_cast_readvariableop_resource:	�L
=mlp_base_batch_normalization_4_cast_1_readvariableop_resource:	�L
=mlp_base_batch_normalization_4_cast_2_readvariableop_resource:	�L
=mlp_base_batch_normalization_4_cast_3_readvariableop_resource:	�C
/mlp_base_dense_6_matmul_readvariableop_resource:
��?
0mlp_base_dense_6_biasadd_readvariableop_resource:	�J
;mlp_base_batch_normalization_5_cast_readvariableop_resource:	�L
=mlp_base_batch_normalization_5_cast_1_readvariableop_resource:	�L
=mlp_base_batch_normalization_5_cast_2_readvariableop_resource:	�L
=mlp_base_batch_normalization_5_cast_3_readvariableop_resource:	�B
/mlp_base_dense_7_matmul_readvariableop_resource:	�>
0mlp_base_dense_7_biasadd_readvariableop_resource:
identity��2mlp_base/batch_normalization_3/Cast/ReadVariableOp�4mlp_base/batch_normalization_3/Cast_1/ReadVariableOp�4mlp_base/batch_normalization_3/Cast_2/ReadVariableOp�4mlp_base/batch_normalization_3/Cast_3/ReadVariableOp�2mlp_base/batch_normalization_4/Cast/ReadVariableOp�4mlp_base/batch_normalization_4/Cast_1/ReadVariableOp�4mlp_base/batch_normalization_4/Cast_2/ReadVariableOp�4mlp_base/batch_normalization_4/Cast_3/ReadVariableOp�2mlp_base/batch_normalization_5/Cast/ReadVariableOp�4mlp_base/batch_normalization_5/Cast_1/ReadVariableOp�4mlp_base/batch_normalization_5/Cast_2/ReadVariableOp�4mlp_base/batch_normalization_5/Cast_3/ReadVariableOp�'mlp_base/dense_4/BiasAdd/ReadVariableOp�&mlp_base/dense_4/MatMul/ReadVariableOp�'mlp_base/dense_5/BiasAdd/ReadVariableOp�&mlp_base/dense_5/MatMul/ReadVariableOp�'mlp_base/dense_6/BiasAdd/ReadVariableOp�&mlp_base/dense_6/MatMul/ReadVariableOp�'mlp_base/dense_7/BiasAdd/ReadVariableOp�&mlp_base/dense_7/MatMul/ReadVariableOpm
mlp_base/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        o
mlp_base/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       o
mlp_base/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
mlp_base/strided_sliceStridedSliceinput_1%mlp_base/strided_slice/stack:output:0'mlp_base/strided_slice/stack_1:output:0'mlp_base/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_maski
mlp_base/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
mlp_base/flatten_2/ReshapeReshapemlp_base/strided_slice:output:0!mlp_base/flatten_2/Const:output:0*
T0*'
_output_shapes
:���������o
mlp_base/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       q
 mlp_base/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       q
 mlp_base/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
mlp_base/strided_slice_1StridedSliceinput_1'mlp_base/strided_slice_1/stack:output:0)mlp_base/strided_slice_1/stack_1:output:0)mlp_base/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_maski
mlp_base/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
mlp_base/flatten_3/ReshapeReshape!mlp_base/strided_slice_1:output:0!mlp_base/flatten_3/Const:output:0*
T0*'
_output_shapes
:���������d
"mlp_base/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
mlp_base/concatenate_1/concatConcatV2#mlp_base/flatten_2/Reshape:output:0#mlp_base/flatten_3/Reshape:output:0+mlp_base/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
&mlp_base/dense_4/MatMul/ReadVariableOpReadVariableOp/mlp_base_dense_4_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
mlp_base/dense_4/MatMulMatMul&mlp_base/concatenate_1/concat:output:0.mlp_base/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'mlp_base/dense_4/BiasAdd/ReadVariableOpReadVariableOp0mlp_base_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
mlp_base/dense_4/BiasAddBiasAdd!mlp_base/dense_4/MatMul:product:0/mlp_base/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
mlp_base/dense_4/TanhTanh!mlp_base/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
2mlp_base/batch_normalization_3/Cast/ReadVariableOpReadVariableOp;mlp_base_batch_normalization_3_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
4mlp_base/batch_normalization_3/Cast_1/ReadVariableOpReadVariableOp=mlp_base_batch_normalization_3_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
4mlp_base/batch_normalization_3/Cast_2/ReadVariableOpReadVariableOp=mlp_base_batch_normalization_3_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
4mlp_base/batch_normalization_3/Cast_3/ReadVariableOpReadVariableOp=mlp_base_batch_normalization_3_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0s
.mlp_base/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
,mlp_base/batch_normalization_3/batchnorm/addAddV2<mlp_base/batch_normalization_3/Cast_1/ReadVariableOp:value:07mlp_base/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
.mlp_base/batch_normalization_3/batchnorm/RsqrtRsqrt0mlp_base/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:��
,mlp_base/batch_normalization_3/batchnorm/mulMul2mlp_base/batch_normalization_3/batchnorm/Rsqrt:y:0<mlp_base/batch_normalization_3/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
.mlp_base/batch_normalization_3/batchnorm/mul_1Mulmlp_base/dense_4/Tanh:y:00mlp_base/batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
.mlp_base/batch_normalization_3/batchnorm/mul_2Mul:mlp_base/batch_normalization_3/Cast/ReadVariableOp:value:00mlp_base/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
,mlp_base/batch_normalization_3/batchnorm/subSub<mlp_base/batch_normalization_3/Cast_2/ReadVariableOp:value:02mlp_base/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
.mlp_base/batch_normalization_3/batchnorm/add_1AddV22mlp_base/batch_normalization_3/batchnorm/mul_1:z:00mlp_base/batch_normalization_3/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
&mlp_base/dense_5/MatMul/ReadVariableOpReadVariableOp/mlp_base_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
mlp_base/dense_5/MatMulMatMul2mlp_base/batch_normalization_3/batchnorm/add_1:z:0.mlp_base/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'mlp_base/dense_5/BiasAdd/ReadVariableOpReadVariableOp0mlp_base_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
mlp_base/dense_5/BiasAddBiasAdd!mlp_base/dense_5/MatMul:product:0/mlp_base/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
mlp_base/dense_5/TanhTanh!mlp_base/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
2mlp_base/batch_normalization_4/Cast/ReadVariableOpReadVariableOp;mlp_base_batch_normalization_4_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
4mlp_base/batch_normalization_4/Cast_1/ReadVariableOpReadVariableOp=mlp_base_batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
4mlp_base/batch_normalization_4/Cast_2/ReadVariableOpReadVariableOp=mlp_base_batch_normalization_4_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
4mlp_base/batch_normalization_4/Cast_3/ReadVariableOpReadVariableOp=mlp_base_batch_normalization_4_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0s
.mlp_base/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
,mlp_base/batch_normalization_4/batchnorm/addAddV2<mlp_base/batch_normalization_4/Cast_1/ReadVariableOp:value:07mlp_base/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
.mlp_base/batch_normalization_4/batchnorm/RsqrtRsqrt0mlp_base/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:��
,mlp_base/batch_normalization_4/batchnorm/mulMul2mlp_base/batch_normalization_4/batchnorm/Rsqrt:y:0<mlp_base/batch_normalization_4/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
.mlp_base/batch_normalization_4/batchnorm/mul_1Mulmlp_base/dense_5/Tanh:y:00mlp_base/batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
.mlp_base/batch_normalization_4/batchnorm/mul_2Mul:mlp_base/batch_normalization_4/Cast/ReadVariableOp:value:00mlp_base/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
,mlp_base/batch_normalization_4/batchnorm/subSub<mlp_base/batch_normalization_4/Cast_2/ReadVariableOp:value:02mlp_base/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
.mlp_base/batch_normalization_4/batchnorm/add_1AddV22mlp_base/batch_normalization_4/batchnorm/mul_1:z:00mlp_base/batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
&mlp_base/dense_6/MatMul/ReadVariableOpReadVariableOp/mlp_base_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
mlp_base/dense_6/MatMulMatMul2mlp_base/batch_normalization_4/batchnorm/add_1:z:0.mlp_base/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'mlp_base/dense_6/BiasAdd/ReadVariableOpReadVariableOp0mlp_base_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
mlp_base/dense_6/BiasAddBiasAdd!mlp_base/dense_6/MatMul:product:0/mlp_base/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
mlp_base/dense_6/TanhTanh!mlp_base/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
2mlp_base/batch_normalization_5/Cast/ReadVariableOpReadVariableOp;mlp_base_batch_normalization_5_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
4mlp_base/batch_normalization_5/Cast_1/ReadVariableOpReadVariableOp=mlp_base_batch_normalization_5_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
4mlp_base/batch_normalization_5/Cast_2/ReadVariableOpReadVariableOp=mlp_base_batch_normalization_5_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
4mlp_base/batch_normalization_5/Cast_3/ReadVariableOpReadVariableOp=mlp_base_batch_normalization_5_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0s
.mlp_base/batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
,mlp_base/batch_normalization_5/batchnorm/addAddV2<mlp_base/batch_normalization_5/Cast_1/ReadVariableOp:value:07mlp_base/batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
.mlp_base/batch_normalization_5/batchnorm/RsqrtRsqrt0mlp_base/batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes	
:��
,mlp_base/batch_normalization_5/batchnorm/mulMul2mlp_base/batch_normalization_5/batchnorm/Rsqrt:y:0<mlp_base/batch_normalization_5/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
.mlp_base/batch_normalization_5/batchnorm/mul_1Mulmlp_base/dense_6/Tanh:y:00mlp_base/batch_normalization_5/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
.mlp_base/batch_normalization_5/batchnorm/mul_2Mul:mlp_base/batch_normalization_5/Cast/ReadVariableOp:value:00mlp_base/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
,mlp_base/batch_normalization_5/batchnorm/subSub<mlp_base/batch_normalization_5/Cast_2/ReadVariableOp:value:02mlp_base/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
.mlp_base/batch_normalization_5/batchnorm/add_1AddV22mlp_base/batch_normalization_5/batchnorm/mul_1:z:00mlp_base/batch_normalization_5/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
&mlp_base/dense_7/MatMul/ReadVariableOpReadVariableOp/mlp_base_dense_7_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
mlp_base/dense_7/MatMulMatMul2mlp_base/batch_normalization_5/batchnorm/add_1:z:0.mlp_base/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'mlp_base/dense_7/BiasAdd/ReadVariableOpReadVariableOp0mlp_base_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
mlp_base/dense_7/BiasAddBiasAdd!mlp_base/dense_7/MatMul:product:0/mlp_base/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������p
IdentityIdentity!mlp_base/dense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp3^mlp_base/batch_normalization_3/Cast/ReadVariableOp5^mlp_base/batch_normalization_3/Cast_1/ReadVariableOp5^mlp_base/batch_normalization_3/Cast_2/ReadVariableOp5^mlp_base/batch_normalization_3/Cast_3/ReadVariableOp3^mlp_base/batch_normalization_4/Cast/ReadVariableOp5^mlp_base/batch_normalization_4/Cast_1/ReadVariableOp5^mlp_base/batch_normalization_4/Cast_2/ReadVariableOp5^mlp_base/batch_normalization_4/Cast_3/ReadVariableOp3^mlp_base/batch_normalization_5/Cast/ReadVariableOp5^mlp_base/batch_normalization_5/Cast_1/ReadVariableOp5^mlp_base/batch_normalization_5/Cast_2/ReadVariableOp5^mlp_base/batch_normalization_5/Cast_3/ReadVariableOp(^mlp_base/dense_4/BiasAdd/ReadVariableOp'^mlp_base/dense_4/MatMul/ReadVariableOp(^mlp_base/dense_5/BiasAdd/ReadVariableOp'^mlp_base/dense_5/MatMul/ReadVariableOp(^mlp_base/dense_6/BiasAdd/ReadVariableOp'^mlp_base/dense_6/MatMul/ReadVariableOp(^mlp_base/dense_7/BiasAdd/ReadVariableOp'^mlp_base/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : 2h
2mlp_base/batch_normalization_3/Cast/ReadVariableOp2mlp_base/batch_normalization_3/Cast/ReadVariableOp2l
4mlp_base/batch_normalization_3/Cast_1/ReadVariableOp4mlp_base/batch_normalization_3/Cast_1/ReadVariableOp2l
4mlp_base/batch_normalization_3/Cast_2/ReadVariableOp4mlp_base/batch_normalization_3/Cast_2/ReadVariableOp2l
4mlp_base/batch_normalization_3/Cast_3/ReadVariableOp4mlp_base/batch_normalization_3/Cast_3/ReadVariableOp2h
2mlp_base/batch_normalization_4/Cast/ReadVariableOp2mlp_base/batch_normalization_4/Cast/ReadVariableOp2l
4mlp_base/batch_normalization_4/Cast_1/ReadVariableOp4mlp_base/batch_normalization_4/Cast_1/ReadVariableOp2l
4mlp_base/batch_normalization_4/Cast_2/ReadVariableOp4mlp_base/batch_normalization_4/Cast_2/ReadVariableOp2l
4mlp_base/batch_normalization_4/Cast_3/ReadVariableOp4mlp_base/batch_normalization_4/Cast_3/ReadVariableOp2h
2mlp_base/batch_normalization_5/Cast/ReadVariableOp2mlp_base/batch_normalization_5/Cast/ReadVariableOp2l
4mlp_base/batch_normalization_5/Cast_1/ReadVariableOp4mlp_base/batch_normalization_5/Cast_1/ReadVariableOp2l
4mlp_base/batch_normalization_5/Cast_2/ReadVariableOp4mlp_base/batch_normalization_5/Cast_2/ReadVariableOp2l
4mlp_base/batch_normalization_5/Cast_3/ReadVariableOp4mlp_base/batch_normalization_5/Cast_3/ReadVariableOp2R
'mlp_base/dense_4/BiasAdd/ReadVariableOp'mlp_base/dense_4/BiasAdd/ReadVariableOp2P
&mlp_base/dense_4/MatMul/ReadVariableOp&mlp_base/dense_4/MatMul/ReadVariableOp2R
'mlp_base/dense_5/BiasAdd/ReadVariableOp'mlp_base/dense_5/BiasAdd/ReadVariableOp2P
&mlp_base/dense_5/MatMul/ReadVariableOp&mlp_base/dense_5/MatMul/ReadVariableOp2R
'mlp_base/dense_6/BiasAdd/ReadVariableOp'mlp_base/dense_6/BiasAdd/ReadVariableOp2P
&mlp_base/dense_6/MatMul/ReadVariableOp&mlp_base/dense_6/MatMul/ReadVariableOp2R
'mlp_base/dense_7/BiasAdd/ReadVariableOp'mlp_base/dense_7/BiasAdd/ReadVariableOp2P
&mlp_base/dense_7/MatMul/ReadVariableOp&mlp_base/dense_7/MatMul/ReadVariableOp:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
)__inference_dense_4_layer_call_fn_1131246

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
GPU 2J 8� *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_1130090p
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
�$
�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1129864

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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1129981

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
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1131157

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
D__inference_dense_7_layer_call_and_return_conditional_losses_1130167

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
)__inference_dense_7_layer_call_fn_1131306

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
GPU 2J 8� *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_1130167o
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
�=
�	
E__inference_mlp_base_layer_call_and_return_conditional_losses_1130597
input_1"
dense_4_1130549:	�
dense_4_1130551:	�,
batch_normalization_3_1130554:	�,
batch_normalization_3_1130556:	�,
batch_normalization_3_1130558:	�,
batch_normalization_3_1130560:	�#
dense_5_1130563:
��
dense_5_1130565:	�,
batch_normalization_4_1130568:	�,
batch_normalization_4_1130570:	�,
batch_normalization_4_1130572:	�,
batch_normalization_4_1130574:	�#
dense_6_1130577:
��
dense_6_1130579:	�,
batch_normalization_5_1130582:	�,
batch_normalization_5_1130584:	�,
batch_normalization_5_1130586:	�,
batch_normalization_5_1130588:	�"
dense_7_1130591:	�
dense_7_1130593:
identity��-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCalld
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
flatten_2/PartitionedCallPartitionedCallstrided_slice:output:0*
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
GPU 2J 8� *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_1130056f
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
flatten_3/PartitionedCallPartitionedCallstrided_slice_1:output:0*
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
GPU 2J 8� *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_1130068�
concatenate_1/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0*
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
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1130077�
dense_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_4_1130549dense_4_1130551*
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
GPU 2J 8� *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_1130090�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_3_1130554batch_normalization_3_1130556batch_normalization_3_1130558batch_normalization_3_1130560*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1129864�
dense_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_5_1130563dense_5_1130565*
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
GPU 2J 8� *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_1130116�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_4_1130568batch_normalization_4_1130570batch_normalization_4_1130572batch_normalization_4_1130574*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1129946�
dense_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_6_1130577dense_6_1130579*
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
GPU 2J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_1130142�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_5_1130582batch_normalization_5_1130584batch_normalization_5_1130586batch_normalization_5_1130588*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1130028�
dense_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_7_1130591dense_7_1130593*
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
GPU 2J 8� *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_1130167w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1131043

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
�
G
+__inference_flatten_2_layer_call_fn_1130967

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
GPU 2J 8� *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_1130056`
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
�
�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1129817

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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1131077

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
�=
�	
E__inference_mlp_base_layer_call_and_return_conditional_losses_1130535
input_1"
dense_4_1130487:	�
dense_4_1130489:	�,
batch_normalization_3_1130492:	�,
batch_normalization_3_1130494:	�,
batch_normalization_3_1130496:	�,
batch_normalization_3_1130498:	�#
dense_5_1130501:
��
dense_5_1130503:	�,
batch_normalization_4_1130506:	�,
batch_normalization_4_1130508:	�,
batch_normalization_4_1130510:	�,
batch_normalization_4_1130512:	�#
dense_6_1130515:
��
dense_6_1130517:	�,
batch_normalization_5_1130520:	�,
batch_normalization_5_1130522:	�,
batch_normalization_5_1130524:	�,
batch_normalization_5_1130526:	�"
dense_7_1130529:	�
dense_7_1130531:
identity��-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCalld
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
flatten_2/PartitionedCallPartitionedCallstrided_slice:output:0*
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
GPU 2J 8� *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_1130056f
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
flatten_3/PartitionedCallPartitionedCallstrided_slice_1:output:0*
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
GPU 2J 8� *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_1130068�
concatenate_1/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0*
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
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1130077�
dense_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_4_1130487dense_4_1130489*
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
GPU 2J 8� *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_1130090�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_3_1130492batch_normalization_3_1130494batch_normalization_3_1130496batch_normalization_3_1130498*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1129817�
dense_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_5_1130501dense_5_1130503*
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
GPU 2J 8� *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_1130116�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_4_1130506batch_normalization_4_1130508batch_normalization_4_1130510batch_normalization_4_1130512*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1129899�
dense_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_6_1130515dense_6_1130517*
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
GPU 2J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_1130142�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_5_1130520batch_normalization_5_1130522batch_normalization_5_1130524batch_normalization_5_1130526*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1129981�
dense_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_7_1130529dense_7_1130531*
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
GPU 2J 8� *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_1130167w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
b
F__inference_flatten_2_layer_call_and_return_conditional_losses_1130973

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
D__inference_dense_5_layer_call_and_return_conditional_losses_1131277

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
�
�
%__inference_signature_wrapper_1130644
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
"__inference__wrapped_model_1129793o
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
�
*__inference_mlp_base_layer_call_fn_1130473
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
GPU 2J 8� *N
fIRG
E__inference_mlp_base_layer_call_and_return_conditional_losses_1130385o
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
7__inference_batch_normalization_3_layer_call_fn_1131010

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1129817p
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
)__inference_dense_5_layer_call_fn_1131266

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
GPU 2J 8� *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_1130116p
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
�

�
D__inference_dense_5_layer_call_and_return_conditional_losses_1130116

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
�

�
D__inference_dense_4_layer_call_and_return_conditional_losses_1131257

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
�$
�
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1129946

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
�
v
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1130997
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
�S
�
#__inference__traced_restore_1131469
file_prefixD
5assignvariableop_mlp_base_batch_normalization_3_gamma:	�E
6assignvariableop_1_mlp_base_batch_normalization_3_beta:	�L
=assignvariableop_2_mlp_base_batch_normalization_3_moving_mean:	�P
Aassignvariableop_3_mlp_base_batch_normalization_3_moving_variance:	�F
7assignvariableop_4_mlp_base_batch_normalization_4_gamma:	�E
6assignvariableop_5_mlp_base_batch_normalization_4_beta:	�L
=assignvariableop_6_mlp_base_batch_normalization_4_moving_mean:	�P
Aassignvariableop_7_mlp_base_batch_normalization_4_moving_variance:	�F
7assignvariableop_8_mlp_base_batch_normalization_5_gamma:	�E
6assignvariableop_9_mlp_base_batch_normalization_5_beta:	�M
>assignvariableop_10_mlp_base_batch_normalization_5_moving_mean:	�Q
Bassignvariableop_11_mlp_base_batch_normalization_5_moving_variance:	�>
+assignvariableop_12_mlp_base_dense_4_kernel:	�8
)assignvariableop_13_mlp_base_dense_4_bias:	�?
+assignvariableop_14_mlp_base_dense_5_kernel:
��8
)assignvariableop_15_mlp_base_dense_5_bias:	�?
+assignvariableop_16_mlp_base_dense_6_kernel:
��8
)assignvariableop_17_mlp_base_dense_6_bias:	�>
+assignvariableop_18_mlp_base_dense_7_kernel:	�7
)assignvariableop_19_mlp_base_dense_7_bias:
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
AssignVariableOpAssignVariableOp5assignvariableop_mlp_base_batch_normalization_3_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp6assignvariableop_1_mlp_base_batch_normalization_3_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp=assignvariableop_2_mlp_base_batch_normalization_3_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpAassignvariableop_3_mlp_base_batch_normalization_3_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp7assignvariableop_4_mlp_base_batch_normalization_4_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp6assignvariableop_5_mlp_base_batch_normalization_4_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp=assignvariableop_6_mlp_base_batch_normalization_4_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpAassignvariableop_7_mlp_base_batch_normalization_4_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp7assignvariableop_8_mlp_base_batch_normalization_5_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp6assignvariableop_9_mlp_base_batch_normalization_5_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp>assignvariableop_10_mlp_base_batch_normalization_5_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpBassignvariableop_11_mlp_base_batch_normalization_5_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp+assignvariableop_12_mlp_base_dense_4_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp)assignvariableop_13_mlp_base_dense_4_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp+assignvariableop_14_mlp_base_dense_5_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp)assignvariableop_15_mlp_base_dense_5_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp+assignvariableop_16_mlp_base_dense_6_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp)assignvariableop_17_mlp_base_dense_6_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp+assignvariableop_18_mlp_base_dense_7_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp)assignvariableop_19_mlp_base_dense_7_biasIdentity_19:output:0"/device:CPU:0*
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

�
D__inference_dense_4_layer_call_and_return_conditional_losses_1130090

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
D__inference_dense_7_layer_call_and_return_conditional_losses_1131316

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
�
G
+__inference_flatten_3_layer_call_fn_1130978

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
GPU 2J 8� *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_1130068`
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
�
�
*__inference_mlp_base_layer_call_fn_1130734

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
GPU 2J 8� *N
fIRG
E__inference_mlp_base_layer_call_and_return_conditional_losses_1130385o
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
�
�
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1131203

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
b
F__inference_flatten_3_layer_call_and_return_conditional_losses_1130984

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
�w
�
E__inference_mlp_base_layer_call_and_return_conditional_losses_1130827

inputs9
&dense_4_matmul_readvariableop_resource:	�6
'dense_4_biasadd_readvariableop_resource:	�A
2batch_normalization_3_cast_readvariableop_resource:	�C
4batch_normalization_3_cast_1_readvariableop_resource:	�C
4batch_normalization_3_cast_2_readvariableop_resource:	�C
4batch_normalization_3_cast_3_readvariableop_resource:	�:
&dense_5_matmul_readvariableop_resource:
��6
'dense_5_biasadd_readvariableop_resource:	�A
2batch_normalization_4_cast_readvariableop_resource:	�C
4batch_normalization_4_cast_1_readvariableop_resource:	�C
4batch_normalization_4_cast_2_readvariableop_resource:	�C
4batch_normalization_4_cast_3_readvariableop_resource:	�:
&dense_6_matmul_readvariableop_resource:
��6
'dense_6_biasadd_readvariableop_resource:	�A
2batch_normalization_5_cast_readvariableop_resource:	�C
4batch_normalization_5_cast_1_readvariableop_resource:	�C
4batch_normalization_5_cast_2_readvariableop_resource:	�C
4batch_normalization_5_cast_3_readvariableop_resource:	�9
&dense_7_matmul_readvariableop_resource:	�5
'dense_7_biasadd_readvariableop_resource:
identity��)batch_normalization_3/Cast/ReadVariableOp�+batch_normalization_3/Cast_1/ReadVariableOp�+batch_normalization_3/Cast_2/ReadVariableOp�+batch_normalization_3/Cast_3/ReadVariableOp�)batch_normalization_4/Cast/ReadVariableOp�+batch_normalization_4/Cast_1/ReadVariableOp�+batch_normalization_4/Cast_2/ReadVariableOp�+batch_normalization_4/Cast_3/ReadVariableOp�)batch_normalization_5/Cast/ReadVariableOp�+batch_normalization_5/Cast_1/ReadVariableOp�+batch_normalization_5/Cast_2/ReadVariableOp�+batch_normalization_5/Cast_3/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�dense_6/BiasAdd/ReadVariableOp�dense_6/MatMul/ReadVariableOp�dense_7/BiasAdd/ReadVariableOp�dense_7/MatMul/ReadVariableOpd
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
shrink_axis_mask`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten_2/ReshapeReshapestrided_slice:output:0flatten_2/Const:output:0*
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
shrink_axis_mask`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten_3/ReshapeReshapestrided_slice_1:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:���������[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_1/concatConcatV2flatten_2/Reshape:output:0flatten_3/Reshape:output:0"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_4/MatMulMatMulconcatenate_1/concat:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_4/TanhTanhdense_4/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
)batch_normalization_3/Cast/ReadVariableOpReadVariableOp2batch_normalization_3_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_3/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_3_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_3/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_3_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_3/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_3_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0j
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_3/batchnorm/addAddV23batch_normalization_3/Cast_1/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�}
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:��
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:03batch_normalization_3/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
%batch_normalization_3/batchnorm/mul_1Muldense_4/Tanh:y:0'batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
%batch_normalization_3/batchnorm/mul_2Mul1batch_normalization_3/Cast/ReadVariableOp:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
#batch_normalization_3/batchnorm/subSub3batch_normalization_3/Cast_2/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_5/MatMulMatMul)batch_normalization_3/batchnorm/add_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_5/TanhTanhdense_5/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
)batch_normalization_4/Cast/ReadVariableOpReadVariableOp2batch_normalization_4_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_4/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_4/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_4_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_4/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_4_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0j
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_4/batchnorm/addAddV23batch_normalization_4/Cast_1/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�}
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:��
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:03batch_normalization_4/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
%batch_normalization_4/batchnorm/mul_1Muldense_5/Tanh:y:0'batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
%batch_normalization_4/batchnorm/mul_2Mul1batch_normalization_4/Cast/ReadVariableOp:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
#batch_normalization_4/batchnorm/subSub3batch_normalization_4/Cast_2/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_6/MatMulMatMul)batch_normalization_4/batchnorm/add_1:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_6/TanhTanhdense_6/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
)batch_normalization_5/Cast/ReadVariableOpReadVariableOp2batch_normalization_5_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_5/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_5_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_5/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_5_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_5/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_5_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0j
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_5/batchnorm/addAddV23batch_normalization_5/Cast_1/ReadVariableOp:value:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�}
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes	
:��
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:03batch_normalization_5/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
%batch_normalization_5/batchnorm/mul_1Muldense_6/Tanh:y:0'batch_normalization_5/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
%batch_normalization_5/batchnorm/mul_2Mul1batch_normalization_5/Cast/ReadVariableOp:value:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
#batch_normalization_5/batchnorm/subSub3batch_normalization_5/Cast_2/ReadVariableOp:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_7/MatMulMatMul)batch_normalization_5/batchnorm/add_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������g
IdentityIdentitydense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp*^batch_normalization_3/Cast/ReadVariableOp,^batch_normalization_3/Cast_1/ReadVariableOp,^batch_normalization_3/Cast_2/ReadVariableOp,^batch_normalization_3/Cast_3/ReadVariableOp*^batch_normalization_4/Cast/ReadVariableOp,^batch_normalization_4/Cast_1/ReadVariableOp,^batch_normalization_4/Cast_2/ReadVariableOp,^batch_normalization_4/Cast_3/ReadVariableOp*^batch_normalization_5/Cast/ReadVariableOp,^batch_normalization_5/Cast_1/ReadVariableOp,^batch_normalization_5/Cast_2/ReadVariableOp,^batch_normalization_5/Cast_3/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : 2V
)batch_normalization_3/Cast/ReadVariableOp)batch_normalization_3/Cast/ReadVariableOp2Z
+batch_normalization_3/Cast_1/ReadVariableOp+batch_normalization_3/Cast_1/ReadVariableOp2Z
+batch_normalization_3/Cast_2/ReadVariableOp+batch_normalization_3/Cast_2/ReadVariableOp2Z
+batch_normalization_3/Cast_3/ReadVariableOp+batch_normalization_3/Cast_3/ReadVariableOp2V
)batch_normalization_4/Cast/ReadVariableOp)batch_normalization_4/Cast/ReadVariableOp2Z
+batch_normalization_4/Cast_1/ReadVariableOp+batch_normalization_4/Cast_1/ReadVariableOp2Z
+batch_normalization_4/Cast_2/ReadVariableOp+batch_normalization_4/Cast_2/ReadVariableOp2Z
+batch_normalization_4/Cast_3/ReadVariableOp+batch_normalization_4/Cast_3/ReadVariableOp2V
)batch_normalization_5/Cast/ReadVariableOp)batch_normalization_5/Cast/ReadVariableOp2Z
+batch_normalization_5/Cast_1/ReadVariableOp+batch_normalization_5/Cast_1/ReadVariableOp2Z
+batch_normalization_5/Cast_2/ReadVariableOp+batch_normalization_5/Cast_2/ReadVariableOp2Z
+batch_normalization_5/Cast_3/ReadVariableOp+batch_normalization_5/Cast_3/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1129899

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
7__inference_batch_normalization_4_layer_call_fn_1131090

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1129899p
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
b
F__inference_flatten_2_layer_call_and_return_conditional_losses_1130056

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
t
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1130077

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
�
�
7__inference_batch_normalization_5_layer_call_fn_1131170

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1129981p
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
�
*__inference_mlp_base_layer_call_fn_1130689

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
GPU 2J 8� *N
fIRG
E__inference_mlp_base_layer_call_and_return_conditional_losses_1130174o
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
�
E__inference_mlp_base_layer_call_and_return_conditional_losses_1130962

inputs9
&dense_4_matmul_readvariableop_resource:	�6
'dense_4_biasadd_readvariableop_resource:	�L
=batch_normalization_3_assignmovingavg_readvariableop_resource:	�N
?batch_normalization_3_assignmovingavg_1_readvariableop_resource:	�A
2batch_normalization_3_cast_readvariableop_resource:	�C
4batch_normalization_3_cast_1_readvariableop_resource:	�:
&dense_5_matmul_readvariableop_resource:
��6
'dense_5_biasadd_readvariableop_resource:	�L
=batch_normalization_4_assignmovingavg_readvariableop_resource:	�N
?batch_normalization_4_assignmovingavg_1_readvariableop_resource:	�A
2batch_normalization_4_cast_readvariableop_resource:	�C
4batch_normalization_4_cast_1_readvariableop_resource:	�:
&dense_6_matmul_readvariableop_resource:
��6
'dense_6_biasadd_readvariableop_resource:	�L
=batch_normalization_5_assignmovingavg_readvariableop_resource:	�N
?batch_normalization_5_assignmovingavg_1_readvariableop_resource:	�A
2batch_normalization_5_cast_readvariableop_resource:	�C
4batch_normalization_5_cast_1_readvariableop_resource:	�9
&dense_7_matmul_readvariableop_resource:	�5
'dense_7_biasadd_readvariableop_resource:
identity��%batch_normalization_3/AssignMovingAvg�4batch_normalization_3/AssignMovingAvg/ReadVariableOp�'batch_normalization_3/AssignMovingAvg_1�6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_3/Cast/ReadVariableOp�+batch_normalization_3/Cast_1/ReadVariableOp�%batch_normalization_4/AssignMovingAvg�4batch_normalization_4/AssignMovingAvg/ReadVariableOp�'batch_normalization_4/AssignMovingAvg_1�6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_4/Cast/ReadVariableOp�+batch_normalization_4/Cast_1/ReadVariableOp�%batch_normalization_5/AssignMovingAvg�4batch_normalization_5/AssignMovingAvg/ReadVariableOp�'batch_normalization_5/AssignMovingAvg_1�6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_5/Cast/ReadVariableOp�+batch_normalization_5/Cast_1/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�dense_6/BiasAdd/ReadVariableOp�dense_6/MatMul/ReadVariableOp�dense_7/BiasAdd/ReadVariableOp�dense_7/MatMul/ReadVariableOpd
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
shrink_axis_mask`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten_2/ReshapeReshapestrided_slice:output:0flatten_2/Const:output:0*
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
shrink_axis_mask`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten_3/ReshapeReshapestrided_slice_1:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:���������[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_1/concatConcatV2flatten_2/Reshape:output:0flatten_3/Reshape:output:0"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_4/MatMulMatMulconcatenate_1/concat:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_4/TanhTanhdense_4/BiasAdd:output:0*
T0*(
_output_shapes
:����������~
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_3/moments/meanMeandense_4/Tanh:y:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:	��
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_4/Tanh:y:03batch_normalization_3/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 p
+batch_normalization_3/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
%batch_normalization_3/AssignMovingAvgAssignSubVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
'batch_normalization_3/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
)batch_normalization_3/Cast/ReadVariableOpReadVariableOp2batch_normalization_3_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_3/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_3_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0j
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�}
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:��
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:03batch_normalization_3/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
%batch_normalization_3/batchnorm/mul_1Muldense_4/Tanh:y:0'batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
#batch_normalization_3/batchnorm/subSub1batch_normalization_3/Cast/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_5/MatMulMatMul)batch_normalization_3/batchnorm/add_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_5/TanhTanhdense_5/BiasAdd:output:0*
T0*(
_output_shapes
:����������~
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_4/moments/meanMeandense_5/Tanh:y:0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:	��
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencedense_5/Tanh:y:03batch_normalization_4/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 p
+batch_normalization_4/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
%batch_normalization_4/AssignMovingAvgAssignSubVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_4/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
'batch_normalization_4/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
)batch_normalization_4/Cast/ReadVariableOpReadVariableOp2batch_normalization_4_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_4/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0j
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�}
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:��
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:03batch_normalization_4/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
%batch_normalization_4/batchnorm/mul_1Muldense_5/Tanh:y:0'batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
#batch_normalization_4/batchnorm/subSub1batch_normalization_4/Cast/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_6/MatMulMatMul)batch_normalization_4/batchnorm/add_1:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_6/TanhTanhdense_6/BiasAdd:output:0*
T0*(
_output_shapes
:����������~
4batch_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_5/moments/meanMeandense_6/Tanh:y:0=batch_normalization_5/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
*batch_normalization_5/moments/StopGradientStopGradient+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes
:	��
/batch_normalization_5/moments/SquaredDifferenceSquaredDifferencedense_6/Tanh:y:03batch_normalization_5/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
8batch_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_5/moments/varianceMean3batch_normalization_5/moments/SquaredDifference:z:0Abatch_normalization_5/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
%batch_normalization_5/moments/SqueezeSqueeze+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
'batch_normalization_5/moments/Squeeze_1Squeeze/batch_normalization_5/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 p
+batch_normalization_5/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_5_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)batch_normalization_5/AssignMovingAvg/subSub<batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_5/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
)batch_normalization_5/AssignMovingAvg/mulMul-batch_normalization_5/AssignMovingAvg/sub:z:04batch_normalization_5/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
%batch_normalization_5/AssignMovingAvgAssignSubVariableOp=batch_normalization_5_assignmovingavg_readvariableop_resource-batch_normalization_5/AssignMovingAvg/mul:z:05^batch_normalization_5/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_5/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_5_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_5/AssignMovingAvg_1/subSub>batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_5/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
+batch_normalization_5/AssignMovingAvg_1/mulMul/batch_normalization_5/AssignMovingAvg_1/sub:z:06batch_normalization_5/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
'batch_normalization_5/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_5_assignmovingavg_1_readvariableop_resource/batch_normalization_5/AssignMovingAvg_1/mul:z:07^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
)batch_normalization_5/Cast/ReadVariableOpReadVariableOp2batch_normalization_5_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_5/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_5_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0j
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_5/batchnorm/addAddV20batch_normalization_5/moments/Squeeze_1:output:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�}
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes	
:��
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:03batch_normalization_5/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
%batch_normalization_5/batchnorm/mul_1Muldense_6/Tanh:y:0'batch_normalization_5/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
%batch_normalization_5/batchnorm/mul_2Mul.batch_normalization_5/moments/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
#batch_normalization_5/batchnorm/subSub1batch_normalization_5/Cast/ReadVariableOp:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_7/MatMulMatMul)batch_normalization_5/batchnorm/add_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������g
IdentityIdentitydense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������	
NoOpNoOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_3/Cast/ReadVariableOp,^batch_normalization_3/Cast_1/ReadVariableOp&^batch_normalization_4/AssignMovingAvg5^batch_normalization_4/AssignMovingAvg/ReadVariableOp(^batch_normalization_4/AssignMovingAvg_17^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_4/Cast/ReadVariableOp,^batch_normalization_4/Cast_1/ReadVariableOp&^batch_normalization_5/AssignMovingAvg5^batch_normalization_5/AssignMovingAvg/ReadVariableOp(^batch_normalization_5/AssignMovingAvg_17^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_5/Cast/ReadVariableOp,^batch_normalization_5/Cast_1/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_3/AssignMovingAvg%batch_normalization_3/AssignMovingAvg2l
4batch_normalization_3/AssignMovingAvg/ReadVariableOp4batch_normalization_3/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_3/AssignMovingAvg_1'batch_normalization_3/AssignMovingAvg_12p
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_3/Cast/ReadVariableOp)batch_normalization_3/Cast/ReadVariableOp2Z
+batch_normalization_3/Cast_1/ReadVariableOp+batch_normalization_3/Cast_1/ReadVariableOp2N
%batch_normalization_4/AssignMovingAvg%batch_normalization_4/AssignMovingAvg2l
4batch_normalization_4/AssignMovingAvg/ReadVariableOp4batch_normalization_4/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_4/AssignMovingAvg_1'batch_normalization_4/AssignMovingAvg_12p
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_4/Cast/ReadVariableOp)batch_normalization_4/Cast/ReadVariableOp2Z
+batch_normalization_4/Cast_1/ReadVariableOp+batch_normalization_4/Cast_1/ReadVariableOp2N
%batch_normalization_5/AssignMovingAvg%batch_normalization_5/AssignMovingAvg2l
4batch_normalization_5/AssignMovingAvg/ReadVariableOp4batch_normalization_5/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_5/AssignMovingAvg_1'batch_normalization_5/AssignMovingAvg_12p
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_5/Cast/ReadVariableOp)batch_normalization_5/Cast/ReadVariableOp2Z
+batch_normalization_5/Cast_1/ReadVariableOp+batch_normalization_5/Cast_1/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_4_layer_call_fn_1131103

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1129946p
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
)__inference_dense_6_layer_call_fn_1131286

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
GPU 2J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_1130142p
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
�
�
7__inference_batch_normalization_3_layer_call_fn_1131023

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1129864p
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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1131237

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
*__inference_mlp_base_layer_call_fn_1130217
*__inference_mlp_base_layer_call_fn_1130689
*__inference_mlp_base_layer_call_fn_1130734
*__inference_mlp_base_layer_call_fn_1130473�
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
E__inference_mlp_base_layer_call_and_return_conditional_losses_1130827
E__inference_mlp_base_layer_call_and_return_conditional_losses_1130962
E__inference_mlp_base_layer_call_and_return_conditional_losses_1130535
E__inference_mlp_base_layer_call_and_return_conditional_losses_1130597�
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
"__inference__wrapped_model_1129793input_1"�
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
3:1�2$mlp_base/batch_normalization_3/gamma
2:0�2#mlp_base/batch_normalization_3/beta
;:9� (2*mlp_base/batch_normalization_3/moving_mean
?:=� (2.mlp_base/batch_normalization_3/moving_variance
3:1�2$mlp_base/batch_normalization_4/gamma
2:0�2#mlp_base/batch_normalization_4/beta
;:9� (2*mlp_base/batch_normalization_4/moving_mean
?:=� (2.mlp_base/batch_normalization_4/moving_variance
3:1�2$mlp_base/batch_normalization_5/gamma
2:0�2#mlp_base/batch_normalization_5/beta
;:9� (2*mlp_base/batch_normalization_5/moving_mean
?:=� (2.mlp_base/batch_normalization_5/moving_variance
*:(	�2mlp_base/dense_4/kernel
$:"�2mlp_base/dense_4/bias
+:)
��2mlp_base/dense_5/kernel
$:"�2mlp_base/dense_5/bias
+:)
��2mlp_base/dense_6/kernel
$:"�2mlp_base/dense_6/bias
*:(	�2mlp_base/dense_7/kernel
#:!2mlp_base/dense_7/bias
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
*__inference_mlp_base_layer_call_fn_1130217input_1"�
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
*__inference_mlp_base_layer_call_fn_1130689inputs"�
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
*__inference_mlp_base_layer_call_fn_1130734inputs"�
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
*__inference_mlp_base_layer_call_fn_1130473input_1"�
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
E__inference_mlp_base_layer_call_and_return_conditional_losses_1130827inputs"�
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
E__inference_mlp_base_layer_call_and_return_conditional_losses_1130962inputs"�
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
E__inference_mlp_base_layer_call_and_return_conditional_losses_1130535input_1"�
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
E__inference_mlp_base_layer_call_and_return_conditional_losses_1130597input_1"�
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
+__inference_flatten_2_layer_call_fn_1130967�
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
F__inference_flatten_2_layer_call_and_return_conditional_losses_1130973�
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
+__inference_flatten_3_layer_call_fn_1130978�
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
F__inference_flatten_3_layer_call_and_return_conditional_losses_1130984�
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
/__inference_concatenate_1_layer_call_fn_1130990�
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
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1130997�
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
7__inference_batch_normalization_3_layer_call_fn_1131010
7__inference_batch_normalization_3_layer_call_fn_1131023�
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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1131043
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1131077�
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
7__inference_batch_normalization_4_layer_call_fn_1131090
7__inference_batch_normalization_4_layer_call_fn_1131103�
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
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1131123
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1131157�
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
7__inference_batch_normalization_5_layer_call_fn_1131170
7__inference_batch_normalization_5_layer_call_fn_1131183�
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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1131203
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1131237�
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
)__inference_dense_4_layer_call_fn_1131246�
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
D__inference_dense_4_layer_call_and_return_conditional_losses_1131257�
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
)__inference_dense_5_layer_call_fn_1131266�
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
D__inference_dense_5_layer_call_and_return_conditional_losses_1131277�
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
)__inference_dense_6_layer_call_fn_1131286�
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
D__inference_dense_6_layer_call_and_return_conditional_losses_1131297�
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
)__inference_dense_7_layer_call_fn_1131306�
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
D__inference_dense_7_layer_call_and_return_conditional_losses_1131316�
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
%__inference_signature_wrapper_1130644input_1"�
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
+__inference_flatten_2_layer_call_fn_1130967inputs"�
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
F__inference_flatten_2_layer_call_and_return_conditional_losses_1130973inputs"�
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
+__inference_flatten_3_layer_call_fn_1130978inputs"�
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
F__inference_flatten_3_layer_call_and_return_conditional_losses_1130984inputs"�
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
/__inference_concatenate_1_layer_call_fn_1130990inputs/0inputs/1"�
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
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1130997inputs/0inputs/1"�
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
7__inference_batch_normalization_3_layer_call_fn_1131010inputs"�
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
7__inference_batch_normalization_3_layer_call_fn_1131023inputs"�
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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1131043inputs"�
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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1131077inputs"�
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
7__inference_batch_normalization_4_layer_call_fn_1131090inputs"�
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
7__inference_batch_normalization_4_layer_call_fn_1131103inputs"�
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
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1131123inputs"�
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
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1131157inputs"�
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
7__inference_batch_normalization_5_layer_call_fn_1131170inputs"�
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
7__inference_batch_normalization_5_layer_call_fn_1131183inputs"�
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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1131203inputs"�
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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1131237inputs"�
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
)__inference_dense_4_layer_call_fn_1131246inputs"�
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
D__inference_dense_4_layer_call_and_return_conditional_losses_1131257inputs"�
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
)__inference_dense_5_layer_call_fn_1131266inputs"�
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
D__inference_dense_5_layer_call_and_return_conditional_losses_1131277inputs"�
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
)__inference_dense_6_layer_call_fn_1131286inputs"�
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
D__inference_dense_6_layer_call_and_return_conditional_losses_1131297inputs"�
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
)__inference_dense_7_layer_call_fn_1131306inputs"�
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
D__inference_dense_7_layer_call_and_return_conditional_losses_1131316inputs"�
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
"__inference__wrapped_model_1129793� !"#$%&'4�1
*�'
%�"
input_1���������
� "3�0
.
output_1"�
output_1����������
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1131043d4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1131077d4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
7__inference_batch_normalization_3_layer_call_fn_1131010W4�1
*�'
!�
inputs����������
p 
� "������������
7__inference_batch_normalization_3_layer_call_fn_1131023W4�1
*�'
!�
inputs����������
p
� "������������
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1131123d4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1131157d4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
7__inference_batch_normalization_4_layer_call_fn_1131090W4�1
*�'
!�
inputs����������
p 
� "������������
7__inference_batch_normalization_4_layer_call_fn_1131103W4�1
*�'
!�
inputs����������
p
� "������������
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1131203d4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1131237d4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
7__inference_batch_normalization_5_layer_call_fn_1131170W4�1
*�'
!�
inputs����������
p 
� "������������
7__inference_batch_normalization_5_layer_call_fn_1131183W4�1
*�'
!�
inputs����������
p
� "������������
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1130997�Z�W
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
/__inference_concatenate_1_layer_call_fn_1130990vZ�W
P�M
K�H
"�
inputs/0���������
"�
inputs/1���������
� "�����������
D__inference_dense_4_layer_call_and_return_conditional_losses_1131257] !/�,
%�"
 �
inputs���������
� "&�#
�
0����������
� }
)__inference_dense_4_layer_call_fn_1131246P !/�,
%�"
 �
inputs���������
� "������������
D__inference_dense_5_layer_call_and_return_conditional_losses_1131277^"#0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� ~
)__inference_dense_5_layer_call_fn_1131266Q"#0�-
&�#
!�
inputs����������
� "������������
D__inference_dense_6_layer_call_and_return_conditional_losses_1131297^$%0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� ~
)__inference_dense_6_layer_call_fn_1131286Q$%0�-
&�#
!�
inputs����������
� "������������
D__inference_dense_7_layer_call_and_return_conditional_losses_1131316]&'0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� }
)__inference_dense_7_layer_call_fn_1131306P&'0�-
&�#
!�
inputs����������
� "�����������
F__inference_flatten_2_layer_call_and_return_conditional_losses_1130973X/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� z
+__inference_flatten_2_layer_call_fn_1130967K/�,
%�"
 �
inputs���������
� "�����������
F__inference_flatten_3_layer_call_and_return_conditional_losses_1130984X/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� z
+__inference_flatten_3_layer_call_fn_1130978K/�,
%�"
 �
inputs���������
� "�����������
E__inference_mlp_base_layer_call_and_return_conditional_losses_1130535w !"#$%&'8�5
.�+
%�"
input_1���������
p 
� "%�"
�
0���������
� �
E__inference_mlp_base_layer_call_and_return_conditional_losses_1130597w !"#$%&'8�5
.�+
%�"
input_1���������
p
� "%�"
�
0���������
� �
E__inference_mlp_base_layer_call_and_return_conditional_losses_1130827v !"#$%&'7�4
-�*
$�!
inputs���������
p 
� "%�"
�
0���������
� �
E__inference_mlp_base_layer_call_and_return_conditional_losses_1130962v !"#$%&'7�4
-�*
$�!
inputs���������
p
� "%�"
�
0���������
� �
*__inference_mlp_base_layer_call_fn_1130217j !"#$%&'8�5
.�+
%�"
input_1���������
p 
� "�����������
*__inference_mlp_base_layer_call_fn_1130473j !"#$%&'8�5
.�+
%�"
input_1���������
p
� "�����������
*__inference_mlp_base_layer_call_fn_1130689i !"#$%&'7�4
-�*
$�!
inputs���������
p 
� "�����������
*__inference_mlp_base_layer_call_fn_1130734i !"#$%&'7�4
-�*
$�!
inputs���������
p
� "�����������
%__inference_signature_wrapper_1130644� !"#$%&'?�<
� 
5�2
0
input_1%�"
input_1���������"3�0
.
output_1"�
output_1���������
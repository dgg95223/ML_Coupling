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
private_mlp_4/dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameprivate_mlp_4/dense_19/bias
�
/private_mlp_4/dense_19/bias/Read/ReadVariableOpReadVariableOpprivate_mlp_4/dense_19/bias*
_output_shapes
:*
dtype0
�
private_mlp_4/dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*.
shared_nameprivate_mlp_4/dense_19/kernel
�
1private_mlp_4/dense_19/kernel/Read/ReadVariableOpReadVariableOpprivate_mlp_4/dense_19/kernel*
_output_shapes
:	�*
dtype0
�
private_mlp_4/dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_nameprivate_mlp_4/dense_18/bias
�
/private_mlp_4/dense_18/bias/Read/ReadVariableOpReadVariableOpprivate_mlp_4/dense_18/bias*
_output_shapes	
:�*
dtype0
�
private_mlp_4/dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*.
shared_nameprivate_mlp_4/dense_18/kernel
�
1private_mlp_4/dense_18/kernel/Read/ReadVariableOpReadVariableOpprivate_mlp_4/dense_18/kernel* 
_output_shapes
:
��*
dtype0
�
private_mlp_4/dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_nameprivate_mlp_4/dense_17/bias
�
/private_mlp_4/dense_17/bias/Read/ReadVariableOpReadVariableOpprivate_mlp_4/dense_17/bias*
_output_shapes	
:�*
dtype0
�
private_mlp_4/dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*.
shared_nameprivate_mlp_4/dense_17/kernel
�
1private_mlp_4/dense_17/kernel/Read/ReadVariableOpReadVariableOpprivate_mlp_4/dense_17/kernel* 
_output_shapes
:
��*
dtype0
�
private_mlp_4/dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_nameprivate_mlp_4/dense_16/bias
�
/private_mlp_4/dense_16/bias/Read/ReadVariableOpReadVariableOpprivate_mlp_4/dense_16/bias*
_output_shapes	
:�*
dtype0
�
private_mlp_4/dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*.
shared_nameprivate_mlp_4/dense_16/kernel
�
1private_mlp_4/dense_16/kernel/Read/ReadVariableOpReadVariableOpprivate_mlp_4/dense_16/kernel* 
_output_shapes
:
��*
dtype0
�
4private_mlp_4/batch_normalization_14/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*E
shared_name64private_mlp_4/batch_normalization_14/moving_variance
�
Hprivate_mlp_4/batch_normalization_14/moving_variance/Read/ReadVariableOpReadVariableOp4private_mlp_4/batch_normalization_14/moving_variance*
_output_shapes	
:�*
dtype0
�
0private_mlp_4/batch_normalization_14/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*A
shared_name20private_mlp_4/batch_normalization_14/moving_mean
�
Dprivate_mlp_4/batch_normalization_14/moving_mean/Read/ReadVariableOpReadVariableOp0private_mlp_4/batch_normalization_14/moving_mean*
_output_shapes	
:�*
dtype0
�
)private_mlp_4/batch_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*:
shared_name+)private_mlp_4/batch_normalization_14/beta
�
=private_mlp_4/batch_normalization_14/beta/Read/ReadVariableOpReadVariableOp)private_mlp_4/batch_normalization_14/beta*
_output_shapes	
:�*
dtype0
�
*private_mlp_4/batch_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*;
shared_name,*private_mlp_4/batch_normalization_14/gamma
�
>private_mlp_4/batch_normalization_14/gamma/Read/ReadVariableOpReadVariableOp*private_mlp_4/batch_normalization_14/gamma*
_output_shapes	
:�*
dtype0
�
4private_mlp_4/batch_normalization_13/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*E
shared_name64private_mlp_4/batch_normalization_13/moving_variance
�
Hprivate_mlp_4/batch_normalization_13/moving_variance/Read/ReadVariableOpReadVariableOp4private_mlp_4/batch_normalization_13/moving_variance*
_output_shapes	
:�*
dtype0
�
0private_mlp_4/batch_normalization_13/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*A
shared_name20private_mlp_4/batch_normalization_13/moving_mean
�
Dprivate_mlp_4/batch_normalization_13/moving_mean/Read/ReadVariableOpReadVariableOp0private_mlp_4/batch_normalization_13/moving_mean*
_output_shapes	
:�*
dtype0
�
)private_mlp_4/batch_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*:
shared_name+)private_mlp_4/batch_normalization_13/beta
�
=private_mlp_4/batch_normalization_13/beta/Read/ReadVariableOpReadVariableOp)private_mlp_4/batch_normalization_13/beta*
_output_shapes	
:�*
dtype0
�
*private_mlp_4/batch_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*;
shared_name,*private_mlp_4/batch_normalization_13/gamma
�
>private_mlp_4/batch_normalization_13/gamma/Read/ReadVariableOpReadVariableOp*private_mlp_4/batch_normalization_13/gamma*
_output_shapes	
:�*
dtype0
�
4private_mlp_4/batch_normalization_12/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*E
shared_name64private_mlp_4/batch_normalization_12/moving_variance
�
Hprivate_mlp_4/batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp4private_mlp_4/batch_normalization_12/moving_variance*
_output_shapes	
:�*
dtype0
�
0private_mlp_4/batch_normalization_12/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*A
shared_name20private_mlp_4/batch_normalization_12/moving_mean
�
Dprivate_mlp_4/batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp0private_mlp_4/batch_normalization_12/moving_mean*
_output_shapes	
:�*
dtype0
�
)private_mlp_4/batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*:
shared_name+)private_mlp_4/batch_normalization_12/beta
�
=private_mlp_4/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOp)private_mlp_4/batch_normalization_12/beta*
_output_shapes	
:�*
dtype0
�
*private_mlp_4/batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*;
shared_name,*private_mlp_4/batch_normalization_12/gamma
�
>private_mlp_4/batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOp*private_mlp_4/batch_normalization_12/gamma*
_output_shapes	
:�*
dtype0

NoOpNoOp
�G
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�G
value�GB�F B�F
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
jd
VARIABLE_VALUE*private_mlp_4/batch_normalization_12/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)private_mlp_4/batch_normalization_12/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE0private_mlp_4/batch_normalization_12/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE4private_mlp_4/batch_normalization_12/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE*private_mlp_4/batch_normalization_13/gamma&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)private_mlp_4/batch_normalization_13/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE0private_mlp_4/batch_normalization_13/moving_mean&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE4private_mlp_4/batch_normalization_13/moving_variance&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE*private_mlp_4/batch_normalization_14/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)private_mlp_4/batch_normalization_14/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE0private_mlp_4/batch_normalization_14/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE4private_mlp_4/batch_normalization_14/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEprivate_mlp_4/dense_16/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEprivate_mlp_4/dense_16/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEprivate_mlp_4/dense_17/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEprivate_mlp_4/dense_17/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEprivate_mlp_4/dense_18/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEprivate_mlp_4/dense_18/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEprivate_mlp_4/dense_19/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEprivate_mlp_4/dense_19/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
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
serving_default_input_1Placeholder*/
_output_shapes
:���������*
dtype0*$
shape:���������
�	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1private_mlp_4/dense_16/kernelprivate_mlp_4/dense_16/bias0private_mlp_4/batch_normalization_12/moving_mean4private_mlp_4/batch_normalization_12/moving_variance)private_mlp_4/batch_normalization_12/beta*private_mlp_4/batch_normalization_12/gammaprivate_mlp_4/dense_17/kernelprivate_mlp_4/dense_17/bias0private_mlp_4/batch_normalization_13/moving_mean4private_mlp_4/batch_normalization_13/moving_variance)private_mlp_4/batch_normalization_13/beta*private_mlp_4/batch_normalization_13/gammaprivate_mlp_4/dense_18/kernelprivate_mlp_4/dense_18/bias0private_mlp_4/batch_normalization_14/moving_mean4private_mlp_4/batch_normalization_14/moving_variance)private_mlp_4/batch_normalization_14/beta*private_mlp_4/batch_normalization_14/gammaprivate_mlp_4/dense_19/kernelprivate_mlp_4/dense_19/bias* 
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
GPU 2J 8� */
f*R(
&__inference_signature_wrapper_14117703
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename>private_mlp_4/batch_normalization_12/gamma/Read/ReadVariableOp=private_mlp_4/batch_normalization_12/beta/Read/ReadVariableOpDprivate_mlp_4/batch_normalization_12/moving_mean/Read/ReadVariableOpHprivate_mlp_4/batch_normalization_12/moving_variance/Read/ReadVariableOp>private_mlp_4/batch_normalization_13/gamma/Read/ReadVariableOp=private_mlp_4/batch_normalization_13/beta/Read/ReadVariableOpDprivate_mlp_4/batch_normalization_13/moving_mean/Read/ReadVariableOpHprivate_mlp_4/batch_normalization_13/moving_variance/Read/ReadVariableOp>private_mlp_4/batch_normalization_14/gamma/Read/ReadVariableOp=private_mlp_4/batch_normalization_14/beta/Read/ReadVariableOpDprivate_mlp_4/batch_normalization_14/moving_mean/Read/ReadVariableOpHprivate_mlp_4/batch_normalization_14/moving_variance/Read/ReadVariableOp1private_mlp_4/dense_16/kernel/Read/ReadVariableOp/private_mlp_4/dense_16/bias/Read/ReadVariableOp1private_mlp_4/dense_17/kernel/Read/ReadVariableOp/private_mlp_4/dense_17/bias/Read/ReadVariableOp1private_mlp_4/dense_18/kernel/Read/ReadVariableOp/private_mlp_4/dense_18/bias/Read/ReadVariableOp1private_mlp_4/dense_19/kernel/Read/ReadVariableOp/private_mlp_4/dense_19/bias/Read/ReadVariableOpConst*!
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
GPU 2J 8� **
f%R#
!__inference__traced_save_14118458
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename*private_mlp_4/batch_normalization_12/gamma)private_mlp_4/batch_normalization_12/beta0private_mlp_4/batch_normalization_12/moving_mean4private_mlp_4/batch_normalization_12/moving_variance*private_mlp_4/batch_normalization_13/gamma)private_mlp_4/batch_normalization_13/beta0private_mlp_4/batch_normalization_13/moving_mean4private_mlp_4/batch_normalization_13/moving_variance*private_mlp_4/batch_normalization_14/gamma)private_mlp_4/batch_normalization_14/beta0private_mlp_4/batch_normalization_14/moving_mean4private_mlp_4/batch_normalization_14/moving_varianceprivate_mlp_4/dense_16/kernelprivate_mlp_4/dense_16/biasprivate_mlp_4/dense_17/kernelprivate_mlp_4/dense_17/biasprivate_mlp_4/dense_18/kernelprivate_mlp_4/dense_18/biasprivate_mlp_4/dense_19/kernelprivate_mlp_4/dense_19/bias* 
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
GPU 2J 8� *-
f(R&
$__inference__traced_restore_14118528��
�4
�
!__inference__traced_save_14118458
file_prefixI
Esavev2_private_mlp_4_batch_normalization_12_gamma_read_readvariableopH
Dsavev2_private_mlp_4_batch_normalization_12_beta_read_readvariableopO
Ksavev2_private_mlp_4_batch_normalization_12_moving_mean_read_readvariableopS
Osavev2_private_mlp_4_batch_normalization_12_moving_variance_read_readvariableopI
Esavev2_private_mlp_4_batch_normalization_13_gamma_read_readvariableopH
Dsavev2_private_mlp_4_batch_normalization_13_beta_read_readvariableopO
Ksavev2_private_mlp_4_batch_normalization_13_moving_mean_read_readvariableopS
Osavev2_private_mlp_4_batch_normalization_13_moving_variance_read_readvariableopI
Esavev2_private_mlp_4_batch_normalization_14_gamma_read_readvariableopH
Dsavev2_private_mlp_4_batch_normalization_14_beta_read_readvariableopO
Ksavev2_private_mlp_4_batch_normalization_14_moving_mean_read_readvariableopS
Osavev2_private_mlp_4_batch_normalization_14_moving_variance_read_readvariableop<
8savev2_private_mlp_4_dense_16_kernel_read_readvariableop:
6savev2_private_mlp_4_dense_16_bias_read_readvariableop<
8savev2_private_mlp_4_dense_17_kernel_read_readvariableop:
6savev2_private_mlp_4_dense_17_bias_read_readvariableop<
8savev2_private_mlp_4_dense_18_kernel_read_readvariableop:
6savev2_private_mlp_4_dense_18_bias_read_readvariableop<
8savev2_private_mlp_4_dense_19_kernel_read_readvariableop:
6savev2_private_mlp_4_dense_19_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Esavev2_private_mlp_4_batch_normalization_12_gamma_read_readvariableopDsavev2_private_mlp_4_batch_normalization_12_beta_read_readvariableopKsavev2_private_mlp_4_batch_normalization_12_moving_mean_read_readvariableopOsavev2_private_mlp_4_batch_normalization_12_moving_variance_read_readvariableopEsavev2_private_mlp_4_batch_normalization_13_gamma_read_readvariableopDsavev2_private_mlp_4_batch_normalization_13_beta_read_readvariableopKsavev2_private_mlp_4_batch_normalization_13_moving_mean_read_readvariableopOsavev2_private_mlp_4_batch_normalization_13_moving_variance_read_readvariableopEsavev2_private_mlp_4_batch_normalization_14_gamma_read_readvariableopDsavev2_private_mlp_4_batch_normalization_14_beta_read_readvariableopKsavev2_private_mlp_4_batch_normalization_14_moving_mean_read_readvariableopOsavev2_private_mlp_4_batch_normalization_14_moving_variance_read_readvariableop8savev2_private_mlp_4_dense_16_kernel_read_readvariableop6savev2_private_mlp_4_dense_16_bias_read_readvariableop8savev2_private_mlp_4_dense_17_kernel_read_readvariableop6savev2_private_mlp_4_dense_17_bias_read_readvariableop8savev2_private_mlp_4_dense_18_kernel_read_readvariableop6savev2_private_mlp_4_dense_18_bias_read_readvariableop8savev2_private_mlp_4_dense_19_kernel_read_readvariableop6savev2_private_mlp_4_dense_19_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
�: :�:�:�:�:�:�:�:�:�:�:�:�:
��:�:
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
:�:&"
 
_output_shapes
:
��:!
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
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_14118182

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
�U
�
$__inference__traced_restore_14118528
file_prefixJ
;assignvariableop_private_mlp_4_batch_normalization_12_gamma:	�K
<assignvariableop_1_private_mlp_4_batch_normalization_12_beta:	�R
Cassignvariableop_2_private_mlp_4_batch_normalization_12_moving_mean:	�V
Gassignvariableop_3_private_mlp_4_batch_normalization_12_moving_variance:	�L
=assignvariableop_4_private_mlp_4_batch_normalization_13_gamma:	�K
<assignvariableop_5_private_mlp_4_batch_normalization_13_beta:	�R
Cassignvariableop_6_private_mlp_4_batch_normalization_13_moving_mean:	�V
Gassignvariableop_7_private_mlp_4_batch_normalization_13_moving_variance:	�L
=assignvariableop_8_private_mlp_4_batch_normalization_14_gamma:	�K
<assignvariableop_9_private_mlp_4_batch_normalization_14_beta:	�S
Dassignvariableop_10_private_mlp_4_batch_normalization_14_moving_mean:	�W
Hassignvariableop_11_private_mlp_4_batch_normalization_14_moving_variance:	�E
1assignvariableop_12_private_mlp_4_dense_16_kernel:
��>
/assignvariableop_13_private_mlp_4_dense_16_bias:	�E
1assignvariableop_14_private_mlp_4_dense_17_kernel:
��>
/assignvariableop_15_private_mlp_4_dense_17_bias:	�E
1assignvariableop_16_private_mlp_4_dense_18_kernel:
��>
/assignvariableop_17_private_mlp_4_dense_18_bias:	�D
1assignvariableop_18_private_mlp_4_dense_19_kernel:	�=
/assignvariableop_19_private_mlp_4_dense_19_bias:
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
AssignVariableOpAssignVariableOp;assignvariableop_private_mlp_4_batch_normalization_12_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp<assignvariableop_1_private_mlp_4_batch_normalization_12_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpCassignvariableop_2_private_mlp_4_batch_normalization_12_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpGassignvariableop_3_private_mlp_4_batch_normalization_12_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp=assignvariableop_4_private_mlp_4_batch_normalization_13_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp<assignvariableop_5_private_mlp_4_batch_normalization_13_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpCassignvariableop_6_private_mlp_4_batch_normalization_13_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpGassignvariableop_7_private_mlp_4_batch_normalization_13_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp=assignvariableop_8_private_mlp_4_batch_normalization_14_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp<assignvariableop_9_private_mlp_4_batch_normalization_14_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpDassignvariableop_10_private_mlp_4_batch_normalization_14_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpHassignvariableop_11_private_mlp_4_batch_normalization_14_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp1assignvariableop_12_private_mlp_4_dense_16_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp/assignvariableop_13_private_mlp_4_dense_16_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp1assignvariableop_14_private_mlp_4_dense_17_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp/assignvariableop_15_private_mlp_4_dense_17_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp1assignvariableop_16_private_mlp_4_dense_18_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp/assignvariableop_17_private_mlp_4_dense_18_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp1assignvariableop_18_private_mlp_4_dense_19_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp/assignvariableop_19_private_mlp_4_dense_19_biasIdentity_19:output:0"/device:CPU:0*
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
�
�
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_14116876

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
0__inference_private_mlp_4_layer_call_fn_14117748

inputs
unknown:
��
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
GPU 2J 8� *T
fORM
K__inference_private_mlp_4_layer_call_and_return_conditional_losses_14117233o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�?
�	
K__inference_private_mlp_4_layer_call_and_return_conditional_losses_14117656
input_1%
dense_16_14117608:
�� 
dense_16_14117610:	�.
batch_normalization_12_14117613:	�.
batch_normalization_12_14117615:	�.
batch_normalization_12_14117617:	�.
batch_normalization_12_14117619:	�%
dense_17_14117622:
�� 
dense_17_14117624:	�.
batch_normalization_13_14117627:	�.
batch_normalization_13_14117629:	�.
batch_normalization_13_14117631:	�.
batch_normalization_13_14117633:	�%
dense_18_14117636:
�� 
dense_18_14117638:	�.
batch_normalization_14_14117641:	�.
batch_normalization_14_14117643:	�.
batch_normalization_14_14117645:	�.
batch_normalization_14_14117647:	�$
dense_19_14117650:	�
dense_19_14117652:
identity��.batch_normalization_12/StatefulPartitionedCall�.batch_normalization_13/StatefulPartitionedCall�.batch_normalization_14/StatefulPartitionedCall� dense_16/StatefulPartitionedCall� dense_17/StatefulPartitionedCall� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCalld
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
valueB"      �
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
flatten_8/PartitionedCallPartitionedCallstrided_slice:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_8_layer_call_and_return_conditional_losses_14117115f
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
T0*+
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
flatten_9/PartitionedCallPartitionedCallstrided_slice_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_9_layer_call_and_return_conditional_losses_14117127�
concatenate_4/PartitionedCallPartitionedCall"flatten_8/PartitionedCall:output:0"flatten_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_concatenate_4_layer_call_and_return_conditional_losses_14117136�
 dense_16/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0dense_16_14117608dense_16_14117610*
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
GPU 2J 8� *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_14117149�
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0batch_normalization_12_14117613batch_normalization_12_14117615batch_normalization_12_14117617batch_normalization_12_14117619*
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
GPU 2J 8� *]
fXRV
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_14116923�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0dense_17_14117622dense_17_14117624*
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
GPU 2J 8� *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_14117175�
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0batch_normalization_13_14117627batch_normalization_13_14117629batch_normalization_13_14117631batch_normalization_13_14117633*
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
GPU 2J 8� *]
fXRV
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_14117005�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0dense_18_14117636dense_18_14117638*
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
GPU 2J 8� *O
fJRH
F__inference_dense_18_layer_call_and_return_conditional_losses_14117201�
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0batch_normalization_14_14117641batch_normalization_14_14117643batch_normalization_14_14117645batch_normalization_14_14117647*
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
GPU 2J 8� *]
fXRV
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_14117087�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0dense_19_14117650dense_19_14117652*
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
GPU 2J 8� *O
fJRH
F__inference_dense_19_layer_call_and_return_conditional_losses_14117226x
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�

�
F__inference_dense_18_layer_call_and_return_conditional_losses_14118356

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
�
H
,__inference_flatten_9_layer_call_fn_14118037

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
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_9_layer_call_and_return_conditional_losses_14117127`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
0__inference_private_mlp_4_layer_call_fn_14117276
input_1
unknown:
��
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
GPU 2J 8� *T
fORM
K__inference_private_mlp_4_layer_call_and_return_conditional_losses_14117233o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�
c
G__inference_flatten_9_layer_call_and_return_conditional_losses_14118043

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������@X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
0__inference_private_mlp_4_layer_call_fn_14117532
input_1
unknown:
��
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
GPU 2J 8� *T
fORM
K__inference_private_mlp_4_layer_call_and_return_conditional_losses_14117444o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�

�
F__inference_dense_16_layer_call_and_return_conditional_losses_14117149

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
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
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
G__inference_flatten_8_layer_call_and_return_conditional_losses_14117115

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������@X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
F__inference_dense_18_layer_call_and_return_conditional_losses_14117201

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
F__inference_dense_16_layer_call_and_return_conditional_losses_14118316

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
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
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_14118262

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
0__inference_private_mlp_4_layer_call_fn_14117793

inputs
unknown:
��
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
GPU 2J 8� *T
fORM
K__inference_private_mlp_4_layer_call_and_return_conditional_losses_14117444o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�$
�
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_14118296

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
F__inference_dense_17_layer_call_and_return_conditional_losses_14118336

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
F__inference_dense_17_layer_call_and_return_conditional_losses_14117175

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
+__inference_dense_19_layer_call_fn_14118365

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
GPU 2J 8� *O
fJRH
F__inference_dense_19_layer_call_and_return_conditional_losses_14117226o
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
�
�
#__inference__wrapped_model_14116852
input_1I
5private_mlp_4_dense_16_matmul_readvariableop_resource:
��E
6private_mlp_4_dense_16_biasadd_readvariableop_resource:	�P
Aprivate_mlp_4_batch_normalization_12_cast_readvariableop_resource:	�R
Cprivate_mlp_4_batch_normalization_12_cast_1_readvariableop_resource:	�R
Cprivate_mlp_4_batch_normalization_12_cast_2_readvariableop_resource:	�R
Cprivate_mlp_4_batch_normalization_12_cast_3_readvariableop_resource:	�I
5private_mlp_4_dense_17_matmul_readvariableop_resource:
��E
6private_mlp_4_dense_17_biasadd_readvariableop_resource:	�P
Aprivate_mlp_4_batch_normalization_13_cast_readvariableop_resource:	�R
Cprivate_mlp_4_batch_normalization_13_cast_1_readvariableop_resource:	�R
Cprivate_mlp_4_batch_normalization_13_cast_2_readvariableop_resource:	�R
Cprivate_mlp_4_batch_normalization_13_cast_3_readvariableop_resource:	�I
5private_mlp_4_dense_18_matmul_readvariableop_resource:
��E
6private_mlp_4_dense_18_biasadd_readvariableop_resource:	�P
Aprivate_mlp_4_batch_normalization_14_cast_readvariableop_resource:	�R
Cprivate_mlp_4_batch_normalization_14_cast_1_readvariableop_resource:	�R
Cprivate_mlp_4_batch_normalization_14_cast_2_readvariableop_resource:	�R
Cprivate_mlp_4_batch_normalization_14_cast_3_readvariableop_resource:	�H
5private_mlp_4_dense_19_matmul_readvariableop_resource:	�D
6private_mlp_4_dense_19_biasadd_readvariableop_resource:
identity��8private_mlp_4/batch_normalization_12/Cast/ReadVariableOp�:private_mlp_4/batch_normalization_12/Cast_1/ReadVariableOp�:private_mlp_4/batch_normalization_12/Cast_2/ReadVariableOp�:private_mlp_4/batch_normalization_12/Cast_3/ReadVariableOp�8private_mlp_4/batch_normalization_13/Cast/ReadVariableOp�:private_mlp_4/batch_normalization_13/Cast_1/ReadVariableOp�:private_mlp_4/batch_normalization_13/Cast_2/ReadVariableOp�:private_mlp_4/batch_normalization_13/Cast_3/ReadVariableOp�8private_mlp_4/batch_normalization_14/Cast/ReadVariableOp�:private_mlp_4/batch_normalization_14/Cast_1/ReadVariableOp�:private_mlp_4/batch_normalization_14/Cast_2/ReadVariableOp�:private_mlp_4/batch_normalization_14/Cast_3/ReadVariableOp�-private_mlp_4/dense_16/BiasAdd/ReadVariableOp�,private_mlp_4/dense_16/MatMul/ReadVariableOp�-private_mlp_4/dense_17/BiasAdd/ReadVariableOp�,private_mlp_4/dense_17/MatMul/ReadVariableOp�-private_mlp_4/dense_18/BiasAdd/ReadVariableOp�,private_mlp_4/dense_18/MatMul/ReadVariableOp�-private_mlp_4/dense_19/BiasAdd/ReadVariableOp�,private_mlp_4/dense_19/MatMul/ReadVariableOpr
!private_mlp_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        t
#private_mlp_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#private_mlp_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
private_mlp_4/strided_sliceStridedSliceinput_1*private_mlp_4/strided_slice/stack:output:0,private_mlp_4/strided_slice/stack_1:output:0,private_mlp_4/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_maskn
private_mlp_4/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   �
private_mlp_4/flatten_8/ReshapeReshape$private_mlp_4/strided_slice:output:0&private_mlp_4/flatten_8/Const:output:0*
T0*'
_output_shapes
:���������@t
#private_mlp_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       v
%private_mlp_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       v
%private_mlp_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
private_mlp_4/strided_slice_1StridedSliceinput_1,private_mlp_4/strided_slice_1/stack:output:0.private_mlp_4/strided_slice_1/stack_1:output:0.private_mlp_4/strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_maskn
private_mlp_4/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   �
private_mlp_4/flatten_9/ReshapeReshape&private_mlp_4/strided_slice_1:output:0&private_mlp_4/flatten_9/Const:output:0*
T0*'
_output_shapes
:���������@i
'private_mlp_4/concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
"private_mlp_4/concatenate_4/concatConcatV2(private_mlp_4/flatten_8/Reshape:output:0(private_mlp_4/flatten_9/Reshape:output:00private_mlp_4/concatenate_4/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
,private_mlp_4/dense_16/MatMul/ReadVariableOpReadVariableOp5private_mlp_4_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
private_mlp_4/dense_16/MatMulMatMul+private_mlp_4/concatenate_4/concat:output:04private_mlp_4/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-private_mlp_4/dense_16/BiasAdd/ReadVariableOpReadVariableOp6private_mlp_4_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
private_mlp_4/dense_16/BiasAddBiasAdd'private_mlp_4/dense_16/MatMul:product:05private_mlp_4/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
private_mlp_4/dense_16/TanhTanh'private_mlp_4/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
8private_mlp_4/batch_normalization_12/Cast/ReadVariableOpReadVariableOpAprivate_mlp_4_batch_normalization_12_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:private_mlp_4/batch_normalization_12/Cast_1/ReadVariableOpReadVariableOpCprivate_mlp_4_batch_normalization_12_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:private_mlp_4/batch_normalization_12/Cast_2/ReadVariableOpReadVariableOpCprivate_mlp_4_batch_normalization_12_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:private_mlp_4/batch_normalization_12/Cast_3/ReadVariableOpReadVariableOpCprivate_mlp_4_batch_normalization_12_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0y
4private_mlp_4/batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
2private_mlp_4/batch_normalization_12/batchnorm/addAddV2Bprivate_mlp_4/batch_normalization_12/Cast_1/ReadVariableOp:value:0=private_mlp_4/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
4private_mlp_4/batch_normalization_12/batchnorm/RsqrtRsqrt6private_mlp_4/batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes	
:��
2private_mlp_4/batch_normalization_12/batchnorm/mulMul8private_mlp_4/batch_normalization_12/batchnorm/Rsqrt:y:0Bprivate_mlp_4/batch_normalization_12/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
4private_mlp_4/batch_normalization_12/batchnorm/mul_1Mulprivate_mlp_4/dense_16/Tanh:y:06private_mlp_4/batch_normalization_12/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
4private_mlp_4/batch_normalization_12/batchnorm/mul_2Mul@private_mlp_4/batch_normalization_12/Cast/ReadVariableOp:value:06private_mlp_4/batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
2private_mlp_4/batch_normalization_12/batchnorm/subSubBprivate_mlp_4/batch_normalization_12/Cast_2/ReadVariableOp:value:08private_mlp_4/batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
4private_mlp_4/batch_normalization_12/batchnorm/add_1AddV28private_mlp_4/batch_normalization_12/batchnorm/mul_1:z:06private_mlp_4/batch_normalization_12/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
,private_mlp_4/dense_17/MatMul/ReadVariableOpReadVariableOp5private_mlp_4_dense_17_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
private_mlp_4/dense_17/MatMulMatMul8private_mlp_4/batch_normalization_12/batchnorm/add_1:z:04private_mlp_4/dense_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-private_mlp_4/dense_17/BiasAdd/ReadVariableOpReadVariableOp6private_mlp_4_dense_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
private_mlp_4/dense_17/BiasAddBiasAdd'private_mlp_4/dense_17/MatMul:product:05private_mlp_4/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
private_mlp_4/dense_17/TanhTanh'private_mlp_4/dense_17/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
8private_mlp_4/batch_normalization_13/Cast/ReadVariableOpReadVariableOpAprivate_mlp_4_batch_normalization_13_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:private_mlp_4/batch_normalization_13/Cast_1/ReadVariableOpReadVariableOpCprivate_mlp_4_batch_normalization_13_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:private_mlp_4/batch_normalization_13/Cast_2/ReadVariableOpReadVariableOpCprivate_mlp_4_batch_normalization_13_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:private_mlp_4/batch_normalization_13/Cast_3/ReadVariableOpReadVariableOpCprivate_mlp_4_batch_normalization_13_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0y
4private_mlp_4/batch_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
2private_mlp_4/batch_normalization_13/batchnorm/addAddV2Bprivate_mlp_4/batch_normalization_13/Cast_1/ReadVariableOp:value:0=private_mlp_4/batch_normalization_13/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
4private_mlp_4/batch_normalization_13/batchnorm/RsqrtRsqrt6private_mlp_4/batch_normalization_13/batchnorm/add:z:0*
T0*
_output_shapes	
:��
2private_mlp_4/batch_normalization_13/batchnorm/mulMul8private_mlp_4/batch_normalization_13/batchnorm/Rsqrt:y:0Bprivate_mlp_4/batch_normalization_13/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
4private_mlp_4/batch_normalization_13/batchnorm/mul_1Mulprivate_mlp_4/dense_17/Tanh:y:06private_mlp_4/batch_normalization_13/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
4private_mlp_4/batch_normalization_13/batchnorm/mul_2Mul@private_mlp_4/batch_normalization_13/Cast/ReadVariableOp:value:06private_mlp_4/batch_normalization_13/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
2private_mlp_4/batch_normalization_13/batchnorm/subSubBprivate_mlp_4/batch_normalization_13/Cast_2/ReadVariableOp:value:08private_mlp_4/batch_normalization_13/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
4private_mlp_4/batch_normalization_13/batchnorm/add_1AddV28private_mlp_4/batch_normalization_13/batchnorm/mul_1:z:06private_mlp_4/batch_normalization_13/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
,private_mlp_4/dense_18/MatMul/ReadVariableOpReadVariableOp5private_mlp_4_dense_18_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
private_mlp_4/dense_18/MatMulMatMul8private_mlp_4/batch_normalization_13/batchnorm/add_1:z:04private_mlp_4/dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-private_mlp_4/dense_18/BiasAdd/ReadVariableOpReadVariableOp6private_mlp_4_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
private_mlp_4/dense_18/BiasAddBiasAdd'private_mlp_4/dense_18/MatMul:product:05private_mlp_4/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
private_mlp_4/dense_18/TanhTanh'private_mlp_4/dense_18/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
8private_mlp_4/batch_normalization_14/Cast/ReadVariableOpReadVariableOpAprivate_mlp_4_batch_normalization_14_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:private_mlp_4/batch_normalization_14/Cast_1/ReadVariableOpReadVariableOpCprivate_mlp_4_batch_normalization_14_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:private_mlp_4/batch_normalization_14/Cast_2/ReadVariableOpReadVariableOpCprivate_mlp_4_batch_normalization_14_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:private_mlp_4/batch_normalization_14/Cast_3/ReadVariableOpReadVariableOpCprivate_mlp_4_batch_normalization_14_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0y
4private_mlp_4/batch_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
2private_mlp_4/batch_normalization_14/batchnorm/addAddV2Bprivate_mlp_4/batch_normalization_14/Cast_1/ReadVariableOp:value:0=private_mlp_4/batch_normalization_14/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
4private_mlp_4/batch_normalization_14/batchnorm/RsqrtRsqrt6private_mlp_4/batch_normalization_14/batchnorm/add:z:0*
T0*
_output_shapes	
:��
2private_mlp_4/batch_normalization_14/batchnorm/mulMul8private_mlp_4/batch_normalization_14/batchnorm/Rsqrt:y:0Bprivate_mlp_4/batch_normalization_14/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
4private_mlp_4/batch_normalization_14/batchnorm/mul_1Mulprivate_mlp_4/dense_18/Tanh:y:06private_mlp_4/batch_normalization_14/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
4private_mlp_4/batch_normalization_14/batchnorm/mul_2Mul@private_mlp_4/batch_normalization_14/Cast/ReadVariableOp:value:06private_mlp_4/batch_normalization_14/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
2private_mlp_4/batch_normalization_14/batchnorm/subSubBprivate_mlp_4/batch_normalization_14/Cast_2/ReadVariableOp:value:08private_mlp_4/batch_normalization_14/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
4private_mlp_4/batch_normalization_14/batchnorm/add_1AddV28private_mlp_4/batch_normalization_14/batchnorm/mul_1:z:06private_mlp_4/batch_normalization_14/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
,private_mlp_4/dense_19/MatMul/ReadVariableOpReadVariableOp5private_mlp_4_dense_19_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
private_mlp_4/dense_19/MatMulMatMul8private_mlp_4/batch_normalization_14/batchnorm/add_1:z:04private_mlp_4/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-private_mlp_4/dense_19/BiasAdd/ReadVariableOpReadVariableOp6private_mlp_4_dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
private_mlp_4/dense_19/BiasAddBiasAdd'private_mlp_4/dense_19/MatMul:product:05private_mlp_4/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
IdentityIdentity'private_mlp_4/dense_19/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������	
NoOpNoOp9^private_mlp_4/batch_normalization_12/Cast/ReadVariableOp;^private_mlp_4/batch_normalization_12/Cast_1/ReadVariableOp;^private_mlp_4/batch_normalization_12/Cast_2/ReadVariableOp;^private_mlp_4/batch_normalization_12/Cast_3/ReadVariableOp9^private_mlp_4/batch_normalization_13/Cast/ReadVariableOp;^private_mlp_4/batch_normalization_13/Cast_1/ReadVariableOp;^private_mlp_4/batch_normalization_13/Cast_2/ReadVariableOp;^private_mlp_4/batch_normalization_13/Cast_3/ReadVariableOp9^private_mlp_4/batch_normalization_14/Cast/ReadVariableOp;^private_mlp_4/batch_normalization_14/Cast_1/ReadVariableOp;^private_mlp_4/batch_normalization_14/Cast_2/ReadVariableOp;^private_mlp_4/batch_normalization_14/Cast_3/ReadVariableOp.^private_mlp_4/dense_16/BiasAdd/ReadVariableOp-^private_mlp_4/dense_16/MatMul/ReadVariableOp.^private_mlp_4/dense_17/BiasAdd/ReadVariableOp-^private_mlp_4/dense_17/MatMul/ReadVariableOp.^private_mlp_4/dense_18/BiasAdd/ReadVariableOp-^private_mlp_4/dense_18/MatMul/ReadVariableOp.^private_mlp_4/dense_19/BiasAdd/ReadVariableOp-^private_mlp_4/dense_19/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : 2t
8private_mlp_4/batch_normalization_12/Cast/ReadVariableOp8private_mlp_4/batch_normalization_12/Cast/ReadVariableOp2x
:private_mlp_4/batch_normalization_12/Cast_1/ReadVariableOp:private_mlp_4/batch_normalization_12/Cast_1/ReadVariableOp2x
:private_mlp_4/batch_normalization_12/Cast_2/ReadVariableOp:private_mlp_4/batch_normalization_12/Cast_2/ReadVariableOp2x
:private_mlp_4/batch_normalization_12/Cast_3/ReadVariableOp:private_mlp_4/batch_normalization_12/Cast_3/ReadVariableOp2t
8private_mlp_4/batch_normalization_13/Cast/ReadVariableOp8private_mlp_4/batch_normalization_13/Cast/ReadVariableOp2x
:private_mlp_4/batch_normalization_13/Cast_1/ReadVariableOp:private_mlp_4/batch_normalization_13/Cast_1/ReadVariableOp2x
:private_mlp_4/batch_normalization_13/Cast_2/ReadVariableOp:private_mlp_4/batch_normalization_13/Cast_2/ReadVariableOp2x
:private_mlp_4/batch_normalization_13/Cast_3/ReadVariableOp:private_mlp_4/batch_normalization_13/Cast_3/ReadVariableOp2t
8private_mlp_4/batch_normalization_14/Cast/ReadVariableOp8private_mlp_4/batch_normalization_14/Cast/ReadVariableOp2x
:private_mlp_4/batch_normalization_14/Cast_1/ReadVariableOp:private_mlp_4/batch_normalization_14/Cast_1/ReadVariableOp2x
:private_mlp_4/batch_normalization_14/Cast_2/ReadVariableOp:private_mlp_4/batch_normalization_14/Cast_2/ReadVariableOp2x
:private_mlp_4/batch_normalization_14/Cast_3/ReadVariableOp:private_mlp_4/batch_normalization_14/Cast_3/ReadVariableOp2^
-private_mlp_4/dense_16/BiasAdd/ReadVariableOp-private_mlp_4/dense_16/BiasAdd/ReadVariableOp2\
,private_mlp_4/dense_16/MatMul/ReadVariableOp,private_mlp_4/dense_16/MatMul/ReadVariableOp2^
-private_mlp_4/dense_17/BiasAdd/ReadVariableOp-private_mlp_4/dense_17/BiasAdd/ReadVariableOp2\
,private_mlp_4/dense_17/MatMul/ReadVariableOp,private_mlp_4/dense_17/MatMul/ReadVariableOp2^
-private_mlp_4/dense_18/BiasAdd/ReadVariableOp-private_mlp_4/dense_18/BiasAdd/ReadVariableOp2\
,private_mlp_4/dense_18/MatMul/ReadVariableOp,private_mlp_4/dense_18/MatMul/ReadVariableOp2^
-private_mlp_4/dense_19/BiasAdd/ReadVariableOp-private_mlp_4/dense_19/BiasAdd/ReadVariableOp2\
,private_mlp_4/dense_19/MatMul/ReadVariableOp,private_mlp_4/dense_19/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
9__inference_batch_normalization_12_layer_call_fn_14118082

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
GPU 2J 8� *]
fXRV
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_14116923p
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
&__inference_signature_wrapper_14117703
input_1
unknown:
��
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
GPU 2J 8� *,
f'R%
#__inference__wrapped_model_14116852o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�
H
,__inference_flatten_8_layer_call_fn_14118026

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
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_8_layer_call_and_return_conditional_losses_14117115`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�?
�	
K__inference_private_mlp_4_layer_call_and_return_conditional_losses_14117233

inputs%
dense_16_14117150:
�� 
dense_16_14117152:	�.
batch_normalization_12_14117155:	�.
batch_normalization_12_14117157:	�.
batch_normalization_12_14117159:	�.
batch_normalization_12_14117161:	�%
dense_17_14117176:
�� 
dense_17_14117178:	�.
batch_normalization_13_14117181:	�.
batch_normalization_13_14117183:	�.
batch_normalization_13_14117185:	�.
batch_normalization_13_14117187:	�%
dense_18_14117202:
�� 
dense_18_14117204:	�.
batch_normalization_14_14117207:	�.
batch_normalization_14_14117209:	�.
batch_normalization_14_14117211:	�.
batch_normalization_14_14117213:	�$
dense_19_14117227:	�
dense_19_14117229:
identity��.batch_normalization_12/StatefulPartitionedCall�.batch_normalization_13/StatefulPartitionedCall�.batch_normalization_14/StatefulPartitionedCall� dense_16/StatefulPartitionedCall� dense_17/StatefulPartitionedCall� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCalld
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
valueB"      �
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
flatten_8/PartitionedCallPartitionedCallstrided_slice:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_8_layer_call_and_return_conditional_losses_14117115f
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
T0*+
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
flatten_9/PartitionedCallPartitionedCallstrided_slice_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_9_layer_call_and_return_conditional_losses_14117127�
concatenate_4/PartitionedCallPartitionedCall"flatten_8/PartitionedCall:output:0"flatten_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_concatenate_4_layer_call_and_return_conditional_losses_14117136�
 dense_16/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0dense_16_14117150dense_16_14117152*
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
GPU 2J 8� *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_14117149�
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0batch_normalization_12_14117155batch_normalization_12_14117157batch_normalization_12_14117159batch_normalization_12_14117161*
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
GPU 2J 8� *]
fXRV
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_14116876�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0dense_17_14117176dense_17_14117178*
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
GPU 2J 8� *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_14117175�
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0batch_normalization_13_14117181batch_normalization_13_14117183batch_normalization_13_14117185batch_normalization_13_14117187*
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
GPU 2J 8� *]
fXRV
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_14116958�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0dense_18_14117202dense_18_14117204*
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
GPU 2J 8� *O
fJRH
F__inference_dense_18_layer_call_and_return_conditional_losses_14117201�
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0batch_normalization_14_14117207batch_normalization_14_14117209batch_normalization_14_14117211batch_normalization_14_14117213*
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
GPU 2J 8� *]
fXRV
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_14117040�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0dense_19_14117227dense_19_14117229*
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
GPU 2J 8� *O
fJRH
F__inference_dense_19_layer_call_and_return_conditional_losses_14117226x
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�?
�	
K__inference_private_mlp_4_layer_call_and_return_conditional_losses_14117594
input_1%
dense_16_14117546:
�� 
dense_16_14117548:	�.
batch_normalization_12_14117551:	�.
batch_normalization_12_14117553:	�.
batch_normalization_12_14117555:	�.
batch_normalization_12_14117557:	�%
dense_17_14117560:
�� 
dense_17_14117562:	�.
batch_normalization_13_14117565:	�.
batch_normalization_13_14117567:	�.
batch_normalization_13_14117569:	�.
batch_normalization_13_14117571:	�%
dense_18_14117574:
�� 
dense_18_14117576:	�.
batch_normalization_14_14117579:	�.
batch_normalization_14_14117581:	�.
batch_normalization_14_14117583:	�.
batch_normalization_14_14117585:	�$
dense_19_14117588:	�
dense_19_14117590:
identity��.batch_normalization_12/StatefulPartitionedCall�.batch_normalization_13/StatefulPartitionedCall�.batch_normalization_14/StatefulPartitionedCall� dense_16/StatefulPartitionedCall� dense_17/StatefulPartitionedCall� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCalld
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
valueB"      �
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
flatten_8/PartitionedCallPartitionedCallstrided_slice:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_8_layer_call_and_return_conditional_losses_14117115f
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
T0*+
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
flatten_9/PartitionedCallPartitionedCallstrided_slice_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_9_layer_call_and_return_conditional_losses_14117127�
concatenate_4/PartitionedCallPartitionedCall"flatten_8/PartitionedCall:output:0"flatten_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_concatenate_4_layer_call_and_return_conditional_losses_14117136�
 dense_16/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0dense_16_14117546dense_16_14117548*
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
GPU 2J 8� *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_14117149�
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0batch_normalization_12_14117551batch_normalization_12_14117553batch_normalization_12_14117555batch_normalization_12_14117557*
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
GPU 2J 8� *]
fXRV
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_14116876�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0dense_17_14117560dense_17_14117562*
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
GPU 2J 8� *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_14117175�
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0batch_normalization_13_14117565batch_normalization_13_14117567batch_normalization_13_14117569batch_normalization_13_14117571*
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
GPU 2J 8� *]
fXRV
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_14116958�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0dense_18_14117574dense_18_14117576*
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
GPU 2J 8� *O
fJRH
F__inference_dense_18_layer_call_and_return_conditional_losses_14117201�
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0batch_normalization_14_14117579batch_normalization_14_14117581batch_normalization_14_14117583batch_normalization_14_14117585*
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
GPU 2J 8� *]
fXRV
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_14117040�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0dense_19_14117588dense_19_14117590*
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
GPU 2J 8� *O
fJRH
F__inference_dense_19_layer_call_and_return_conditional_losses_14117226x
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�$
�
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_14117005

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
�
c
G__inference_flatten_9_layer_call_and_return_conditional_losses_14117127

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������@X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
G__inference_flatten_8_layer_call_and_return_conditional_losses_14118032

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������@X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
9__inference_batch_normalization_14_layer_call_fn_14118229

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
GPU 2J 8� *]
fXRV
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_14117040p
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
�
�
9__inference_batch_normalization_13_layer_call_fn_14118162

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
GPU 2J 8� *]
fXRV
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_14117005p
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
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_14118216

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
�
�
+__inference_dense_18_layer_call_fn_14118345

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
GPU 2J 8� *O
fJRH
F__inference_dense_18_layer_call_and_return_conditional_losses_14117201p
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
�$
�
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_14118136

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
�
�
+__inference_dense_16_layer_call_fn_14118305

inputs
unknown:
��
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
GPU 2J 8� *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_14117149p
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
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
9__inference_batch_normalization_13_layer_call_fn_14118149

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
GPU 2J 8� *]
fXRV
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_14116958p
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
F__inference_dense_19_layer_call_and_return_conditional_losses_14118375

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
��
�
K__inference_private_mlp_4_layer_call_and_return_conditional_losses_14118021

inputs;
'dense_16_matmul_readvariableop_resource:
��7
(dense_16_biasadd_readvariableop_resource:	�M
>batch_normalization_12_assignmovingavg_readvariableop_resource:	�O
@batch_normalization_12_assignmovingavg_1_readvariableop_resource:	�B
3batch_normalization_12_cast_readvariableop_resource:	�D
5batch_normalization_12_cast_1_readvariableop_resource:	�;
'dense_17_matmul_readvariableop_resource:
��7
(dense_17_biasadd_readvariableop_resource:	�M
>batch_normalization_13_assignmovingavg_readvariableop_resource:	�O
@batch_normalization_13_assignmovingavg_1_readvariableop_resource:	�B
3batch_normalization_13_cast_readvariableop_resource:	�D
5batch_normalization_13_cast_1_readvariableop_resource:	�;
'dense_18_matmul_readvariableop_resource:
��7
(dense_18_biasadd_readvariableop_resource:	�M
>batch_normalization_14_assignmovingavg_readvariableop_resource:	�O
@batch_normalization_14_assignmovingavg_1_readvariableop_resource:	�B
3batch_normalization_14_cast_readvariableop_resource:	�D
5batch_normalization_14_cast_1_readvariableop_resource:	�:
'dense_19_matmul_readvariableop_resource:	�6
(dense_19_biasadd_readvariableop_resource:
identity��&batch_normalization_12/AssignMovingAvg�5batch_normalization_12/AssignMovingAvg/ReadVariableOp�(batch_normalization_12/AssignMovingAvg_1�7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_12/Cast/ReadVariableOp�,batch_normalization_12/Cast_1/ReadVariableOp�&batch_normalization_13/AssignMovingAvg�5batch_normalization_13/AssignMovingAvg/ReadVariableOp�(batch_normalization_13/AssignMovingAvg_1�7batch_normalization_13/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_13/Cast/ReadVariableOp�,batch_normalization_13/Cast_1/ReadVariableOp�&batch_normalization_14/AssignMovingAvg�5batch_normalization_14/AssignMovingAvg/ReadVariableOp�(batch_normalization_14/AssignMovingAvg_1�7batch_normalization_14/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_14/Cast/ReadVariableOp�,batch_normalization_14/Cast_1/ReadVariableOp�dense_16/BiasAdd/ReadVariableOp�dense_16/MatMul/ReadVariableOp�dense_17/BiasAdd/ReadVariableOp�dense_17/MatMul/ReadVariableOp�dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOpd
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
valueB"      �
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask`
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   �
flatten_8/ReshapeReshapestrided_slice:output:0flatten_8/Const:output:0*
T0*'
_output_shapes
:���������@f
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
T0*+
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask`
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   �
flatten_9/ReshapeReshapestrided_slice_1:output:0flatten_9/Const:output:0*
T0*'
_output_shapes
:���������@[
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_4/concatConcatV2flatten_8/Reshape:output:0flatten_9/Reshape:output:0"concatenate_4/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_16/MatMulMatMulconcatenate_4/concat:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_16/TanhTanhdense_16/BiasAdd:output:0*
T0*(
_output_shapes
:����������
5batch_normalization_12/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_12/moments/meanMeandense_16/Tanh:y:0>batch_normalization_12/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
+batch_normalization_12/moments/StopGradientStopGradient,batch_normalization_12/moments/mean:output:0*
T0*
_output_shapes
:	��
0batch_normalization_12/moments/SquaredDifferenceSquaredDifferencedense_16/Tanh:y:04batch_normalization_12/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
9batch_normalization_12/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_12/moments/varianceMean4batch_normalization_12/moments/SquaredDifference:z:0Bbatch_normalization_12/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
&batch_normalization_12/moments/SqueezeSqueeze,batch_normalization_12/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
(batch_normalization_12/moments/Squeeze_1Squeeze0batch_normalization_12/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 q
,batch_normalization_12/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_12/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_12_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*batch_normalization_12/AssignMovingAvg/subSub=batch_normalization_12/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_12/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
*batch_normalization_12/AssignMovingAvg/mulMul.batch_normalization_12/AssignMovingAvg/sub:z:05batch_normalization_12/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
&batch_normalization_12/AssignMovingAvgAssignSubVariableOp>batch_normalization_12_assignmovingavg_readvariableop_resource.batch_normalization_12/AssignMovingAvg/mul:z:06^batch_normalization_12/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_12/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_12_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_12/AssignMovingAvg_1/subSub?batch_normalization_12/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_12/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
,batch_normalization_12/AssignMovingAvg_1/mulMul0batch_normalization_12/AssignMovingAvg_1/sub:z:07batch_normalization_12/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
(batch_normalization_12/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_12_assignmovingavg_1_readvariableop_resource0batch_normalization_12/AssignMovingAvg_1/mul:z:08^batch_normalization_12/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
*batch_normalization_12/Cast/ReadVariableOpReadVariableOp3batch_normalization_12_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_12/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_12_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_12/batchnorm/addAddV21batch_normalization_12/moments/Squeeze_1:output:0/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_12/batchnorm/RsqrtRsqrt(batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_12/batchnorm/mulMul*batch_normalization_12/batchnorm/Rsqrt:y:04batch_normalization_12/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_12/batchnorm/mul_1Muldense_16/Tanh:y:0(batch_normalization_12/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
&batch_normalization_12/batchnorm/mul_2Mul/batch_normalization_12/moments/Squeeze:output:0(batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
$batch_normalization_12/batchnorm/subSub2batch_normalization_12/Cast/ReadVariableOp:value:0*batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
&batch_normalization_12/batchnorm/add_1AddV2*batch_normalization_12/batchnorm/mul_1:z:0(batch_normalization_12/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_17/MatMulMatMul*batch_normalization_12/batchnorm/add_1:z:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_17/TanhTanhdense_17/BiasAdd:output:0*
T0*(
_output_shapes
:����������
5batch_normalization_13/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_13/moments/meanMeandense_17/Tanh:y:0>batch_normalization_13/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
+batch_normalization_13/moments/StopGradientStopGradient,batch_normalization_13/moments/mean:output:0*
T0*
_output_shapes
:	��
0batch_normalization_13/moments/SquaredDifferenceSquaredDifferencedense_17/Tanh:y:04batch_normalization_13/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
9batch_normalization_13/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_13/moments/varianceMean4batch_normalization_13/moments/SquaredDifference:z:0Bbatch_normalization_13/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
&batch_normalization_13/moments/SqueezeSqueeze,batch_normalization_13/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
(batch_normalization_13/moments/Squeeze_1Squeeze0batch_normalization_13/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 q
,batch_normalization_13/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_13/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_13_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*batch_normalization_13/AssignMovingAvg/subSub=batch_normalization_13/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_13/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
*batch_normalization_13/AssignMovingAvg/mulMul.batch_normalization_13/AssignMovingAvg/sub:z:05batch_normalization_13/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
&batch_normalization_13/AssignMovingAvgAssignSubVariableOp>batch_normalization_13_assignmovingavg_readvariableop_resource.batch_normalization_13/AssignMovingAvg/mul:z:06^batch_normalization_13/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_13/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_13/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_13_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_13/AssignMovingAvg_1/subSub?batch_normalization_13/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_13/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
,batch_normalization_13/AssignMovingAvg_1/mulMul0batch_normalization_13/AssignMovingAvg_1/sub:z:07batch_normalization_13/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
(batch_normalization_13/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_13_assignmovingavg_1_readvariableop_resource0batch_normalization_13/AssignMovingAvg_1/mul:z:08^batch_normalization_13/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
*batch_normalization_13/Cast/ReadVariableOpReadVariableOp3batch_normalization_13_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_13/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_13_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_13/batchnorm/addAddV21batch_normalization_13/moments/Squeeze_1:output:0/batch_normalization_13/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_13/batchnorm/RsqrtRsqrt(batch_normalization_13/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_13/batchnorm/mulMul*batch_normalization_13/batchnorm/Rsqrt:y:04batch_normalization_13/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_13/batchnorm/mul_1Muldense_17/Tanh:y:0(batch_normalization_13/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
&batch_normalization_13/batchnorm/mul_2Mul/batch_normalization_13/moments/Squeeze:output:0(batch_normalization_13/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
$batch_normalization_13/batchnorm/subSub2batch_normalization_13/Cast/ReadVariableOp:value:0*batch_normalization_13/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
&batch_normalization_13/batchnorm/add_1AddV2*batch_normalization_13/batchnorm/mul_1:z:0(batch_normalization_13/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_18/MatMulMatMul*batch_normalization_13/batchnorm/add_1:z:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_18/TanhTanhdense_18/BiasAdd:output:0*
T0*(
_output_shapes
:����������
5batch_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_14/moments/meanMeandense_18/Tanh:y:0>batch_normalization_14/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
+batch_normalization_14/moments/StopGradientStopGradient,batch_normalization_14/moments/mean:output:0*
T0*
_output_shapes
:	��
0batch_normalization_14/moments/SquaredDifferenceSquaredDifferencedense_18/Tanh:y:04batch_normalization_14/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
9batch_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_14/moments/varianceMean4batch_normalization_14/moments/SquaredDifference:z:0Bbatch_normalization_14/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
&batch_normalization_14/moments/SqueezeSqueeze,batch_normalization_14/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
(batch_normalization_14/moments/Squeeze_1Squeeze0batch_normalization_14/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 q
,batch_normalization_14/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_14/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_14_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*batch_normalization_14/AssignMovingAvg/subSub=batch_normalization_14/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_14/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
*batch_normalization_14/AssignMovingAvg/mulMul.batch_normalization_14/AssignMovingAvg/sub:z:05batch_normalization_14/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
&batch_normalization_14/AssignMovingAvgAssignSubVariableOp>batch_normalization_14_assignmovingavg_readvariableop_resource.batch_normalization_14/AssignMovingAvg/mul:z:06^batch_normalization_14/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_14/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_14/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_14_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_14/AssignMovingAvg_1/subSub?batch_normalization_14/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_14/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
,batch_normalization_14/AssignMovingAvg_1/mulMul0batch_normalization_14/AssignMovingAvg_1/sub:z:07batch_normalization_14/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
(batch_normalization_14/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_14_assignmovingavg_1_readvariableop_resource0batch_normalization_14/AssignMovingAvg_1/mul:z:08^batch_normalization_14/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
*batch_normalization_14/Cast/ReadVariableOpReadVariableOp3batch_normalization_14_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_14/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_14_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_14/batchnorm/addAddV21batch_normalization_14/moments/Squeeze_1:output:0/batch_normalization_14/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_14/batchnorm/RsqrtRsqrt(batch_normalization_14/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_14/batchnorm/mulMul*batch_normalization_14/batchnorm/Rsqrt:y:04batch_normalization_14/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_14/batchnorm/mul_1Muldense_18/Tanh:y:0(batch_normalization_14/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
&batch_normalization_14/batchnorm/mul_2Mul/batch_normalization_14/moments/Squeeze:output:0(batch_normalization_14/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
$batch_normalization_14/batchnorm/subSub2batch_normalization_14/Cast/ReadVariableOp:value:0*batch_normalization_14/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
&batch_normalization_14/batchnorm/add_1AddV2*batch_normalization_14/batchnorm/mul_1:z:0(batch_normalization_14/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_19/MatMulMatMul*batch_normalization_14/batchnorm/add_1:z:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_19/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������	
NoOpNoOp'^batch_normalization_12/AssignMovingAvg6^batch_normalization_12/AssignMovingAvg/ReadVariableOp)^batch_normalization_12/AssignMovingAvg_18^batch_normalization_12/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_12/Cast/ReadVariableOp-^batch_normalization_12/Cast_1/ReadVariableOp'^batch_normalization_13/AssignMovingAvg6^batch_normalization_13/AssignMovingAvg/ReadVariableOp)^batch_normalization_13/AssignMovingAvg_18^batch_normalization_13/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_13/Cast/ReadVariableOp-^batch_normalization_13/Cast_1/ReadVariableOp'^batch_normalization_14/AssignMovingAvg6^batch_normalization_14/AssignMovingAvg/ReadVariableOp)^batch_normalization_14/AssignMovingAvg_18^batch_normalization_14/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_14/Cast/ReadVariableOp-^batch_normalization_14/Cast_1/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_12/AssignMovingAvg&batch_normalization_12/AssignMovingAvg2n
5batch_normalization_12/AssignMovingAvg/ReadVariableOp5batch_normalization_12/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_12/AssignMovingAvg_1(batch_normalization_12/AssignMovingAvg_12r
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_12/Cast/ReadVariableOp*batch_normalization_12/Cast/ReadVariableOp2\
,batch_normalization_12/Cast_1/ReadVariableOp,batch_normalization_12/Cast_1/ReadVariableOp2P
&batch_normalization_13/AssignMovingAvg&batch_normalization_13/AssignMovingAvg2n
5batch_normalization_13/AssignMovingAvg/ReadVariableOp5batch_normalization_13/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_13/AssignMovingAvg_1(batch_normalization_13/AssignMovingAvg_12r
7batch_normalization_13/AssignMovingAvg_1/ReadVariableOp7batch_normalization_13/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_13/Cast/ReadVariableOp*batch_normalization_13/Cast/ReadVariableOp2\
,batch_normalization_13/Cast_1/ReadVariableOp,batch_normalization_13/Cast_1/ReadVariableOp2P
&batch_normalization_14/AssignMovingAvg&batch_normalization_14/AssignMovingAvg2n
5batch_normalization_14/AssignMovingAvg/ReadVariableOp5batch_normalization_14/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_14/AssignMovingAvg_1(batch_normalization_14/AssignMovingAvg_12r
7batch_normalization_14/AssignMovingAvg_1/ReadVariableOp7batch_normalization_14/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_14/Cast/ReadVariableOp*batch_normalization_14/Cast/ReadVariableOp2\
,batch_normalization_14/Cast_1/ReadVariableOp,batch_normalization_14/Cast_1/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�$
�
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_14116923

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
\
0__inference_concatenate_4_layer_call_fn_14118049
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_concatenate_4_layer_call_and_return_conditional_losses_14117136a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:���������@:Q M
'
_output_shapes
:���������@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
inputs/1
�	
�
F__inference_dense_19_layer_call_and_return_conditional_losses_14117226

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
�
�
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_14116958

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
�?
�	
K__inference_private_mlp_4_layer_call_and_return_conditional_losses_14117444

inputs%
dense_16_14117396:
�� 
dense_16_14117398:	�.
batch_normalization_12_14117401:	�.
batch_normalization_12_14117403:	�.
batch_normalization_12_14117405:	�.
batch_normalization_12_14117407:	�%
dense_17_14117410:
�� 
dense_17_14117412:	�.
batch_normalization_13_14117415:	�.
batch_normalization_13_14117417:	�.
batch_normalization_13_14117419:	�.
batch_normalization_13_14117421:	�%
dense_18_14117424:
�� 
dense_18_14117426:	�.
batch_normalization_14_14117429:	�.
batch_normalization_14_14117431:	�.
batch_normalization_14_14117433:	�.
batch_normalization_14_14117435:	�$
dense_19_14117438:	�
dense_19_14117440:
identity��.batch_normalization_12/StatefulPartitionedCall�.batch_normalization_13/StatefulPartitionedCall�.batch_normalization_14/StatefulPartitionedCall� dense_16/StatefulPartitionedCall� dense_17/StatefulPartitionedCall� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCalld
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
valueB"      �
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
flatten_8/PartitionedCallPartitionedCallstrided_slice:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_8_layer_call_and_return_conditional_losses_14117115f
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
T0*+
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
flatten_9/PartitionedCallPartitionedCallstrided_slice_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_9_layer_call_and_return_conditional_losses_14117127�
concatenate_4/PartitionedCallPartitionedCall"flatten_8/PartitionedCall:output:0"flatten_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_concatenate_4_layer_call_and_return_conditional_losses_14117136�
 dense_16/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0dense_16_14117396dense_16_14117398*
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
GPU 2J 8� *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_14117149�
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0batch_normalization_12_14117401batch_normalization_12_14117403batch_normalization_12_14117405batch_normalization_12_14117407*
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
GPU 2J 8� *]
fXRV
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_14116923�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0dense_17_14117410dense_17_14117412*
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
GPU 2J 8� *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_14117175�
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0batch_normalization_13_14117415batch_normalization_13_14117417batch_normalization_13_14117419batch_normalization_13_14117421*
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
GPU 2J 8� *]
fXRV
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_14117005�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0dense_18_14117424dense_18_14117426*
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
GPU 2J 8� *O
fJRH
F__inference_dense_18_layer_call_and_return_conditional_losses_14117201�
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0batch_normalization_14_14117429batch_normalization_14_14117431batch_normalization_14_14117433batch_normalization_14_14117435*
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
GPU 2J 8� *]
fXRV
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_14117087�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0dense_19_14117438dense_19_14117440*
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
GPU 2J 8� *O
fJRH
F__inference_dense_19_layer_call_and_return_conditional_losses_14117226x
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_14118102

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
9__inference_batch_normalization_12_layer_call_fn_14118069

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
GPU 2J 8� *]
fXRV
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_14116876p
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
�y
�
K__inference_private_mlp_4_layer_call_and_return_conditional_losses_14117886

inputs;
'dense_16_matmul_readvariableop_resource:
��7
(dense_16_biasadd_readvariableop_resource:	�B
3batch_normalization_12_cast_readvariableop_resource:	�D
5batch_normalization_12_cast_1_readvariableop_resource:	�D
5batch_normalization_12_cast_2_readvariableop_resource:	�D
5batch_normalization_12_cast_3_readvariableop_resource:	�;
'dense_17_matmul_readvariableop_resource:
��7
(dense_17_biasadd_readvariableop_resource:	�B
3batch_normalization_13_cast_readvariableop_resource:	�D
5batch_normalization_13_cast_1_readvariableop_resource:	�D
5batch_normalization_13_cast_2_readvariableop_resource:	�D
5batch_normalization_13_cast_3_readvariableop_resource:	�;
'dense_18_matmul_readvariableop_resource:
��7
(dense_18_biasadd_readvariableop_resource:	�B
3batch_normalization_14_cast_readvariableop_resource:	�D
5batch_normalization_14_cast_1_readvariableop_resource:	�D
5batch_normalization_14_cast_2_readvariableop_resource:	�D
5batch_normalization_14_cast_3_readvariableop_resource:	�:
'dense_19_matmul_readvariableop_resource:	�6
(dense_19_biasadd_readvariableop_resource:
identity��*batch_normalization_12/Cast/ReadVariableOp�,batch_normalization_12/Cast_1/ReadVariableOp�,batch_normalization_12/Cast_2/ReadVariableOp�,batch_normalization_12/Cast_3/ReadVariableOp�*batch_normalization_13/Cast/ReadVariableOp�,batch_normalization_13/Cast_1/ReadVariableOp�,batch_normalization_13/Cast_2/ReadVariableOp�,batch_normalization_13/Cast_3/ReadVariableOp�*batch_normalization_14/Cast/ReadVariableOp�,batch_normalization_14/Cast_1/ReadVariableOp�,batch_normalization_14/Cast_2/ReadVariableOp�,batch_normalization_14/Cast_3/ReadVariableOp�dense_16/BiasAdd/ReadVariableOp�dense_16/MatMul/ReadVariableOp�dense_17/BiasAdd/ReadVariableOp�dense_17/MatMul/ReadVariableOp�dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOpd
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
valueB"      �
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask`
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   �
flatten_8/ReshapeReshapestrided_slice:output:0flatten_8/Const:output:0*
T0*'
_output_shapes
:���������@f
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
T0*+
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask`
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   �
flatten_9/ReshapeReshapestrided_slice_1:output:0flatten_9/Const:output:0*
T0*'
_output_shapes
:���������@[
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_4/concatConcatV2flatten_8/Reshape:output:0flatten_9/Reshape:output:0"concatenate_4/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_16/MatMulMatMulconcatenate_4/concat:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_16/TanhTanhdense_16/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
*batch_normalization_12/Cast/ReadVariableOpReadVariableOp3batch_normalization_12_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_12/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_12_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_12/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_12_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_12/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_12_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_12/batchnorm/addAddV24batch_normalization_12/Cast_1/ReadVariableOp:value:0/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_12/batchnorm/RsqrtRsqrt(batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_12/batchnorm/mulMul*batch_normalization_12/batchnorm/Rsqrt:y:04batch_normalization_12/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_12/batchnorm/mul_1Muldense_16/Tanh:y:0(batch_normalization_12/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
&batch_normalization_12/batchnorm/mul_2Mul2batch_normalization_12/Cast/ReadVariableOp:value:0(batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
$batch_normalization_12/batchnorm/subSub4batch_normalization_12/Cast_2/ReadVariableOp:value:0*batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
&batch_normalization_12/batchnorm/add_1AddV2*batch_normalization_12/batchnorm/mul_1:z:0(batch_normalization_12/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_17/MatMulMatMul*batch_normalization_12/batchnorm/add_1:z:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_17/TanhTanhdense_17/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
*batch_normalization_13/Cast/ReadVariableOpReadVariableOp3batch_normalization_13_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_13/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_13_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_13/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_13_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_13/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_13_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_13/batchnorm/addAddV24batch_normalization_13/Cast_1/ReadVariableOp:value:0/batch_normalization_13/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_13/batchnorm/RsqrtRsqrt(batch_normalization_13/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_13/batchnorm/mulMul*batch_normalization_13/batchnorm/Rsqrt:y:04batch_normalization_13/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_13/batchnorm/mul_1Muldense_17/Tanh:y:0(batch_normalization_13/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
&batch_normalization_13/batchnorm/mul_2Mul2batch_normalization_13/Cast/ReadVariableOp:value:0(batch_normalization_13/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
$batch_normalization_13/batchnorm/subSub4batch_normalization_13/Cast_2/ReadVariableOp:value:0*batch_normalization_13/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
&batch_normalization_13/batchnorm/add_1AddV2*batch_normalization_13/batchnorm/mul_1:z:0(batch_normalization_13/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_18/MatMulMatMul*batch_normalization_13/batchnorm/add_1:z:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_18/TanhTanhdense_18/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
*batch_normalization_14/Cast/ReadVariableOpReadVariableOp3batch_normalization_14_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_14/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_14_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_14/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_14_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_14/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_14_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_14/batchnorm/addAddV24batch_normalization_14/Cast_1/ReadVariableOp:value:0/batch_normalization_14/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_14/batchnorm/RsqrtRsqrt(batch_normalization_14/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_14/batchnorm/mulMul*batch_normalization_14/batchnorm/Rsqrt:y:04batch_normalization_14/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_14/batchnorm/mul_1Muldense_18/Tanh:y:0(batch_normalization_14/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
&batch_normalization_14/batchnorm/mul_2Mul2batch_normalization_14/Cast/ReadVariableOp:value:0(batch_normalization_14/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
$batch_normalization_14/batchnorm/subSub4batch_normalization_14/Cast_2/ReadVariableOp:value:0*batch_normalization_14/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
&batch_normalization_14/batchnorm/add_1AddV2*batch_normalization_14/batchnorm/mul_1:z:0(batch_normalization_14/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_19/MatMulMatMul*batch_normalization_14/batchnorm/add_1:z:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_19/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp+^batch_normalization_12/Cast/ReadVariableOp-^batch_normalization_12/Cast_1/ReadVariableOp-^batch_normalization_12/Cast_2/ReadVariableOp-^batch_normalization_12/Cast_3/ReadVariableOp+^batch_normalization_13/Cast/ReadVariableOp-^batch_normalization_13/Cast_1/ReadVariableOp-^batch_normalization_13/Cast_2/ReadVariableOp-^batch_normalization_13/Cast_3/ReadVariableOp+^batch_normalization_14/Cast/ReadVariableOp-^batch_normalization_14/Cast_1/ReadVariableOp-^batch_normalization_14/Cast_2/ReadVariableOp-^batch_normalization_14/Cast_3/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : 2X
*batch_normalization_12/Cast/ReadVariableOp*batch_normalization_12/Cast/ReadVariableOp2\
,batch_normalization_12/Cast_1/ReadVariableOp,batch_normalization_12/Cast_1/ReadVariableOp2\
,batch_normalization_12/Cast_2/ReadVariableOp,batch_normalization_12/Cast_2/ReadVariableOp2\
,batch_normalization_12/Cast_3/ReadVariableOp,batch_normalization_12/Cast_3/ReadVariableOp2X
*batch_normalization_13/Cast/ReadVariableOp*batch_normalization_13/Cast/ReadVariableOp2\
,batch_normalization_13/Cast_1/ReadVariableOp,batch_normalization_13/Cast_1/ReadVariableOp2\
,batch_normalization_13/Cast_2/ReadVariableOp,batch_normalization_13/Cast_2/ReadVariableOp2\
,batch_normalization_13/Cast_3/ReadVariableOp,batch_normalization_13/Cast_3/ReadVariableOp2X
*batch_normalization_14/Cast/ReadVariableOp*batch_normalization_14/Cast/ReadVariableOp2\
,batch_normalization_14/Cast_1/ReadVariableOp,batch_normalization_14/Cast_1/ReadVariableOp2\
,batch_normalization_14/Cast_2/ReadVariableOp,batch_normalization_14/Cast_2/ReadVariableOp2\
,batch_normalization_14/Cast_3/ReadVariableOp,batch_normalization_14/Cast_3/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
w
K__inference_concatenate_4_layer_call_and_return_conditional_losses_14118056
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :x
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:���������@:Q M
'
_output_shapes
:���������@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
inputs/1
�
�
+__inference_dense_17_layer_call_fn_14118325

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
GPU 2J 8� *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_14117175p
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
�
�
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_14117040

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
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_14117087

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
9__inference_batch_normalization_14_layer_call_fn_14118242

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
GPU 2J 8� *]
fXRV
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_14117087p
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
�
u
K__inference_concatenate_4_layer_call_and_return_conditional_losses_14117136

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :v
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������@
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_18
serving_default_input_1:0���������<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:ف
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
0__inference_private_mlp_4_layer_call_fn_14117276
0__inference_private_mlp_4_layer_call_fn_14117748
0__inference_private_mlp_4_layer_call_fn_14117793
0__inference_private_mlp_4_layer_call_fn_14117532�
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
K__inference_private_mlp_4_layer_call_and_return_conditional_losses_14117886
K__inference_private_mlp_4_layer_call_and_return_conditional_losses_14118021
K__inference_private_mlp_4_layer_call_and_return_conditional_losses_14117594
K__inference_private_mlp_4_layer_call_and_return_conditional_losses_14117656�
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
#__inference__wrapped_model_14116852input_1"�
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
9:7�2*private_mlp_4/batch_normalization_12/gamma
8:6�2)private_mlp_4/batch_normalization_12/beta
A:?� (20private_mlp_4/batch_normalization_12/moving_mean
E:C� (24private_mlp_4/batch_normalization_12/moving_variance
9:7�2*private_mlp_4/batch_normalization_13/gamma
8:6�2)private_mlp_4/batch_normalization_13/beta
A:?� (20private_mlp_4/batch_normalization_13/moving_mean
E:C� (24private_mlp_4/batch_normalization_13/moving_variance
9:7�2*private_mlp_4/batch_normalization_14/gamma
8:6�2)private_mlp_4/batch_normalization_14/beta
A:?� (20private_mlp_4/batch_normalization_14/moving_mean
E:C� (24private_mlp_4/batch_normalization_14/moving_variance
1:/
��2private_mlp_4/dense_16/kernel
*:(�2private_mlp_4/dense_16/bias
1:/
��2private_mlp_4/dense_17/kernel
*:(�2private_mlp_4/dense_17/bias
1:/
��2private_mlp_4/dense_18/kernel
*:(�2private_mlp_4/dense_18/bias
0:.	�2private_mlp_4/dense_19/kernel
):'2private_mlp_4/dense_19/bias
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
0__inference_private_mlp_4_layer_call_fn_14117276input_1"�
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
0__inference_private_mlp_4_layer_call_fn_14117748inputs"�
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
0__inference_private_mlp_4_layer_call_fn_14117793inputs"�
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
0__inference_private_mlp_4_layer_call_fn_14117532input_1"�
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
K__inference_private_mlp_4_layer_call_and_return_conditional_losses_14117886inputs"�
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
K__inference_private_mlp_4_layer_call_and_return_conditional_losses_14118021inputs"�
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
K__inference_private_mlp_4_layer_call_and_return_conditional_losses_14117594input_1"�
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
K__inference_private_mlp_4_layer_call_and_return_conditional_losses_14117656input_1"�
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
,__inference_flatten_8_layer_call_fn_14118026�
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
G__inference_flatten_8_layer_call_and_return_conditional_losses_14118032�
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
,__inference_flatten_9_layer_call_fn_14118037�
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
G__inference_flatten_9_layer_call_and_return_conditional_losses_14118043�
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
0__inference_concatenate_4_layer_call_fn_14118049�
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
K__inference_concatenate_4_layer_call_and_return_conditional_losses_14118056�
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
9__inference_batch_normalization_12_layer_call_fn_14118069
9__inference_batch_normalization_12_layer_call_fn_14118082�
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
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_14118102
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_14118136�
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
9__inference_batch_normalization_13_layer_call_fn_14118149
9__inference_batch_normalization_13_layer_call_fn_14118162�
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
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_14118182
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_14118216�
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
9__inference_batch_normalization_14_layer_call_fn_14118229
9__inference_batch_normalization_14_layer_call_fn_14118242�
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
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_14118262
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_14118296�
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
+__inference_dense_16_layer_call_fn_14118305�
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
F__inference_dense_16_layer_call_and_return_conditional_losses_14118316�
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
+__inference_dense_17_layer_call_fn_14118325�
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
F__inference_dense_17_layer_call_and_return_conditional_losses_14118336�
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
+__inference_dense_18_layer_call_fn_14118345�
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
F__inference_dense_18_layer_call_and_return_conditional_losses_14118356�
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
+__inference_dense_19_layer_call_fn_14118365�
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
F__inference_dense_19_layer_call_and_return_conditional_losses_14118375�
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
&__inference_signature_wrapper_14117703input_1"�
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
,__inference_flatten_8_layer_call_fn_14118026inputs"�
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
G__inference_flatten_8_layer_call_and_return_conditional_losses_14118032inputs"�
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
,__inference_flatten_9_layer_call_fn_14118037inputs"�
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
G__inference_flatten_9_layer_call_and_return_conditional_losses_14118043inputs"�
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
0__inference_concatenate_4_layer_call_fn_14118049inputs/0inputs/1"�
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
K__inference_concatenate_4_layer_call_and_return_conditional_losses_14118056inputs/0inputs/1"�
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
9__inference_batch_normalization_12_layer_call_fn_14118069inputs"�
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
9__inference_batch_normalization_12_layer_call_fn_14118082inputs"�
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
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_14118102inputs"�
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
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_14118136inputs"�
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
9__inference_batch_normalization_13_layer_call_fn_14118149inputs"�
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
9__inference_batch_normalization_13_layer_call_fn_14118162inputs"�
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
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_14118182inputs"�
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
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_14118216inputs"�
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
9__inference_batch_normalization_14_layer_call_fn_14118229inputs"�
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
9__inference_batch_normalization_14_layer_call_fn_14118242inputs"�
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
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_14118262inputs"�
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
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_14118296inputs"�
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
+__inference_dense_16_layer_call_fn_14118305inputs"�
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
F__inference_dense_16_layer_call_and_return_conditional_losses_14118316inputs"�
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
+__inference_dense_17_layer_call_fn_14118325inputs"�
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
F__inference_dense_17_layer_call_and_return_conditional_losses_14118336inputs"�
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
+__inference_dense_18_layer_call_fn_14118345inputs"�
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
F__inference_dense_18_layer_call_and_return_conditional_losses_14118356inputs"�
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
+__inference_dense_19_layer_call_fn_14118365inputs"�
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
F__inference_dense_19_layer_call_and_return_conditional_losses_14118375inputs"�
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
#__inference__wrapped_model_14116852� !"#$%&'8�5
.�+
)�&
input_1���������
� "3�0
.
output_1"�
output_1����������
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_14118102d4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_14118136d4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
9__inference_batch_normalization_12_layer_call_fn_14118069W4�1
*�'
!�
inputs����������
p 
� "������������
9__inference_batch_normalization_12_layer_call_fn_14118082W4�1
*�'
!�
inputs����������
p
� "������������
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_14118182d4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_14118216d4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
9__inference_batch_normalization_13_layer_call_fn_14118149W4�1
*�'
!�
inputs����������
p 
� "������������
9__inference_batch_normalization_13_layer_call_fn_14118162W4�1
*�'
!�
inputs����������
p
� "������������
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_14118262d4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_14118296d4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
9__inference_batch_normalization_14_layer_call_fn_14118229W4�1
*�'
!�
inputs����������
p 
� "������������
9__inference_batch_normalization_14_layer_call_fn_14118242W4�1
*�'
!�
inputs����������
p
� "������������
K__inference_concatenate_4_layer_call_and_return_conditional_losses_14118056�Z�W
P�M
K�H
"�
inputs/0���������@
"�
inputs/1���������@
� "&�#
�
0����������
� �
0__inference_concatenate_4_layer_call_fn_14118049wZ�W
P�M
K�H
"�
inputs/0���������@
"�
inputs/1���������@
� "������������
F__inference_dense_16_layer_call_and_return_conditional_losses_14118316^ !0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
+__inference_dense_16_layer_call_fn_14118305Q !0�-
&�#
!�
inputs����������
� "������������
F__inference_dense_17_layer_call_and_return_conditional_losses_14118336^"#0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
+__inference_dense_17_layer_call_fn_14118325Q"#0�-
&�#
!�
inputs����������
� "������������
F__inference_dense_18_layer_call_and_return_conditional_losses_14118356^$%0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
+__inference_dense_18_layer_call_fn_14118345Q$%0�-
&�#
!�
inputs����������
� "������������
F__inference_dense_19_layer_call_and_return_conditional_losses_14118375]&'0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� 
+__inference_dense_19_layer_call_fn_14118365P&'0�-
&�#
!�
inputs����������
� "�����������
G__inference_flatten_8_layer_call_and_return_conditional_losses_14118032\3�0
)�&
$�!
inputs���������
� "%�"
�
0���������@
� 
,__inference_flatten_8_layer_call_fn_14118026O3�0
)�&
$�!
inputs���������
� "����������@�
G__inference_flatten_9_layer_call_and_return_conditional_losses_14118043\3�0
)�&
$�!
inputs���������
� "%�"
�
0���������@
� 
,__inference_flatten_9_layer_call_fn_14118037O3�0
)�&
$�!
inputs���������
� "����������@�
K__inference_private_mlp_4_layer_call_and_return_conditional_losses_14117594{ !"#$%&'<�9
2�/
)�&
input_1���������
p 
� "%�"
�
0���������
� �
K__inference_private_mlp_4_layer_call_and_return_conditional_losses_14117656{ !"#$%&'<�9
2�/
)�&
input_1���������
p
� "%�"
�
0���������
� �
K__inference_private_mlp_4_layer_call_and_return_conditional_losses_14117886z !"#$%&';�8
1�.
(�%
inputs���������
p 
� "%�"
�
0���������
� �
K__inference_private_mlp_4_layer_call_and_return_conditional_losses_14118021z !"#$%&';�8
1�.
(�%
inputs���������
p
� "%�"
�
0���������
� �
0__inference_private_mlp_4_layer_call_fn_14117276n !"#$%&'<�9
2�/
)�&
input_1���������
p 
� "�����������
0__inference_private_mlp_4_layer_call_fn_14117532n !"#$%&'<�9
2�/
)�&
input_1���������
p
� "�����������
0__inference_private_mlp_4_layer_call_fn_14117748m !"#$%&';�8
1�.
(�%
inputs���������
p 
� "�����������
0__inference_private_mlp_4_layer_call_fn_14117793m !"#$%&';�8
1�.
(�%
inputs���������
p
� "�����������
&__inference_signature_wrapper_14117703� !"#$%&'C�@
� 
9�6
4
input_1)�&
input_1���������"3�0
.
output_1"�
output_1���������
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
private_mlp_5/dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameprivate_mlp_5/dense_23/bias
�
/private_mlp_5/dense_23/bias/Read/ReadVariableOpReadVariableOpprivate_mlp_5/dense_23/bias*
_output_shapes
:*
dtype0
�
private_mlp_5/dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*.
shared_nameprivate_mlp_5/dense_23/kernel
�
1private_mlp_5/dense_23/kernel/Read/ReadVariableOpReadVariableOpprivate_mlp_5/dense_23/kernel*
_output_shapes
:	�*
dtype0
�
private_mlp_5/dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_nameprivate_mlp_5/dense_22/bias
�
/private_mlp_5/dense_22/bias/Read/ReadVariableOpReadVariableOpprivate_mlp_5/dense_22/bias*
_output_shapes	
:�*
dtype0
�
private_mlp_5/dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*.
shared_nameprivate_mlp_5/dense_22/kernel
�
1private_mlp_5/dense_22/kernel/Read/ReadVariableOpReadVariableOpprivate_mlp_5/dense_22/kernel* 
_output_shapes
:
��*
dtype0
�
private_mlp_5/dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_nameprivate_mlp_5/dense_21/bias
�
/private_mlp_5/dense_21/bias/Read/ReadVariableOpReadVariableOpprivate_mlp_5/dense_21/bias*
_output_shapes	
:�*
dtype0
�
private_mlp_5/dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*.
shared_nameprivate_mlp_5/dense_21/kernel
�
1private_mlp_5/dense_21/kernel/Read/ReadVariableOpReadVariableOpprivate_mlp_5/dense_21/kernel* 
_output_shapes
:
��*
dtype0
�
private_mlp_5/dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_nameprivate_mlp_5/dense_20/bias
�
/private_mlp_5/dense_20/bias/Read/ReadVariableOpReadVariableOpprivate_mlp_5/dense_20/bias*
_output_shapes	
:�*
dtype0
�
private_mlp_5/dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*.
shared_nameprivate_mlp_5/dense_20/kernel
�
1private_mlp_5/dense_20/kernel/Read/ReadVariableOpReadVariableOpprivate_mlp_5/dense_20/kernel* 
_output_shapes
:
��*
dtype0
�
4private_mlp_5/batch_normalization_17/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*E
shared_name64private_mlp_5/batch_normalization_17/moving_variance
�
Hprivate_mlp_5/batch_normalization_17/moving_variance/Read/ReadVariableOpReadVariableOp4private_mlp_5/batch_normalization_17/moving_variance*
_output_shapes	
:�*
dtype0
�
0private_mlp_5/batch_normalization_17/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*A
shared_name20private_mlp_5/batch_normalization_17/moving_mean
�
Dprivate_mlp_5/batch_normalization_17/moving_mean/Read/ReadVariableOpReadVariableOp0private_mlp_5/batch_normalization_17/moving_mean*
_output_shapes	
:�*
dtype0
�
)private_mlp_5/batch_normalization_17/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*:
shared_name+)private_mlp_5/batch_normalization_17/beta
�
=private_mlp_5/batch_normalization_17/beta/Read/ReadVariableOpReadVariableOp)private_mlp_5/batch_normalization_17/beta*
_output_shapes	
:�*
dtype0
�
*private_mlp_5/batch_normalization_17/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*;
shared_name,*private_mlp_5/batch_normalization_17/gamma
�
>private_mlp_5/batch_normalization_17/gamma/Read/ReadVariableOpReadVariableOp*private_mlp_5/batch_normalization_17/gamma*
_output_shapes	
:�*
dtype0
�
4private_mlp_5/batch_normalization_16/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*E
shared_name64private_mlp_5/batch_normalization_16/moving_variance
�
Hprivate_mlp_5/batch_normalization_16/moving_variance/Read/ReadVariableOpReadVariableOp4private_mlp_5/batch_normalization_16/moving_variance*
_output_shapes	
:�*
dtype0
�
0private_mlp_5/batch_normalization_16/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*A
shared_name20private_mlp_5/batch_normalization_16/moving_mean
�
Dprivate_mlp_5/batch_normalization_16/moving_mean/Read/ReadVariableOpReadVariableOp0private_mlp_5/batch_normalization_16/moving_mean*
_output_shapes	
:�*
dtype0
�
)private_mlp_5/batch_normalization_16/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*:
shared_name+)private_mlp_5/batch_normalization_16/beta
�
=private_mlp_5/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOp)private_mlp_5/batch_normalization_16/beta*
_output_shapes	
:�*
dtype0
�
*private_mlp_5/batch_normalization_16/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*;
shared_name,*private_mlp_5/batch_normalization_16/gamma
�
>private_mlp_5/batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOp*private_mlp_5/batch_normalization_16/gamma*
_output_shapes	
:�*
dtype0
�
4private_mlp_5/batch_normalization_15/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*E
shared_name64private_mlp_5/batch_normalization_15/moving_variance
�
Hprivate_mlp_5/batch_normalization_15/moving_variance/Read/ReadVariableOpReadVariableOp4private_mlp_5/batch_normalization_15/moving_variance*
_output_shapes	
:�*
dtype0
�
0private_mlp_5/batch_normalization_15/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*A
shared_name20private_mlp_5/batch_normalization_15/moving_mean
�
Dprivate_mlp_5/batch_normalization_15/moving_mean/Read/ReadVariableOpReadVariableOp0private_mlp_5/batch_normalization_15/moving_mean*
_output_shapes	
:�*
dtype0
�
)private_mlp_5/batch_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*:
shared_name+)private_mlp_5/batch_normalization_15/beta
�
=private_mlp_5/batch_normalization_15/beta/Read/ReadVariableOpReadVariableOp)private_mlp_5/batch_normalization_15/beta*
_output_shapes	
:�*
dtype0
�
*private_mlp_5/batch_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*;
shared_name,*private_mlp_5/batch_normalization_15/gamma
�
>private_mlp_5/batch_normalization_15/gamma/Read/ReadVariableOpReadVariableOp*private_mlp_5/batch_normalization_15/gamma*
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
VARIABLE_VALUE*private_mlp_5/batch_normalization_15/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)private_mlp_5/batch_normalization_15/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE0private_mlp_5/batch_normalization_15/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE4private_mlp_5/batch_normalization_15/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE*private_mlp_5/batch_normalization_16/gamma&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)private_mlp_5/batch_normalization_16/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE0private_mlp_5/batch_normalization_16/moving_mean&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE4private_mlp_5/batch_normalization_16/moving_variance&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE*private_mlp_5/batch_normalization_17/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)private_mlp_5/batch_normalization_17/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE0private_mlp_5/batch_normalization_17/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE4private_mlp_5/batch_normalization_17/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEprivate_mlp_5/dense_20/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEprivate_mlp_5/dense_20/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEprivate_mlp_5/dense_21/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEprivate_mlp_5/dense_21/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEprivate_mlp_5/dense_22/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEprivate_mlp_5/dense_22/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEprivate_mlp_5/dense_23/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEprivate_mlp_5/dense_23/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1private_mlp_5/dense_20/kernelprivate_mlp_5/dense_20/bias0private_mlp_5/batch_normalization_15/moving_mean4private_mlp_5/batch_normalization_15/moving_variance)private_mlp_5/batch_normalization_15/beta*private_mlp_5/batch_normalization_15/gammaprivate_mlp_5/dense_21/kernelprivate_mlp_5/dense_21/bias0private_mlp_5/batch_normalization_16/moving_mean4private_mlp_5/batch_normalization_16/moving_variance)private_mlp_5/batch_normalization_16/beta*private_mlp_5/batch_normalization_16/gammaprivate_mlp_5/dense_22/kernelprivate_mlp_5/dense_22/bias0private_mlp_5/batch_normalization_17/moving_mean4private_mlp_5/batch_normalization_17/moving_variance)private_mlp_5/batch_normalization_17/beta*private_mlp_5/batch_normalization_17/gammaprivate_mlp_5/dense_23/kernelprivate_mlp_5/dense_23/bias* 
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
%__inference_signature_wrapper_6789470
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename>private_mlp_5/batch_normalization_15/gamma/Read/ReadVariableOp=private_mlp_5/batch_normalization_15/beta/Read/ReadVariableOpDprivate_mlp_5/batch_normalization_15/moving_mean/Read/ReadVariableOpHprivate_mlp_5/batch_normalization_15/moving_variance/Read/ReadVariableOp>private_mlp_5/batch_normalization_16/gamma/Read/ReadVariableOp=private_mlp_5/batch_normalization_16/beta/Read/ReadVariableOpDprivate_mlp_5/batch_normalization_16/moving_mean/Read/ReadVariableOpHprivate_mlp_5/batch_normalization_16/moving_variance/Read/ReadVariableOp>private_mlp_5/batch_normalization_17/gamma/Read/ReadVariableOp=private_mlp_5/batch_normalization_17/beta/Read/ReadVariableOpDprivate_mlp_5/batch_normalization_17/moving_mean/Read/ReadVariableOpHprivate_mlp_5/batch_normalization_17/moving_variance/Read/ReadVariableOp1private_mlp_5/dense_20/kernel/Read/ReadVariableOp/private_mlp_5/dense_20/bias/Read/ReadVariableOp1private_mlp_5/dense_21/kernel/Read/ReadVariableOp/private_mlp_5/dense_21/bias/Read/ReadVariableOp1private_mlp_5/dense_22/kernel/Read/ReadVariableOp/private_mlp_5/dense_22/bias/Read/ReadVariableOp1private_mlp_5/dense_23/kernel/Read/ReadVariableOp/private_mlp_5/dense_23/bias/Read/ReadVariableOpConst*!
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
 __inference__traced_save_6790225
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename*private_mlp_5/batch_normalization_15/gamma)private_mlp_5/batch_normalization_15/beta0private_mlp_5/batch_normalization_15/moving_mean4private_mlp_5/batch_normalization_15/moving_variance*private_mlp_5/batch_normalization_16/gamma)private_mlp_5/batch_normalization_16/beta0private_mlp_5/batch_normalization_16/moving_mean4private_mlp_5/batch_normalization_16/moving_variance*private_mlp_5/batch_normalization_17/gamma)private_mlp_5/batch_normalization_17/beta0private_mlp_5/batch_normalization_17/moving_mean4private_mlp_5/batch_normalization_17/moving_varianceprivate_mlp_5/dense_20/kernelprivate_mlp_5/dense_20/biasprivate_mlp_5/dense_21/kernelprivate_mlp_5/dense_21/biasprivate_mlp_5/dense_22/kernelprivate_mlp_5/dense_22/biasprivate_mlp_5/dense_23/kernelprivate_mlp_5/dense_23/bias* 
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
#__inference__traced_restore_6790295ý
�
v
J__inference_concatenate_5_layer_call_and_return_conditional_losses_6789823
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
�
c
G__inference_flatten_10_layer_call_and_return_conditional_losses_6789799

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
�
�
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_6789869

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
/__inference_private_mlp_5_layer_call_fn_6789043
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
GPU 2J 8� *S
fNRL
J__inference_private_mlp_5_layer_call_and_return_conditional_losses_6789000o
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
�
/__inference_private_mlp_5_layer_call_fn_6789515

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
GPU 2J 8� *S
fNRL
J__inference_private_mlp_5_layer_call_and_return_conditional_losses_6789000o
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
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_6790063

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
��
�
J__inference_private_mlp_5_layer_call_and_return_conditional_losses_6789788

inputs;
'dense_20_matmul_readvariableop_resource:
��7
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
valueB"      �
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_maska
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   �
flatten_10/ReshapeReshapestrided_slice:output:0flatten_10/Const:output:0*
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
shrink_axis_maska
flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   �
flatten_11/ReshapeReshapestrided_slice_1:output:0flatten_11/Const:output:0*
T0*'
_output_shapes
:���������@[
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_5/concatConcatV2flatten_10/Reshape:output:0flatten_11/Reshape:output:0"concatenate_5/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource* 
_output_shapes
:
��*
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
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : 2P
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
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_15_layer_call_fn_6789836

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
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_6788643p
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
E__inference_dense_20_layer_call_and_return_conditional_losses_6790083

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
�

�
E__inference_dense_21_layer_call_and_return_conditional_losses_6790103

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
E__inference_dense_22_layer_call_and_return_conditional_losses_6790123

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
�U
�
#__inference__traced_restore_6790295
file_prefixJ
;assignvariableop_private_mlp_5_batch_normalization_15_gamma:	�K
<assignvariableop_1_private_mlp_5_batch_normalization_15_beta:	�R
Cassignvariableop_2_private_mlp_5_batch_normalization_15_moving_mean:	�V
Gassignvariableop_3_private_mlp_5_batch_normalization_15_moving_variance:	�L
=assignvariableop_4_private_mlp_5_batch_normalization_16_gamma:	�K
<assignvariableop_5_private_mlp_5_batch_normalization_16_beta:	�R
Cassignvariableop_6_private_mlp_5_batch_normalization_16_moving_mean:	�V
Gassignvariableop_7_private_mlp_5_batch_normalization_16_moving_variance:	�L
=assignvariableop_8_private_mlp_5_batch_normalization_17_gamma:	�K
<assignvariableop_9_private_mlp_5_batch_normalization_17_beta:	�S
Dassignvariableop_10_private_mlp_5_batch_normalization_17_moving_mean:	�W
Hassignvariableop_11_private_mlp_5_batch_normalization_17_moving_variance:	�E
1assignvariableop_12_private_mlp_5_dense_20_kernel:
��>
/assignvariableop_13_private_mlp_5_dense_20_bias:	�E
1assignvariableop_14_private_mlp_5_dense_21_kernel:
��>
/assignvariableop_15_private_mlp_5_dense_21_bias:	�E
1assignvariableop_16_private_mlp_5_dense_22_kernel:
��>
/assignvariableop_17_private_mlp_5_dense_22_bias:	�D
1assignvariableop_18_private_mlp_5_dense_23_kernel:	�=
/assignvariableop_19_private_mlp_5_dense_23_bias:
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
AssignVariableOpAssignVariableOp;assignvariableop_private_mlp_5_batch_normalization_15_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp<assignvariableop_1_private_mlp_5_batch_normalization_15_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpCassignvariableop_2_private_mlp_5_batch_normalization_15_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpGassignvariableop_3_private_mlp_5_batch_normalization_15_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp=assignvariableop_4_private_mlp_5_batch_normalization_16_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp<assignvariableop_5_private_mlp_5_batch_normalization_16_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpCassignvariableop_6_private_mlp_5_batch_normalization_16_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpGassignvariableop_7_private_mlp_5_batch_normalization_16_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp=assignvariableop_8_private_mlp_5_batch_normalization_17_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp<assignvariableop_9_private_mlp_5_batch_normalization_17_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpDassignvariableop_10_private_mlp_5_batch_normalization_17_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpHassignvariableop_11_private_mlp_5_batch_normalization_17_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp1assignvariableop_12_private_mlp_5_dense_20_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp/assignvariableop_13_private_mlp_5_dense_20_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp1assignvariableop_14_private_mlp_5_dense_21_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp/assignvariableop_15_private_mlp_5_dense_21_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp1assignvariableop_16_private_mlp_5_dense_22_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp/assignvariableop_17_private_mlp_5_dense_22_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp1assignvariableop_18_private_mlp_5_dense_23_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp/assignvariableop_19_private_mlp_5_dense_23_biasIdentity_19:output:0"/device:CPU:0*
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
�
�
8__inference_batch_normalization_16_layer_call_fn_6789916

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
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_6788725p
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
*__inference_dense_22_layer_call_fn_6790112

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
E__inference_dense_22_layer_call_and_return_conditional_losses_6788968p
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
�
c
G__inference_flatten_10_layer_call_and_return_conditional_losses_6788882

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
�
�
*__inference_dense_20_layer_call_fn_6790072

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
GPU 2J 8� *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_6788916p
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
�>
�	
J__inference_private_mlp_5_layer_call_and_return_conditional_losses_6789361
input_1$
dense_20_6789313:
��
dense_20_6789315:	�-
batch_normalization_15_6789318:	�-
batch_normalization_15_6789320:	�-
batch_normalization_15_6789322:	�-
batch_normalization_15_6789324:	�$
dense_21_6789327:
��
dense_21_6789329:	�-
batch_normalization_16_6789332:	�-
batch_normalization_16_6789334:	�-
batch_normalization_16_6789336:	�-
batch_normalization_16_6789338:	�$
dense_22_6789341:
��
dense_22_6789343:	�-
batch_normalization_17_6789346:	�-
batch_normalization_17_6789348:	�-
batch_normalization_17_6789350:	�-
batch_normalization_17_6789352:	�#
dense_23_6789355:	�
dense_23_6789357:
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
valueB"      �
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

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
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_10_layer_call_and_return_conditional_losses_6788882f
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
flatten_11/PartitionedCallPartitionedCallstrided_slice_1:output:0*
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
G__inference_flatten_11_layer_call_and_return_conditional_losses_6788894�
concatenate_5/PartitionedCallPartitionedCall#flatten_10/PartitionedCall:output:0#flatten_11/PartitionedCall:output:0*
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
GPU 2J 8� *S
fNRL
J__inference_concatenate_5_layer_call_and_return_conditional_losses_6788903�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0dense_20_6789313dense_20_6789315*
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
E__inference_dense_20_layer_call_and_return_conditional_losses_6788916�
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0batch_normalization_15_6789318batch_normalization_15_6789320batch_normalization_15_6789322batch_normalization_15_6789324*
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
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_6788643�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0dense_21_6789327dense_21_6789329*
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
E__inference_dense_21_layer_call_and_return_conditional_losses_6788942�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0batch_normalization_16_6789332batch_normalization_16_6789334batch_normalization_16_6789336batch_normalization_16_6789338*
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
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_6788725�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0dense_22_6789341dense_22_6789343*
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
E__inference_dense_22_layer_call_and_return_conditional_losses_6788968�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0batch_normalization_17_6789346batch_normalization_17_6789348batch_normalization_17_6789350batch_normalization_17_6789352*
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
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_6788807�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0dense_23_6789355dense_23_6789357*
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
E__inference_dense_23_layer_call_and_return_conditional_losses_6788993x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�
[
/__inference_concatenate_5_layer_call_fn_6789816
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
GPU 2J 8� *S
fNRL
J__inference_concatenate_5_layer_call_and_return_conditional_losses_6788903a
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
�
�
8__inference_batch_normalization_17_layer_call_fn_6789996

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
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_6788807p
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
*__inference_dense_21_layer_call_fn_6790092

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
E__inference_dense_21_layer_call_and_return_conditional_losses_6788942p
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
�y
�
J__inference_private_mlp_5_layer_call_and_return_conditional_losses_6789653

inputs;
'dense_20_matmul_readvariableop_resource:
��7
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
valueB"      �
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_maska
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   �
flatten_10/ReshapeReshapestrided_slice:output:0flatten_10/Const:output:0*
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
shrink_axis_maska
flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   �
flatten_11/ReshapeReshapestrided_slice_1:output:0flatten_11/Const:output:0*
T0*'
_output_shapes
:���������@[
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_5/concatConcatV2flatten_10/Reshape:output:0flatten_11/Reshape:output:0"concatenate_5/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource* 
_output_shapes
:
��*
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
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : 2X
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
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_6788807

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
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_6788772

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
�$
�
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_6789903

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
E__inference_dense_20_layer_call_and_return_conditional_losses_6788916

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
�>
�	
J__inference_private_mlp_5_layer_call_and_return_conditional_losses_6789000

inputs$
dense_20_6788917:
��
dense_20_6788919:	�-
batch_normalization_15_6788922:	�-
batch_normalization_15_6788924:	�-
batch_normalization_15_6788926:	�-
batch_normalization_15_6788928:	�$
dense_21_6788943:
��
dense_21_6788945:	�-
batch_normalization_16_6788948:	�-
batch_normalization_16_6788950:	�-
batch_normalization_16_6788952:	�-
batch_normalization_16_6788954:	�$
dense_22_6788969:
��
dense_22_6788971:	�-
batch_normalization_17_6788974:	�-
batch_normalization_17_6788976:	�-
batch_normalization_17_6788978:	�-
batch_normalization_17_6788980:	�#
dense_23_6788994:	�
dense_23_6788996:
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
valueB"      �
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

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
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_10_layer_call_and_return_conditional_losses_6788882f
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
flatten_11/PartitionedCallPartitionedCallstrided_slice_1:output:0*
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
G__inference_flatten_11_layer_call_and_return_conditional_losses_6788894�
concatenate_5/PartitionedCallPartitionedCall#flatten_10/PartitionedCall:output:0#flatten_11/PartitionedCall:output:0*
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
GPU 2J 8� *S
fNRL
J__inference_concatenate_5_layer_call_and_return_conditional_losses_6788903�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0dense_20_6788917dense_20_6788919*
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
E__inference_dense_20_layer_call_and_return_conditional_losses_6788916�
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0batch_normalization_15_6788922batch_normalization_15_6788924batch_normalization_15_6788926batch_normalization_15_6788928*
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
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_6788643�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0dense_21_6788943dense_21_6788945*
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
E__inference_dense_21_layer_call_and_return_conditional_losses_6788942�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0batch_normalization_16_6788948batch_normalization_16_6788950batch_normalization_16_6788952batch_normalization_16_6788954*
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
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_6788725�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0dense_22_6788969dense_22_6788971*
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
E__inference_dense_22_layer_call_and_return_conditional_losses_6788968�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0batch_normalization_17_6788974batch_normalization_17_6788976batch_normalization_17_6788978batch_normalization_17_6788980*
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
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_6788807�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0dense_23_6788994dense_23_6788996*
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
E__inference_dense_23_layer_call_and_return_conditional_losses_6788993x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_6788643

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
/__inference_private_mlp_5_layer_call_fn_6789299
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
GPU 2J 8� *S
fNRL
J__inference_private_mlp_5_layer_call_and_return_conditional_losses_6789211o
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
G__inference_flatten_11_layer_call_and_return_conditional_losses_6788894

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
/__inference_private_mlp_5_layer_call_fn_6789560

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
GPU 2J 8� *S
fNRL
J__inference_private_mlp_5_layer_call_and_return_conditional_losses_6789211o
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
�
H
,__inference_flatten_10_layer_call_fn_6789793

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
G__inference_flatten_10_layer_call_and_return_conditional_losses_6788882`
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
�
�
8__inference_batch_normalization_15_layer_call_fn_6789849

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
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_6788690p
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
�
�
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_6789949

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
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_6789983

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
�$
�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_6788854

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
�
�
"__inference__wrapped_model_6788619
input_1I
5private_mlp_5_dense_20_matmul_readvariableop_resource:
��E
6private_mlp_5_dense_20_biasadd_readvariableop_resource:	�P
Aprivate_mlp_5_batch_normalization_15_cast_readvariableop_resource:	�R
Cprivate_mlp_5_batch_normalization_15_cast_1_readvariableop_resource:	�R
Cprivate_mlp_5_batch_normalization_15_cast_2_readvariableop_resource:	�R
Cprivate_mlp_5_batch_normalization_15_cast_3_readvariableop_resource:	�I
5private_mlp_5_dense_21_matmul_readvariableop_resource:
��E
6private_mlp_5_dense_21_biasadd_readvariableop_resource:	�P
Aprivate_mlp_5_batch_normalization_16_cast_readvariableop_resource:	�R
Cprivate_mlp_5_batch_normalization_16_cast_1_readvariableop_resource:	�R
Cprivate_mlp_5_batch_normalization_16_cast_2_readvariableop_resource:	�R
Cprivate_mlp_5_batch_normalization_16_cast_3_readvariableop_resource:	�I
5private_mlp_5_dense_22_matmul_readvariableop_resource:
��E
6private_mlp_5_dense_22_biasadd_readvariableop_resource:	�P
Aprivate_mlp_5_batch_normalization_17_cast_readvariableop_resource:	�R
Cprivate_mlp_5_batch_normalization_17_cast_1_readvariableop_resource:	�R
Cprivate_mlp_5_batch_normalization_17_cast_2_readvariableop_resource:	�R
Cprivate_mlp_5_batch_normalization_17_cast_3_readvariableop_resource:	�H
5private_mlp_5_dense_23_matmul_readvariableop_resource:	�D
6private_mlp_5_dense_23_biasadd_readvariableop_resource:
identity��8private_mlp_5/batch_normalization_15/Cast/ReadVariableOp�:private_mlp_5/batch_normalization_15/Cast_1/ReadVariableOp�:private_mlp_5/batch_normalization_15/Cast_2/ReadVariableOp�:private_mlp_5/batch_normalization_15/Cast_3/ReadVariableOp�8private_mlp_5/batch_normalization_16/Cast/ReadVariableOp�:private_mlp_5/batch_normalization_16/Cast_1/ReadVariableOp�:private_mlp_5/batch_normalization_16/Cast_2/ReadVariableOp�:private_mlp_5/batch_normalization_16/Cast_3/ReadVariableOp�8private_mlp_5/batch_normalization_17/Cast/ReadVariableOp�:private_mlp_5/batch_normalization_17/Cast_1/ReadVariableOp�:private_mlp_5/batch_normalization_17/Cast_2/ReadVariableOp�:private_mlp_5/batch_normalization_17/Cast_3/ReadVariableOp�-private_mlp_5/dense_20/BiasAdd/ReadVariableOp�,private_mlp_5/dense_20/MatMul/ReadVariableOp�-private_mlp_5/dense_21/BiasAdd/ReadVariableOp�,private_mlp_5/dense_21/MatMul/ReadVariableOp�-private_mlp_5/dense_22/BiasAdd/ReadVariableOp�,private_mlp_5/dense_22/MatMul/ReadVariableOp�-private_mlp_5/dense_23/BiasAdd/ReadVariableOp�,private_mlp_5/dense_23/MatMul/ReadVariableOpr
!private_mlp_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        t
#private_mlp_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#private_mlp_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
private_mlp_5/strided_sliceStridedSliceinput_1*private_mlp_5/strided_slice/stack:output:0,private_mlp_5/strided_slice/stack_1:output:0,private_mlp_5/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_masko
private_mlp_5/flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   �
 private_mlp_5/flatten_10/ReshapeReshape$private_mlp_5/strided_slice:output:0'private_mlp_5/flatten_10/Const:output:0*
T0*'
_output_shapes
:���������@t
#private_mlp_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       v
%private_mlp_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       v
%private_mlp_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
private_mlp_5/strided_slice_1StridedSliceinput_1,private_mlp_5/strided_slice_1/stack:output:0.private_mlp_5/strided_slice_1/stack_1:output:0.private_mlp_5/strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_masko
private_mlp_5/flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   �
 private_mlp_5/flatten_11/ReshapeReshape&private_mlp_5/strided_slice_1:output:0'private_mlp_5/flatten_11/Const:output:0*
T0*'
_output_shapes
:���������@i
'private_mlp_5/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
"private_mlp_5/concatenate_5/concatConcatV2)private_mlp_5/flatten_10/Reshape:output:0)private_mlp_5/flatten_11/Reshape:output:00private_mlp_5/concatenate_5/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
,private_mlp_5/dense_20/MatMul/ReadVariableOpReadVariableOp5private_mlp_5_dense_20_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
private_mlp_5/dense_20/MatMulMatMul+private_mlp_5/concatenate_5/concat:output:04private_mlp_5/dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-private_mlp_5/dense_20/BiasAdd/ReadVariableOpReadVariableOp6private_mlp_5_dense_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
private_mlp_5/dense_20/BiasAddBiasAdd'private_mlp_5/dense_20/MatMul:product:05private_mlp_5/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
private_mlp_5/dense_20/TanhTanh'private_mlp_5/dense_20/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
8private_mlp_5/batch_normalization_15/Cast/ReadVariableOpReadVariableOpAprivate_mlp_5_batch_normalization_15_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:private_mlp_5/batch_normalization_15/Cast_1/ReadVariableOpReadVariableOpCprivate_mlp_5_batch_normalization_15_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:private_mlp_5/batch_normalization_15/Cast_2/ReadVariableOpReadVariableOpCprivate_mlp_5_batch_normalization_15_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:private_mlp_5/batch_normalization_15/Cast_3/ReadVariableOpReadVariableOpCprivate_mlp_5_batch_normalization_15_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0y
4private_mlp_5/batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
2private_mlp_5/batch_normalization_15/batchnorm/addAddV2Bprivate_mlp_5/batch_normalization_15/Cast_1/ReadVariableOp:value:0=private_mlp_5/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
4private_mlp_5/batch_normalization_15/batchnorm/RsqrtRsqrt6private_mlp_5/batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes	
:��
2private_mlp_5/batch_normalization_15/batchnorm/mulMul8private_mlp_5/batch_normalization_15/batchnorm/Rsqrt:y:0Bprivate_mlp_5/batch_normalization_15/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
4private_mlp_5/batch_normalization_15/batchnorm/mul_1Mulprivate_mlp_5/dense_20/Tanh:y:06private_mlp_5/batch_normalization_15/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
4private_mlp_5/batch_normalization_15/batchnorm/mul_2Mul@private_mlp_5/batch_normalization_15/Cast/ReadVariableOp:value:06private_mlp_5/batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
2private_mlp_5/batch_normalization_15/batchnorm/subSubBprivate_mlp_5/batch_normalization_15/Cast_2/ReadVariableOp:value:08private_mlp_5/batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
4private_mlp_5/batch_normalization_15/batchnorm/add_1AddV28private_mlp_5/batch_normalization_15/batchnorm/mul_1:z:06private_mlp_5/batch_normalization_15/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
,private_mlp_5/dense_21/MatMul/ReadVariableOpReadVariableOp5private_mlp_5_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
private_mlp_5/dense_21/MatMulMatMul8private_mlp_5/batch_normalization_15/batchnorm/add_1:z:04private_mlp_5/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-private_mlp_5/dense_21/BiasAdd/ReadVariableOpReadVariableOp6private_mlp_5_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
private_mlp_5/dense_21/BiasAddBiasAdd'private_mlp_5/dense_21/MatMul:product:05private_mlp_5/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
private_mlp_5/dense_21/TanhTanh'private_mlp_5/dense_21/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
8private_mlp_5/batch_normalization_16/Cast/ReadVariableOpReadVariableOpAprivate_mlp_5_batch_normalization_16_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:private_mlp_5/batch_normalization_16/Cast_1/ReadVariableOpReadVariableOpCprivate_mlp_5_batch_normalization_16_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:private_mlp_5/batch_normalization_16/Cast_2/ReadVariableOpReadVariableOpCprivate_mlp_5_batch_normalization_16_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:private_mlp_5/batch_normalization_16/Cast_3/ReadVariableOpReadVariableOpCprivate_mlp_5_batch_normalization_16_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0y
4private_mlp_5/batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
2private_mlp_5/batch_normalization_16/batchnorm/addAddV2Bprivate_mlp_5/batch_normalization_16/Cast_1/ReadVariableOp:value:0=private_mlp_5/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
4private_mlp_5/batch_normalization_16/batchnorm/RsqrtRsqrt6private_mlp_5/batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes	
:��
2private_mlp_5/batch_normalization_16/batchnorm/mulMul8private_mlp_5/batch_normalization_16/batchnorm/Rsqrt:y:0Bprivate_mlp_5/batch_normalization_16/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
4private_mlp_5/batch_normalization_16/batchnorm/mul_1Mulprivate_mlp_5/dense_21/Tanh:y:06private_mlp_5/batch_normalization_16/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
4private_mlp_5/batch_normalization_16/batchnorm/mul_2Mul@private_mlp_5/batch_normalization_16/Cast/ReadVariableOp:value:06private_mlp_5/batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
2private_mlp_5/batch_normalization_16/batchnorm/subSubBprivate_mlp_5/batch_normalization_16/Cast_2/ReadVariableOp:value:08private_mlp_5/batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
4private_mlp_5/batch_normalization_16/batchnorm/add_1AddV28private_mlp_5/batch_normalization_16/batchnorm/mul_1:z:06private_mlp_5/batch_normalization_16/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
,private_mlp_5/dense_22/MatMul/ReadVariableOpReadVariableOp5private_mlp_5_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
private_mlp_5/dense_22/MatMulMatMul8private_mlp_5/batch_normalization_16/batchnorm/add_1:z:04private_mlp_5/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-private_mlp_5/dense_22/BiasAdd/ReadVariableOpReadVariableOp6private_mlp_5_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
private_mlp_5/dense_22/BiasAddBiasAdd'private_mlp_5/dense_22/MatMul:product:05private_mlp_5/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
private_mlp_5/dense_22/TanhTanh'private_mlp_5/dense_22/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
8private_mlp_5/batch_normalization_17/Cast/ReadVariableOpReadVariableOpAprivate_mlp_5_batch_normalization_17_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:private_mlp_5/batch_normalization_17/Cast_1/ReadVariableOpReadVariableOpCprivate_mlp_5_batch_normalization_17_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:private_mlp_5/batch_normalization_17/Cast_2/ReadVariableOpReadVariableOpCprivate_mlp_5_batch_normalization_17_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:private_mlp_5/batch_normalization_17/Cast_3/ReadVariableOpReadVariableOpCprivate_mlp_5_batch_normalization_17_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0y
4private_mlp_5/batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
2private_mlp_5/batch_normalization_17/batchnorm/addAddV2Bprivate_mlp_5/batch_normalization_17/Cast_1/ReadVariableOp:value:0=private_mlp_5/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
4private_mlp_5/batch_normalization_17/batchnorm/RsqrtRsqrt6private_mlp_5/batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes	
:��
2private_mlp_5/batch_normalization_17/batchnorm/mulMul8private_mlp_5/batch_normalization_17/batchnorm/Rsqrt:y:0Bprivate_mlp_5/batch_normalization_17/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
4private_mlp_5/batch_normalization_17/batchnorm/mul_1Mulprivate_mlp_5/dense_22/Tanh:y:06private_mlp_5/batch_normalization_17/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
4private_mlp_5/batch_normalization_17/batchnorm/mul_2Mul@private_mlp_5/batch_normalization_17/Cast/ReadVariableOp:value:06private_mlp_5/batch_normalization_17/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
2private_mlp_5/batch_normalization_17/batchnorm/subSubBprivate_mlp_5/batch_normalization_17/Cast_2/ReadVariableOp:value:08private_mlp_5/batch_normalization_17/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
4private_mlp_5/batch_normalization_17/batchnorm/add_1AddV28private_mlp_5/batch_normalization_17/batchnorm/mul_1:z:06private_mlp_5/batch_normalization_17/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
,private_mlp_5/dense_23/MatMul/ReadVariableOpReadVariableOp5private_mlp_5_dense_23_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
private_mlp_5/dense_23/MatMulMatMul8private_mlp_5/batch_normalization_17/batchnorm/add_1:z:04private_mlp_5/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-private_mlp_5/dense_23/BiasAdd/ReadVariableOpReadVariableOp6private_mlp_5_dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
private_mlp_5/dense_23/BiasAddBiasAdd'private_mlp_5/dense_23/MatMul:product:05private_mlp_5/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
IdentityIdentity'private_mlp_5/dense_23/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������	
NoOpNoOp9^private_mlp_5/batch_normalization_15/Cast/ReadVariableOp;^private_mlp_5/batch_normalization_15/Cast_1/ReadVariableOp;^private_mlp_5/batch_normalization_15/Cast_2/ReadVariableOp;^private_mlp_5/batch_normalization_15/Cast_3/ReadVariableOp9^private_mlp_5/batch_normalization_16/Cast/ReadVariableOp;^private_mlp_5/batch_normalization_16/Cast_1/ReadVariableOp;^private_mlp_5/batch_normalization_16/Cast_2/ReadVariableOp;^private_mlp_5/batch_normalization_16/Cast_3/ReadVariableOp9^private_mlp_5/batch_normalization_17/Cast/ReadVariableOp;^private_mlp_5/batch_normalization_17/Cast_1/ReadVariableOp;^private_mlp_5/batch_normalization_17/Cast_2/ReadVariableOp;^private_mlp_5/batch_normalization_17/Cast_3/ReadVariableOp.^private_mlp_5/dense_20/BiasAdd/ReadVariableOp-^private_mlp_5/dense_20/MatMul/ReadVariableOp.^private_mlp_5/dense_21/BiasAdd/ReadVariableOp-^private_mlp_5/dense_21/MatMul/ReadVariableOp.^private_mlp_5/dense_22/BiasAdd/ReadVariableOp-^private_mlp_5/dense_22/MatMul/ReadVariableOp.^private_mlp_5/dense_23/BiasAdd/ReadVariableOp-^private_mlp_5/dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : 2t
8private_mlp_5/batch_normalization_15/Cast/ReadVariableOp8private_mlp_5/batch_normalization_15/Cast/ReadVariableOp2x
:private_mlp_5/batch_normalization_15/Cast_1/ReadVariableOp:private_mlp_5/batch_normalization_15/Cast_1/ReadVariableOp2x
:private_mlp_5/batch_normalization_15/Cast_2/ReadVariableOp:private_mlp_5/batch_normalization_15/Cast_2/ReadVariableOp2x
:private_mlp_5/batch_normalization_15/Cast_3/ReadVariableOp:private_mlp_5/batch_normalization_15/Cast_3/ReadVariableOp2t
8private_mlp_5/batch_normalization_16/Cast/ReadVariableOp8private_mlp_5/batch_normalization_16/Cast/ReadVariableOp2x
:private_mlp_5/batch_normalization_16/Cast_1/ReadVariableOp:private_mlp_5/batch_normalization_16/Cast_1/ReadVariableOp2x
:private_mlp_5/batch_normalization_16/Cast_2/ReadVariableOp:private_mlp_5/batch_normalization_16/Cast_2/ReadVariableOp2x
:private_mlp_5/batch_normalization_16/Cast_3/ReadVariableOp:private_mlp_5/batch_normalization_16/Cast_3/ReadVariableOp2t
8private_mlp_5/batch_normalization_17/Cast/ReadVariableOp8private_mlp_5/batch_normalization_17/Cast/ReadVariableOp2x
:private_mlp_5/batch_normalization_17/Cast_1/ReadVariableOp:private_mlp_5/batch_normalization_17/Cast_1/ReadVariableOp2x
:private_mlp_5/batch_normalization_17/Cast_2/ReadVariableOp:private_mlp_5/batch_normalization_17/Cast_2/ReadVariableOp2x
:private_mlp_5/batch_normalization_17/Cast_3/ReadVariableOp:private_mlp_5/batch_normalization_17/Cast_3/ReadVariableOp2^
-private_mlp_5/dense_20/BiasAdd/ReadVariableOp-private_mlp_5/dense_20/BiasAdd/ReadVariableOp2\
,private_mlp_5/dense_20/MatMul/ReadVariableOp,private_mlp_5/dense_20/MatMul/ReadVariableOp2^
-private_mlp_5/dense_21/BiasAdd/ReadVariableOp-private_mlp_5/dense_21/BiasAdd/ReadVariableOp2\
,private_mlp_5/dense_21/MatMul/ReadVariableOp,private_mlp_5/dense_21/MatMul/ReadVariableOp2^
-private_mlp_5/dense_22/BiasAdd/ReadVariableOp-private_mlp_5/dense_22/BiasAdd/ReadVariableOp2\
,private_mlp_5/dense_22/MatMul/ReadVariableOp,private_mlp_5/dense_22/MatMul/ReadVariableOp2^
-private_mlp_5/dense_23/BiasAdd/ReadVariableOp-private_mlp_5/dense_23/BiasAdd/ReadVariableOp2\
,private_mlp_5/dense_23/MatMul/ReadVariableOp,private_mlp_5/dense_23/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�
H
,__inference_flatten_11_layer_call_fn_6789804

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
G__inference_flatten_11_layer_call_and_return_conditional_losses_6788894`
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
�
E__inference_dense_23_layer_call_and_return_conditional_losses_6790142

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
�

�
E__inference_dense_21_layer_call_and_return_conditional_losses_6788942

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
E__inference_dense_22_layer_call_and_return_conditional_losses_6788968

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
J__inference_private_mlp_5_layer_call_and_return_conditional_losses_6789423
input_1$
dense_20_6789375:
��
dense_20_6789377:	�-
batch_normalization_15_6789380:	�-
batch_normalization_15_6789382:	�-
batch_normalization_15_6789384:	�-
batch_normalization_15_6789386:	�$
dense_21_6789389:
��
dense_21_6789391:	�-
batch_normalization_16_6789394:	�-
batch_normalization_16_6789396:	�-
batch_normalization_16_6789398:	�-
batch_normalization_16_6789400:	�$
dense_22_6789403:
��
dense_22_6789405:	�-
batch_normalization_17_6789408:	�-
batch_normalization_17_6789410:	�-
batch_normalization_17_6789412:	�-
batch_normalization_17_6789414:	�#
dense_23_6789417:	�
dense_23_6789419:
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
valueB"      �
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

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
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_10_layer_call_and_return_conditional_losses_6788882f
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
flatten_11/PartitionedCallPartitionedCallstrided_slice_1:output:0*
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
G__inference_flatten_11_layer_call_and_return_conditional_losses_6788894�
concatenate_5/PartitionedCallPartitionedCall#flatten_10/PartitionedCall:output:0#flatten_11/PartitionedCall:output:0*
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
GPU 2J 8� *S
fNRL
J__inference_concatenate_5_layer_call_and_return_conditional_losses_6788903�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0dense_20_6789375dense_20_6789377*
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
E__inference_dense_20_layer_call_and_return_conditional_losses_6788916�
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0batch_normalization_15_6789380batch_normalization_15_6789382batch_normalization_15_6789384batch_normalization_15_6789386*
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
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_6788690�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0dense_21_6789389dense_21_6789391*
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
E__inference_dense_21_layer_call_and_return_conditional_losses_6788942�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0batch_normalization_16_6789394batch_normalization_16_6789396batch_normalization_16_6789398batch_normalization_16_6789400*
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
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_6788772�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0dense_22_6789403dense_22_6789405*
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
E__inference_dense_22_layer_call_and_return_conditional_losses_6788968�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0batch_normalization_17_6789408batch_normalization_17_6789410batch_normalization_17_6789412batch_normalization_17_6789414*
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
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_6788854�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0dense_23_6789417dense_23_6789419*
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
E__inference_dense_23_layer_call_and_return_conditional_losses_6788993x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
8__inference_batch_normalization_17_layer_call_fn_6790009

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
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_6788854p
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
�>
�	
J__inference_private_mlp_5_layer_call_and_return_conditional_losses_6789211

inputs$
dense_20_6789163:
��
dense_20_6789165:	�-
batch_normalization_15_6789168:	�-
batch_normalization_15_6789170:	�-
batch_normalization_15_6789172:	�-
batch_normalization_15_6789174:	�$
dense_21_6789177:
��
dense_21_6789179:	�-
batch_normalization_16_6789182:	�-
batch_normalization_16_6789184:	�-
batch_normalization_16_6789186:	�-
batch_normalization_16_6789188:	�$
dense_22_6789191:
��
dense_22_6789193:	�-
batch_normalization_17_6789196:	�-
batch_normalization_17_6789198:	�-
batch_normalization_17_6789200:	�-
batch_normalization_17_6789202:	�#
dense_23_6789205:	�
dense_23_6789207:
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
valueB"      �
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

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
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_10_layer_call_and_return_conditional_losses_6788882f
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
flatten_11/PartitionedCallPartitionedCallstrided_slice_1:output:0*
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
G__inference_flatten_11_layer_call_and_return_conditional_losses_6788894�
concatenate_5/PartitionedCallPartitionedCall#flatten_10/PartitionedCall:output:0#flatten_11/PartitionedCall:output:0*
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
GPU 2J 8� *S
fNRL
J__inference_concatenate_5_layer_call_and_return_conditional_losses_6788903�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0dense_20_6789163dense_20_6789165*
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
E__inference_dense_20_layer_call_and_return_conditional_losses_6788916�
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0batch_normalization_15_6789168batch_normalization_15_6789170batch_normalization_15_6789172batch_normalization_15_6789174*
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
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_6788690�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0dense_21_6789177dense_21_6789179*
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
E__inference_dense_21_layer_call_and_return_conditional_losses_6788942�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0batch_normalization_16_6789182batch_normalization_16_6789184batch_normalization_16_6789186batch_normalization_16_6789188*
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
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_6788772�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0dense_22_6789191dense_22_6789193*
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
E__inference_dense_22_layer_call_and_return_conditional_losses_6788968�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0batch_normalization_17_6789196batch_normalization_17_6789198batch_normalization_17_6789200batch_normalization_17_6789202*
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
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_6788854�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0dense_23_6789205dense_23_6789207*
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
E__inference_dense_23_layer_call_and_return_conditional_losses_6788993x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_16_layer_call_fn_6789929

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
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_6788772p
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
�
�
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_6788725

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
G__inference_flatten_11_layer_call_and_return_conditional_losses_6789810

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
�$
�
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_6788690

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
%__inference_signature_wrapper_6789470
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
GPU 2J 8� *+
f&R$
"__inference__wrapped_model_6788619o
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
t
J__inference_concatenate_5_layer_call_and_return_conditional_losses_6788903

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
 
_user_specified_nameinputs
�	
�
E__inference_dense_23_layer_call_and_return_conditional_losses_6788993

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
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_6790029

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
�4
�
 __inference__traced_save_6790225
file_prefixI
Esavev2_private_mlp_5_batch_normalization_15_gamma_read_readvariableopH
Dsavev2_private_mlp_5_batch_normalization_15_beta_read_readvariableopO
Ksavev2_private_mlp_5_batch_normalization_15_moving_mean_read_readvariableopS
Osavev2_private_mlp_5_batch_normalization_15_moving_variance_read_readvariableopI
Esavev2_private_mlp_5_batch_normalization_16_gamma_read_readvariableopH
Dsavev2_private_mlp_5_batch_normalization_16_beta_read_readvariableopO
Ksavev2_private_mlp_5_batch_normalization_16_moving_mean_read_readvariableopS
Osavev2_private_mlp_5_batch_normalization_16_moving_variance_read_readvariableopI
Esavev2_private_mlp_5_batch_normalization_17_gamma_read_readvariableopH
Dsavev2_private_mlp_5_batch_normalization_17_beta_read_readvariableopO
Ksavev2_private_mlp_5_batch_normalization_17_moving_mean_read_readvariableopS
Osavev2_private_mlp_5_batch_normalization_17_moving_variance_read_readvariableop<
8savev2_private_mlp_5_dense_20_kernel_read_readvariableop:
6savev2_private_mlp_5_dense_20_bias_read_readvariableop<
8savev2_private_mlp_5_dense_21_kernel_read_readvariableop:
6savev2_private_mlp_5_dense_21_bias_read_readvariableop<
8savev2_private_mlp_5_dense_22_kernel_read_readvariableop:
6savev2_private_mlp_5_dense_22_bias_read_readvariableop<
8savev2_private_mlp_5_dense_23_kernel_read_readvariableop:
6savev2_private_mlp_5_dense_23_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Esavev2_private_mlp_5_batch_normalization_15_gamma_read_readvariableopDsavev2_private_mlp_5_batch_normalization_15_beta_read_readvariableopKsavev2_private_mlp_5_batch_normalization_15_moving_mean_read_readvariableopOsavev2_private_mlp_5_batch_normalization_15_moving_variance_read_readvariableopEsavev2_private_mlp_5_batch_normalization_16_gamma_read_readvariableopDsavev2_private_mlp_5_batch_normalization_16_beta_read_readvariableopKsavev2_private_mlp_5_batch_normalization_16_moving_mean_read_readvariableopOsavev2_private_mlp_5_batch_normalization_16_moving_variance_read_readvariableopEsavev2_private_mlp_5_batch_normalization_17_gamma_read_readvariableopDsavev2_private_mlp_5_batch_normalization_17_beta_read_readvariableopKsavev2_private_mlp_5_batch_normalization_17_moving_mean_read_readvariableopOsavev2_private_mlp_5_batch_normalization_17_moving_variance_read_readvariableop8savev2_private_mlp_5_dense_20_kernel_read_readvariableop6savev2_private_mlp_5_dense_20_bias_read_readvariableop8savev2_private_mlp_5_dense_21_kernel_read_readvariableop6savev2_private_mlp_5_dense_21_bias_read_readvariableop8savev2_private_mlp_5_dense_22_kernel_read_readvariableop6savev2_private_mlp_5_dense_22_bias_read_readvariableop8savev2_private_mlp_5_dense_23_kernel_read_readvariableop6savev2_private_mlp_5_dense_23_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
�
�
*__inference_dense_23_layer_call_fn_6790132

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
E__inference_dense_23_layer_call_and_return_conditional_losses_6788993o
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
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
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
/__inference_private_mlp_5_layer_call_fn_6789043
/__inference_private_mlp_5_layer_call_fn_6789515
/__inference_private_mlp_5_layer_call_fn_6789560
/__inference_private_mlp_5_layer_call_fn_6789299�
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
J__inference_private_mlp_5_layer_call_and_return_conditional_losses_6789653
J__inference_private_mlp_5_layer_call_and_return_conditional_losses_6789788
J__inference_private_mlp_5_layer_call_and_return_conditional_losses_6789361
J__inference_private_mlp_5_layer_call_and_return_conditional_losses_6789423�
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
"__inference__wrapped_model_6788619input_1"�
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
9:7�2*private_mlp_5/batch_normalization_15/gamma
8:6�2)private_mlp_5/batch_normalization_15/beta
A:?� (20private_mlp_5/batch_normalization_15/moving_mean
E:C� (24private_mlp_5/batch_normalization_15/moving_variance
9:7�2*private_mlp_5/batch_normalization_16/gamma
8:6�2)private_mlp_5/batch_normalization_16/beta
A:?� (20private_mlp_5/batch_normalization_16/moving_mean
E:C� (24private_mlp_5/batch_normalization_16/moving_variance
9:7�2*private_mlp_5/batch_normalization_17/gamma
8:6�2)private_mlp_5/batch_normalization_17/beta
A:?� (20private_mlp_5/batch_normalization_17/moving_mean
E:C� (24private_mlp_5/batch_normalization_17/moving_variance
1:/
��2private_mlp_5/dense_20/kernel
*:(�2private_mlp_5/dense_20/bias
1:/
��2private_mlp_5/dense_21/kernel
*:(�2private_mlp_5/dense_21/bias
1:/
��2private_mlp_5/dense_22/kernel
*:(�2private_mlp_5/dense_22/bias
0:.	�2private_mlp_5/dense_23/kernel
):'2private_mlp_5/dense_23/bias
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
/__inference_private_mlp_5_layer_call_fn_6789043input_1"�
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
/__inference_private_mlp_5_layer_call_fn_6789515inputs"�
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
/__inference_private_mlp_5_layer_call_fn_6789560inputs"�
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
/__inference_private_mlp_5_layer_call_fn_6789299input_1"�
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
J__inference_private_mlp_5_layer_call_and_return_conditional_losses_6789653inputs"�
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
J__inference_private_mlp_5_layer_call_and_return_conditional_losses_6789788inputs"�
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
J__inference_private_mlp_5_layer_call_and_return_conditional_losses_6789361input_1"�
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
J__inference_private_mlp_5_layer_call_and_return_conditional_losses_6789423input_1"�
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
,__inference_flatten_10_layer_call_fn_6789793�
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
G__inference_flatten_10_layer_call_and_return_conditional_losses_6789799�
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
,__inference_flatten_11_layer_call_fn_6789804�
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
G__inference_flatten_11_layer_call_and_return_conditional_losses_6789810�
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
/__inference_concatenate_5_layer_call_fn_6789816�
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
J__inference_concatenate_5_layer_call_and_return_conditional_losses_6789823�
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
8__inference_batch_normalization_15_layer_call_fn_6789836
8__inference_batch_normalization_15_layer_call_fn_6789849�
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
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_6789869
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_6789903�
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
8__inference_batch_normalization_16_layer_call_fn_6789916
8__inference_batch_normalization_16_layer_call_fn_6789929�
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
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_6789949
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_6789983�
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
8__inference_batch_normalization_17_layer_call_fn_6789996
8__inference_batch_normalization_17_layer_call_fn_6790009�
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
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_6790029
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_6790063�
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
*__inference_dense_20_layer_call_fn_6790072�
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
E__inference_dense_20_layer_call_and_return_conditional_losses_6790083�
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
*__inference_dense_21_layer_call_fn_6790092�
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
E__inference_dense_21_layer_call_and_return_conditional_losses_6790103�
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
*__inference_dense_22_layer_call_fn_6790112�
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
E__inference_dense_22_layer_call_and_return_conditional_losses_6790123�
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
*__inference_dense_23_layer_call_fn_6790132�
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
E__inference_dense_23_layer_call_and_return_conditional_losses_6790142�
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
%__inference_signature_wrapper_6789470input_1"�
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
,__inference_flatten_10_layer_call_fn_6789793inputs"�
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
G__inference_flatten_10_layer_call_and_return_conditional_losses_6789799inputs"�
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
,__inference_flatten_11_layer_call_fn_6789804inputs"�
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
G__inference_flatten_11_layer_call_and_return_conditional_losses_6789810inputs"�
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
/__inference_concatenate_5_layer_call_fn_6789816inputs/0inputs/1"�
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
J__inference_concatenate_5_layer_call_and_return_conditional_losses_6789823inputs/0inputs/1"�
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
8__inference_batch_normalization_15_layer_call_fn_6789836inputs"�
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
8__inference_batch_normalization_15_layer_call_fn_6789849inputs"�
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
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_6789869inputs"�
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
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_6789903inputs"�
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
8__inference_batch_normalization_16_layer_call_fn_6789916inputs"�
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
8__inference_batch_normalization_16_layer_call_fn_6789929inputs"�
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
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_6789949inputs"�
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
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_6789983inputs"�
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
8__inference_batch_normalization_17_layer_call_fn_6789996inputs"�
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
8__inference_batch_normalization_17_layer_call_fn_6790009inputs"�
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
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_6790029inputs"�
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
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_6790063inputs"�
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
*__inference_dense_20_layer_call_fn_6790072inputs"�
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
E__inference_dense_20_layer_call_and_return_conditional_losses_6790083inputs"�
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
*__inference_dense_21_layer_call_fn_6790092inputs"�
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
E__inference_dense_21_layer_call_and_return_conditional_losses_6790103inputs"�
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
*__inference_dense_22_layer_call_fn_6790112inputs"�
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
E__inference_dense_22_layer_call_and_return_conditional_losses_6790123inputs"�
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
*__inference_dense_23_layer_call_fn_6790132inputs"�
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
E__inference_dense_23_layer_call_and_return_conditional_losses_6790142inputs"�
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
"__inference__wrapped_model_6788619� !"#$%&'8�5
.�+
)�&
input_1���������
� "3�0
.
output_1"�
output_1����������
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_6789869d4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_6789903d4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
8__inference_batch_normalization_15_layer_call_fn_6789836W4�1
*�'
!�
inputs����������
p 
� "������������
8__inference_batch_normalization_15_layer_call_fn_6789849W4�1
*�'
!�
inputs����������
p
� "������������
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_6789949d4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_6789983d4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
8__inference_batch_normalization_16_layer_call_fn_6789916W4�1
*�'
!�
inputs����������
p 
� "������������
8__inference_batch_normalization_16_layer_call_fn_6789929W4�1
*�'
!�
inputs����������
p
� "������������
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_6790029d4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_6790063d4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
8__inference_batch_normalization_17_layer_call_fn_6789996W4�1
*�'
!�
inputs����������
p 
� "������������
8__inference_batch_normalization_17_layer_call_fn_6790009W4�1
*�'
!�
inputs����������
p
� "������������
J__inference_concatenate_5_layer_call_and_return_conditional_losses_6789823�Z�W
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
/__inference_concatenate_5_layer_call_fn_6789816wZ�W
P�M
K�H
"�
inputs/0���������@
"�
inputs/1���������@
� "������������
E__inference_dense_20_layer_call_and_return_conditional_losses_6790083^ !0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� 
*__inference_dense_20_layer_call_fn_6790072Q !0�-
&�#
!�
inputs����������
� "������������
E__inference_dense_21_layer_call_and_return_conditional_losses_6790103^"#0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� 
*__inference_dense_21_layer_call_fn_6790092Q"#0�-
&�#
!�
inputs����������
� "������������
E__inference_dense_22_layer_call_and_return_conditional_losses_6790123^$%0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� 
*__inference_dense_22_layer_call_fn_6790112Q$%0�-
&�#
!�
inputs����������
� "������������
E__inference_dense_23_layer_call_and_return_conditional_losses_6790142]&'0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� ~
*__inference_dense_23_layer_call_fn_6790132P&'0�-
&�#
!�
inputs����������
� "�����������
G__inference_flatten_10_layer_call_and_return_conditional_losses_6789799\3�0
)�&
$�!
inputs���������
� "%�"
�
0���������@
� 
,__inference_flatten_10_layer_call_fn_6789793O3�0
)�&
$�!
inputs���������
� "����������@�
G__inference_flatten_11_layer_call_and_return_conditional_losses_6789810\3�0
)�&
$�!
inputs���������
� "%�"
�
0���������@
� 
,__inference_flatten_11_layer_call_fn_6789804O3�0
)�&
$�!
inputs���������
� "����������@�
J__inference_private_mlp_5_layer_call_and_return_conditional_losses_6789361{ !"#$%&'<�9
2�/
)�&
input_1���������
p 
� "%�"
�
0���������
� �
J__inference_private_mlp_5_layer_call_and_return_conditional_losses_6789423{ !"#$%&'<�9
2�/
)�&
input_1���������
p
� "%�"
�
0���������
� �
J__inference_private_mlp_5_layer_call_and_return_conditional_losses_6789653z !"#$%&';�8
1�.
(�%
inputs���������
p 
� "%�"
�
0���������
� �
J__inference_private_mlp_5_layer_call_and_return_conditional_losses_6789788z !"#$%&';�8
1�.
(�%
inputs���������
p
� "%�"
�
0���������
� �
/__inference_private_mlp_5_layer_call_fn_6789043n !"#$%&'<�9
2�/
)�&
input_1���������
p 
� "�����������
/__inference_private_mlp_5_layer_call_fn_6789299n !"#$%&'<�9
2�/
)�&
input_1���������
p
� "�����������
/__inference_private_mlp_5_layer_call_fn_6789515m !"#$%&';�8
1�.
(�%
inputs���������
p 
� "�����������
/__inference_private_mlp_5_layer_call_fn_6789560m !"#$%&';�8
1�.
(�%
inputs���������
p
� "�����������
%__inference_signature_wrapper_6789470� !"#$%&'C�@
� 
9�6
4
input_1)�&
input_1���������"3�0
.
output_1"�
output_1���������